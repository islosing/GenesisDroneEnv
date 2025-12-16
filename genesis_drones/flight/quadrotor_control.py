import numpy as np
import torch
import roma
import yaml
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """
    Quadrotor trajectory tracking controller based on https://ieeexplore.ieee.org/document/5717652 
    with Hopf Fibration based attitude control from Control of Quadrotors Using the Hopf Fibration on SO(3)
    
    """


    def __init__(self, yaml_path: str):

        with open(yaml_path, "r") as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        # =====================
        # Inertia
        # =====================
        self.mass = cfg["inertia"]["mass"]
        self.Ixx  = cfg["inertia"]["Ixx"]
        self.Iyy  = cfg["inertia"]["Iyy"]
        self.Izz  = cfg["inertia"]["Izz"]
        self.Ixy  = cfg["inertia"]["Ixy"]
        self.Ixz  = cfg["inertia"]["Ixz"]
        self.Iyz  = cfg["inertia"]["Iyz"]

        self.inertia = np.array([
            [self.Ixx, self.Ixy, self.Ixz],
            [self.Ixy, self.Iyy, self.Iyz],
            [self.Ixz, self.Iyz, self.Izz]
        ])

        self.g = 9.81

        # =====================
        # Gains
        # =====================
        self.kp_pos = np.array([16, 16, 16])
        self.kd_pos = np.array([5, 5, 7])
        self.kp_att = 20.44
        self.kd_att = 0.1
        self.kp_vel = 0.1 * self.kp_pos

        # =====================
        # Rotor geometry
        # =====================
        d = cfg["arm_length"]

        self.num_rotors = cfg["geometry"]["num_rotors"]

        self.rotor_pos = {
            k: d * np.array(v, dtype=float)
            for k, v in cfg["geometry"]["rotor_pos"].items()
        }

        self.rotor_dir = np.array(
            cfg["geometry"]["rotor_directions"], dtype=float
        )

        # =====================
        # Rotor / motor parameters
        # =====================
        self.k_eta = cfg["rotor"]["k_eta"]
        self.k_m   = cfg["rotor"]["k_m"]

        # =====================
        # Allocation matrix
        # =====================
        k = self.k_m / self.k_eta

        self.f_to_TM = np.vstack((
            np.ones((1, self.num_rotors)),
            np.hstack([
                np.cross(self.rotor_pos[key], np.array([0, 0, 1]))[:2].reshape(-1, 1)
                for key in self.rotor_pos
            ]),
            (k * self.rotor_dir).reshape(1, -1)
        ))

        self.TM_to_f = np.linalg.inv(self.f_to_TM)

    # ------------------------
    # HFCA QUATERNION MULTIPLY
    # ------------------------
    @staticmethod
    def quat_mul(q1, q2):
        w1,x1,y1,z1 = q1
        w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-9: return v
        return v / n

    @staticmethod
    def vee(S):
        return np.array([-S[1,2], S[0,2], -S[0,1]])

    # ============================================================
    #                       MAIN UPDATE()
    # ============================================================
    def update(self, t, state, flat):

        # --------------------
        # 1. DESIRED ACC ζ
        # --------------------
        pos_err = state['x'] - flat['x']
        vel_err = state['v'] - flat['x_dot']

        zeta = (- self.kp_pos * pos_err
                - self.kd_pos * vel_err
                + flat['x_ddot']
                + np.array([0,0,self.g]))

        F_des = self.mass * zeta

        # Current attitude
        R = Rotation.from_quat(state['q']).as_matrix()
        b3 = R @ np.array([0,0,1])

        # Scalar thrust (same as original SE3 controller)
        u1 = np.dot(F_des, b3)

        # ---------------------------
        # 2. HOPF FIBRATION ATTITUDE
        # ---------------------------
        eps = 1e-6
        zeta_norm = np.linalg.norm(zeta)

        if zeta_norm < eps:
            # fallback to classic SE3 control
            b3_des = np.array([0,0,1])
            yaw = flat['yaw']
            c1 = np.array([np.cos(yaw), np.sin(yaw), 0])
            b2 = self.normalize(np.cross(b3_des, c1))
            b1 = np.cross(b2, b3_des)
            R_des = np.stack([b1, b2, b3_des]).T
            w_des = np.array([0,0, flat['yaw_dot']])

        else:

            # S^2 PART: thrust direction
            s = zeta / zeta_norm
            a, b, c = s

            # s_dot via normalization derivative
            zeta_dot = flat['x_dddot']
            I3 = np.eye(3)
            P = (zeta_norm**2 * I3 - np.outer(zeta, zeta)) / (zeta_norm**3)
            s_dot = P @ zeta_dot
            a_dot, b_dot, c_dot = s_dot

            # --- HOPF QUATERNION q_abc ---
            denom = np.sqrt(2*(1+c))
            if denom < eps: denom = eps

            q_abc = np.array([
                (1+c)/denom,
                -b/denom,
                a/denom,
                0.0
            ])  # [w,x,y,z]

            # --- YAW QUATERNION q_ψ ---
            psi = flat['yaw']
            psi_dot = flat['yaw_dot']
            half = 0.5 * psi
            q_psi = np.array([np.cos(half), 0, 0, np.sin(half)])

            # total quaternion
            q_tot = self.quat_mul(q_abc, q_psi)

            # convert to scipy format [x,y,z,w]
            q_scipy = np.array([q_tot[1], q_tot[2], q_tot[3], q_tot[0]])
            R_des = Rotation.from_quat(q_scipy).as_matrix()

            # =========================
            # 3. HFCA ω_des FORMULAS
            # =========================
            one_plus_c = max(1+c, eps)
            sinp = np.sin(psi)
            cosp = np.cos(psi)

            omega1 = ( sinp*a_dot - cosp*b_dot
                      - (a*sinp - b*cosp)*(c_dot/one_plus_c) )

            omega2 = ( cosp*a_dot + sinp*b_dot
                      - (a*cosp + b*sinp)*(c_dot/one_plus_c) )

            omega3 = ( (b*a_dot - a*b_dot)/one_plus_c + psi_dot )

            w_des = np.array([omega1, omega2, omega3])

        # -----------------------
        # 4. ATTITUDE PD CONTROL
        # -----------------------
        S_err = 0.5*(R_des.T@R - R.T@R_des)
        att_err = self.vee(S_err)
        w_err  = state['w'] - w_des

        u2 = ( self.inertia @ (-self.kp_att * att_err - self.kd_att * w_err)
               + np.cross(state['w'], self.inertia @ state['w']) )

        # body-rate command (optional)
        cmd_w = -self.kp_att*att_err - self.kd_att*w_err

        # -----------------------
        # 5. MOTOR ALLOCATION
        # -----------------------
        TM = np.array([u1, u2[0], u2[1], u2[2]])
        rotor_thrusts = self.TM_to_f @ TM

        motor_speeds = rotor_thrusts / self.k_eta
        motor_speeds = np.sign(motor_speeds)*np.sqrt(np.abs(motor_speeds))

        # -----------------------
        # OUTPUTS
        # -----------------------
        cmd_q = Rotation.from_matrix(R_des).as_quat()
        #print("cmd_q:", cmd_q)
        return {
            'cmd_motor_speeds': motor_speeds,
            'cmd_motor_thrusts': rotor_thrusts,
            'cmd_thrust': u1,
            'cmd_moment': u2,
            'cmd_q': cmd_q,
            'cmd_w': cmd_w,
            'cmd_v': -self.kp_vel*pos_err + flat['x_dot'],
            'cmd_acc': F_des/self.mass
        }
    
class BatchedSE3Control(object):
    def __init__(self, batch_params, num_drones, device, kp_pos=None, kd_pos=None, kp_att=None, kd_att=None):
        '''
        batch_params, BatchedMultirotorParams object 
        num_drones: int, number of drones in the batch
        device: torch.device("cpu") or torch.device("cuda")

        kp_pos: torch.Tensor of shape (num_drones, 3)
        kd_pos: torch.Tensor of shape (num_drones, 3)
        kp_att: torch.Tensor of shape (num_drones, 1)
        kd_att: torch.Tensor of shape (num_drones, 1)
        '''
        assert batch_params.device == device
        self.params = batch_params
        self.device = device
        # Quadrotor physical parameters

        # Gains
        if kp_pos is None:
            self.kp_pos = torch.tensor([6.5, 6.5, 15], device=self.device).repeat(num_drones, 1).double()
        else:
            self.kp_pos = kp_pos.to(self.device).double()
        if kd_pos is None:
            self.kd_pos = torch.tensor([4.0, 4.0, 9], device=self.device).repeat(num_drones, 1).double()
        else:
            self.kd_pos = kd_pos.to(self.device).double()
        if kp_att is None:
            self.kp_att = torch.tensor([544], device=device).repeat(num_drones, 1).double()
        else:
            self.kp_att = kp_att.to(self.device).double()
            if len(self.kp_att.shape) < 2:
                self.kp_att = self.kp_att.unsqueeze(-1)
        if kd_att is None:
            self.kd_att = torch.tensor([46.64], device=device).repeat(num_drones, 1).double()
        else:
            self.kd_att = kd_att.to(self.device).double()
            if len(self.kd_att.shape) < 2:
                self.kd_att = self.kd_att.unsqueeze(-1)

        self.kp_vel = 0.1 * self.kp_pos

    def normalize(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True)

    def update(self, t, states, flat_outputs, idxs=None):
        '''
        Computes a batch of control outputs for the drones specified by idxs
        :param states: a dictionary of pytorch tensors containing the states of the quadrotors (expects double precision)
        :param flat_outputs: a dictionary of pytorch tensors containing the reference trajectories for each quad. (expects double precision)
        :param idxs: a list of which drones to update
        :return:
        '''
        if idxs is None:
            idxs = [i for i in range(states['x'].shape[0])]
        pos_err = states['x'][idxs].double() - flat_outputs['x'][idxs].double()
        dpos_err = states['v'][idxs].double() - flat_outputs['x_dot'][idxs].double()

        F_des = self.params.mass[idxs] * (-self.kp_pos[idxs] * pos_err
                             - self.kd_pos[idxs] * dpos_err
                             + flat_outputs['x_ddot'][idxs].double()
                             + torch.tensor([0, 0, self.params.g], device=self.device))


        R = roma.unitquat_to_rotmat(states['q'][idxs]).double()
        b3 = R @ torch.tensor([0.0, 0.0, 1.0], device=self.device).double()
        u1 = torch.sum(F_des * b3, dim=-1).double()

        b3_des = self.normalize(F_des)
        yaw_des = flat_outputs['yaw'][idxs].double()
        c1_des = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros_like(yaw_des)], dim=-1)
        b2_des = self.normalize(torch.cross(b3_des, c1_des, dim=-1))
        b1_des = torch.cross(b2_des, b3_des, dim=-1)
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

        S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
        att_err = torch.stack([-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1)

        w_des = torch.stack([torch.zeros_like(yaw_des), torch.zeros_like(yaw_des), flat_outputs['yaw_dot'][idxs].double()], dim=-1).to(self.device)
        w_err = states['w'][idxs].double()- w_des

        Iw = self.params.inertia[idxs] @ states['w'][idxs].unsqueeze(-1).double()
        tmp = -self.kp_att[idxs] * att_err - self.kd_att[idxs] * w_err
        u2 = (self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1) + torch.cross(states['w'][idxs].double(), Iw.squeeze(-1), dim=-1)

        TM = torch.cat([u1.unsqueeze(-1), u2], dim=-1)
        cmd_rotor_thrusts = (self.params.TM_to_f[idxs] @ TM.unsqueeze(1).transpose(-1, -2)).squeeze(-1)
        cmd_motor_speeds = cmd_rotor_thrusts / self.params.k_eta[idxs]
        cmd_motor_speeds = torch.sign(cmd_motor_speeds) * torch.sqrt(torch.abs(cmd_motor_speeds))

        cmd_q = roma.rotmat_to_unitquat(R_des)
        cmd_v = -self.kp_vel[idxs] * pos_err + flat_outputs['x_dot'][idxs].double()

        control_inputs = BatchedSE3Control._unpack_control(cmd_motor_speeds,
                                                           cmd_rotor_thrusts,
                                                           u1.unsqueeze(-1),
                                                           u2,
                                                           cmd_q,
                                                           -self.kp_att[idxs] * att_err - self.kd_att[idxs] * w_err,
                                                           cmd_v,
                                                           F_des/self.params.mass[idxs],
                                                           idxs,
                                                           states['x'].shape[0])

        return control_inputs

    @classmethod
    def _unpack_control(cls, cmd_motor_speeds, cmd_motor_thrusts,
                        u1, u2, cmd_q, cmd_w, cmd_v, cmd_acc, idxs, num_drones):
        device = cmd_motor_speeds.device
        # fill state with zeros, then replace with appropriate indexes.
        ctrl = {'cmd_motor_speeds': torch.zeros(num_drones, 4, dtype=torch.double, device=device),
                'cmd_motor_thrusts': torch.zeros(num_drones, 4, dtype=torch.double, device=device),
                 'cmd_thrust': torch.zeros(num_drones, 1, dtype=torch.double, device=device),
                 'cmd_moment': torch.zeros(num_drones, 3, dtype=torch.double, device=device),
                 'cmd_q': torch.zeros(num_drones, 4, dtype=torch.double, device=device),
                 'cmd_w': torch.zeros(num_drones, 3, dtype=torch.double, device=device),
                 'cmd_v': torch.zeros(num_drones, 3, dtype=torch.double, device=device),
                 'cmd_acc': torch.zeros(num_drones, 3, dtype=torch.double, device=device)}

        ctrl['cmd_motor_speeds'][idxs] = cmd_motor_speeds
        ctrl['cmd_motor_thrusts'][idxs] = cmd_motor_thrusts
        ctrl['cmd_thrust'][idxs] = u1
        ctrl['cmd_moment'][idxs] = u2
        ctrl['cmd_q'][idxs] = cmd_q
        ctrl['cmd_w'][idxs] = cmd_w
        ctrl['cmd_v'][idxs] = cmd_v
        ctrl['cmd_acc'][idxs] = cmd_acc
        return ctrl
