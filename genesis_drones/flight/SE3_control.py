import numpy as np
import torch
import roma
import yaml
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """
    Quadrotor trajectory tracking controller based on https://ieeexplore.ieee.org/document/5717652
    with Hopf Fibration based attitude control from
    'Control of Quadrotors Using the Hopf Fibration on SO(3)'

    Fix: handle c -> -1 singularity by flipping the S^2 chart (force c >= 0),
         compute Hopf quantities in the stable chart, then flip back R_des and ω_des.
    """

    def __init__(self, yaml_path: str):

        with open(yaml_path, "r") as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float

        # =====================
        # Inertia
        # =====================
        self.mass = cfg["inertia"]["mass"]
        self.Ixx = cfg["inertia"]["Ixx"]
        self.Iyy = cfg["inertia"]["Iyy"]
        self.Izz = cfg["inertia"]["Izz"]
        self.Ixy = cfg["inertia"]["Ixy"]
        self.Ixz = cfg["inertia"]["Ixz"]
        self.Iyz = cfg["inertia"]["Iyz"]

        self.inertia = torch.tensor(
            [
                [self.Ixx, self.Ixy, self.Ixz],
                [self.Ixy, self.Iyy, self.Iyz],
                [self.Ixz, self.Iyz, self.Izz],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        self.g = cfg["flight"]["g"]
        self.omega_z_limit = cfg["flight"]["omega_z_limit"]
        self.omega_xy_limit = cfg["flight"]["omega_xy_limit"]

        # =====================
        # Gains
        # =====================
        self.kp_pos = torch.tensor(
            cfg["gains"]["kp_pos"], dtype=self.dtype, device=self.device
        )
        self.kd_pos = torch.tensor(
            cfg["gains"]["kd_pos"], dtype=self.dtype, device=self.device
        )
        self.kp_att = cfg["gains"]["kp_att"]
        self.kd_att = cfg["gains"]["kd_att"]
        self.kp_vel = 0.1 * self.kp_pos

        # =====================
        # Rotor geometry
        # =====================
        d = cfg["arm_length"]

        self.num_rotors = cfg["geometry"]["num_rotors"]

        self.rotor_pos = {
            k: torch.tensor(v, dtype=self.dtype, device=self.device) * d
            for k, v in cfg["geometry"]["rotor_pos"].items()
        }

        self.rotor_dir = torch.tensor(
            cfg["geometry"]["rotor_directions"], dtype=self.dtype, device=self.device
        )

        # =====================
        # Rotor / motor parameters
        # =====================
        self.k_eta = cfg["rotor"]["k_eta"]
        self.k_m = cfg["rotor"]["k_m"]

        # =====================
        # Allocation matrix
        # =====================
        k = self.k_m / self.k_eta

        e3 = torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        self.f_to_TM = torch.vstack(
            (
                torch.ones((1, self.num_rotors), dtype=self.dtype, device=self.device),
                torch.hstack(
                    [
                        torch.cross(self.rotor_pos[key], e3, dim=0)[:2].reshape(-1, 1)
                        for key in self.rotor_pos
                    ]
                ),
                (k * self.rotor_dir).reshape(1, -1),
            )
        )

        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

    # ------------------------
    # HFCA QUATERNION MULTIPLY
    # ------------------------
    @staticmethod
    def quat_mul(q1, q2):
        """Quaternion multiply (Hamilton product), q = q1 ⊗ q2.
        Quaternions are [w,x,y,z].
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=0,
        )

    @staticmethod
    def normalize(v, eps=1e-9):
        n = torch.norm(v)
        if n < eps:
            return v
        return v / n

    @staticmethod
    def vee(S):
        """vee-map for so(3) -> R^3 for skew-symmetric matrix S."""
        return torch.stack([-S[1, 2], S[0, 2], -S[0, 1]], dim=0)

    @staticmethod
    def _quat_wxyz_to_rotmat(q):
        # q: (4,) [w,x,y,z]
        w, x, y, z = q
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        R = torch.stack(
            [
                torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=0),
                torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=0),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=0),
            ],
            dim=0,
        )
        return R

    @staticmethod
    def _rotmat_to_quat_wxyz(R, eps=1e-12):
        # robust-ish rotmat -> quat [w,x,y,z]
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0.0:
            S = torch.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / (S + eps)
            y = (R[0, 2] - R[2, 0]) / (S + eps)
            z = (R[1, 0] - R[0, 1]) / (S + eps)
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / (S + eps)
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / (S + eps)
                z = (R[0, 2] + R[2, 0]) / (S + eps)
            elif R[1, 1] > R[2, 2]:
                S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / (S + eps)
                x = (R[0, 1] + R[1, 0]) / (S + eps)
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / (S + eps)
            else:
                S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / (S + eps)
                x = (R[0, 2] + R[2, 0]) / (S + eps)
                y = (R[1, 2] + R[2, 1]) / (S + eps)
                z = 0.25 * S

        q = torch.stack([w, x, y, z], dim=0)
        q = q / (torch.norm(q) + eps)
        return q

    @staticmethod
    def _safe_hopf_attitude_and_omega(
        zeta,
        zeta_dot,
        yaw,
        yaw_dot,
        eps=1e-6,
        omega_z_limit=5.0,  # rad/s
        omega_xy_limit=10.0,  # rad/s
    ):
        """
        Compute R_des and w_des from Hopf fibration formulas robustly near c=-1 by:
        - Build s = zeta/||zeta|| (desired b3)
        - If c < 0, flip s and s_dot to make c >= 0 (stable chart)
        - Compute Hopf tilt quaternion q_abc, then apply yaw quaternion q_psi
        - Compute omega via HFCA closed-form
        - If flipped, flip back R_des (columns 0 and 2) and w_des (x and z)
        """

        zeta_norm = torch.norm(zeta)
        if zeta_norm < eps:
            # caller should handle fallback; return None to indicate failure
            return None, None

        # s on S^2
        s = zeta / zeta_norm
        a, b, c = s

        # s_dot via normalization derivative: s_dot = (I - s s^T) zeta_dot / ||zeta||
        # Your original P-form is equivalent.
        I3 = torch.eye(3, dtype=zeta.dtype, device=zeta.device)
        P = (zeta_norm**2 * I3 - torch.outer(zeta, zeta)) / (zeta_norm**3 + eps)
        s_dot = P @ zeta_dot
        a_dot, b_dot, c_dot = s_dot

        # ---- flip chart if in south hemisphere (c < 0) ----
        flip = 1.0
        if c < 0.0:
            flip = -1.0
            a, b, c = -a, -b, -c
            a_dot, b_dot, c_dot = -a_dot, -b_dot, -c_dot

        # ---- Hopf tilt quaternion q_abc (stable since c >= 0 => 1+c >= 1) ----
        one_plus_c = torch.clamp(1.0 + c, min=eps)
        denom = torch.sqrt(2.0 * one_plus_c)

        q_abc = torch.stack(
            [
                (1.0 + c) / denom,
                -b / denom,
                a / denom,
                torch.zeros((), dtype=zeta.dtype, device=zeta.device),
            ],
            dim=0,
        )  # [w,x,y,z]

        # ---- LIMIT: yaw wrap + yaw rate clamp ----
        yaw = torch.remainder(yaw + torch.pi, 2.0 * torch.pi) - torch.pi

        # yaw quaternion q_psi about +z
        half = 0.5 * yaw
        q_psi = torch.stack(
            [
                torch.cos(half),
                torch.zeros_like(half),
                torch.zeros_like(half),
                torch.sin(half),
            ],
            dim=0,
        )

        # total quaternion
        q_tot = SE3Control.quat_mul(q_abc, q_psi)  # [w,x,y,z]

        R_des = SE3Control._quat_wxyz_to_rotmat(q_tot)

        # ---- omega formulas (stable since one_plus_c >= 1) ----
        sinp = torch.sin(yaw)
        cosp = torch.cos(yaw)

        # ---- LIMIT: protect c_dot / (1 + c) ----
        omg_term = c_dot / one_plus_c

        omega1 = sinp * a_dot - cosp * b_dot - (a * sinp - b * cosp) * omg_term
        omega2 = cosp * a_dot + sinp * b_dot - (a * cosp + b * sinp) * omg_term
        omega3 = (b * a_dot - a * b_dot) / one_plus_c + yaw_dot

        # limit w_des with separate z/xy limits
        w_des = torch.stack([omega1, omega2, omega3], dim=0)
        w_des[0:2] = torch.clamp(w_des[0:2], -omega_xy_limit, omega_xy_limit)
        w_des[2] = torch.clamp(w_des[2], -omega_z_limit, omega_z_limit)

        # ---- flip back to recover original (unflipped) b3_des = zeta/||zeta|| ----
        if flip < 0.0:
            # keep det +1 by flipping first and third columns
            R_des[:, 0] *= -1.0
            R_des[:, 2] *= -1.0
            # flip omega_x and omega_z like your torch reference
            w_des[0] *= -1.0
            w_des[2] *= -1.0

        return R_des, w_des

    # ============================================================
    #                       MAIN UPDATE()
    # ============================================================
    def update(self, t, state, flat, omega_cmd=None, quat_format="wxyz"):
        """
        state:
            x: position (3,)
            v: velocity (3,)
            q: quaternion in scipy format [x,y,z,w]
            w: body rates (3,) in body frame

        flat:
            x, x_dot, x_ddot, x_dddot: (3,)
            yaw, yaw_dot: scalars
        """

        # --------------------
        # 1. DESIRED ACC ζ
        # --------------------
        # Convert inputs to torch (do not change external formats)
        q_input = state["q"].reshape(4)

        # Keep same behavior: rewrite state["q"] as scipy [x,y,z,w] for downstream if someone expects it
        if quat_format.lower() == "wxyz":
            # [w, x, y, z] -> [x, y, z, w]
            q_xyzw = torch.stack(
                [
                    state["q"].reshape(4)[1],
                    state["q"].reshape(4)[2],
                    state["q"].reshape(4)[3],
                    state["q"].reshape(4)[0],
                ],
                dim=0,
            )
        elif quat_format.lower() == "xyzw":
            # already scipy format
            q_xyzw = state["q"].reshape(4)
        else:
            raise ValueError(
                f"Unknown quat_format: {quat_format}, expected 'wxyz' or 'xyzw'"
            )
        # state["q"] = q_xyzw.detach().cpu().numpy()

        # For internal rotation math, use wxyz
        q_wxyz = torch.stack([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dim=0)

        pos_err = state["x"] - flat["x"]
        vel_err = state["v"] - flat["x_dot"]

        zeta = (
            -self.kp_pos * pos_err
            - self.kd_pos * vel_err
            + flat["x_ddot"]
            + torch.tensor([0.0, 0.0, self.g], dtype=self.dtype, device=self.device)
        )

        F_des = self.mass * zeta

        # Current attitude
        R = self._quat_wxyz_to_rotmat(q_wxyz)
        b3 = R @ torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)

        # Scalar thrust
        u1 = torch.dot(F_des, b3)

        # ---------------------------
        # 2. DESIRED ATTITUDE (Hopf, robust near c=-1)
        # ---------------------------
        eps = 1e-6
        zeta_norm = torch.norm(zeta)

        if zeta_norm < eps:
            # fallback to classic SE3 direction + yaw
            b3_des = torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
            c1 = torch.stack(
                [
                    torch.cos(flat["yaw"]),
                    torch.sin(flat["yaw"]),
                    torch.zeros_like(flat["yaw"]),
                ],
                dim=0,
            )
            b2 = self.normalize(torch.cross(b3_des, c1, dim=0))
            b1 = torch.cross(b2, b3_des, dim=0)
            R_des = torch.stack([b1, b2, b3_des], dim=1)
            w_des = torch.stack(
                [
                    torch.zeros_like(flat["yaw_dot"]),
                    torch.zeros_like(flat["yaw_dot"]),
                    flat["yaw_dot"],
                ],
                dim=0,
            )
        else:
            R_des, w_des = self._safe_hopf_attitude_and_omega(
                zeta=zeta,
                zeta_dot=flat["x_dddot"],
                yaw=flat["yaw"],
                yaw_dot=flat["yaw_dot"],
                eps=1e-6,
                omega_z_limit=self.omega_z_limit,  # rad/s
                omega_xy_limit=self.omega_xy_limit,  # rad/s
            )
            # ultra-safe fallback (should not happen)
            if R_des is None or w_des is None:
                b3_des = zeta / zeta_norm
                c1 = torch.stack(
                    [
                        torch.cos(flat["yaw"]),
                        torch.sin(flat["yaw"]),
                        torch.zeros_like(flat["yaw"]),
                    ],
                    dim=0,
                )
                b2 = self.normalize(torch.cross(b3_des, c1, dim=0))
                b1 = torch.cross(b2, b3_des, dim=0)
                R_des = torch.stack([b1, b2, b3_des], dim=1)
                w_des = torch.stack(
                    [
                        torch.zeros_like(flat["yaw_dot"]),
                        torch.zeros_like(flat["yaw_dot"]),
                        flat["yaw_dot"],
                    ],
                    dim=0,
                )

        if omega_cmd is not None:
            w_des = omega_cmd  # override omega command

        # -----------------------
        # 3. ATTITUDE PD CONTROL
        # -----------------------
        S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
        att_err = self.vee(S_err)
        w_err = state["w"] - w_des

        u2 = self.inertia @ (
            -self.kp_att * att_err - self.kd_att * w_err
        ) + torch.cross(state["w"], self.inertia @ state["w"], dim=0)

        # body-rate command (optional)
        cmd_w = -self.kp_att * att_err - self.kd_att * w_err

        # -----------------------
        # 4. MOTOR ALLOCATION
        # -----------------------
        TM = torch.stack([u1, u2[0], u2[1], u2[2]], dim=0)
        rotor_thrusts = self.TM_to_f @ TM

        motor_speeds = rotor_thrusts / self.k_eta
        motor_speeds = torch.sign(motor_speeds) * torch.sqrt(torch.abs(motor_speeds))

        # -----------------------
        # OUTPUTS
        # -----------------------
        cmd_q_wxyz = self._rotmat_to_quat_wxyz(R_des)  # [w,x,y,z]
        cmd_q = torch.stack(
            [cmd_q_wxyz[1], cmd_q_wxyz[2], cmd_q_wxyz[3], cmd_q_wxyz[0]], dim=0
        )  # scipy format [x,y,z,w]

        return {
            "cmd_motor_speeds": motor_speeds,  # torch
            "cmd_motor_thrusts": rotor_thrusts,  # torch
            "cmd_thrust": u1,  # torch scalar
            "cmd_moment": u2,  # torch
            "cmd_q": cmd_q,  # torch
            "cmd_w": cmd_w,  # torch
            "cmd_v": (-self.kp_vel * pos_err + flat["x_dot"]),
            "cmd_acc": (F_des / self.mass),
        }

    def action(
        self,
        control,
        flight_config,
        env_config,
        device,
    ):
        """
        normlize the ctrl output to action space
        """
        # -------- thrust normlize --------
        # ensure tensors on device
        cmd_thrust = control["cmd_thrust"]
        cmd_w = control["cmd_w"]
        cmd_q = control["cmd_q"]
        min_t = flight_config["min_t"]
        max_t = flight_config["max_t"]
        thrust_norm = (cmd_thrust - min_t) / (max_t - min_t)
        thrust_norm = thrust_norm * 2.0 - 1.0
        if env_config["controller"] == "rate":
            # -------- rate normlize --------
            wx, wy, wz = cmd_w[0], cmd_w[1], cmd_w[2]
            roll_norm = torch.clamp(wx / flight_config["max_roll_rate"], -1.0, 1.0)
            pitch_norm = torch.clamp(wy / flight_config["max_pitch_rate"], -1.0, 1.0)
            yaw_norm = torch.clamp(wz / flight_config["max_yaw_rate"], -1.0, 1.0)

            action = torch.stack(
                [roll_norm, pitch_norm, yaw_norm, thrust_norm]
            ).reshape(1, -1)

        elif env_config["controller"] == "angle":
            # -------- angle normlize --------
            # keep external behavior: scipy expects [x,y,z,w]
            from scipy.spatial.transform import Rotation

            eulers = Rotation.from_quat(cmd_q.detach().cpu().numpy()).as_euler("xyz")
            eulers = torch.as_tensor(eulers, device=device).float()

            action = torch.cat([eulers, thrust_norm.reshape(1)]).reshape(1, -1)

        return action


class BatchedSE3Control(object):
    def __init__(
        self,
        batch_params,
        num_drones,
        device,
        kp_pos=None,
        kd_pos=None,
        kp_att=None,
        kd_att=None,
    ):
        """
        batch_params, BatchedMultirotorParams object
        num_drones: int, number of drones in the batch
        device: torch.device("cpu") or torch.device("cuda")

        kp_pos: torch.Tensor of shape (num_drones, 3)
        kd_pos: torch.Tensor of shape (num_drones, 3)
        kp_att: torch.Tensor of shape (num_drones, 1)
        kd_att: torch.Tensor of shape (num_drones, 1)
        """
        assert batch_params.device == device
        self.params = batch_params
        self.device = device
        # Quadrotor physical parameters

        # Gains
        if kp_pos is None:
            self.kp_pos = (
                torch.tensor([6.5, 6.5, 15], device=self.device)
                .repeat(num_drones, 1)
                .double()
            )
        else:
            self.kp_pos = kp_pos.to(self.device).double()
        if kd_pos is None:
            self.kd_pos = (
                torch.tensor([4.0, 4.0, 9], device=self.device)
                .repeat(num_drones, 1)
                .double()
            )
        else:
            self.kd_pos = kd_pos.to(self.device).double()
        if kp_att is None:
            self.kp_att = (
                torch.tensor([544], device=device).repeat(num_drones, 1).double()
            )
        else:
            self.kp_att = kp_att.to(self.device).double()
            if len(self.kp_att.shape) < 2:
                self.kp_att = self.kp_att.unsqueeze(-1)
        if kd_att is None:
            self.kd_att = (
                torch.tensor([46.64], device=device).repeat(num_drones, 1).double()
            )
        else:
            self.kd_att = kd_att.to(self.device).double()
            if len(self.kd_att.shape) < 2:
                self.kd_att = self.kd_att.unsqueeze(-1)

        self.kp_vel = 0.1 * self.kp_pos

    def normalize(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True)

    def update(self, t, states, flat_outputs, idxs=None):
        """
        Computes a batch of control outputs for the drones specified by idxs
        :param states: a dictionary of pytorch tensors containing the states of the quadrotors (expects double precision)
        :param flat_outputs: a dictionary of pytorch tensors containing the reference trajectories for each quad. (expects double precision)
        :param idxs: a list of which drones to update
        :return:
        """
        if idxs is None:
            idxs = [i for i in range(states["x"].shape[0])]
        pos_err = states["x"][idxs].double() - flat_outputs["x"][idxs].double()
        dpos_err = states["v"][idxs].double() - flat_outputs["x_dot"][idxs].double()

        F_des = self.params.mass[idxs] * (
            -self.kp_pos[idxs] * pos_err
            - self.kd_pos[idxs] * dpos_err
            + flat_outputs["x_ddot"][idxs].double()
            + torch.tensor([0, 0, self.params.g], device=self.device)
        )

        R = roma.unitquat_to_rotmat(states["q"][idxs]).double()
        b3 = R @ torch.tensor([0.0, 0.0, 1.0], device=self.device).double()
        u1 = torch.sum(F_des * b3, dim=-1).double()

        b3_des = self.normalize(F_des)
        yaw_des = flat_outputs["yaw"][idxs].double()
        c1_des = torch.stack(
            [torch.cos(yaw_des), torch.sin(yaw_des), torch.zeros_like(yaw_des)], dim=-1
        )
        b2_des = self.normalize(torch.cross(b3_des, c1_des, dim=-1))
        b1_des = torch.cross(b2_des, b3_des, dim=-1)
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)

        S_err = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
        att_err = torch.stack(
            [-S_err[:, 1, 2], S_err[:, 0, 2], -S_err[:, 0, 1]], dim=-1
        )

        w_des = torch.stack(
            [
                torch.zeros_like(yaw_des),
                torch.zeros_like(yaw_des),
                flat_outputs["yaw_dot"][idxs].double(),
            ],
            dim=-1,
        ).to(self.device)
        w_err = states["w"][idxs].double() - w_des

        Iw = self.params.inertia[idxs] @ states["w"][idxs].unsqueeze(-1).double()
        tmp = -self.kp_att[idxs] * att_err - self.kd_att[idxs] * w_err
        u2 = (self.params.inertia[idxs] @ tmp.unsqueeze(-1)).squeeze(-1) + torch.cross(
            states["w"][idxs].double(), Iw.squeeze(-1), dim=-1
        )

        TM = torch.cat([u1.unsqueeze(-1), u2], dim=-1)
        cmd_rotor_thrusts = (
            self.params.TM_to_f[idxs] @ TM.unsqueeze(1).transpose(-1, -2)
        ).squeeze(-1)
        cmd_motor_speeds = cmd_rotor_thrusts / self.params.k_eta[idxs]
        cmd_motor_speeds = torch.sign(cmd_motor_speeds) * torch.sqrt(
            torch.abs(cmd_motor_speeds)
        )

        cmd_q = roma.rotmat_to_unitquat(R_des)
        cmd_v = -self.kp_vel[idxs] * pos_err + flat_outputs["x_dot"][idxs].double()

        control_inputs = BatchedSE3Control._unpack_control(
            cmd_motor_speeds,
            cmd_rotor_thrusts,
            u1.unsqueeze(-1),
            u2,
            cmd_q,
            -self.kp_att[idxs] * att_err - self.kd_att[idxs] * w_err,
            cmd_v,
            F_des / self.params.mass[idxs],
            idxs,
            states["x"].shape[0],
        )

        return control_inputs

    @classmethod
    def _unpack_control(
        cls,
        cmd_motor_speeds,
        cmd_motor_thrusts,
        u1,
        u2,
        cmd_q,
        cmd_w,
        cmd_v,
        cmd_acc,
        idxs,
        num_drones,
    ):
        device = cmd_motor_speeds.device
        # fill state with zeros, then replace with appropriate indexes.
        ctrl = {
            "cmd_motor_speeds": torch.zeros(
                num_drones, 4, dtype=torch.double, device=device
            ),
            "cmd_motor_thrusts": torch.zeros(
                num_drones, 4, dtype=torch.double, device=device
            ),
            "cmd_thrust": torch.zeros(num_drones, 1, dtype=torch.double, device=device),
            "cmd_moment": torch.zeros(num_drones, 3, dtype=torch.double, device=device),
            "cmd_q": torch.zeros(num_drones, 4, dtype=torch.double, device=device),
            "cmd_w": torch.zeros(num_drones, 3, dtype=torch.double, device=device),
            "cmd_v": torch.zeros(num_drones, 3, dtype=torch.double, device=device),
            "cmd_acc": torch.zeros(num_drones, 3, dtype=torch.double, device=device),
        }

        ctrl["cmd_motor_speeds"][idxs] = cmd_motor_speeds
        ctrl["cmd_motor_thrusts"][idxs] = cmd_motor_thrusts
        ctrl["cmd_thrust"][idxs] = u1
        ctrl["cmd_moment"][idxs] = u2
        ctrl["cmd_q"][idxs] = cmd_q
        ctrl["cmd_w"][idxs] = cmd_w
        ctrl["cmd_v"][idxs] = cmd_v
        ctrl["cmd_acc"][idxs] = cmd_acc
        return ctrl
