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

        self.inertia = np.array(
            [
                [self.Ixx, self.Ixy, self.Ixz],
                [self.Ixy, self.Iyy, self.Iyz],
                [self.Ixz, self.Iyz, self.Izz],
            ]
        )

        self.g = cfg["flight"]["g"]
        self.omega_z_limit = cfg["flight"]["omega_z_limit"]
        self.omega_xy_limit = cfg["flight"]["omega_xy_limit"]

        # =====================
        # Gains
        # =====================
        self.kp_pos = np.array(cfg["gains"]["kp_pos"])
        self.kd_pos = np.array(cfg["gains"]["kd_pos"])
        self.kp_att = np.array(cfg["gains"]["kp_att"])
        self.kd_att = np.array(cfg["gains"]["kd_att"])
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

        self.rotor_dir = np.array(cfg["geometry"]["rotor_directions"], dtype=float)

        # =====================
        # Rotor / motor parameters
        # =====================
        self.k_eta = cfg["rotor"]["k_eta"]
        self.k_m = cfg["rotor"]["k_m"]

        # =====================
        # Allocation matrix
        # =====================
        k = self.k_m / self.k_eta

        self.f_to_TM = np.vstack(
            (
                np.ones((1, self.num_rotors)),
                np.hstack(
                    [
                        np.cross(self.rotor_pos[key], np.array([0, 0, 1]))[:2].reshape(
                            -1, 1
                        )
                        for key in self.rotor_pos
                    ]
                ),
                (k * self.rotor_dir).reshape(1, -1),
            )
        )

        self.TM_to_f = np.linalg.inv(self.f_to_TM)

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
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    @staticmethod
    def normalize(v, eps=1e-9):
        n = np.linalg.norm(v)
        if n < eps:
            return v
        return v / n

    @staticmethod
    def vee(S):
        """vee-map for so(3) -> R^3 for skew-symmetric matrix S."""
        return np.array([-S[1, 2], S[0, 2], -S[0, 1]], dtype=float)

    @staticmethod
    def _safe_hopf_attitude_and_omega(
        zeta,
        zeta_dot,
        yaw,
        yaw_dot,
        eps=1e-6,
        yaw_rate_limit=3.0,  # rad/s
        omega_limit=10.0,  # rad/s
    ):
        """
        Compute R_des and w_des from Hopf fibration formulas robustly near c=-1 by:
        - Build s = zeta/||zeta|| (desired b3)
        - If c < 0, flip s and s_dot to make c >= 0 (stable chart)
        - Compute Hopf tilt quaternion q_abc, then apply yaw quaternion q_psi
        - Compute omega via HFCA closed-form
        - If flipped, flip back R_des (columns 0 and 2) and w_des (x and z)
        """

        zeta_norm = np.linalg.norm(zeta)
        if zeta_norm < eps:
            # caller should handle fallback; return None to indicate failure
            return None, None

        # s on S^2
        s = zeta / zeta_norm
        a, b, c = s

        # s_dot via normalization derivative: s_dot = (I - s s^T) zeta_dot / ||zeta||
        # Your original P-form is equivalent.
        I3 = np.eye(3)
        P = (zeta_norm**2 * I3 - np.outer(zeta, zeta)) / (zeta_norm**3 + eps)
        s_dot = P @ zeta_dot
        a_dot, b_dot, c_dot = s_dot

        # ---- flip chart if in south hemisphere (c < 0) ----
        flip = 1.0
        if c < 0.0:
            flip = -1.0
            a, b, c = -a, -b, -c
            a_dot, b_dot, c_dot = -a_dot, -b_dot, -c_dot

        # ---- Hopf tilt quaternion q_abc (stable since c >= 0 => 1+c >= 1) ----
        one_plus_c = max(1.0 + c, eps)
        denom = np.sqrt(2.0 * one_plus_c)

        q_abc = np.array(
            [(1.0 + c) / denom, -b / denom, a / denom, 0.0], dtype=float
        )  # [w,x,y,z]

        # ---- LIMIT: yaw wrap + yaw rate clamp ----
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        # yaw_dot = np.clip(yaw_dot, -yaw_rate_limit, yaw_rate_limit)

        # yaw quaternion q_psi about +z
        half = 0.5 * yaw
        q_psi = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)

        # total quaternion
        q_tot = SE3Control.quat_mul(q_abc, q_psi)  # [w,x,y,z]

        # scipy wants [x,y,z,w]
        q_scipy = np.array([q_tot[1], q_tot[2], q_tot[3], q_tot[0]], dtype=float)
        R_des = Rotation.from_quat(q_scipy).as_matrix()

        # ---- omega formulas (stable since one_plus_c >= 1) ----
        sinp = np.sin(yaw)
        cosp = np.cos(yaw)

        # ---- LIMIT: protect c_dot / (1 + c) ----
        omg_term = c_dot / one_plus_c
        # omg_term = np.clip(omg_term, -omega_limit, omega_limit)

        omega1 = sinp * a_dot - cosp * b_dot - (a * sinp - b * cosp) * omg_term
        omega2 = cosp * a_dot + sinp * b_dot - (a * cosp + b * sinp) * omg_term
        omega3 = (b * a_dot - a * b_dot) / one_plus_c + yaw_dot
        omega3 = np.clip(omega3, -yaw_rate_limit, yaw_rate_limit)
        w_des = np.array([omega1, omega2, omega3], dtype=float)
        w_des = np.clip(w_des, -omega_limit, omega_limit)
        # ---- LIMIT: global omega saturation (final safety net) ----
        # omega_norm = np.linalg.norm(w_des)
        # if omega_norm > omega_limit:
        #     w_des *= omega_limit / (omega_norm + eps)

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
    def update(self, t, state, flat, omega_cmd):
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
        pos_err = state["x"] - flat["x"]
        vel_err = state["v"] - flat["x_dot"]

        zeta = (
            -self.kp_pos * pos_err
            - self.kd_pos * vel_err
            + flat["x_ddot"]
            + np.array([0.0, 0.0, self.g])
        )

        F_des = self.mass * zeta

        # Current attitude
        R = Rotation.from_quat(state["q"]).as_matrix()
        b3 = R @ np.array([0.0, 0.0, 1.0])

        # Scalar thrust
        u1 = float(np.dot(F_des, b3))
        # u1 = thrust_cmd  # override thrust command
        # ---------------------------
        # 2. DESIRED ATTITUDE (Hopf, robust near c=-1)
        # ---------------------------
        eps = 1e-6
        zeta_norm = np.linalg.norm(zeta)

        if zeta_norm < eps:
            # fallback to classic SE3 direction + yaw
            b3_des = np.array([0.0, 0.0, 1.0])
            yaw = float(flat["yaw"])
            c1 = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            b2 = self.normalize(np.cross(b3_des, c1))
            b1 = np.cross(b2, b3_des)
            R_des = np.stack([b1, b2, b3_des], axis=1)
            w_des = np.array([0.0, 0.0, float(flat["yaw_dot"])], dtype=float)
        else:
            R_des, w_des = self._safe_hopf_attitude_and_omega(
                zeta=zeta,
                zeta_dot=flat["x_dddot"],
                yaw=float(flat["yaw"]),
                yaw_dot=float(flat["yaw_dot"]),
                eps=1e-6,
                yaw_rate_limit=self.omega_z_limit,  # rad/s
                omega_limit=self.omega_xy_limit,  # rad/s
            )
            # ultra-safe fallback (should not happen)
            if R_des is None or w_des is None:
                b3_des = zeta / zeta_norm
                yaw = float(flat["yaw"])
                c1 = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                b2 = self.normalize(np.cross(b3_des, c1))
                b1 = np.cross(b2, b3_des)
                R_des = np.stack([b1, b2, b3_des], axis=1)
                w_des = np.array([0.0, 0.0, float(flat["yaw_dot"])], dtype=float)
        # w_des = omega_cmd  # override omega command
        # -----------------------
        # 3. ATTITUDE PD CONTROL
        # -----------------------
        S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
        att_err = self.vee(S_err)
        w_err = state["w"] - w_des

        u2 = self.inertia @ (-self.kp_att * att_err - self.kd_att * w_err) + np.cross(
            state["w"], self.inertia @ state["w"]
        )

        # body-rate command (optional)
        cmd_w = w_des - self.kp_att * att_err - self.kd_att * w_err

        # -----------------------
        # 4. MOTOR ALLOCATION
        # -----------------------
        TM = np.array([u1, u2[0], u2[1], u2[2]], dtype=float)
        rotor_thrusts = self.TM_to_f @ TM

        motor_speeds = rotor_thrusts / self.k_eta
        motor_speeds = np.sign(motor_speeds) * np.sqrt(np.abs(motor_speeds))

        # -----------------------
        # OUTPUTS
        # -----------------------
        cmd_q = Rotation.from_matrix(R_des).as_quat()  # scipy format [x,y,z,w]

        return {
            "cmd_motor_speeds": motor_speeds,
            "cmd_motor_thrusts": rotor_thrusts,
            "cmd_thrust": u1,
            "cmd_moment": u2,
            "cmd_q": cmd_q,
            "cmd_w": cmd_w,
            "cmd_v": -self.kp_vel * pos_err + flat["x_dot"],
            "cmd_acc": F_des / self.mass,
        }


class TorchSE3Control(object):
    """
    PyTorch-based Vectorized SE3 Controller.
    Fully differentiable and GPU-accelerated.
    """

    def __init__(self, yaml_path: str, device):
        with open(yaml_path, "r") as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.device = device

        # =====================
        # Inertia (3, 3)
        # =====================
        self.nominal_mass = float(cfg["inertia"]["mass"])

        # Construct Inertia Matrix directly as Tensor
        Ixx = cfg["inertia"]["Ixx"]
        Iyy = cfg["inertia"]["Iyy"]
        Izz = cfg["inertia"]["Izz"]
        Ixy = cfg["inertia"]["Ixy"]
        Ixz = cfg["inertia"]["Ixz"]
        Iyz = cfg["inertia"]["Iyz"]

        self.nominal_inertia = torch.tensor(
            [
                [Ixx, Ixy, Ixz],
                [Ixy, Iyy, Iyz],
                [Ixz, Iyz, Izz],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.g = float(cfg["flight"]["g"])
        self.omega_z_limit = float(cfg["flight"]["omega_z_limit"])
        self.omega_xy_limit = float(cfg["flight"]["omega_xy_limit"])

        # =====================
        # Gains
        # =====================
        self.nominal_kp_pos = torch.tensor(cfg["gains"]["kp_pos"], device=self.device)
        self.nominal_kd_pos = torch.tensor(cfg["gains"]["kd_pos"], device=self.device)
        self.nominal_kp_att = torch.tensor(cfg["gains"]["kp_att"], device=self.device)
        self.nominal_kd_att = torch.tensor(cfg["gains"]["kd_att"], device=self.device)
        # self.kp_vel = 0.1 * self.kp_pos
        self.mass = torch.tensor([self.nominal_mass], device=self.device)
        self.inertia = self.nominal_inertia.unsqueeze(0)  # (1, 3, 3)
        self.kp_pos = self.nominal_kp_pos.unsqueeze(0)  # (1, 3)
        self.kd_pos = self.nominal_kd_pos.unsqueeze(0)
        self.kp_att = self.nominal_kp_att.unsqueeze(0)
        self.kd_att = self.nominal_kd_att.unsqueeze(0)
        # =====================
        # Allocation
        # =====================
        d = cfg["arm_length"]
        self.num_rotors = cfg["geometry"]["num_rotors"]

        rotor_pos_np = np.array(
            [
                cfg["geometry"]["rotor_pos"][k]
                for k in sorted(cfg["geometry"]["rotor_pos"].keys())
            ]
        )
        rotor_pos_np = d * rotor_pos_np  # Scale by arm length

        rotor_dir = np.array(cfg["geometry"]["rotor_directions"])

        k_eta = cfg["rotor"]["k_eta"]
        k_m = cfg["rotor"]["k_m"]
        k = k_m / k_eta
        self.k_eta = k_eta

        # Build Allocation Matrix (NumPy first, then convert to Torch)
        f_to_TM = np.vstack(
            (
                np.ones((1, self.num_rotors)),
                np.hstack(
                    [
                        np.cross(r, np.array([0, 0, 1]))[:2].reshape(-1, 1)
                        for r in rotor_pos_np
                    ]
                ),
                (k * rotor_dir).reshape(1, -1),
            )
        )
        TM_to_f_np = np.linalg.inv(f_to_TM)

        # Convert to Tensor (4, 4)
        self.TM_to_f = torch.from_numpy(TM_to_f_np).to(
            device=self.device, dtype=torch.float32
        )

    # ------------------------
    # TORCH HELPERS
    # ------------------------
    @staticmethod
    def quat_mul(q1, q2):
        """
        q1 * q2 (Hamilton product).
        q: (B, 4) [w, x, y, z]
        """
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    @staticmethod
    def vee(S):
        """
        S: (B, 3, 3) skew-symmetric
        Returns: (B, 3) vector
        """
        return torch.stack([-S[:, 1, 2], S[:, 0, 2], -S[:, 0, 1]], dim=1)

    @staticmethod
    def quat_to_rot_matrix(q):
        """
        q: (B, 4) [x, y, z, w] (Scipy convention input)
        """
        x, y, z, w = q.unbind(-1)

        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        # Row 0
        r00 = 1 - 2 * (y2 + z2)
        r01 = 2 * (xy - wz)
        r02 = 2 * (xz + wy)

        # Row 1
        r10 = 2 * (xy + wz)
        r11 = 1 - 2 * (x2 + z2)
        r12 = 2 * (yz - wx)

        # Row 2
        r20 = 2 * (xz - wy)
        r21 = 2 * (yz + wx)
        r22 = 1 - 2 * (x2 + y2)

        # Stack to (B, 3, 3)
        row0 = torch.stack([r00, r01, r02], dim=1)
        row1 = torch.stack([r10, r11, r12], dim=1)
        row2 = torch.stack([r20, r21, r22], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    # ------------------------
    # HOPF LOGIC (TORCH)
    # ------------------------
    def _batch_safe_hopf(self, zeta, zeta_dot, yaw, yaw_dot, eps=1e-6):
        """
        All inputs are Tensors on GPU.
        zeta: (B, 3)
        yaw: (B, 1) or (B,)
        """
        # Ensure shapes
        if yaw.dim() == 1:
            yaw = yaw.unsqueeze(1)  # (B, 1)
        if yaw_dot.dim() == 1:
            yaw_dot = yaw_dot.unsqueeze(1)

        zeta_norm = torch.norm(zeta, dim=1, keepdim=True)  # (B, 1)
        zeta_norm_safe = torch.clamp(zeta_norm, min=eps)

        s = zeta / zeta_norm_safe  # (B, 3)
        a, b, c = s[:, 0], s[:, 1], s[:, 2]  # (B,) views

        # s_dot calculation
        s_dot_zeta = torch.sum(s * zeta_dot, dim=1, keepdim=True)
        s_dot = (zeta_dot - s * s_dot_zeta) / zeta_norm_safe

        a_dot, b_dot, c_dot = s_dot[:, 0], s_dot[:, 1], s_dot[:, 2]

        # ---- Flip Chart Logic (c < 0) ----
        # No branching, use masking
        mask_neg = c < 0.0  # Bool tensor
        flip_sign = torch.where(mask_neg, -1.0, 1.0)  # (B,)

        # Apply flip
        a = a * flip_sign
        b = b * flip_sign
        c = c * flip_sign
        a_dot = a_dot * flip_sign
        b_dot = b_dot * flip_sign
        c_dot = c_dot * flip_sign

        # ---- Hopf Quaternion q_abc [w, x, y, z] ----
        one_plus_c = torch.clamp(1.0 + c, min=eps)
        denom = torch.sqrt(2.0 * one_plus_c)

        qw = (1.0 + c) / denom
        qx = -b / denom
        qy = a / denom
        qz = torch.zeros_like(qw)
        q_abc = torch.stack([qw, qx, qy, qz], dim=1)

        # ---- Yaw Quaternion q_psi [w, x, y, z] ----
        # yaw is (B, 1), flatten to (B,)
        yaw = yaw.view(-1)
        yaw = (yaw + torch.pi) % (2.0 * torch.pi) - torch.pi
        half_yaw = 0.5 * yaw

        zeros = torch.zeros_like(yaw)
        q_psi = torch.stack(
            [torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)], dim=1
        )

        # ---- Total Quaternion ----
        q_tot = self.quat_mul(q_abc, q_psi)  # [w, x, y, z]

        # Output q in Scipy order [x, y, z, w] for compatibility if needed
        q_scipy = torch.stack(
            [q_tot[:, 1], q_tot[:, 2], q_tot[:, 3], q_tot[:, 0]], dim=1
        )

        # R_des
        R_des = self.quat_to_rot_matrix(q_scipy)

        # ---- Omega Calculation ----
        sinp = torch.sin(yaw)
        cosp = torch.cos(yaw)

        omg_term = c_dot / one_plus_c

        omega1 = sinp * a_dot - cosp * b_dot - (a * sinp - b * cosp) * omg_term
        omega2 = cosp * a_dot + sinp * b_dot - (a * cosp + b * sinp) * omg_term
        omega3 = (b * a_dot - a * b_dot) / one_plus_c + yaw_dot.view(-1)

        omega3 = torch.clamp(omega3, -self.omega_z_limit, self.omega_z_limit)
        w_des = torch.stack([omega1, omega2, omega3], dim=1)
        w_des = torch.clamp(w_des, -self.omega_xy_limit, self.omega_xy_limit)

        # ---- Flip Back Logic ----
        # R_des columns 0 and 2, w_des x and z need flipping if mask was true
        f_sign = flip_sign.view(-1, 1, 1)  # (B, 1, 1) for broadcasting

        # Create a flip matrix or just operate on slices (torch slices are views, careful)
        # R_des is (B, 3, 3). We want to multiply col 0 and 2 by f_sign
        R_des = R_des.clone()  # Avoid in-place modification errors in gradients
        R_des[:, :, 0] = R_des[:, :, 0] * f_sign.view(-1, 1)
        R_des[:, :, 2] = R_des[:, :, 2] * f_sign.view(-1, 1)

        w_des = w_des.clone()
        w_des[:, 0] = w_des[:, 0] * flip_sign
        w_des[:, 2] = w_des[:, 2] * flip_sign

        return R_des, w_des, q_scipy

    def randomize_params(self, num_envs, mass_std=0.05, pid_scale_range=(0.8, 1.2)):
        """
        num_envs: Number of environments (Batch Size)
        mass_std: Standard deviation for mass Gaussian distribution (kg)
        pid_scale_range: Uniform distribution range for PID parameter scaling (min, max)
        """

        # 1. Mass Randomization (Gaussian)
        # shape: (B, 1) to support broadcasting
        mass_noise = torch.normal(
            mean=0.0, std=mass_std, size=(num_envs, 1), device=self.device
        )
        self.mass = self.nominal_mass + mass_noise

        # Safety clamp, avoid mass <= 0
        self.mass = torch.clamp(self.mass, min=0.01)

        # 2. Inertia Matrix Randomization
        # Physically, inertia matrix usually scales linearly with mass: J_new = J_nom * (m_new / m_nom)
        mass_ratio = self.mass / self.nominal_mass  # (B, 1)

        # nominal_inertia: (3, 3) -> (1, 3, 3)
        # mass_ratio: (B, 1) -> (B, 1, 1)
        # Result: (B, 3, 3)
        self.inertia = self.nominal_inertia.unsqueeze(0) * mass_ratio.unsqueeze(-1)

        # 3. PID Parameter Randomization (Uniform distribution scaling)
        # Generate random scale matrix (B, 3)
        low, high = pid_scale_range

        def get_rand_gains(nominal_gain):
            # random scale shape: (B, 3) -> independent random for x, y, z axes
            scale = torch.rand((num_envs, 3), device=self.device) * (high - low) + low
            return nominal_gain.unsqueeze(0) * scale  # (1, 3) * (B, 3) -> (B, 3)

        self.kp_pos = get_rand_gains(self.nominal_kp_pos)
        self.kd_pos = get_rand_gains(self.nominal_kd_pos)
        self.kp_att = get_rand_gains(self.nominal_kp_att)
        self.kd_att = get_rand_gains(self.nominal_kd_att)
        self.kp_vel = 0.1 * self.kp_pos

    # ============================================================
    #                       MAIN UPDATE
    # ============================================================
    def update(self, t, state, flat, omega_cmd=None):
        """
        Inputs should be Tensors on GPU.
        state: {'x': (B,3), 'v': (B,3), 'q': (B,4), 'w': (B,3)}
        flat:  {'x': (B,3), ... 'yaw': (B,1)}
        """

        # 1. Desired Force
        pos_err = state["x"] - flat["x"]
        vel_err = state["v"] - flat["x_dot"]

        target_acc = -self.kp_pos * pos_err - self.kd_pos * vel_err + flat["x_ddot"]
        target_acc[:, 2] += self.g

        F_des = self.mass * target_acc  # (B, 3)

        # 2. Current Attitude
        R = self.quat_to_rot_matrix(state["q"])  # (B, 3, 3)
        b3 = R[:, :, 2]  # (B, 3)

        # 3. Thrust (u1)
        u1 = torch.sum(F_des * b3, dim=1)  # (B,)

        # 4. Desired Att
        R_des, w_des, q_des = self._batch_safe_hopf(
            target_acc, flat["x_dddot"], flat["yaw"], flat["yaw_dot"]
        )
        # w_des = omega_cmd
        # 5. Att Control
        # R_des^T * R - R^T * R_des
        R_des_T = R_des.transpose(1, 2)
        R_T = R.transpose(1, 2)

        R_err_mat = torch.matmul(R_des_T, R) - torch.matmul(R_T, R_des)
        att_err = self.vee(0.5 * R_err_mat)  # (B, 3)

        w_err = state["w"] - w_des

        # Torque u2
        # (J @ att_err)
        term1 = -self.kp_att * att_err - self.kd_att * w_err
        J_term1 = torch.matmul(self.inertia, term1.unsqueeze(-1)).squeeze(-1)

        # w x Jw
        Jw = torch.matmul(self.inertia, state["w"].unsqueeze(-1)).squeeze(-1)
        w_cross_Jw = torch.linalg.cross(state["w"], Jw, dim=1)

        u2 = J_term1 + w_cross_Jw  # (B, 3)
        cmd_w = w_des - self.kp_att * att_err - self.kd_att * w_err
        # 6. Motor Allocation
        # TM: (4, B)
        TM = torch.stack([u1, u2[:, 0], u2[:, 1], u2[:, 2]], dim=0)

        # thrusts = InvAlloc @ TM
        rotor_thrusts = torch.matmul(self.TM_to_f, TM)  # (4, 4) @ (4, B) -> (4, B)

        # Speeds
        motor_speeds_sq = rotor_thrusts / self.k_eta
        motor_speeds = torch.sign(motor_speeds_sq) * torch.sqrt(
            torch.abs(motor_speeds_sq)
        )

        return {
            "cmd_motor_speeds": motor_speeds.T,  # (B, 4)
            "cmd_motor_thrusts": rotor_thrusts.T,  # (B, 4)
            "cmd_thrust": u1,  # (B,)
            "cmd_moment": u2,  # (B, 3)
            "cmd_q": q_des,  # (B, 4)
            "cmd_w": cmd_w,  # (B, 3)
            "cmd_v": -self.kp_vel * pos_err + flat["x_dot"],  # (B, 3)
            "cmd_acc": F_des / self.mass,  # (B, 3)
        }


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
