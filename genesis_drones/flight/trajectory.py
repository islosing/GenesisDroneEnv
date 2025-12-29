import numpy as np
import torch
from genesis_drones.flight.compute_altitude import compute_altitude, _yaw_cmd


def circular_trajectory(t, R, omega, center, device=None, dtype=torch.float32):
    cx, cy, cz = center
    device = device or torch.device("cpu")
    t = torch.as_tensor(t, dtype=dtype, device=device)

    R = torch.as_tensor(R, dtype=dtype, device=device)
    omega = torch.as_tensor(omega, dtype=dtype, device=device)
    cx, cy, cz = [torch.as_tensor(v, dtype=dtype, device=device) for v in (cx, cy, cz)]

    x_ref = torch.stack(
        [
            cx + R * torch.cos(omega * t),
            cy,
            cz + R * torch.sin(omega * t),
        ]
    )
    x_dot_ref = torch.stack(
        [
            -R * omega * torch.sin(omega * t),
            torch.zeros_like(t),
            R * omega * torch.cos(omega * t),
        ]
    )
    x_ddot_ref = torch.stack(
        [
            -R * omega**2 * torch.cos(omega * t),
            torch.zeros_like(t),
            -R * omega**2 * torch.sin(omega * t),
        ]
    )
    x_dddot_ref = torch.stack(
        [
            R * omega**3 * torch.sin(omega * t),
            torch.zeros_like(t),
            -R * omega**3 * torch.cos(omega * t),
        ]
    )

    psi_info_cmd, omega_cmd = _yaw_cmd(
        x_dot_ref, x_ddot_ref, x_dddot_ref, device=device
    )
    yaw = psi_info_cmd["psi"]
    yaw_dot = psi_info_cmd["dpsi"]

    return x_ref, x_dot_ref, x_ddot_ref, x_dddot_ref, yaw, yaw_dot
