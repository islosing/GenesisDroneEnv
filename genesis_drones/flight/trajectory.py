import numpy as np

def circular_trajectory(t, R, omega, center):
    """
    Generate a circular trajectory in 3D space.

    Args:
        t (float): Time variable.
        R (float): Radius of the circle.
        omega (float): Angular velocity.
        center (tuple): Center of the circle (cx, cy, cz).
    """
    cx, cy, cz = center

    x_ref = [
        cx + R * np.cos(omega * t),  # x
        cy,  # y
        cz + R * np.sin(omega * t),  # z
    ]
    x_dot_ref = [
        -R * omega * np.sin(omega * t),
        0.0,
        R * omega * np.cos(omega * t),
    ]
    x_ddot_ref = [
        -R * omega**2 * np.cos(omega * t),
        0.0,
        -R * omega**2 * np.sin(omega * t),
    ]
    x_dddot_ref = [
        R * omega**3 * np.sin(omega * t),
        0.0,
        -R * omega**3 * np.cos(omega * t),
    ]
    yaw = 0.0
    yaw_dot = 0.0

    return x_ref, x_dot_ref, x_ddot_ref, x_dddot_ref, yaw, yaw_dot