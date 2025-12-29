import argparse
import torch
import yaml
import genesis as gs
import numpy as np
from genesis_drones.env.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task
from genesis_drones.flight.SO3_control import SE3Control
from genesis_drones.flight.trajectory import circular_trajectory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--use-trajectory",
        action="store_true",
        help="Use reference trajectory instead of command buffer",
    )
    p.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Do not use reference trajectory",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.use_trajectory and args.no_trajectory:
        raise ValueError("Cannot set both --use-trajectory and --no-trajectory")

    if args.use_trajectory:
        use_trajectory = True
    elif args.no_trajectory:
        use_trajectory = False
    else:
        use_trajectory = True
    gs.init(logging_level="warning")
    max_sim_step = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("config/track_rl/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/track_rl/rl_env.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/track_rl/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)

    task_config = rl_config["task"]
    train_config = rl_config["train"]

    genesis_env = Genesis_env(
        env_config=env_config,
        flight_config=flight_config,
        num_envs=1,
    )

    track_task = Track_task(
        genesis_env=genesis_env,
        env_config=env_config,
        task_config=task_config,
        train_config=train_config,
        num_envs=1,
    )
    controller = SE3Control("config/controller_track/minco_params.yaml")
    obs = track_task.reset()  # tensordict

    with torch.no_grad():
        for step in range(max_sim_step):
            state = {
                "x": genesis_env.drone.odom.world_pos.flatten(),
                "v": genesis_env.drone.odom.world_linear_vel.flatten(),
                "q": genesis_env.drone.odom.body_quat.flatten(),
                "w": genesis_env.drone.odom.body_ang_vel.flatten(),
            }
            if use_trajectory:
                dt = env_config["dt"]

                t = step * dt

                # circular trajectory
                R = 0.5
                omega = 3
                cx, cy, cz = 0.0, 0.0, 0.8  # center of the circle
                x_ref, x_dot_ref, x_ddot_ref, x_dddot_ref, yaw, yaw_dot = (
                    circular_trajectory(
                        t, R, omega, (cx, cy, cz), device, torch.float32
                    )
                )
            else:
                x_ref = track_task.command_buf.squeeze()
                x_dot_ref = torch.zeros(3, device=device)
                x_ddot_ref = torch.zeros(3, device=device)
                x_dddot_ref = torch.zeros(3, device=device)
                yaw = torch.tensor(0.0, device=device)
                yaw_dot = torch.tensor(0.0, device=device)

            flat = {
                "x": x_ref,
                "x_dot": x_dot_ref,
                "x_ddot": x_ddot_ref,
                "x_dddot": x_dddot_ref,
                "yaw": yaw,
                "yaw_dot": yaw_dot,
            }
            ctrl = controller.update(0, state, flat, None, "wxyz")
            action = controller.action(
                control=ctrl,
                flight_config=flight_config,
                env_config=env_config,
                device=device,
            )

            obs, rews, dones, infos = track_task.step(action)


if __name__ == "__main__":
    main()
