import torch
import yaml
import genesis as gs
import numpy as np
from genesis_drones.env.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task
from genesis_drones.flight.quadrotor_control import SE3Control

def main():
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
        env_config = env_config, 
        flight_config = flight_config,
        num_envs=1,
    )

    track_task = Track_task(
        genesis_env = genesis_env, 
        env_config = env_config, 
        task_config = task_config,
        train_config = train_config,
        num_envs=1,
    )
    controller = SE3Control("config/controller_track/minco_params.yaml")
    obs = track_task.reset()    # tensordict

    with torch.no_grad():
        for step in range(max_sim_step):
            wxyz_quat = genesis_env.drone.odom.body_quat.cpu().numpy().flatten()  
            xyzw_quat = wxyz_quat[1:].tolist() + [wxyz_quat[0]]
                
            state = {
            "x": genesis_env.drone.odom.world_pos.cpu().numpy().flatten(),
            "v": genesis_env.drone.odom.world_linear_vel.cpu().numpy().flatten(),
            "q": xyzw_quat,
            "w": genesis_env.drone.odom.body_ang_vel.cpu().numpy().flatten()
            }
            flat = {
            "x": track_task.command_buf.squeeze().tolist(),
            "x_dot": [0, 0, 0],
            "x_ddot": [0, 0, 0],
            "x_dddot": [0, 0, 0],
            "yaw": 0.0,
            "yaw_dot": 0.0
            }
            ctrl = controller.update(0, state, flat)
            action = controller.action(
                control=ctrl,
                flight_config=flight_config,
                env_config = env_config,
                device=device,
            )

            obs, rews, dones, infos = track_task.step(action)

if __name__ == "__main__":
    main()  