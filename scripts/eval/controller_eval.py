import torch
import yaml
import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation as R
from genesis_drones.env.genesis_env import Genesis_env
from genesis_drones.tasks.track_task import Track_task
from genesis_drones.flight.quadrotor_control import SE3Control
from genesis_drones.flight.minco_params import quad_params
from rsl_rl.runners import OnPolicyRunner

def main():
    gs.init(logging_level="warning")
    max_sim_step = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("config/track_rl/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config/track_rl/rl_env.yaml", "r") as file:
        rl_config = yaml.load(file, Loader=yaml.FullLoader)

    with open("config//track_rl/flight.yaml", "r") as file:
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
    controller = SE3Control(quad_params)
    obs = track_task.reset()    # tensordict

    with torch.no_grad():
        for step in range(max_sim_step):
            wxyz_quat = genesis_env.drone.odom.body_quat.cpu().numpy().flatten()  
            print(genesis_env.drone.odom.world_pos)
            xyzw_quat = wxyz_quat[1:].tolist() + [wxyz_quat[0]]
                
            state = {
            "x": genesis_env.drone.odom.world_pos.cpu().numpy().flatten(),
            "v": genesis_env.drone.odom.world_linear_vel.cpu().numpy().flatten(),
            "q": xyzw_quat,
            "w": genesis_env.drone.odom.body_ang_vel.cpu().numpy().flatten()
            }
            flat= {
            "x": track_task.command_buf.squeeze().tolist(),
            "x_dot": [0, 0, 0],
            "x_ddot": [0, 0, 0],
            "x_dddot": [0, 0, 0],
            "yaw": 0.0,
            "yaw_dot": 0.0
            }
            ctrl = controller.update(0, state, flat)
            min_t = 0  
            max_t =25
            thrust_norm = (ctrl["cmd_thrust"] - min_t) / ((max_t - min_t))
            thrust_norm = thrust_norm * 2 - 1
            wx, wy, wz = ctrl["cmd_w"]   # 期望机体系角速度，单位 rad/s
            roll_norm  = wx / 10
            pitch_norm = wy / 10
            yaw_norm   = wz / 7
            roll_norm  = np.clip(roll_norm,  -1.0, 1.0)
            pitch_norm = np.clip(pitch_norm, -1.0, 1.0)
            yaw_norm   = np.clip(yaw_norm,   -1.0, 1.0)
            # # -------------------------
            # # 3. 拼成 action 向量
            # # -------------------------
            action = np.hstack([roll_norm, pitch_norm, yaw_norm,thrust_norm ]).reshape(1, -1)
            action_tensor = torch.from_numpy(action).to(device).float()

            obs, rews, dones, infos = track_task.step(action_tensor)

if __name__ == "__main__" :
    main()


    