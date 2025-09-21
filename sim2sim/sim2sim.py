import argparse
import time
import numpy as np
import mujoco
import torch
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import torchvision
import cv2

def get_obj_default(task_name):
    if task_name == "kick_ball":
        return np.array([0.75, 0.0, 0.103, 1, 0, 0, 0])
    elif task_name == "kick_box":
        return np.array([0.75, 0.0, 0.1905, 1, 0, 0, 0])
    elif task_name == "push_box":
        return np.array([1.0, 0.0, 0.5, 1, 0, 0, 0])
    elif task_name == "lift_box":
        return np.array([1.2, 0.0, 0.254, 1, 0, 0, 0])
    else:
        raise NotImplementedError
# -------------------------------------------------------------------
# Any helper functions for sim or arrow drawing as needed
# -------------------------------------------------------------------
def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    eulerVec[1] = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)

    return eulerVec

# -------------------------------------------------------------------
# Main low-level policy controller that:
#   - feeds into policy
#   - runs the sim
# -------------------------------------------------------------------

class CameraConfig:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.near_clip = 0.1
        self.far_clip = 2.0
        self.dis_noise = 0.0
        self.resized = (80, 45)
        self.buffer_len = 2
        self.update_interval = 5


def process_depth_image(depth_image, cfg):
    resize_transform = torchvision.transforms.Resize(
        (cfg.resized[1], cfg.resized[0]),
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        antialias=True
    )
    depth_image = resize_transform(depth_image.unsqueeze(0)).squeeze(0)
    depth_image = depth_image - 0.5
    return depth_image

def update_depth_buffer(renderer, envs_data, depth_buffer, global_counter):
    cfg = CameraConfig()
    if global_counter % cfg.update_interval != 0:
        return depth_buffer

    renderer.update_scene(envs_data, camera="d435_camera")
    renderer.enable_depth_rendering()
    depth_image_raw = renderer.render()
    renderer.disable_depth_rendering()
    depth_normalized = (depth_image_raw - cfg.near_clip) / (cfg.far_clip - cfg.near_clip) + np.random.uniform(-cfg.dis_noise, cfg.dis_noise, size=depth_image_raw.shape)
    depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

    depth_image_tensor = torch.from_numpy(depth_normalized.copy()).float()

    processed_depth = process_depth_image(depth_image_tensor, cfg)
    depth_normalized = processed_depth + 0.5
    depth_normalized = depth_normalized.cpu().numpy()
    depth_8u = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    cv2.imshow("Depth Image Processed", depth_colored)
    cv2.waitKey(1)

    if global_counter < cfg.update_interval:
        depth_buffer = torch.stack([processed_depth] * cfg.buffer_len, dim=0)
    else:
        depth_buffer = torch.cat([depth_buffer[1:], processed_depth.unsqueeze(0)], dim=0)
    return depth_buffer

class RealTimePolicyController:
    def __init__(self,
                 task_name, # kick box, kick ball, push box, lift box
                 device='cuda', 
                 record_video=False,):

        self.device = device
        self.task_name = task_name

        # Load policy
        policy_path = f"../checkpoints/{task_name}.pt"
        self.policy = torch.jit.load(policy_path, map_location=device)
        print(f"Policy loaded from {policy_path}")

        # Create MuJoCo sim
        xml_file = f"../asset/g1/{task_name}.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        cfg = CameraConfig()
        self.renderer = mujoco.Renderer(self.model, height=cfg.height // 2, width=cfg.width // 2)
        resized_h, resized_w = cfg.resized[1], cfg.resized[0]
        self.depth_buffer = torch.zeros((cfg.buffer_len, resized_h, resized_w))

        self.viewer = mjv.launch_passive(self.model, self.data)
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 1
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1
        self.viewer.cam.distance = 4.0

        self.num_actions = 23
        self.num_actions_high_level = 18
        self.sim_duration = 1000.0
        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.command_dim = 1
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action_high_level = np.zeros(self.num_actions_high_level, dtype=np.float32)

        # PD Gains, etc. (adapt as needed)
        self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.4, 0.0, 1.2,
                0.0, -0.4, 0.0, 1.2,
            ])
        
        self.obj_default_state = get_obj_default(task_name)
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([1, 0, 0, 0]),
             np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.2, 0.0, 1.2,
                0.0, -0.2, 0.0, 1.2,])
        ])
        self.mujuco_default_state = np.concatenate([
            self.obj_default_state,
            self.mujoco_default_dof_pos,
        ])
        
        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40,
                40, 40, 40, 40,
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5,
                5, 5, 5, 5,
            ])
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
        self.action_scale = 0.5

        self.n_proprio = 18 + 3 + 2 + 3 * self.num_actions
        self.n_proprio_high_level = 3 + 2 + 2 * self.num_actions + self.num_actions_high_level
        self.proprio_history_buf = deque(maxlen=10)
        self.proprio_history_buf_high_level = deque(maxlen=10)
        self.command_history_buf = deque(maxlen=10)
        for _ in range(10):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
            self.proprio_history_buf_high_level.append(np.zeros(self.n_proprio_high_level))
            self.command_history_buf.append(0)

        self.record_video = record_video

        self.start_timer = 50
        self.command = 0

    def extract_data(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.model.nu:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.model.nu:]
        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        return dof_pos, dof_vel, quat, ang_vel

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, mujuco_default_state=None):
        self.data.qpos = mujuco_default_state
        mujoco.mj_forward(self.model, self.data)

    def run(self):
        # Optionally record video
        if self.record_video:
            import imageio
            video_name = f"{self.task_name}_sim2sim.mp4"
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujuco_default_state)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating...")
        print("Stand still and prepare...")
        for i in pbar:

            t_start = time.time()
            dof_pos, dof_vel, quat, ang_vel = self.extract_data()
            self.depth_buffer = update_depth_buffer(self.renderer, self.data, self.depth_buffer, i)
            
            if i % self.sim_decimation == 0:
                if self.start_timer > 0:
                    self.start_timer -= 1
                elif self.start_timer == 0:
                    self.start_timer -= 1
                    self.command = 1
                    print("Performing task...")

                rpy = quatToEuler(quat)
                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[[4, 5, 10, 11]] = 0.
                obs_proprio = np.concatenate([
                    ang_vel * 0.25,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos),
                    obs_dof_vel * 0.05,
                    self.last_action
                ])

                dummy_high_level_keypoint_command = np.zeros(self.num_actions_high_level, dtype=np.float32)
                obs_full = np.concatenate([dummy_high_level_keypoint_command, obs_proprio])

                obs_hist = np.array(self.proprio_history_buf).flatten()
                obs_buf = np.concatenate([obs_full, obs_hist])

                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                obs_tensor = obs_tensor[:, self.num_actions_high_level:] # remove the first 18 elements (high-level actions)

                depth_image = self.depth_buffer[-1].clone()  # shape: [H, W], range: [-0.5, 0.5]
                depth_image = depth_image.flatten().unsqueeze(0).to(self.device)  # [1, H*W]

                command_history = np.array(self.command_history_buf).flatten()
                command = np.array(self.command, dtype=np.float32).reshape(1)  # [1]

                obs_proprio_high_level = np.concatenate([
                    ang_vel * 0.25,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos),
                    obs_dof_vel * 0.05,
                    self.last_action_high_level
                ])

                command_obs = np.concatenate([
                    command,
                    command_history,
                    obs_proprio_high_level
                ])
                obs_hist_high_level = np.array(self.proprio_history_buf_high_level).flatten()
                obs_proprio_high_level_command = np.concatenate([command_obs, obs_hist_high_level])
                obs_proprio_high_level_command = torch.tensor(obs_proprio_high_level_command).unsqueeze(0).to(self.device)  # [1, obs_dim]
                obs_high_level_all = torch.cat([depth_image, obs_proprio_high_level_command], dim=1)  # [1, H*W + 1]

                obs_tensor = torch.cat((obs_high_level_all, obs_tensor), dim=1)  # [1, obs_dim + H*W]
                with torch.no_grad():
                    raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                self.last_action_high_level = raw_action[:self.num_actions_high_level]
                self.last_action = raw_action[self.num_actions_high_level:]

                obs_full_ = obs_full.copy()
                obs_full_[:self.num_actions_high_level] = self.last_action_high_level
                if i == 0:
                    for _ in range(10):
                        self.proprio_history_buf.append(obs_full_)
                        self.proprio_history_buf_high_level.append(obs_proprio_high_level)
                        self.command_history_buf.append(self.command)
                else:
                    self.proprio_history_buf.append(obs_full_)
                    self.proprio_history_buf_high_level.append(obs_proprio_high_level)
                    self.command_history_buf.append(self.command)

                raw_action = np.clip(raw_action, -10., 10.)
                scaled_actions = raw_action[self.num_actions_high_level:] * self.action_scale
                pd_target = scaled_actions + self.default_dof_pos

                # make camera follow the pelvis
                pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                self.viewer.cam.lookat = pelvis_pos
                self.viewer.sync()
                if mp4_writer is not None:
                    self.renderer.update_scene(self.data, camera=self.viewer.cam)
                    rgb_img = self.renderer.render()
                    processed_depth_tensor = self.depth_buffer[-1]  # range: [-0.5, 0.5]
                    depth_0_to_1 = processed_depth_tensor + 0.5
                    depth_0_to_255 = depth_0_to_1 * 255.0
                    depth_numpy = depth_0_to_255.cpu().numpy().astype(np.uint8)
                    depth_visual_bgr = cv2.applyColorMap(depth_numpy, cv2.COLORMAP_JET)
                    depth_visual_rgb = cv2.cvtColor(depth_visual_bgr, cv2.COLOR_BGR2RGB)
                    main_h, main_w, _ = rgb_img.shape
                    inset_w = main_w // 5
                    inset_h = int(inset_w * (depth_visual_rgb.shape[0] / depth_visual_rgb.shape[1]))
                    depth_inset = cv2.resize(depth_visual_rgb, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
                    rgb_img[0:inset_h, 0:inset_w] = depth_inset
                    text = f"Command: {self.command}"
                    rgb_img = cv2.putText(rgb_img, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    mp4_writer.append_data(rgb_img)

            # PD control
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            torque *= 1.0
            self.data.ctrl = torque
            mujoco.mj_step(self.model, self.data)
            # sleep to maintain real-time pace
            elapsed = time.time() - t_start
            if elapsed < self.sim_dt:
                time.sleep(self.sim_dt - elapsed)
        print("Simulation finished")
        if mp4_writer is not None:
            mp4_writer.close()
            print("Video saved")
        self.viewer.close()


def main_sim(args):
    controller = RealTimePolicyController(
        task_name=args.task_name,
        device='cuda',
        record_video=args.record_video,
    )
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",  help="Path to the policy", choices=["kick_ball", "kick_box", "push_box", "lift_box"], default="kick_ball")
    parser.add_argument("--record_video", action="store_true", help="Record a video")
    args = parser.parse_args()
    main_sim(args)
