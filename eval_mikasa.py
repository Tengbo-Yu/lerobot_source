#!/usr/bin/env python
"""
Evaluate lerobot-trained models (ACT, Groot, etc.) in Mikasa environment.

Compatibility Guide:
- ACT: Uses dataset_stats for Normalization/Unnormalization.
- Groot: Uses AutoProcessor for VLM encoding and overrides for Normalization.
- Post-processing: Automatically unnormalizes actions back to physical space.
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import defaultdict
import time
import numpy as np
import torch
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt

from lerobot.policies.xvla.utils import quat2axisangle

# Add MIKASA and lerobot paths
sys.path.append("/media/raid/workspace/tengbo/lerobot/third_party/MIKASA-Robo/baselines/openvla")
sys.path.append("/media/raid/workspace/tengbo/lerobot/src")

# --- Import LeRobot Factory ---
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from mikasa_utils import get_mikasa_eval_env
from robot_utils import normalize_gripper_action, invert_gripper_action
import math
@dataclass
class MikasaEvalConfig:
    """Mikasa Environment Configuration"""
    env_id: str = "RememberColor3-v0"
    num_eval_steps: int = 60
    num_eval_episodes: int = 100
    seed: int = 0
    save_video: bool = True
    info_on_video: bool = False
    camera_width: int = 128
    camera_height: int = 128
    control_mode: str = "pd_joint_delta_pos"
    # control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "all"
    
    # LeRobot Compatible Configuration
    include_rgb: bool = True
    include_joints: bool = False
    include_state: bool = True
    include_oracle: bool = False

    project_name: str = "lerobot"
    model_id: str = ""
    TIME_STAMP: str = ""
    policy_type: str = "act"  # act, groot, smolvla, etc.
    task_prompt: str = ""  # Optional prompt for the

class MikasaEnvWrapper:
    """Wrap Mikasa environment to be compatible with LeRobot Processor format"""
    
    def __init__(self, env, config: MikasaEvalConfig):
        self.env = env
        self.config = config
        self.num_envs = 1
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_observation(obs), reward, terminated, truncated, info
    
    def _convert_observation(self, obs: Dict) -> Dict[str, Any]:
        """
        Convert Mikasa observation format to LeRobot standard format.
        
        Requirements for Official Processor:
        1. Float Tensor in [0, 1] range.
        2. Channel First (C, H, W).
        3. NO BATCH DIMENSION (Processor will add it).
        """
        lerobot_obs = {}
        
        # Helper function to process image
        def process_image(img_data):
            if isinstance(img_data, np.ndarray):
                img_data = torch.from_numpy(img_data)
            
            # Handle Dimensions: (H, W, C) -> (C, H, W)
            if img_data.ndim == 3 and img_data.shape[-1] == 3: 
                img_data = img_data.permute(2, 0, 1)
            # Handle potential Batch dim: (1, H, W, C) -> (C, H, W)
            elif img_data.ndim == 4:
                img_data = img_data.permute(0, 3, 1, 2).squeeze(0)
            
            # Normalize uint8 [0, 255] -> float [0, 1]
            if img_data.dtype == torch.uint8:
                img_data = img_data.float() / 255.0
            return img_data
        
        # print(obs.keys())

        # Process primary image
        if "image_primary" in obs:
            lerobot_obs["observation.images.image"] = process_image(obs["image_primary"])
            
        # Process wrist image
        if "image_wrist" in obs:
            lerobot_obs["observation.images.wrist_image"] = process_image(obs["image_wrist"])
        
        # Process state: (D,) Float Tensor
        if "state" in obs:
            state = obs["state"]
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)
            
            # Ensure 1D: (1, D) -> (D,)
            if state.ndim == 2:
                state = state.squeeze(0)
            
            # Take first 8 dims (Joints + Gripper)
            lerobot_obs["observation.state"] = state.float()
            # lerobot_obs["observation.state"] = state.float()
            # lerobot_obs["observation.state"] = torch.zeros_like(state[:8]).float()
        
        if self.config.policy_type == "smolvla" and self.config.task_prompt:
            # Add task prompt for Smolvla
            lerobot_obs["task"] = self.config.task_prompt
          
        return lerobot_obs
    
    def call(self, method, *args, **kwargs):
        return getattr(self.env, method)(*args, **kwargs)
    
    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()
        
    @property
    def unwrapped(self):
        return self.env.unwrapped


def make_mikasa_env(cfg: MikasaEvalConfig):
    """Create Mikasa environment"""
    class Args:
        def __init__(self, cfg: MikasaEvalConfig):
            self.env_id = cfg.env_id
            self.include_rgb = cfg.include_rgb
            self.include_joints = cfg.include_joints
            self.include_state = cfg.include_state
            self.camera_width = cfg.camera_width
            self.camera_height = cfg.camera_height
            self.control_mode = cfg.control_mode
            self.render_mode = cfg.render_mode
            self.save_video = cfg.save_video
            self.info_on_video = cfg.info_on_video
            self.noop_steps = 1
            self.shader = "default"
            self.sim_backend = "gpu"
            self.include_oracle = cfg.include_oracle
            self.project_name = cfg.project_name
            self.model_id = cfg.model_id
            self.TIME_STAMP = cfg.TIME_STAMP
            self.policy_type = cfg.policy_type
            self.task_prompt = cfg.task_prompt

    args = Args(cfg)
    env = get_mikasa_eval_env(args)
    return MikasaEnvWrapper(env, cfg)


def evaluate_lerobot_on_mikasa(
    policy_path: str,
    ds_meta_path: str,
    env_id: str = "RememberColor3-v0",
    n_episodes: int = 100,
    n_steps: int = 60,
    seed: int = 0,
    batch_size: int = 1,
    device: str = "cuda:0",
    project_name: str = "lerobot",
    model_id: str = "",
    policy_type: str = "act",
    task_prompt: str = "",
):
    """Evaluate lerobot-trained models in Mikasa environment"""
    TIME_STAMP = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. Configuration
    config = MikasaEvalConfig(
        env_id=env_id,
        num_eval_steps=n_steps,
        num_eval_episodes=n_episodes,
        seed=seed,
        project_name=project_name,
        model_id=model_id,
        TIME_STAMP=TIME_STAMP,
        policy_type=policy_type,
        task_prompt=task_prompt
    )
    control_mode = config.control_mode
    # 2. Setup device and seed
    device_obj = get_safe_torch_device(device)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    
    # 3. Create environment
    print(f"Creating Mikasa environment: {env_id}")
    env = make_mikasa_env(config)
    
    # 4. Load dataset metadata (Crucial for Stats!)
    print(f"Loading dataset metadata: {ds_meta_path}")
    ds_meta = LeRobotDatasetMetadata(repo_id="local", root=ds_meta_path)
    
    # 5. Load model
    print(f"Loading model: {policy_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)

    # original_n_action_steps = policy_cfg.n_action_steps
    # policy_cfg.n_action_steps = 1  # 每步都重新观察
    # print(f"Overriding n_action_steps: {original_n_action_steps} -> {policy_cfg.n_action_steps}")
    
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=ds_meta,
        rename_map={},
    )
    policy.to(device_obj)
    policy.eval()
    
    # 6. Create Processor (Unified Interface)
    print(f"Creating processor for policy type: {policy_cfg.type}")
    
    processor_args = {
        "policy_cfg": policy_cfg,
        "pretrained_path": policy_path,
        "dataset_stats": ds_meta.stats, # Groot & ACT need this
    }

    processor_args["preprocessor_overrides"] = {
        "device_processor": {"device": device_obj.type},
        "normalizer_processor": {
            "stats": ds_meta.stats,
            "features": {**policy_cfg.input_features, **policy_cfg.output_features},
            "norm_map": policy_cfg.normalization_mapping,
        },
    }
    processor_args["postprocessor_overrides"] = {
        "unnormalizer_processor": {
            "stats": ds_meta.stats,
            "features": policy_cfg.output_features,
            "norm_map": policy_cfg.normalization_mapping,
        },
    }

    # Create BOTH processors
    preprocessor, postprocessor = make_pre_post_processors(**processor_args)
    
    # 7. Evaluation loop
    eval_metrics = defaultdict(list)
    output_dir = Path(f"outputs/eval/{project_name}/{model_id}/{TIME_STAMP}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"n_action_steps: {policy_cfg.n_action_steps}")
    print(f"chunk_size: {policy_cfg.chunk_size}")

    for ep_idx in trange(n_episodes, desc="Evaluating episodes"):
        obs, info = env.reset(seed=[seed + ep_idx], options={})
        policy.reset()
        
        success = False
        
        # Storage for plotting
        episode_states = []
        episode_actions = []
        
        for step_idx in range(n_steps):
            # Store raw state before preprocessing (for plotting)
            if "observation.state" in obs:
                raw_state = obs["observation.state"].cpu().numpy().copy()
                episode_states.append(raw_state)
                # print(obs["observation.state"].shape)
            
            # A. Preprocess
            # obs = preprocessor(obs)
            processed_obs = preprocessor(obs)
            # B. Model Inference
            with torch.inference_mode():
                # action = policy.select_action(obs)
                action = policy.select_action(processed_obs)

            # C. Postprocess 
            action = postprocessor(action)

            # D. Convert to Numpy for Environment
            action_np = action.to("cpu").numpy()

            # Remove Batch dim (1, D) -> (D,)
            # print(action_np.shape)
            if action_np.ndim == 2:
                action_np = action_np[0]
            
            if control_mode == "pd_ee_delta_pose":
                eef_pos = action_np[:3]
                eef_axisangle = quat2axisangle(action_np[3:7])
                gripper = action_np[7]
                action_np = np.concatenate([eef_pos, eef_axisangle, [gripper]], axis=-1)

            # Store action for plotting
            episode_actions.append(action_np.copy())
                
            # E. Robot Specific Utils (Optional, depending on your setup)
            action_np = normalize_gripper_action(action_np)
            action_np = invert_gripper_action(action_np)

            # action_np = action_np[:-1]

            # F. Execute
            obs, reward, done, trunc, info = env.step(action_np)
            
            if done or trunc:
                # Success Check
                if "is_success" in info:
                    success = info["is_success"]
                elif "success" in info:
                    success = info["success"]
                elif "final_info" in info and "is_success" in info["final_info"]:
                    success = info["final_info"]["is_success"]
                
                if hasattr(success, "item"): 
                    success = success.item()
                
                eval_metrics["success"].append(success)
                print(f"Episode {ep_idx}: success={success}")
                break
        
        if not (done or trunc):
             eval_metrics["success"].append(False)
             print(f"Episode {ep_idx}: success=False (Timeout)")
        
        # Plot state-action for this episode
        # if episode_states and episode_actions:
        #     plot_state_action(
        #         states=episode_states,
        #         actions=episode_actions,
        #         episode_idx=ep_idx,
        #         output_dir=output_dir,
        #     )
    
    # 8. Results
    if eval_metrics["success"]:
        success_rate = np.mean(eval_metrics["success"]) * 100
        print(f"\n===== Evaluation Results =====")
        print(f"Environment: {env_id}")
        print(f"Policy Type: {policy_cfg.type}")
        print(f"Success rate: {success_rate:.2f}%")
        
        with open(output_dir / "success_rate.txt", "w") as f:
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            f.write(f"Total Episodes: {n_episodes}\n")
    else:
        print("\nWARNING: No success metrics were recorded.")
    
    env.close()
    return eval_metrics


def plot_state_action(
    states: List[np.ndarray], 
    actions: List[np.ndarray], 
    episode_idx: int, 
    output_dir: Path,
    joint_names: List[str] = None
):
    """
    Plot observation.state and action for each joint in a single figure with 8 subplots.
    
    Args:
        states: List of state arrays, each of shape (8,)
        actions: List of action arrays, each of shape (8,)
        episode_idx: Episode index for filename
        output_dir: Directory to save the plot
        joint_names: Optional list of joint names for subplot titles
    """
    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(7)] + ["Gripper"]
    
    # Convert to numpy arrays
    states_arr = np.array(states)  # Shape: (T, 8)
    actions_arr = np.array(actions)  # Shape: (T, 8)
    
    timesteps = np.arange(len(states))
    
    # Create figure with 8 subplots (2 rows x 4 cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(8):
        ax = axes[i]
        
        # Plot state and action for this joint
        ax.plot(timesteps, states_arr[:, i], label='State', color='blue', linewidth=1.5)
        ax.plot(timesteps, actions_arr[:, i], label='Action', color='red', linewidth=1.5, linestyle='--')
        
        ax.set_title(joint_names[i], fontsize=12)
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Episode {episode_idx}: State vs Action', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plot_dir = output_dir / "state_action_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"episode_{episode_idx:04d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate lerobot models in Mikasa environment")
    parser.add_argument("--policy.path", type=str, required=True, help="Model path")
    parser.add_argument("--policy.ds_meta_path", type=str, required=True, help="Dataset metadata path")
    parser.add_argument("--env.id", type=str, default="RememberColor3-v0", help="Environment ID")
    parser.add_argument("--eval.n_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--eval.n_steps", type=int, default=60, help="Max steps per episode")
    parser.add_argument("--eval.seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval.batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--policy.device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--project.name", type=str, default="lerobot", help="Project name")
    parser.add_argument("--model.id", type=str, default="", help="Model ID")
    parser.add_argument("--policy.type", type=str, default="act", help="Policy type (act, groot, smolvla, etc.)")
    parser.add_argument("--eval.task_prompt", type=str, default="", help="Optional task prompt for the evaluation")
    args = parser.parse_args()
    
    evaluate_lerobot_on_mikasa(
        policy_path=getattr(args, "policy.path"),
        ds_meta_path=getattr(args, "policy.ds_meta_path"),
        env_id=getattr(args, "env.id"),
        n_episodes=getattr(args, "eval.n_episodes"),
        n_steps=getattr(args, "eval.n_steps"),
        seed=getattr(args, "eval.seed"),
        batch_size=getattr(args, "eval.batch_size"),
        device=getattr(args, "policy.device"),
        project_name=getattr(args, "project.name"),
        model_id=getattr(args, "model.id"),
        policy_type=getattr(args, "policy.type"),
        task_prompt=getattr(args, "eval.task_prompt")
    )

if __name__ == "__main__":
    main()