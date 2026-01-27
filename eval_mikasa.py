#!/usr/bin/env python
"""
Evaluate lerobot-trained models in Mikasa environment

Usage Example:
python eval_lerobot_on_mikasa.py \
    --policy.path=outputs/train/groot_n1/checkpoints/pretrained_model \
    --env.id=RememberColor3-v0 \
    --eval.n_episodes=100 \
    --eval.batch_size=1
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from collections import defaultdict

import numpy as np
import torch
import gymnasium as gym
from tqdm import trange

# Add MIKASA and lerobot paths
sys.path.append("/media/raid/workspace/tengbo/lerobot/third_party/MIKASA-Robo/baselines/openvla")
sys.path.append("/media/raid/workspace/tengbo/lerobot/src")

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.configs import parser
from lerobot.configs.eval import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.envs.utils import preprocess_observation
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from mikasa_utils import get_mikasa_eval_env

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
    render_mode: str = "all"
    
    # LeRobot Compatible Configuration
    include_rgb: bool = True
    include_joints: bool = False
    include_state: bool = True
    include_oracle: bool = False

    project_name: str = "lerobot"


class MikasaEnvWrapper:
    """Wrap Mikasa environment to be compatible with lerobot format"""
    
    def __init__(self, env, config: MikasaEvalConfig):
        self.env = env
        self.config = config
        self.num_envs = 1
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # print("obs: ", obs.keys())
        # print("info: ", info.keys())
        # print("state: ", obs['state'])
        return self._convert_observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_observation(obs), reward, terminated, truncated, info
    
    def _convert_observation(self, obs: Dict) -> Dict[str, Any]:
        """Convert Mikasa observation format to lerobot format"""
        # Mikasa raw observation format (from CameraWrapper):
        # - image_primary: (B, H, W, C) uint8 or (H, W, C) uint8
        # - image_wrist: (B, H, W, C) uint8 or (H, W, C) uint8
        
        # Convert to lerobot expected format (key: need explicit batch dimension):
        # - observation.images.{camera_name}: (B, C, H, W) uint8
        # - observation.state: (B, D) float32
        
        lerobot_obs = {}
        
        # Process primary image: keep batch dimension
        if "image_primary" in obs:
            img_primary = obs["image_primary"]  # (1, H, W, C) or (H, W, C)
            if img_primary.ndim == 4:
                # (B, H, W, C) -> (B, C, H, W)
                img = img_primary.permute(0, 3, 1, 2)
            elif img_primary.ndim == 3:
                # (H, W, C) -> (1, C, H, W) - add batch=1 dimension
                img = img_primary.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {img_primary.shape}")
            lerobot_obs["observation.images.image"] = img
            
        # Process wrist image
        if "image_wrist" in obs:
            img_wrist = obs["image_wrist"]  # (1, H, W, C) or (H, W, C)
            if img_wrist.ndim == 4:
                img = img_wrist.permute(0, 3, 1, 2)
            elif img_wrist.ndim == 3:
                img = img_wrist.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {img_wrist.shape}")
            lerobot_obs["observation.images.wrist_image"] = img
        
        # Process state: ensure batch dimension
        if "state" in obs:
            state = obs["state"]  # (1, D) or (D,)
            if state.ndim == 2:
                state = state[:, :8]  # (B, 8)
            elif state.ndim == 1:
                state = state[:8].unsqueeze(0)  # (1, 8) - add batch dimension
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
            lerobot_obs["observation.state"] = state
            
        return lerobot_obs
            
        return lerobot_obs
    
    @property
    def num_envs(self):
        return self._num_envs
    
    @num_envs.setter
    def num_envs(self, value):
        self._num_envs = value
        
    def call(self, method, *args, **kwargs):
        """Support VectorEnv call method"""
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
    # Use lerobot config structure, but set MIKASA-specific parameters
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
            
    args = Args(cfg)
    env = get_mikasa_eval_env(args)
    return MikasaEnvWrapper(env, cfg)


def make_mikasa_policy_preprocessor(policy: PreTrainedConfig, pretrained_path: str):
    """
    Create custom preprocessor for Mikasa environment
    Handle conversion from Mikasa observation format to model input format
    """
    
    class MikasaPreprocessor(PolicyProcessorPipeline):
        """Custom preprocessor: convert Mikasa observation to model input"""
        
        def __init__(self):
            super().__init__(steps=[], name="mikasa_preprocessor")
            
        def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            
            # Process primary image
            if "observation.images.image" in obs:
                img = obs["observation.images.image"]
                if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
                    img = img.float() / 255.0
                result["observation.images.image"] = img
                
            # Process wrist image
            if "observation.images.wrist_image" in obs:
                img = obs["observation.images.wrist_image"]
                if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
                    img = img.float() / 255.0
                result["observation.images.wrist_image"] = img
            
            # Keep state
            if "observation.state" in obs:
                result["observation.state"] = obs["observation.state"]
                
            return result
    
    return MikasaPreprocessor(), None


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
):
    """
    Evaluate lerobot-trained models in Mikasa environment
    
    Args:
        policy_path: Model path (containing model.safetensors and config.json)
        ds_meta_path: Dataset metadata path
        env_id: Mikasa environment ID
        n_episodes: Number of evaluation episodes
        n_steps: Max steps per episode
        seed: Random seed
        batch_size: Batch size (Mikasa currently only supports 1)
        device: Device
    """
    
    # 1. Configuration
    config = MikasaEvalConfig(
        env_id=env_id,
        num_eval_steps=n_steps,
        num_eval_episodes=n_episodes,
        seed=seed,
        project_name=project_name,
    )
    
    # 2. Setup device and seed
    device = get_safe_torch_device(device)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    
    # 3. Create environment
    print(f"Creating Mikasa environment: {env_id}")
    env = make_mikasa_env(config)
    
    # 4. Load dataset metadata
    print(f"Loading dataset metadata: {ds_meta_path}")
    ds_meta = LeRobotDatasetMetadata(repo_id="local", root=ds_meta_path)
    
    # 5. Load model
    print(f"Loading model: {policy_path}")
    policy = make_policy(
        cfg=PreTrainedConfig.from_pretrained(policy_path),
        ds_meta=ds_meta,
        rename_map={},
    )
    policy.to(device)
    policy.eval()
    
    # 5. Create preprocessor (handle observation format conversion)
    preprocessor = make_mikasa_policy_preprocessor(
        policy.config, 
        policy_path
    )[0]
    
    # 6. Evaluation loop
    eval_metrics = defaultdict(list)
    
    for ep_idx in trange(n_episodes, desc="Evaluating episodes"):
        # Reset environment
        obs, info = env.reset(seed=[seed + ep_idx], options={})
        print("obs keys: ", list(obs.keys()))
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: type={type(v)}")
        print("info keys: ", list(info.keys()))
        # Reset policy
        policy.reset()
        
        for step_idx in range(n_steps):
            # 1. Preprocess observation
            # obs = preprocess_observation(obs)
            obs = preprocessor(obs)
            
            # 2. Move to device
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    obs[k] = v.to(device)
                    
            # 3. Model inference
            with torch.inference_mode():
                action = policy.select_action(obs)
                
            # 4. Post-process action
            action_np = action.to("cpu").numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]  # remove batch dimension
                
            # 5. Execute action
            obs, reward, done, trunc, info = env.step(action_np)
            
            # 6. Check termination
            if done or trunc:
                break
        
        # Record metrics
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict) and "is_success" in final_info:
                success = final_info["is_success"].item()
                eval_metrics["success"].append(success)
                print(f"Episode {ep_idx}: success={success}")
    
    # 7. Calculate and print results
    if eval_metrics["success"]:
        success_rate = np.mean(eval_metrics["success"]) * 100
        print(f"\n===== Evaluation Results =====")
        print(f"Environment: {env_id}")
        print(f"Total episodes: {n_episodes}")
        print(f"Success count: {sum(eval_metrics['success'])}")
        print(f"Success rate: {success_rate:.2f}%")
    
    env.close()
    
    return eval_metrics


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
    )


if __name__ == "__main__":
    main()
