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
import time
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
from robot_utils import normalize_gripper_action, invert_gripper_action

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
    model_id: str = ""
    TIME_STAMP: str = ""

class MikasaEnvWrapper:
    """Wrap Mikasa environment to be compatible with lerobot format"""
    
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
        """Convert Mikasa observation format to lerobot format"""
        lerobot_obs = {}
        
        # Process primary image
        if "image_primary" in obs:
            img_primary = obs["image_primary"]
            if img_primary.ndim == 4:
                img = img_primary.permute(0, 3, 1, 2)
            elif img_primary.ndim == 3:
                img = img_primary.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {img_primary.shape}")
            lerobot_obs["observation.images.image"] = img
            
        # Process wrist image
        if "image_wrist" in obs:
            img_wrist = obs["image_wrist"]
            if img_wrist.ndim == 4:
                img = img_wrist.permute(0, 3, 1, 2)
            elif img_wrist.ndim == 3:
                img = img_wrist.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected image shape: {img_wrist.shape}")
            lerobot_obs["observation.images.wrist_image"] = img
        
        # Process state
        if "state" in obs:
            state = obs["state"]
            if state.ndim == 2:
                state = state[:, :8]
            elif state.ndim == 1:
                state = state[:8].unsqueeze(0)
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
            lerobot_obs["observation.state"] = state
            
        return lerobot_obs
    
    @property
    def num_envs(self):
        return self._num_envs
    
    @num_envs.setter
    def num_envs(self, value):
        self._num_envs = value
        
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

    args = Args(cfg)
    env = get_mikasa_eval_env(args)
    return MikasaEnvWrapper(env, cfg)

def make_mikasa_policy_preprocessor(policy: PreTrainedConfig, pretrained_path: str):
    """Create custom preprocessor for Mikasa environment"""
    class MikasaPreprocessor(PolicyProcessorPipeline):
        def __init__(self):
            super().__init__(steps=[], name="mikasa_preprocessor")
            
        def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            if "observation.images.image" in obs:
                img = obs["observation.images.image"]
                if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
                    img = img.float() / 255.0
                result["observation.images.image"] = img
                
            if "observation.images.wrist_image" in obs:
                img = obs["observation.images.wrist_image"]
                if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
                    img = img.float() / 255.0
                result["observation.images.wrist_image"] = img
            
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
    model_id: str = "",
):
    TIME_STAMP = time.strftime("%Y%m%d_%H%M%S")
    config = MikasaEvalConfig(
        env_id=env_id,
        num_eval_steps=n_steps,
        num_eval_episodes=n_episodes,
        seed=seed,
        project_name=project_name,
        model_id=model_id,
        TIME_STAMP=TIME_STAMP,
    )
    
    device = get_safe_torch_device(device)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    
    print(f"Creating Mikasa environment: {env_id}")
    env = make_mikasa_env(config)
    
    print(f"Loading dataset metadata: {ds_meta_path}")
    ds_meta = LeRobotDatasetMetadata(repo_id="local", root=ds_meta_path)
    
    print(f"Loading model: {policy_path}")
    policy = make_policy(
        cfg=PreTrainedConfig.from_pretrained(policy_path),
        ds_meta=ds_meta,
        rename_map={},
    )
    policy.to(device)
    policy.eval()
    
    preprocessor = make_mikasa_policy_preprocessor(policy.config, policy_path)[0]
    eval_metrics = defaultdict(list)
    
    for ep_idx in trange(n_episodes, desc="Evaluating episodes"):
        obs, info = env.reset(seed=[seed + ep_idx], options={})
        policy.reset()
        
        for step_idx in range(n_steps):
            obs = preprocessor(obs)
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    obs[k] = v.to(device)
            
            with torch.inference_mode():
                action = policy.select_action(obs)
                
            action_np = action.to("cpu").numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]
            action_np = normalize_gripper_action(action_np)
            action_np = invert_gripper_action(action_np)

            obs, reward, done, trunc, info = env.step(action_np)
            if done or trunc:
                break
        
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict) and "is_success" in final_info:
                success = final_info["is_success"].item()
                eval_metrics["success"].append(success)
                print(f"Episode {ep_idx}: success={success}")
    
    if eval_metrics["success"]:
        success_rate = np.mean(eval_metrics["success"]) * 100
        print(f"\n===== Evaluation Results =====")
        print(f"Environment: {env_id}")
        print(f"Total episodes: {n_episodes}")
        print(f"Success count: {sum(eval_metrics['success'])}")
        print(f"Success rate: {success_rate:.2f}%")
        
        # 注意：此处原代码 success_rate 是 float，没有 .to_csv 方法
        # 如果需要保存，建议使用简单的文件写入或 pandas
        save_path = Path(f"outputs/eval/{project_name}/{model_id}/{TIME_STAMP}")
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "success_rate.txt", "w") as f:
            f.write(str(success_rate))
    
    env.close()
    return eval_metrics

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate lerobot models in Mikasa environment")
    parser.add_argument("--policy.path", type=str, required=True)
    parser.add_argument("--policy.ds_meta_path", type=str, required=True)
    parser.add_argument("--env.id", type=str, default="RememberColor3-v0")
    parser.add_argument("--eval.n_episodes", type=int, default=100)
    parser.add_argument("--eval.n_steps", type=int, default=60)
    parser.add_argument("--eval.seed", type=int, default=0)
    parser.add_argument("--eval.batch_size", type=int, default=1)
    parser.add_argument("--policy.device", type=str, default="cuda:0")
    parser.add_argument("--project.name", type=str, default="lerobot")
    parser.add_argument("--model.id", type=str, default="")
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
    )

if __name__ == "__main__":
    main()