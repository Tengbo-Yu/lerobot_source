#!/usr/bin/env python
"""
评估在 lerobot 中训练的模型在 Mikasa 环境中的表现

使用示例:
python eval_mikasa.py \
    --policy.path=outputs/train/groot_n1/checkpoints/pretrained_model \
    --policy.type=groot \
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

# 添加 MIKASA 和 lerobot 的路径
sys.path.append("/media/raid/workspace/tengbo/lerobot/third_party/MIKASA-Robo/baselines/openvla")
sys.path.append("/media/raid/workspace/tengbo/lerobot/src")

from lerobot.policies.factory import make_policy, make_pre_post_processors, make_policy_config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.configs import parser
from lerobot.configs.eval import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device, set_seed
from lerobot.envs.utils import preprocess_observation

from mikasa_utils import get_mikasa_eval_env


@dataclass
class MikasaEvalConfig:
    """Mikasa 环境配置"""
    env_id: str = "RememberColor3-v0"
    num_eval_steps: int = 60
    num_eval_episodes: int = 100
    seed: int = 0
    save_video: bool = True
    info_on_video: bool = False
    camera_width: int = 128
    camera_height: int = 128
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "all"
    
    # Lerobot 兼容配置
    include_rgb: bool = True
    include_joints: bool = False
    include_state: bool = True


class MikasaEnvWrapper:
    """将 Mikasa 环境包装为兼容 lerobot 格式的环境"""
    
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
        """将 Mikasa 观察格式转换为 lerobot 格式"""
        lerobot_obs = {}
        
        # 处理主视角图像: (1, H, W, C) -> (C, H, W)
        if "image_primary" in obs:
            img_primary = obs["image_primary"]  # (1, H, W, C)
            img = img_primary[0].permute(2, 0, 1)  # (C, H, W)
            lerobot_obs["observation.images.primary"] = img
            
        # 处理手腕图像
        if "image_wrist" in obs:
            img_wrist = obs["image_wrist"]  # (1, H, W, C)
            img = img_wrist[0].permute(2, 0, 1)  # (C, H, W)
            lerobot_obs["observation.images.wrist"] = img
        
        # 处理状态信息
        if "state" in obs:
            state = obs["state"]
            lerobot_obs["observation.state"] = state
            
        return lerobot_obs
    
    @property
    def num_envs(self):
        return self._num_envs
    
    @num_envs.setter
    def num_envs(self, value):
        self._num_envs = value
        
    def call(self, method, *args, **kwargs):
        """支持 VectorEnv 的 call 方法"""
        return getattr(self.env, method)(*args, **kwargs)
    
    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()
        
    @property
    def unwrapped(self):
        return self.env.unwrapped


def make_mikasa_env(cfg: MikasaEvalConfig):
    """创建 Mikasa 环境"""
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
            
    args = Args(cfg)
    env = get_mikasa_eval_env(args)
    return MikasaEnvWrapper(env, cfg)


def make_mikasa_policy_preprocessor(policy_type: str, pretrained_path: str):
    """
    为 Mikasa 环境创建自定义的预处理器
    """
    
    class MikasaPreprocessor(PolicyProcessorPipeline):
        """自定义预处理器：将 Mikasa 观察转换为模型输入"""
        
        def __init__(self):
            super().__init__(steps=[], name="mikasa_preprocessor")
            
        def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            
            # 重命名图像键
            if "observation.images.primary" in obs:
                result["observation.images.main_camera"] = obs["observation.images.primary"]
            if "observation.images.wrist" in obs:
                result["observation.images.hand_camera"] = obs["observation.images.wrist"]
                
            # 保留状态
            if "observation.state" in obs:
                result["observation.state"] = obs["observation.state"]
                
            return result
    
    return MikasaPreprocessor(), None


def evaluate_lerobot_on_mikasa(
    policy_path: str,
    policy_type: str,
    env_id: str = "RememberColor3-v0",
    n_episodes: int = 100,
    n_steps: int = 60,
    seed: int = 0,
    batch_size: int = 1,
    device: str = "cuda:0",
):
    """
    在 Mikasa 环境中评估 lerobot 训练的模型
    
    Args:
        policy_path: 模型路径 (包含 model.safetensors 和 config.json)
        policy_type: 策略类型 (如 "groot", "pi0", "diffusion", "act" 等)
        env_id: Mikasa 环境 ID
        n_episodes: 评估 episodes 数量
        n_steps: 每个 episode 的最大步数
        seed: 随机种子
        batch_size: 批大小 (Mikasa 当前仅支持 1)
        device: 设备
    """
    
    # 1. 配置
    config = MikasaEvalConfig(
        env_id=env_id,
        num_eval_steps=n_steps,
        num_eval_episodes=n_episodes,
        seed=seed,
    )
    
    # 2. 设置设备和种子
    device = get_safe_torch_device(device)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    
    # 3. 创建环境
    print(f"创建 Mikasa 环境: {env_id}")
    env = make_mikasa_env(config)
    
    # 4. 加载/创建模型配置
    print(f"加载模型: {policy_path} (type: {policy_type})")
    
    # 方法1: 如果有预训练路径，从pretrained_path加载
    if policy_path and os.path.isdir(policy_path):
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        # 允许覆盖 policy_type
        if policy_type:
            policy_cfg.type = policy_type
    else:
        # 方法2: 创建新的配置
        policy_cfg = make_policy_config(policy_type)
        policy_cfg.pretrained_path = policy_path
    
    # 5. 创建策略配置
    policy_cfg.device = device
    
    # 6. 加载模型
    policy = make_policy(
        cfg=policy_cfg,
        env_cfg=None,
        rename_map={},
    )
    policy.eval()
    
    # 7. 创建预处理器
    preprocessor = make_mikasa_policy_preprocessor(policy_type, policy_path)[0]
    
    # 8. 评估循环
    eval_metrics = defaultdict(list)
    
    for ep_idx in trange(n_episodes, desc="评估 episodes"):
        # 重置环境
        obs, info = env.reset(seed=[seed + ep_idx])
        
        # 重置策略
        policy.reset()
        
        for step_idx in range(n_steps):
            # 1. 预处理观察
            obs = preprocess_observation(obs)
            obs = preprocessor(obs)
            
            # 2. 移动到设备
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    obs[k] = v.to(device)
                    
            # 3. 模型推理
            with torch.inference_mode():
                action = policy.select_action(obs)
                
            # 4. 后处理动作
            action_np = action.to("cpu").numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]
                
            # 5. 执行动作
            obs, reward, done, trunc, info = env.step(action_np)
            
            # 6. 检查是否结束
            if done or trunc:
                break
        
        # 记录指标
        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict) and "is_success" in final_info:
                success = final_info["is_success"].item()
                eval_metrics["success"].append(success)
                print(f"Episode {ep_idx}: success={success}")
    
    # 9. 计算并打印结果
    if eval_metrics["success"]:
        success_rate = np.mean(eval_metrics["success"]) * 100
        print(f"\n===== 评估结果 =====")
        print(f"环境: {env_id}")
        print(f"策略类型: {policy_type}")
        print(f"模型路径: {policy_path}")
        print(f"总 episodes: {n_episodes}")
        print(f"成功次数: {sum(eval_metrics['success'])}")
        print(f"成功率: {success_rate:.2f}%")
    
    env.close()
    
    return eval_metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="在 Mikasa 环境中评估 lerobot 模型")
    parser.add_argument("--policy.path", type=str, required=True, help="模型路径")
    parser.add_argument("--policy.type", type=str, required=True, help="策略类型 (groot, pi0, diffusion, act 等)")
    parser.add_argument("--env.id", type=str, default="RememberColor3-v0", help="环境 ID")
    parser.add_argument("--eval.n_episodes", type=int, default=100, help="评估 episodes 数量")
    parser.add_argument("--eval.n_steps", type=int, default=60, help="每个 episode 最大步数")
    parser.add_argument("--eval.seed", type=int, default=0, help="随机种子")
    parser.add_argument("--eval.batch_size", type=int, default=1, help="批大小")
    parser.add_argument("--policy.device", type=str, default="cuda:0", help="设备")
    
    args = parser.parse_args()
    
    evaluate_lerobot_on_mikasa(
        policy_path=getattr(args, "policy.path"),
        policy_type=getattr(args, "policy.type"),
        env_id=getattr(args, "env.id"),
        n_episodes=getattr(args, "eval.n_episodes"),
        n_steps=getattr(args, "eval.n_steps"),
        seed=getattr(args, "eval.seed"),
        batch_size=getattr(args, "eval.batch_size"),
        device=getattr(args, "policy.device"),
    )


if __name__ == "__main__":
    main()