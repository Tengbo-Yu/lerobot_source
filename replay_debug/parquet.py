#!/usr/bin/env python3
"""
Replay a specific episode from LeRobot dataset in Mikasa environment.
"""

import sys
import os
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd
import gymnasium as gym

# Add MIKASA and lerobot paths
# Add MIKASA and lerobot paths
sys.path.append("/media/raid/workspace/tengbo/lerobot/third_party/MIKASA-Robo/baselines/openvla")
sys.path.append("/media/raid/workspace/tengbo/lerobot/src")
sys.path.append("/media/raid/workspace/tengbo/lerobot")

from mikasa_utils import get_mikasa_eval_env
from robot_utils import normalize_gripper_action, invert_gripper_action


class Args:
    def __init__(self, env_id, camera_width, camera_height, control_mode, render_mode, 
                 save_video, project_name, model_id, TIME_STAMP, include_rgb, include_joints,
                 include_state, include_oracle, noop_steps, shader, sim_backend):
        self.env_id = env_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.save_video = save_video
        self.project_name = project_name
        self.model_id = model_id
        self.TIME_STAMP = TIME_STAMP
        self.include_rgb = include_rgb
        self.include_joints = include_joints
        self.include_state = include_state
        self.include_oracle = include_oracle
        self.noop_steps = noop_steps
        self.shader = shader
        self.sim_backend = sim_backend


def find_successful_seed(
    parquet_path: str,
    episode_index: int = 0,
    env_id: str = "InterceptMedium-v0",
    control_mode: str = "pd_joint_delta_pos",
    sim_backend: str = "cpu",
    seed_start: int = 0,
    seed_end: int = 100,
    max_steps: int = 30,
):
    """
    Loop through different seeds to find one that results in successful replay.
    
    Args:
        parquet_path: Path to the chunk parquet file
        episode_index: Which episode to replay (0-indexed)
        env_id: Mikasa environment ID
        control_mode: Control mode
        sim_backend: Simulation backend ("cpu" or "gpu")
        seed_start: Starting seed value
        seed_end: Ending seed value (exclusive)
        max_steps: Max steps to run before declaring failure
    
    Returns:
        int: Successful seed, or -1 if none found
    """
    # Load episode data
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Filter for specific episode
    episode_df = df[df['episode_index'] == episode_index].reset_index(drop=True)
    
    if len(episode_df) == 0:
        raise ValueError(f"Episode {episode_index} not found in dataset!")
    
    print(f"Episode {episode_index}: {len(episode_df)} frames")
    actions = episode_df['action'].values
    
    # Load full actions (not just first max_steps)
    actions = episode_df['action'].values
    
    print(f"Testing seeds from {seed_start} to {seed_end - 1}...")
    print(f"Episode has {len(actions)} total steps, testing with max {max_steps} steps")
    print()
    
    for seed in range(seed_start, seed_end):
        # Create environment
        TIME_STAMP = time.strftime('%Y%m%d_%H%M%S')
        
        args = Args(
            env_id=env_id,
            camera_width=128,
            camera_height=128,
            control_mode=control_mode,
            render_mode=None,  # No render for speed
            save_video=True,  # No video for speed
            project_name="seed_test",
            model_id=f"seed_{seed}",
            TIME_STAMP=TIME_STAMP,
            include_rgb=False,  # No RGB for speed
            include_joints=False,
            include_state=True,
            include_oracle=False,
            noop_steps=1,
            shader="default",
            sim_backend=sim_backend,
        )
        
        try:
            env = get_mikasa_eval_env(args)
            obs, info = env.reset(seed=[seed], options={})
            
            success = False
            
            for step_idx, action in enumerate(actions[:max_steps]):
                # Apply gripper action transformations
                action = normalize_gripper_action(action)
                action = invert_gripper_action(action)
                
                # For ee_delta_pose mode, only use first 7 dims
                if control_mode == "pd_ee_delta_pose":
                    action = action[:-1]
                
                # Execute step
                obs, reward, done, trunc, info = env.step(action)
                
                if done or trunc:
                    # Check success
                    if "is_success" in info:
                        success = bool(info["is_success"].item())
                    elif "success" in info:
                        success = bool(info["success"].item())
                    elif "final_info" in info and "is_success" in info["final_info"]:
                        success = bool(info["final_info"]["is_success"].item())
                    
                    if success:
                        print(f"✓ SUCCESS with seed={seed} at step {step_idx}!")
                        env.close()
                        return seed
                    break
            
            env.close()
            
        except Exception as e:
            print(f"✗ Seed {seed}: Error - {str(e)[:50]}")
            continue
        
        if seed % 10 == 0:
            print(f"  Tested seeds: {seed_start} to {seed}")
    
    print(f"\nNo successful seed found in range [{seed_start}, {seed_end})")
    return -1


def replay_with_seed(
    parquet_path: str,
    episode_index: int = 0,
    env_id: str = "InterceptMedium-v0",
    control_mode: str = "pd_joint_delta_pos",
    seed: int = 0,
    save_video: bool = True,
    render: bool = True,
    speed: float = 1.0,
):
    """Replay episode with a specific seed."""
    # Load episode data
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Filter for specific episode
    episode_df = df[df['episode_index'] == episode_index].reset_index(drop=True)
    
    if len(episode_df) == 0:
        raise ValueError(f"Episode {episode_index} not found in dataset!")
    
    print(f"Episode {episode_index}: {len(episode_df)} frames")
    print(f"Using seed: {seed}")
    
    # Create environment
    TIME_STAMP = time.strftime('%Y%m%d_%H%M%S')
    
    args = Args(
        env_id=env_id,
        camera_width=128,
        camera_height=128,
        control_mode=control_mode,
        render_mode="all" if render else None,
        save_video=save_video,
        project_name="replay",
        model_id=f"episode_{episode_index}_{control_mode}_seed{seed}",
        TIME_STAMP=TIME_STAMP,
        include_rgb=True,
        include_joints=False,
        include_state=True,
        include_oracle=False,
        noop_steps=1,
        shader="default",
        sim_backend="cpu",  # Use CPU for stability
    )
    
    print(f"Creating environment: {env_id}")
    env = get_mikasa_eval_env(args)
    
    # Reset environment
    obs, info = env.reset(seed=[seed], options={})
    print(f"Environment reset complete")
    
    # Get actions from episode data
    actions = episode_df['action'].values
    
    print(f"Replaying {len(actions)} steps...")
    
    for step_idx, action in enumerate(actions):
        # Apply gripper action transformations
        action = normalize_gripper_action(action)
        action = invert_gripper_action(action)
        
        # For ee_delta_pose mode, only use first 7 dims
        if control_mode == "pd_ee_delta_pose":
            action = action[:-1]
        
        # Execute step
        obs, reward, done, trunc, info = env.step(action)
        
        # Optional: render
        if render:
            try:
                env.render()
            except:
                pass
        
        # Control replay speed
        time.sleep(0.02 / speed)
        
        if (step_idx + 1) % 30 == 0:
            print(f"  Step {step_idx + 1}/{len(actions)}")
        
        if done or trunc:
            print(f"  Episode ended at step {step_idx + 1}")
            break
    
    # Check success
    if "is_success" in info:
        print(f"Success: {info['is_success']}")
    elif "success" in info:
        print(f"Success: {info['success']}")
    
    print("Replay complete!")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Replay or find seed for LeRobot dataset in Mikasa")
    parser.add_argument("--data.path", type=str, 
                        default="/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot/data/chunk-000/file-000.parquet",
                        help="Path to chunk parquet file")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--env.id", type=str, default="InterceptMedium-v0", 
                        help="Mikasa environment ID")
    parser.add_argument("--control.mode", type=str, default="pd_joint_delta_pos",
                        choices=["pd_joint_delta_pos", "pd_ee_delta_pose"],
                        help="Control mode (pd_joint_delta_pos: 8 dims, pd_ee_delta_pose: 7 dims)")
    
    # Seed finding options
    parser.add_argument("--find-seed", action="store_true",
                        help="Loop through seeds to find successful one")
    parser.add_argument("--seed.start", type=int, default=0,
                        help="Starting seed for search")
    parser.add_argument("--seed.end", type=int, default=100,
                        help="Ending seed for search (exclusive)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Specific seed to use for replay")
    
    parser.add_argument("--sim.backend", type=str, default="gpu",
                        choices=["cpu", "gpu"],
                        help="Simulation backend (cpu: stable, gpu: fast but may crash)")
    parser.add_argument("--no-video", action="store_true", help="Don't save video")
    parser.add_argument("--no-render", action="store_true", help="Don't render")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    
    args = parser.parse_args()
    
    if getattr(args, "find_seed"):
        # Find successful seed
        successful_seed = find_successful_seed(
            parquet_path=getattr(args, "data.path"),
            episode_index=getattr(args, "episode"),
            env_id=getattr(args, "env.id"),
            control_mode=getattr(args, "control.mode"),
            sim_backend=getattr(args, "sim.backend"),
            seed_start=getattr(args, "seed.start"),
            seed_end=getattr(args, "seed.end"),
        )
        if successful_seed >= 0:
            print(f"\n使用成功的seed重新replay:")
            replay_with_seed(
                parquet_path=getattr(args, "data.path"),
                episode_index=getattr(args, "episode"),
                env_id=getattr(args, "env.id"),
                control_mode=getattr(args, "control.mode"),
                seed=successful_seed,
                save_video=not getattr(args, "no_video"),
                render=not getattr(args, "no_render"),
                speed=getattr(args, "speed"),
            )
    else:
        # Replay with specific seed
        replay_with_seed(
            parquet_path=getattr(args, "data.path"),
            episode_index=getattr(args, "episode"),
            env_id=getattr(args, "env.id"),
            control_mode=getattr(args, "control.mode"),
            seed=getattr(args, "seed"),
            save_video=not getattr(args, "no_video"),
            render=not getattr(args, "no_render"),
            speed=getattr(args, "speed"),
        )


if __name__ == "__main__":
    main()
