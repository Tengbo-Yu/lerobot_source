#!/usr/bin/env python3
"""
Replay a specific episode from RLDS tfrecord dataset in Mikasa environment.
"""

import sys
import os
from pathlib import Path
import argparse
import time
import numpy as np

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


def parse_rlds_example(record):
    """Parse a single RLDS example from tf.train.Example."""
    import tensorflow as tf
    
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    
    # Extract steps data
    steps_data = {
        'action': [],
        'state': [],
        'reward': [],
        'is_terminal': [],
        'is_first': [],
        'is_last': [],
    }
    
    # Get repeated features
    action_list = example.features.feature['steps/action'].float_list.value
    state_list = example.features.feature['steps/observation/state'].float_list.value
    reward_list = example.features.feature['steps/reward'].float_list.value
    terminal_list = example.features.feature['steps/is_terminal'].int64_list.value
    is_first_list = example.features.feature['steps/is_first'].int64_list.value
    is_last_list = example.features.feature['steps/is_last'].int64_list.value
    
    # Determine the number of steps
    n_steps = len(is_first_list)
    
    # Parse each step
    action_dim = len(action_list) // n_steps if n_steps > 0 else 0
    state_dim = len(state_list) // n_steps if n_steps > 0 else 0
    
    for i in range(n_steps):
        steps_data['action'].append(action_list[i*action_dim:(i+1)*action_dim])
        steps_data['state'].append(state_list[i*state_dim:(i+1)*state_dim])
        steps_data['reward'].append(reward_list[i] if i < len(reward_list) else 0)
        steps_data['is_terminal'].append(terminal_list[i] if i < len(terminal_list) else 0)
        steps_data['is_first'].append(is_first_list[i])
        steps_data['is_last'].append(is_last_list[i])
    
    return steps_data


def replay_episode(
    tfrecord_path: str,
    episode_index: int = 0,
    env_id: str = "InterceptMedium-v0",
    control_mode: str = "pd_joint_delta_pos",
    sim_backend: str = "cpu",
    save_video: bool = True,
    render: bool = True,
    speed: float = 1.0,
):
    """
    Replay a specific episode from an RLDS tfrecord file.
    
    Args:
        tfrecord_path: Path to the tfrecord file
        episode_index: Which episode to replay (0-indexed)
        env_id: Mikasa environment ID
        control_mode: Control mode ("pd_joint_delta_pos" or "pd_ee_delta_pose")
        sim_backend: Simulation backend ("cpu" or "gpu")
        save_video: Whether to save video of replay
        render: Whether to render the replay
        speed: Replay speed multiplier (higher = faster)
    """
    import tensorflow as tf
    
    print(f"Loading RLDS data from: {tfrecord_path}")
    
    # Load tfrecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Get the specific episode
    episode_count = 0
    steps_data = None
    
    for i, record in enumerate(dataset):
        if i == episode_index:
            steps_data = parse_rlds_example(record)
            break
        episode_count += 1
    
    if steps_data is None:
        raise ValueError(f"Episode {episode_index} not found! Total episodes scanned: {episode_count}")
    
    n_steps = len(steps_data['action'])
    print(f"Episode {episode_index}: {n_steps} steps")
    print(f"First state: {steps_data['state'][0]}")
    print(f"First action: {steps_data['action'][0]}")
    
    # Create environment
    TIME_STAMP = time.strftime('%Y%m%d_%H%M%S')
    
    args = Args(
        env_id=env_id,
        camera_width=128,
        camera_height=128,
        control_mode=control_mode,
        render_mode="all" if render else None,
        save_video=save_video,
        project_name="replay_rlds",
        model_id=f"episode_{episode_index}_{control_mode}",
        TIME_STAMP=TIME_STAMP,
        include_rgb=True,
        include_joints=False,
        include_state=True,
        include_oracle=False,
        noop_steps=1,
        shader="default",
        sim_backend=sim_backend,
    )
    
    print(f"Creating environment: {env_id}")
    env = get_mikasa_eval_env(args)
    
    # Reset environment
    obs, info = env.reset(seed=[episode_index], options={})
    print(f"Environment reset complete")
    
    # Replay actions
    print(f"Replaying {n_steps} steps with control mode: {control_mode}")
    
    for step_idx in range(n_steps):
        action = np.array(steps_data['action'][step_idx])
        
        # Apply gripper action transformations
        action = normalize_gripper_action(action)
        action = invert_gripper_action(action)
        
        # For ee_delta_pose mode, only use first 7 dims (no gripper)
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
            print(f"  Step {step_idx + 1}/{n_steps}")
        
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
    parser = argparse.ArgumentParser(description="Replay episode from RLDS tfrecord in Mikasa")
    parser.add_argument("--data.path", type=str, 
                        default="/media/raid/workspace/tengbo/any4lerobot/data/rlds_data/ShellGameTouch-100/mikasa_dataset/1.0.0/mikasa_dataset-train.tfrecord-00000-of-00004",
                        help="Path to tfrecord file")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (0-indexed)")
    parser.add_argument("--env.id", type=str, default="ShellGameTouch-v0", 
                        help="Mikasa environment ID")
    parser.add_argument("--control.mode", type=str, default="pd_joint_delta_pos",
                        choices=["pd_joint_delta_pos", "pd_ee_delta_pose"],
                        help="Control mode (pd_joint_delta_pos: 8 dims, pd_ee_delta_pose: 7 dims)")
    parser.add_argument("--sim.backend", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Simulation backend (cpu: stable, gpu: fast but may crash)")
    parser.add_argument("--no-video", action="store_true", help="Don't save video")
    parser.add_argument("--no-render", action="store_true", help="Don't render")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    
    args = parser.parse_args()
    
    replay_episode(
        tfrecord_path=getattr(args, "data.path"),
        episode_index=getattr(args, "episode"),
        env_id=getattr(args, "env.id"),
        control_mode=getattr(args, "control.mode"),
        sim_backend=getattr(args, "sim.backend"),
        save_video=not getattr(args, "no_video"),
        render=not getattr(args, "no_render"),
        speed=getattr(args, "speed"),
    )


if __name__ == "__main__":
    main()
