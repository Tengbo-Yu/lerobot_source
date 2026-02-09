#!/usr/bin/env python3
"""
Replay a specific episode from npz data file in Mikasa environment.
With real-time action visualization.
"""

import sys
import os
from pathlib import Path
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Add MIKASA and lerobot paths
sys.path.append("/media/raid/workspace/tengbo/lerobot/third_party/MIKASA-Robo/baselines/openvla")
sys.path.append("/media/raid/workspace/tengbo/lerobot/src")
sys.path.append("/media/raid/workspace/tengbo/lerobot")

from mikasa_utils import get_mikasa_eval_env


ACTION_DIM_NAMES_EE = [
    "dx", "dy", "dz", "dqx", "dqy", "dqz", "dqw", "gripper"
]

ACTION_DIM_NAMES_JOINT = [
    f"joint_{i}" for i in range(25)
]


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


class ActionVisualizer:
    """Real-time action visualization during replay."""
    
    def __init__(self, all_actions, all_joints=None, control_mode="pd_ee_delta_pose", save_path=None):
        """
        Args:
            all_actions: Full action array (n_steps, action_dim) for reference lines
            all_joints: Full joints array (n_steps, joint_dim), optional
            control_mode: Control mode to determine dim names
            save_path: If set, save final plot to this path
        """
        self.all_actions = all_actions
        self.all_joints = all_joints
        self.save_path = save_path
        self.n_steps = len(all_actions)
        self.action_dim = all_actions.shape[1]
        self.joint_dim = all_joints.shape[1] if all_joints is not None else 0
        
        # Dim names
        if "ee" in control_mode:
            self.dim_names = ACTION_DIM_NAMES_EE[:self.action_dim]
        else:
            self.dim_names = ACTION_DIM_NAMES_JOINT[:self.action_dim]
        
        # Storage for executed actions
        self.executed_steps = []
        self.executed_actions = []
        
        # --- Layout: per-dim subplots + 2 summary subplots at the bottom ---
        n_cols = min(4, self.action_dim)
        n_per_dim_rows = (self.action_dim + n_cols - 1) // n_cols
        # Add 1 extra row for the 2 summary plots (all-actions, all-joints)
        total_rows = n_per_dim_rows + 1
        self.fig, all_axes = plt.subplots(total_rows, n_cols,
                                          figsize=(4 * n_cols, 3 * total_rows))
        all_axes = np.atleast_2d(all_axes)
        
        # Per-dimension axes (first n_per_dim_rows rows)
        self.axes = all_axes[:n_per_dim_rows, :].flatten()
        # Summary axes (last row)
        summary_axes = all_axes[n_per_dim_rows, :]
        
        # Plot reference (full trajectory) as light background
        self.lines_ref = []
        self.lines_exec = []
        self.vlines = []
        time_steps = np.arange(self.n_steps)
        
        # --- Per-dimension subplots ---
        for i in range(self.action_dim):
            ax = self.axes[i]
            line_ref, = ax.plot(time_steps, all_actions[:, i], 'b-', alpha=0.3, linewidth=1, label='recorded')
            self.lines_ref.append(line_ref)
            line_exec, = ax.plot([], [], 'r-', linewidth=1.5, label='executed')
            self.lines_exec.append(line_exec)
            vline = ax.axvline(x=0, color='green', linewidth=1, alpha=0.7, linestyle='--')
            self.vlines.append(vline)
            
            name = self.dim_names[i] if i < len(self.dim_names) else f"dim_{i}"
            ax.set_title(name, fontsize=10)
            ax.set_xlim(0, self.n_steps)
            margin = 0.1 * (all_actions[:, i].max() - all_actions[:, i].min() + 1e-6)
            ax.set_ylim(all_actions[:, i].min() - margin, all_actions[:, i].max() + margin)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, loc='upper right')
        
        # Hide unused per-dim axes
        for i in range(self.action_dim, len(self.axes)):
            self.axes[i].axis('off')
        
        # --- Summary subplot 1: All actions together ---
        cmap_action = plt.cm.tab10
        self.ax_all_actions = summary_axes[0]
        self.lines_all_exec = []
        for i in range(self.action_dim):
            color = cmap_action(i % 10)
            name = self.dim_names[i] if i < len(self.dim_names) else f"dim_{i}"
            self.ax_all_actions.plot(time_steps, all_actions[:, i], '-', color=color, alpha=0.25, linewidth=1)
            line, = self.ax_all_actions.plot([], [], '-', color=color, linewidth=1.2, label=name)
            self.lines_all_exec.append(line)
        self.ax_all_actions.set_title('All Actions (overlay)', fontsize=10)
        self.ax_all_actions.set_xlim(0, self.n_steps)
        act_margin = 0.1 * (all_actions.max() - all_actions.min() + 1e-6)
        self.ax_all_actions.set_ylim(all_actions.min() - act_margin, all_actions.max() + act_margin)
        self.ax_all_actions.grid(True, alpha=0.3)
        self.ax_all_actions.legend(fontsize=6, loc='upper right', ncol=2)
        self.vline_all_actions = self.ax_all_actions.axvline(x=0, color='green', linewidth=1, alpha=0.7, linestyle='--')
        
        # --- Summary subplot 2: All joints together ---
        self.ax_all_joints = summary_axes[1]
        if all_joints is not None:
            cmap_joint = plt.cm.tab20
            for j in range(self.joint_dim):
                color = cmap_joint(j % 20)
                self.ax_all_joints.plot(time_steps, all_joints[:, j], '-', color=color,
                                        linewidth=1, alpha=0.8, label=f'j{j}')
            self.ax_all_joints.set_title('All Joints (overlay)', fontsize=10)
            self.ax_all_joints.set_xlim(0, self.n_steps)
            jt_margin = 0.1 * (all_joints.max() - all_joints.min() + 1e-6)
            self.ax_all_joints.set_ylim(all_joints.min() - jt_margin, all_joints.max() + jt_margin)
            self.ax_all_joints.grid(True, alpha=0.3)
            if self.joint_dim <= 25:
                self.ax_all_joints.legend(fontsize=5, loc='upper right', ncol=3)
        else:
            self.ax_all_joints.set_title('Joints (no data)', fontsize=10)
            self.ax_all_joints.axis('off')
        
        # Hide remaining summary axes
        for i in range(2, n_cols):
            summary_axes[i].axis('off')
        
        self.fig.suptitle('Action & Joint Visualization (blue/light=recorded, red/solid=executed)', fontsize=12)
        plt.tight_layout()
    
    def update(self, step_idx, executed_action):
        """Update the plot with new executed action."""
        self.executed_steps.append(step_idx)
        self.executed_actions.append(executed_action.copy())
        
        exec_arr = np.array(self.executed_actions)
        steps_arr = np.array(self.executed_steps)
        
        for i in range(self.action_dim):
            # Per-dim subplot
            self.lines_exec[i].set_data(steps_arr, exec_arr[:, i])
            self.vlines[i].set_xdata([step_idx])
            # All-actions summary subplot
            self.lines_all_exec[i].set_data(steps_arr, exec_arr[:, i])
        self.vline_all_actions.set_xdata([step_idx])
    
    def finalize(self):
        """Finalize and save the plot."""
        # Final redraw into buffer
        self.fig.canvas.draw()
        
        if self.save_path:
            self.fig.savefig(self.save_path, dpi=150, bbox_inches='tight')
            print(f"Action plot saved to: {self.save_path}")
        else:
            # Default save location if none specified
            fallback = "replay_actions.png"
            self.fig.savefig(fallback, dpi=150, bbox_inches='tight')
            print(f"Action plot saved to: {fallback}")
        
        plt.close(self.fig)


def replay_episode(
    npz_path: str,
    env_id: str = "ShellGameTouch-v0",
    control_mode: str = "pd_ee_delta_pose",
    sim_backend: str = "cpu",
    save_video: bool = True,
    render: bool = True,
    speed: float = 1.0,
    seed: int = 0,
    traj_idx: int = 0,
    vis_action: bool = True,
    save_action_plot: bool = True,
):
    """
    Replay an episode from an npz data file.
    
    Important notes:
    - NPZ data was collected with pd_ee_delta_pose control mode (8 dims: 7 pose + 1 gripper)
    - Actions are raw PPO agent outputs, do NOT apply normalize/invert gripper transforms
    - Seed must match the data collection seed to reproduce the same scene
    - Batched NPZ (3D actions) needs traj_idx to select a trajectory
    
    Args:
        npz_path: Path to the npz file
        env_id: Mikasa environment ID
        control_mode: Control mode (must match data collection, default: pd_ee_delta_pose)
        sim_backend: Simulation backend ("cpu" or "gpu")
        save_video: Whether to save video of replay
        render: Whether to render the replay
        speed: Replay speed multiplier (higher = faster)
        seed: Environment seed (must match data collection seed)
        traj_idx: Trajectory index within batch (for batched npz data)
        vis_action: Whether to visualize actions in real-time
        save_action_plot: Whether to save the final action plot
    """
    print(f"Loading data from: {npz_path}")
    
    # Load npz data
    data = np.load(npz_path, allow_pickle=True)
    
    actions = data['action']
    joints = data['joints'][:,7:14]
    rewards = data['reward']
    successes = data['success']
    dones = data['done']
    
    print(f"actions.ndim: {actions.ndim}")

    # Handle batched data: shape (T, batch_size, ...) -> select one trajectory
    if actions.ndim == 3:
        batch_size = actions.shape[1]
        print(f"Detected BATCHED data: action shape {actions.shape}, batch_size={batch_size}")
        print(f"Selecting trajectory index {traj_idx} out of {batch_size}")
        actions = actions[:, traj_idx, :]
        joints = joints[:, traj_idx, :]
        rewards = rewards[:, traj_idx]
        successes = successes[:, traj_idx]
        dones = dones[:, traj_idx]
    
    n_steps = len(actions)
    print(f"Episode: {n_steps} steps")
    print(f"Action shape: {actions.shape}")
    print(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"Gripper (last dim) range: [{actions[:, -1].min():.4f}, {actions[:, -1].max():.4f}]")
    print(f"First action: {actions[0]}")
    print(f"Recorded success at any step: {successes.any()}")
    
    # Create environment
    TIME_STAMP = time.strftime('%Y%m%d_%H%M%S')
    
    args = Args(
        env_id=env_id,
        camera_width=128,
        camera_height=128,
        control_mode=control_mode,
        render_mode="all" if render else None,
        save_video=save_video,
        project_name="replay_npz",
        model_id=f"episode_{control_mode}_{sim_backend}",
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
    print(f"Control mode: {control_mode}")
    print(f"Seed: {seed}")
    env = get_mikasa_eval_env(args)
    
    # Reset environment - seed must match data collection!
    obs, info = env.reset(seed=[seed], options={})
    print(f"Environment reset complete")
    
    # Setup action visualizer
    visualizer = None
    if vis_action:
        save_plot_path = None
        if save_action_plot:
            output_dir = os.path.join("/media/raid/workspace/tengbo/lerobot/replay_debug/outputs/eval/replay_npz/episode_pd_joint_delta_pos_cpu/action")
            os.makedirs(output_dir, exist_ok=True)
            basename = Path(npz_path).stem
            save_plot_path = os.path.join(output_dir, f"{basename}_traj{traj_idx}_actions.png")
        visualizer = ActionVisualizer(actions, all_joints=joints, control_mode=control_mode, save_path=save_plot_path)
    
    # Replay actions
    print(f"Replaying {n_steps} steps with control mode: {control_mode}")
    
    total_reward = 0.0
    for step_idx in range(n_steps):
        action = actions[step_idx]

        # NPZ stores raw PPO agent outputs - use directly!
        # Do NOT apply normalize_gripper_action / invert_gripper_action
        # (those are for RLDS/OpenVLA pipeline where gripper was remapped to [0,1])
        # Do NOT truncate action dims - pd_ee_delta_pose expects full 8 dims
        
        # Execute step
        obs, reward, done, trunc, info = env.step(action)
        total_reward += float(reward) if np.isscalar(reward) else float(reward.item())
        
        # Update action visualization
        if visualizer is not None:
            visualizer.update(step_idx, action)
        
        # Control replay speed
        time.sleep(0.02 / speed)
        
        if (step_idx + 1) % 30 == 0:
            print(f"  Step {step_idx + 1}/{n_steps}, cumulative reward: {total_reward:.4f}")
        
        # Data was collected with ignore_terminations=True, so full trajectory runs
        # Uncomment below if you want to stop early on done:
        # if done or trunc:
        #     print(f"  Episode ended at step {step_idx + 1}")
        #     break
    
    # Check success
    print(f"\n--- Results ---")
    print(f"Total reward: {total_reward:.4f}")
    if "is_success" in info:
        print(f"Replay success: {info['is_success']}")
    elif "success" in info:
        print(f"Replay success: {info['success']}")
    print(f"Recorded success (from NPZ): {successes.any()}")
    
    # Finalize visualization
    if visualizer is not None:
        visualizer.finalize()
    
    print("Replay complete!")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Replay episode from npz data in Mikasa")
    parser.add_argument("--data.path", type=str, 
                        default="/media/raid/workspace/tengbo/any4lerobot/data/raw_data/InterceptGrabMedium-test/train_data_0.npz",
                        help="Path to npz file")
    parser.add_argument("--env.id", type=str, default="InterceptGrabMedium-v0", 
                        help="Mikasa environment ID")
    parser.add_argument("--control.mode", type=str, default="pd_joint_delta_pos",
                        choices=["pd_joint_delta_pos", "pd_ee_delta_pose"],
                        help="Control mode (must match data collection, default: pd_ee_delta_pose)")
    parser.add_argument("--sim.backend", type=str, default="cpu",
                        choices=["cpu", "gpu"],
                        help="Simulation backend (cpu recommended for reproducibility)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Environment seed (must match data collection seed)")
    parser.add_argument("--traj-idx", type=int, default=0,
                        help="Trajectory index within batch (for batched npz data)")
    parser.add_argument("--no-video", action="store_true", help="Don't save video")
    parser.add_argument("--no-render", action="store_true", help="Don't render")
    parser.add_argument("--no-vis-action", action="store_true", help="Don't visualize actions")
    parser.add_argument("--no-save-action-plot", action="store_true", help="Don't save action plot")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    
    args = parser.parse_args()
    
    replay_episode(
        npz_path=getattr(args, "data.path"),
        env_id=getattr(args, "env.id"),
        control_mode=getattr(args, "control.mode"),
        sim_backend=getattr(args, "sim.backend"),
        save_video=not getattr(args, "no_video"),
        render=not getattr(args, "no_render"),
        speed=getattr(args, "speed"),
        seed=getattr(args, "seed"),
        traj_idx=getattr(args, "traj_idx"),
        vis_action=not getattr(args, "no_vis_action"),
        save_action_plot=not getattr(args, "no_save_action_plot"),
    )


if __name__ == "__main__":
    main()