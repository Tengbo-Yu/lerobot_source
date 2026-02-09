#!/usr/bin/env python3
"""
Compare action values between NPZ and LeRobot (Parquet).
Visualize differences per action dimension.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ======================== Paths ========================
NPZ_PATH = "/media/raid/workspace/tengbo/any4lerobot/data/raw_data/InterceptGrabMedium-test/train_data_0.npz"
PARQUET_PATH = "/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptGrabMedium-test/data/chunk-000/file-000.parquet"
OUTPUT_DIR = "/media/raid/workspace/tengbo/lerobot/replay_debug/outputs/compare"

NPZ_TRAJ_INDEX = 0      # Trajectory 0 from NPZ (if batched)
EPISODE_INDEX = 0       # Episode index for Parquet


# ======================== Load NPZ ========================
def load_npz_actions(path, traj_idx=0):
    print(f"[NPZ] Loading: {path}")
    data = np.load(path, allow_pickle=True)
    actions = data['action']
    if actions.ndim == 3:
        print(f"  Batched NPZ: shape {actions.shape}, selecting traj_idx={traj_idx}")
        actions = actions[:, traj_idx, :]
    print(f"  Actions shape: {actions.shape}")
    print(f"  Range: [{actions.min():.6f}, {actions.max():.6f}]")
    return actions

# ======================== Load Parquet ========================
def load_parquet_actions(path, episode_index=0):
    print(f"[Parquet] Loading: {path}")
    df = pd.read_parquet(path)

    # episode_index column in lerobot parquet
    if 'episode_index' in df.columns:
        episode_df = df[df['episode_index'] == episode_index].reset_index(drop=True)
    else:
        # fallback: treat entire file as one episode
        episode_df = df.reset_index(drop=True)

    if len(episode_df) == 0:
        raise ValueError(f"Episode {episode_index} not found in {path}")

    print(f"  Episode {episode_index}: {len(episode_df)} frames")

    # action column is a list/array per row
    actions_raw = episode_df['action'].values
    actions = np.stack([np.array(a) for a in actions_raw])
    print(f"  Actions shape: {actions.shape}")
    print(f"  Range: [{actions.min():.6f}, {actions.max():.6f}]")
    return actions


# ======================== Compare & Plot ========================
def compare_and_plot(npz_actions, parquet_actions, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  NPZ     : {npz_actions.shape}")
    print(f"  Parquet : {parquet_actions.shape}")

    # Use the minimum length for comparison
    min_steps = min(len(npz_actions), len(parquet_actions))
    min_dim = min(npz_actions.shape[1], parquet_actions.shape[1])
    print(f"  Comparing first {min_steps} steps, {min_dim} dims")

    npz_a = npz_actions[:min_steps, :min_dim]
    parq_a = parquet_actions[:min_steps, :min_dim]

    # ---- Numerical diff ----
    print("\n--- NPZ vs Parquet ---")
    diff = np.abs(npz_a - parq_a)
    print(f"  Max abs diff : {diff.max():.8f}")
    print(f"  Mean abs diff: {diff.mean():.8f}")
    print(f"  Per-dim max  : {diff.max(axis=0)}")

    # ---- First few steps side by side ----
    print("\n--- First 5 steps (all dims) ---")
    for t in range(min(5, min_steps)):
        print(f"  Step {t}:")
        print(f"    NPZ    : {npz_a[t]}")
        print(f"    Parquet: {parq_a[t]}")

    time_steps = np.arange(min_steps)

    # ========== Plot 1: Per-dimension overlay ==========
    n_cols = min(4, min_dim)
    n_rows = (min_dim + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes1 = np.atleast_2d(axes1).reshape(-1)

    for i in range(min_dim):
        ax = axes1[i]
        ax.plot(time_steps, npz_a[:, i], 'b-', linewidth=1.5, alpha=0.8, label='NPZ')
        ax.plot(time_steps, parq_a[:, i], 'r--', linewidth=1.5, alpha=0.8, label='Parquet')
        ax.set_title(f'Action dim {i}', fontsize=10)
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    for i in range(min_dim, len(axes1)):
        axes1[i].axis('off')

    fig1.suptitle('Per-Dimension Action Comparison (NPZ vs Parquet)', fontsize=13)
    plt.tight_layout()
    path1 = os.path.join(output_dir, 'compare_per_dim.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\nSaved: {path1}")

    # ========== Plot 2: All dims overlaid (one subplot per source) ==========
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cmap = plt.cm.tab10
    sources = [('NPZ', npz_a), ('Parquet', parq_a)]
    for ax, (name, data) in zip(axes2, sources):
        for i in range(min_dim):
            ax.plot(time_steps, data[:, i], '-', color=cmap(i % 10), linewidth=1, alpha=0.8, label=f'd{i}')
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc='upper right')

    fig2.suptitle('All Action Dims per Source', fontsize=13)
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'compare_all_dims_per_source.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {path2}")

    # ========== Plot 3: Absolute difference heatmap ==========
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    im = ax3.imshow(diff.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax3.set_title('NPZ vs Parquet (Absolute Difference)', fontsize=12)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Action Dim')
    plt.colorbar(im, ax=ax3, shrink=0.8)

    plt.tight_layout()
    path3 = os.path.join(output_dir, 'compare_diff_heatmap.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {path3}")

    # ========== Plot 4: Per-dim difference curves ==========
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes4 = np.atleast_2d(axes4).reshape(-1)

    for i in range(min_dim):
        ax = axes4[i]
        ax.plot(time_steps, diff[:, i], 'r-', linewidth=1.2, alpha=0.8)
        ax.set_title(f'Dim {i} |NPZ - Parquet|', fontsize=10)
        ax.set_xlabel('Step')
        ax.set_ylabel('Abs Diff')
        ax.grid(True, alpha=0.3)
    for i in range(min_dim, len(axes4)):
        axes4[i].axis('off')

    fig4.suptitle('Per-Dimension Absolute Difference', fontsize=13)
    plt.tight_layout()
    path4 = os.path.join(output_dir, 'compare_per_dim_diff.png')
    fig4.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {path4}")

    print(f"\nAll comparison plots saved to: {output_dir}")


# ======================== Main ========================
def main():
    npz_actions = load_npz_actions(NPZ_PATH, traj_idx=NPZ_TRAJ_INDEX)
    parquet_actions = load_parquet_actions(PARQUET_PATH, episode_index=EPISODE_INDEX)
    compare_and_plot(npz_actions, parquet_actions, OUTPUT_DIR)


if __name__ == "__main__":
    main()
