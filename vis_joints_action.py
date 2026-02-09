import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
npz_path = '/media/raid/workspace/tengbo/any4lerobot/data/raw_data/InterceptMedium-v0/train_data_1.npz'
data = np.load(npz_path)

joints = data['joints']  # shape: (90, 25)
action = data['action']  # shape: (90, 8)

time_steps = np.arange(joints.shape[0])

# Create output directory
output_dir = '/media/raid/workspace/tengbo/lerobot/vis_output'
os.makedirs(output_dir, exist_ok=True)

# Plot each joint separately
print("Plotting joints...")
for i in range(joints.shape[1]):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_steps, joints[:, i], 'b-', linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'Joint {i} Value')
    ax.set_title(f'Joint {i} over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'joint_{i:02d}.png'), dpi=150)
    plt.close()
    print(f"  Saved joint_{i:02d}.png")

# Plot each action separately
print("\nPlotting actions...")
for i in range(action.shape[1]):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_steps, action[:, i], 'r-', linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'Action {i} Value')
    ax.set_title(f'Action {i} over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'action_{i:02d}.png'), dpi=150)
    plt.close()
    print(f"  Saved action_{i:02d}.png")

# Also create combined overview plots
print("\nCreating overview plots...")

# All joints in one figure with subplots
n_cols = 5
n_rows = (joints.shape[1] + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()
for i in range(joints.shape[1]):
    axes[i].plot(time_steps, joints[:, i], 'b-', linewidth=1)
    axes[i].set_title(f'Joint {i}')
    axes[i].grid(True, alpha=0.3)
for i in range(joints.shape[1], len(axes)):
    axes[i].axis('off')
plt.suptitle('All Joints over Time', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_joints_overview.png'), dpi=150)
plt.close()
print("  Saved all_joints_overview.png")

# All actions in one figure with subplots
n_cols = 4
n_rows = (action.shape[1] + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()
for i in range(action.shape[1]):
    axes[i].plot(time_steps, action[:, i], 'r-', linewidth=1)
    axes[i].set_title(f'Action {i}')
    axes[i].grid(True, alpha=0.3)
for i in range(action.shape[1], len(axes)):
    axes[i].axis('off')
plt.suptitle('All Actions over Time', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_actions_overview.png'), dpi=150)
plt.close()

# Compute cumulative action (real values from delta)
action_cumsum = np.cumsum(action, axis=0)

# Plot each cumulative action separately
print("\nPlotting cumulative actions (real values)...")
for i in range(action_cumsum.shape[1]):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_steps, action_cumsum[:, i], 'g-', linewidth=1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel(f'Action {i} Cumulative Value')
    ax.set_title(f'Action {i} Real Value (Cumulative) over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'action_cumsum_{i:02d}.png'), dpi=150)
    plt.close()
    print(f"  Saved action_cumsum_{i:02d}.png")

# All cumulative actions in one figure with subplots
n_cols = 4
n_rows = (action_cumsum.shape[1] + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()
for i in range(action_cumsum.shape[1]):
    axes[i].plot(time_steps, action_cumsum[:, i], 'g-', linewidth=1)
    axes[i].set_title(f'Action {i} (Cumulative)')
    axes[i].grid(True, alpha=0.3)
for i in range(action_cumsum.shape[1], len(axes)):
    axes[i].axis('off')
plt.suptitle('All Actions Real Values (Cumulative) over Time', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_actions_cumsum_overview.png'), dpi=150)
plt.close()
print("  Saved all_actions_cumsum_overview.png")
print("  Saved all_actions_overview.png")

print(f"\nAll plots saved to: {output_dir}")
