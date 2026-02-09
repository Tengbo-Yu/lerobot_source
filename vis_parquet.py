import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_combined_dims(file_path):
    # 1. 读取数据 (确保已安装 pyarrow 或 fastparquet)
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 选取第一个剧集进行分析
    target_episode = df['episode_index'].unique()[0]
    df_ep = df[df['episode_index'] == target_episode].sort_values('timestamp')

    # 3. 解析数据维度
    states = np.array(df_ep['observation.state'].tolist()) # 形状为 (N, 8)
    actions = np.array(df_ep['action'].tolist())           # 形状为 (N, 8)
    timestamps = df_ep['timestamp'].values
    num_dims = states.shape[1]

    # 4. 创建 4x2 的画布
    fig, axes = plt.subplots(4, 2, figsize=(15, 18), sharex=True)
    axes = axes.flatten() # 转为一维数组方便循环

    for i in range(num_dims):
        ax = axes[i]
        
        # 绘制该维度的 State
        ax.plot(timestamps, states[:, i], label=f'State Dim {i}', color='blue', linewidth=1.5)
        
        # 绘制该维度的 Action
        # 注意：如果两者量级差异巨大，建议观察趋势或进行归一化
        ax.plot(timestamps, actions[:, i], label=f'Action Dim {i}', color='red', linestyle='--', linewidth=1.5)
        
        ax.set_title(f'Dimension {i}: State vs Action', fontsize=12)
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 设置坐标轴标签
        if i >= 6:
            ax.set_xlabel('Timestamp (s)')
        if i % 2 == 0:
            ax.set_ylabel('Value')

    plt.suptitle(f'State & Action Comparison (Episode {target_episode})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存并显示
    output_name = 'dim_wise_comparison.png'
    plt.savefig(output_name, bbox_inches='tight')
    print(f"可视化完成！图像已保存为: {output_name}")
    plt.show()


if __name__ == "__main__":
    visualize_combined_dims("/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot/data/chunk-000/file-000.parquet")