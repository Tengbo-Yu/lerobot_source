#!/bin/bash

# 快速测试方法

# 1. 只测试1个seed（快速验证）
echo "=== 快速测试：测试seed=0 ==="
conda run -n lerobot_mikasa python parquet.py --seed 0

echo ""
echo "=== 快速测试：测试seeds 0-10（只测10个）==="
conda run -n lerobot_mikasa python parquet.py --find-seed --seed.start 101 --seed.end 200

# python rlds.py --episode 0 --sim.backend cpu

# python npz.py --sim.backend gpu