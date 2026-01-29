#!/bin/bash

GPU_ID=7
THRESHOLD_MB=32000  # 32GB

echo "Checking GPU ${GPU_ID} free memory, will start training when free >= ${THRESHOLD_MB} MB..."

while true; do
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found, cannot check GPU memory. Exiting."
    exit 1
  fi

  FREE_MEM=$(nvidia-smi --id=${GPU_ID} --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1)

  if [ -z "$FREE_MEM" ]; then
    echo "Failed to query GPU memory for GPU ${GPU_ID}. Retrying in 30s..."
    sleep 30
    continue
  fi

  echo "GPU ${GPU_ID} free memory: ${FREE_MEM} MB"

  if [ "$FREE_MEM" -ge "$THRESHOLD_MB" ]; then
    echo "GPU ${GPU_ID} free memory (${FREE_MEM} MB) >= ${THRESHOLD_MB} MB, starting training..."
    break
  fi

  echo "GPU ${GPU_ID} free memory is less than ${THRESHOLD_MB} MB, waiting 60 seconds..."
  sleep 60
done

# CUDA_VISIBLE_DEVICES=${GPU_ID} python src/lerobot/scripts/lerobot_train.py \
#   --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot \
#   --dataset.repo_id=JackYuuuu/InterceptMedium-v0 \
#   --policy.repo_id=JackYuuuu/act_policy_v0 \
#   --policy.type=act \
#   --output_dir=outputs/train/Act_InterceptMedium-v0\
#   --job_name=Act_InterceptMedium-v0 \
#   --policy.device=cuda \
#   --wandb.enable=true

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/lerobot/scripts/lerobot_train.py \
  --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot \
  --dataset.repo_id=JackYuuuu/InterceptMedium-v0 \
  --policy.repo_id=JackYuuuu/SmolVLA_InterceptMedium_100 \
  --policy.type=smolvla \
  --output_dir=outputs/train/SmolVLA_InterceptMedium-v0\
  --job_name=SmolVLA_InterceptMedium-v0 \
  --policy.device=cuda \
  --wandb.enable=true

# CUDA_VISIBLE_DEVICES=${GPU_ID} python src/lerobot/scripts/lerobot_train.py \
#   --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot \
#   --dataset.repo_id=JackYuuuu/InterceptMedium-v0 \
#   --policy.repo_id=JackYuuuu/Groot_InterceptMedium_100 \
#   --policy.type=groot \
#   --output_dir=outputs/train/Groot_InterceptMedium-v0\
#   --job_name=Groot_InterceptMedium-v0 \
#   --policy.device=cuda \
#   --wandb.enable=true
  