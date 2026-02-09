#!/bin/bash

GPU_ID=3
# THRESHOLD_MB=20000  # 20GB

# echo "Checking GPU ${GPU_ID} free memory, will start training when free >= ${THRESHOLD_MB} MB..."

# while true; do
#   if ! command -v nvidia-smi >/dev/null 2>&1; then
#     echo "nvidia-smi not found, cannot check GPU memory. Exiting."
#     exit 1
#   fi

#   FREE_MEM=$(nvidia-smi --id=${GPU_ID} --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1)

#   if [ -z "$FREE_MEM" ]; then
#     echo "Failed to query GPU memory for GPU ${GPU_ID}. Retrying in 30s..."
#     sleep 30
#     continue
#   fi

#   echo "GPU ${GPU_ID} free memory: ${FREE_MEM} MB"

#   if [ "$FREE_MEM" -ge "$THRESHOLD_MB" ]; then
#     echo "GPU ${GPU_ID} free memory (${FREE_MEM} MB) >= ${THRESHOLD_MB} MB, starting training..."
#     break
#   fi

#   echo "GPU ${GPU_ID} free memory is less than ${THRESHOLD_MB} MB, waiting 60 seconds..."
#   sleep 60
# done

# CUDA_VISIBLE_DEVICES=${GPU_ID} python src/lerobot/scripts/lerobot_train.py \
#   --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/ShellGameTouch-100/mikasa_dataset_1.0.0_lerobot \
#   --dataset.repo_id=JackYuuuu/ShellGameTouch-100 \
#   --policy.repo_id=JackYuuuu/act_policy_v0 \
#   --policy.type=act \
#   --output_dir=outputs/train/ACT_ShellGameTouch-100-8-1\
#   --job_name=Act_ShellGameTouch-100 \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --policy.chunk_size=8 \
#   --policy.n_action_steps=5

# CUDA_VISIBLE_DEVICES=${GPU_ID} python src/lerobot/scripts/lerobot_train.py \
#   --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptGrabSlow-250 \
#   --dataset.repo_id=JackYuuuu/InterceptGrabSlow-250 \
#   --policy.repo_id=JackYuuuu/act_policy_v0 \
#   --policy.type=act \
#   --output_dir=outputs/train/ACT_InterceptGrabSlow-250-8-1 \
#   --job_name=Act_InterceptGrabSlow-250-8-1 \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --policy.chunk_size=8 \
#   --policy.n_action_steps=5

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/lerobot/scripts/lerobot_train.py \
  --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/ShellGameTouch-250 \
  --dataset.repo_id=JackYuuuu/ShellGameTouch-250 \
  --policy.repo_id=JackYuuuu/act_policy_v0 \
  --policy.type=act \
  --output_dir=outputs/train/ACT_ShellGameTouch-250-8-1\
  --job_name=Act_ShellGameTouch-250-8-1 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.chunk_size=8 \
  --policy.n_action_steps=5 \
  --batch_size=32

# CUDA_VISIBLE_DEVICES=${GPU_ID} python src/lerobot/scripts/lerobot_train.py \
#   --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot \
#   --dataset.repo_id=JackYuuuu/InterceptMedium-v0 \
#   --policy.repo_id=JackYuuuu/smolvla_policy_v0 \
#   --policy.type=act \
#   --output_dir=outputs/train/act_InterceptMedium-v0-8-1-rgb\
#   --job_name=Smolvla_InterceptMedium-v0 \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --policy.chunk_size=8 \
#   --policy.n_action_steps=1 \
#   --policy.input_features='{
#         "observation.images.image": {"type": "VISUAL", "shape": [3, 128, 128]},
#         "observation.images.wrist_image": {"type": "VISUAL", "shape": [3, 128, 128]}
#     }' \
#   --batch_size=32 \



# CUDA_VISIBLE_DEVICES=3 python src/lerobot/scripts/lerobot_train.py \
#   --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptGrabMedium-v0/mikasa_dataset_1.0.0_lerobot \
#   --dataset.repo_id=JackYuuuu/InterceptGrabMedium-v0 \
#   --policy.repo_id=JackYuuuu/Groot_InterceptGrabMedium_100 \
#   --policy.type=groot \
#   --output_dir=outputs/train/Groot_InterceptGrabMedium-v0\
#   --job_name=Groot_InterceptGrabMedium-v0 \
#   --policy.device=cuda \
#   --wandb.enable=true


# CUDA_VISIBLE_DEVICES=${GPU_ID} python src/lerobot/scripts/lerobot_train.py\
#   --dataset.root=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot \
#   --dataset.repo_id=JackYuuuu/InterceptMedium-v0 \
#   --policy.repo_id=JackYuuuu/pi05_policy_v0 \
#   --policy.type=pi05 \
#   --output_dir=outputs/train/pi05_InterceptMedium-v0\
#   --job_name=pi05_InterceptMedium-v0 \
#   --policy.device=cuda \
#   --wandb.enable=true \
#   --policy.pretrained_path=/media/raid/workspace/tengbo/lerobot/pi05_base\
#   --policy.compile_model=false \
#   --policy.gradient_checkpointing=true \
#   --policy.dtype=bfloat16 \
#   --policy.freeze_vision_encoder=false \
#   --policy.train_expert_only=true \
#   --steps=30000 \
#   --policy.device=cuda \
#   --batch_size=32
