#!/bin/bash
#SBATCH -J SmolVLA_train
#SBATCH -o SmolVLA_train.out
#SBATCH --exclude=hgx-hyperplane00
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

conda activate lerobot
python src/lerobot/scripts/lerobot_train.py \
  --dataset.root=/share/NSS/yutengbo/lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot \
  --dataset.repo_id=JackYuuuu/InterceptMedium-v0 \
  --policy.repo_id=JackYuuuu/SmolVLA_InterceptMedium_100 \
  --policy.type=smolVLA \
  --output_dir=outputs/train/SmolVLA_InterceptMedium-v0\
  --job_name=SmolVLA_InterceptMedium-v0 \
  --policy.device=cuda \
  --wandb.enable=true