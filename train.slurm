#!/bin/bash
#SBATCH -J SmolVLA_train
#SBATCH -o SmolVLA_train.out
#SBATCH --exclude=hgx-hyperplane00
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source /share/NSS/yutengbo/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export WANDB_API_KEY=wandb_v1_OC5RFtCcvVatTbSCpxR0dKWgYqk
export WANDB_ENTITY=ytb23-peking-university
export WANDB_PROJECT=lerobot

python src/lerobot/scripts/lerobot_train.py \
  --dataset.root=/share/NSS/yutengbo/lerobot/data/lerobot_data/InterceptMedium-v0/mikasa_dataset_1.0.0_lerobot \
  --dataset.repo_id=JackYuuuu/InterceptMedium-v0 \
  --policy.repo_id=JackYuuuu/pi05_InterceptMedium_100 \
  --policy.type=pi05 \
  --output_dir=outputs/train/pi05_InterceptMedium-v0\
  --job_name=pi05_InterceptMedium-v0 \
  --policy.device=cuda \
  --wandb.enable=false