#!/bin/bash
#SBATCH -J eval_mikasa
#SBATCH -o eval.out
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
bash scripts/eval_mikasa.sh