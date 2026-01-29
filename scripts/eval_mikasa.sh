#!/bin/bash
GPU=5
ENV_NAME=InterceptMedium-v0
MODEL_ID=Groot_InterceptMedium-v0
POLICY_PATH=/media/raid/workspace/tengbo/lerobot/outputs/train/${MODEL_ID}/checkpoints/100000/pretrained_model
DS_META_PATH=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/InterceptGrabMedium_100/mikasa_dataset_1.0.0_lerobot
POLICY_TYPE=act
N_EPISODES=100
N_STEPS=60

cd /media/raid/workspace/tengbo/lerobot

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU} python eval_mikasa.py \
    --policy.path=${POLICY_PATH} \
    --policy.ds_meta_path=${DS_META_PATH} \
    --env.id=${ENV_NAME} \
    --project.name=${POLICY_TYPE}+${ENV_NAME} \
    --eval.n_episodes=${N_EPISODES} \
    --eval.n_steps=${N_STEPS} \
    --model.id=${MODEL_ID} 