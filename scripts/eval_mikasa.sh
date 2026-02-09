#!/bin/bash
GPU=0
ENV_NAME=ShellGameTouch-v0
# MODEL_ID=Act_InterceptMedium-v0
MODEL_ID=ACT_ShellGameTouch-250-8-1
NUM=100000
POLICY_PATH=/media/raid/workspace/tengbo/lerobot/outputs/train/${MODEL_ID}/checkpoints/${NUM}/pretrained_model
# DS_META_PATH=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/ShellGameTouch-100/mikasa_dataset_1.0.0_lerobot
DS_META_PATH=/media/raid/workspace/tengbo/any4lerobot/data/lerobot_data/ShellGameTouch-250
POLICY_TYPE=act
N_EPISODES=100
N_STEPS=90
PROMPT="catch the ball."

cd /media/raid/workspace/tengbo/lerobot

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU} python eval_mikasa.py \
    --policy.path=${POLICY_PATH} \
    --policy.ds_meta_path=${DS_META_PATH} \
    --env.id=${ENV_NAME} \
    --project.name=${POLICY_TYPE}+${ENV_NAME} \
    --eval.n_episodes=${N_EPISODES} \
    --eval.n_steps=${N_STEPS} \
    --model.id=${MODEL_ID} \
    --policy.type=${POLICY_TYPE} \
    --eval.task_prompt="${PROMPT}"