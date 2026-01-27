#!/bin/bash
ENV_NAME=InterceptGrabMedium-v0
POLICY_PATH=outputs/train/groot_n1/checkpoints/pretrained_model
POLICY_TYPE=act
N_EPISODES=100
N_STEPS=60
cd /media/raid/workspace/tengbo/lerobot

python eval_lerobot_on_mikasa.py \
    --policy.path=${POLICY_PATH} \
    --policy.type=${POLICY_TYPE} \
    --env.id=${ENV_NAME} \
    --eval.n_episodes=${N_EPISODES} \
    --eval.n_steps=${N_STEPS}