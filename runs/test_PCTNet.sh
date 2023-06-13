#! /bin/bash

# For CNN-based PCTNet
model="CNN_pct"
pretrain_path="./pretrained_models/PCTNet_CNN.pth"

# Uncomment for ViT-based PCTNet 
model="ViT_pct"
pretrain_path="./pretrained_models/PCTNet_ViT.pth"

# Evaluation for Full Resolution 
python3 scripts/evaluate_iHarmony4.py ${model} ${pretrain_path} \
    --resize-strategy Fixed256 \
    --res FR \
    --config-path config_test_FR.yml

## Evaluation for Low Resolution
# python3 scripts/evaluate_iHarmony4.py ${model} ${pretrain_path} \
#     --resize-strategy Fixed256 \
#     --res LR \
#     --config-path config.yml 
