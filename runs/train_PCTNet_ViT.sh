#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore -m torch.distributed.launch --nproc_per_node=2 train.py \
    models/PCTNet_ViT.py \
    --ngpu 2 \
    --workers 8 \
    --batch-size 8