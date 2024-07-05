#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 100 \
    --stage 3 \
    --val_epoch 30 \
    --weight_decay 5e-4 \
    --train_way 20 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --gpu 0
