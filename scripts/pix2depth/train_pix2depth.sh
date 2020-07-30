#!/bin/bash
python train.py \
      --preset Pix2DepthTrain \
      --name pix2depth_train \
      --epoch 12 \
      --batch_size 32 \
      --gpu_ids 0,1,2 \
      --print_iter 10 \
      --num_workers 2 \
      --save_epoch 2 \
      --p2d_lr 0.001 \
