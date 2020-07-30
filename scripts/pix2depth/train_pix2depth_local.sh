#!/bin/bash
python train.py \
      --preset Pix2DepthTrain \
      --name pix2depth_train_local \
      --max_data 20\
      --epoch 6 \
      --batch_size 2 \
      --gpu_ids 0 \
      --print_iter 10 \
      --num_workers 2 \
      --save_epoch 1 \
      --p2d_lr 0.0002 \
