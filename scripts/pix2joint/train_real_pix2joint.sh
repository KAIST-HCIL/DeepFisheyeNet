#!/bin/bash
python train.py \
      --name fish_pix2joint_train \
      --run pix2joint_train \
      --preset RealPix2JointTrain \
      --batch_size 16 \
      --print_iter 10 \
      --gpu_ids 0,1,2 \
      --num_workers 4 \
      --save_epoch 4 \
      --hpe_lr 0.0001 \
      --epoch 12 \
      --hpe_network basic \
      --hpe_pretrained path/to/weight.pth.tar \
