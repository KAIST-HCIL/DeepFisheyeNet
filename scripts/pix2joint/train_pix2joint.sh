#!/bin/bash
python train.py \
      --name pix2joint_train \
      --run pix2joint_train \
      --preset Pix2JointTrain \
      --batch_size 16 \
      --print_iter 10 \
      --gpu_ids 0,1,2 \
      --num_workers 16 \
      --save_epoch 2 \
      --epoch 6 \
      --hpe_network basic  \
      --hpe_lr 0.01 \
      #--show_grad
