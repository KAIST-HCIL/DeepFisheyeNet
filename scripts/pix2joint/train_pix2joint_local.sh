#!/bin/bash
python train.py \
      --name pix2joint_local_train \
      --run pix2joint_train \
      --preset Pix2JointTrain \
      --batch_size 3 \
      --gpu_ids 0 \
      --max_data 500 \
      --num_workers 0 \
      --save_epoch 1 \
      --hpe_network basic \
      --hpe_lr 0.1 \
