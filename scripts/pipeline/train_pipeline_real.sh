#!/bin/bash
python train.py \
      --name pipeline_real_train \
      --run pipeline_train \
      --preset PipelineRealTrain \
      --batch_size 16 \
      --gpu_ids 0,1,2 \
      --num_workers 2 \
      --save_epoch 4 \
      --p2d_lr 0.00001 \
      --hpe_lr 0.0001 \
      --epoch 20 \
      --print_iter 10 \
      --pipeline_pretrained path/to/weight.pth.tar \
