#!/bin/bash
python train.py \
      --name pipeline_real_local_train \
      --run pipeline_train \
      --preset PipelineRealTrain \
      --batch_size 1 \
      --gpu_ids 0 \
      --max_data 20 \
      --num_workers 0 \
      --save_epoch 1 \
      --p2d_lr 0.00002 \
      --hpe_lr 0.001 \
      --epoch 30 \
      --print_iter 10 \
      --pipeline_pretrained path/to/weight.pth.tar \
