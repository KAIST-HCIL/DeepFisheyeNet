#!/bin/bash
python train.py \
      --name pipeline_synth_train \
      --run pipeline_train \
      --preset PipelineSynthTrain \
      --batch_size 16 \
      --gpu_ids 0,1,2 \
      --num_workers 2 \
      --save_epoch 2 \
      --p2d_lr 0.0001 \
      --hpe_lr 0.01 \
      --epoch 6 \
      --print_iter 10 \
      --p2d_pretrained path/to/weight.pth.tar \
