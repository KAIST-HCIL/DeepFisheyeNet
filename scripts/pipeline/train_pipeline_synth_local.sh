#!/bin/bash
python train.py \
      --name pipeline_synth_local_train \
      --run pipeline_train \
      --preset PipelineSynthTrain \
      --batch_size 1 \
      --gpu_ids 0 \
      --max_data 20 \
      --num_workers 0 \
      --save_epoch 100 \
      --p2d_lr 0.0005 \
      --hpe_lr 0.005 \
      --epoch 30 \
      --p2d_pretrained path/to/weight.pth.tar \
