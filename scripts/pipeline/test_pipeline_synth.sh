#!/bin/bash
python test.py \
      --preset PipelineSynthTest \
      --name test_pipeline_synth \
      --batch_size 1 \
      --gpu_ids 0 \
      --num_workers 0 \
      --pipeline_pretrained path/to/weight.pth.tar \
      --no_save_image \
