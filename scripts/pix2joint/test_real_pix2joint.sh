#!/bin/bash
python test.py \
      --name real_pix2joint_test \
      --run pix2joint_test \
      --preset RealPix2JointTest \
      --batch_size 1 \
      --gpu_ids 0 \
      --num_workers 0 \
      --hpe_network basic \
      --hpe_pretrained path/to/weight.pth.tar \
      --no_save_image
