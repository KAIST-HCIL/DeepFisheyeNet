#!/bin/bash
rm -r tensorboard_logs
mkdir tensorboard_logs
tensorboard --logdir tensorboard_logs --port 8008 --bind_all
