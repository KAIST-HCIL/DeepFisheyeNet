import numpy as np
import torch
import torchvision
import time
from torch.utils.tensorboard import SummaryWriter
from .image import blend

from util.image import unnormalize_as_img

class Visualizer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def add_losses(self, tag, losses, epoch):
        for k, v in losses.items():
            self.writer.add_scalar("{}/{}".format(tag, k), v, epoch, walltime = time.time())

    def add_images(self, tag, imgs, epoch, nrow = 5, dataformats = 'NCHW'):
        self.writer.add_images(tag, imgs, epoch, walltime = time.time(), dataformats = dataformats)

    def add_image(self, tag, img, epoch, dataformats = 'CHW'):
        self.writer.add_image(tag, img.squeeze(0), epoch, walltime = time.time(), dataformats = dataformats)

    def add_histogram(self, tag, val, epoch):
        self.writer.add_histogram(tag, val, epoch, walltime = time.time())

    def blend_heatmap(self, img, heatmap):
        img = img.cpu().detach()
        heatmap = heatmap.cpu().detach()
        blended = blend(img, heatmap)
        blended_img = torchvision.transforms.ToPILImage()(blended)
        return blended_img
