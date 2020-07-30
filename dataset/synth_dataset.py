import random
import torch
import torch.nn as nn
import numpy as np
import scipy.ndimage

from dataset.data_model import HandDataModel
from dataset.base_dataset import BaseDataset
import util.filter as filter
from util.image import merge_channel
from util.image import get_center_circle_mask

class SynthDataset(BaseDataset):

    def __init__(self, opt, is_train):
        super().__init__(opt, is_train)
        img_shape = (self.img_size, self.img_size)
        self.fish_mask = get_center_circle_mask(img_shape, dataformats='CHW')

        self.min_depth_thrs = opt.min_depth_thrs

        self.blur = filter.GaussianFilter(channels = 3, kernel_size = 5, sigma = 3, peak_to_one = False)
        self.threshold_depth = nn.Threshold(self.min_depth_thrs, 0)

    def set_hand_list(self):
        self.hand_list = []
        filenames = self._load_filenames('synth')
        for fn in filenames:
            fish_fn, fish_depth_img_fn, fn_joint = fn
            hand = HandDataModel(fish_fn, fish_depth_img_fn, fn_joint)
            self.hand_list.append(hand)

    def __getitem__(self, index):
        hand = self.hand_list[index]

        is_flip = (not self.no_flip) and self.toss_coin()

        hand.load_data(self.img_size, is_flip)
        fish_img = hand.fish_img
        fish_depth_img = hand.fish_depth_img

        if self.is_train:
            fish_img = self.color_transform(fish_img)

        fish_img = self.transform(fish_img)
        fish_img = self._blur_img(fish_img)

        fish_depth = self.transform(fish_depth_img)
        fish_depth = self._mask_fish_area(fish_depth)
        fish_depth = merge_channel(fish_depth, dataformats='CHW')
        fish_depth = self.threshold_depth(fish_depth)

        joint = hand.joints_3d

        fish_segment = self._to_binary(fish_depth, self.min_depth_thrs)

        item = {'fish': fish_img, \
                'fish_depth': fish_depth, \
                'joint': joint
                }

        # this prevents memory explosion.
        hand.unload_data()
        return item

    def _blur_img(self, img):
        img = img.unsqueeze(0)
        img = self.blur(img)
        return img.squeeze(0)

    def _mask_fish_area(self, img):
        return self.fish_mask * img

    def _to_binary(self, x, threshold):
        zero = torch.zeros(x.shape)
        one = torch.ones(x.shape)
        x = torch.where(x > threshold, one, zero)
        return x
