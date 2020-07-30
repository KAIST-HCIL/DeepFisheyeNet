import pathlib
import os
import random

from dataset.base_dataset import BaseDataset
from dataset.data_model import HandDataModel

class RealDataset(BaseDataset):

    def set_hand_list(self):
        self.hand_list = []
        filenames = self._load_filenames('real')
        for fn in filenames:
            img_fn, fn_3d = fn
            hand = HandDataModel(img_fn, None, fn_3d_joints = fn_3d)
            self.hand_list.append(hand)

    def __getitem__(self, index):
        hand = self.hand_list[index]

        if self.is_train:
            is_flip = (not self.no_flip) and self.toss_coin()
        else:
            is_flip = False

        hand.load_data(self.img_size, is_flip = is_flip)
        img = hand.fish_img

        if self.is_train:
            img = self.color_transform(img)
        pix = self.transform(img)

        joint = hand.joints_3d

        item = {'fish': pix, \
                'joint': joint
                }

        # this prevents memory explosion.
        hand.unload_data()
        return item
