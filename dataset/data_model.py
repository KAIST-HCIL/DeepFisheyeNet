from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class HandDataModel:

    def __init__(self, fish_fn, fish_depth_img_fn=None, fn_3d_joints=None):

        self.fish_fn = fish_fn
        self.fish_depth_img_fn = fish_depth_img_fn
        self.fn_3d_joints = fn_3d_joints

    def load_data(self, img_size, is_flip):
        self.load_imgs()

        self.joints_3d = self._parse_joints(self.fn_3d_joints)

        if is_flip:
            self._flip()

    def unload_data(self):
        del self.fish_img
        del self.fish_depth_img
        del self.joints_3d

    def load_imgs(self):
        self.fish_img = self._load_img(self.fish_fn)
        self.fish_depth_img = self._load_img(self.fish_depth_img_fn)

    def _load_img(self, fn):
        if fn is None:
            return None
        return Image.open(fn)

    def _parse_joints(self, filename):
        if filename is None:
            return None
        with open(filename, 'r') as f:
            line = f.readline()
            data = [float(d) for d in line.split(',')]
            joints_3d = np.array(data).reshape((-1,3))
            joints_3d = torch.FloatTensor(joints_3d)
            return joints_3d

    def _flip(self):
        self.fish_img = self._flip_img(self.fish_img)
        self.fish_depth_img = self._flip_img(self.fish_depth_img)
        self.joints_3d[:,0] *= -1

    def _flip_img(self, img):
        if img is None:
            return None
        return transforms.functional.hflip(img)
