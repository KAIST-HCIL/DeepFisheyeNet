from abc import ABC, abstractmethod
import torch
import numpy as np
import cv2
import math
import util.fisheye as fisheye

class BaseProjector(ABC):

    def __init__(self, img_size):
        self.img_size = img_size
        self.img_shape = (img_size, img_size)

    def make_heatmap_seed(self, joint, data_format = 'NJC'):
        mapped_joint = self.convert_to_uv(joint, data_format = data_format)
        heatmap_seed = self.convert_to_heatmap_seed(mapped_joint, self.img_shape)
        return heatmap_seed

    @abstractmethod
    def convert_to_uv(self, xyz):
        pass

    def convert_to_heatmap_seed(self, uv_mat, img_shape):
        assert len(uv_mat.shape) == 3 # N, n_joints, 2

        n_sample, n_joints = uv_mat.shape[:2]

        int_uv_mat = torch.round(uv_mat).long()
        int_uv_mat = torch.clamp(int_uv_mat, min = 0, max = img_shape[0] - 1)

        batch_joint_index = np.arange(0, n_sample * n_joints)

        flatten_idx = int_uv_mat.view(-1, 2)
        x_idx = flatten_idx[:,0]
        y_idx = flatten_idx[:,1]

        heatmaps = torch.zeros(n_sample * n_joints, img_shape[0], img_shape[1])
        heatmaps[batch_joint_index, y_idx, x_idx] = 1
        heatmaps = heatmaps.view(n_sample, n_joints, img_shape[0], img_shape[1])

        return heatmaps

    def check_data_format(self, xyz, data_format):
        if data_format == 'NJC':
            assert len(xyz.shape) == 3, "for 'NC' data format, the shape should be (N,n_joints,3)"
        elif data_format == 'NC':
            assert len(xyz.shape) == 2, "for 'NC' data format, the shape should be (N,3)"
        else:
            raise Exception("Wrong Data format. The data format should be 'NJC' or 'NC'")

class FisheyeProjector(BaseProjector):
    def __init__(self, img_size, fisheye_type = 'equidistant'):
        super().__init__(img_size)

        self.radius = int(self.img_size / 2)
        self.center = (self.radius, self.radius)
        self.fisheye_type = fisheye_type

    def convert_to_uv(self, xyz, data_format='NJC'):
        self.check_data_format(xyz, data_format)

        if data_format == 'NJC':
            n_sample, n_joints = xyz.shape[:2]

        xyz = xyz.view(-1,3)
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]

        theta = torch.atan2(torch.sqrt(x*x + y*y), z)
        phi = torch.atan2(y, x)

        r = fisheye.r_function(self.radius, theta, self.fisheye_type)

        _x = r * torch.cos(phi)
        _y = r * torch.sin(phi)

        fish_x = torch.round(self.center[0] + _x).unsqueeze(1)
        fish_y = torch.round(self.center[1] + _y).unsqueeze(1)

        out = torch.cat((fish_x, fish_y), dim = 1)
        if data_format == 'NJC':
            out = out.view(n_sample, n_joints, -1)
        return out

class FlatProjector(BaseProjector):
    def __init__(self, img_size, space_to_img_ratio):
        super().__init__(img_size)
        self.space_to_img_ratio = space_to_img_ratio
        self.range_3d = img_size / space_to_img_ratio

    def convert_to_uv(self, xyz, data_format = 'NC'):
        self.check_data_format(xyz, data_format)

        if data_format == 'NJC':
            n_sample, n_joints = xyz.shape[:2]

        xyz = xyz.view(-1, 3)

        xy = xyz[:,0:2]

        uv = self.proj_to_img_plane(xy)
        if data_format == 'NJC':
            uv = uv.view(n_sample, n_joints, -1)
        return uv

    def proj_to_img_plane(self, val):
        return torch.round((val / self.range_3d) * self.img_size + (self.img_size/2))
