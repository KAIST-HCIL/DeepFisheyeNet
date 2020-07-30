import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import util.fisheye as fisheye
from util.debug import viz_grad, viz_grad_mean
import network.basic_block as bb

""" Unprojection networks
    1) get keypoints from heatmaps
    2) convert the keypoints to direction vectors
    3) convert the vectors and distance values to 3D points in the cartesian coordinate.
"""

def unravel_index(idx, H, W):
    row = (idx / W).long()
    col = (idx % W).long()
    return row, col

class ArgMax2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        N, C, H, W = tensor.shape
        tensor = tensor.reshape(N, C, H*W)
        max_idx = tensor.argmax(dim = -1)
        row, col = unravel_index(max_idx, H, W)
        idx = torch.stack([row, col], dim = 2)
        idx = idx.float()
        ctx.input_shape = (N, C, H, W)
        ctx.save_for_backward(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_output):

        idx, = ctx.saved_tensors
        N, C, H, W = ctx.input_shape


        grad_input = torch.zeros(N, C, H, W)
        grad_input = grad_input.to(grad_output.device)

        return grad_input

def create_unprojection_net_generator(opt):
    return UnprojectionNetGenerator(opt.img_size)

class UnprojectionNetGenerator:
    def __init__(self, img_size):
        self.img_size = img_size

    def create(self):
        return FisheyeUnprojectionNet(self.img_size)

class FisheyeUnprojectionNet(nn.Module):
    def __init__(self, img_size, fisheye_type = "equidistant"):
        super().__init__()
        self.img_size = img_size
        self.img_radius = img_size / 2
        self.argmax_2d = ArgMax2d.apply
        self.fisheye_type = fisheye_type

    def forward(self, heatmaps, dists):
        max_idx = self.argmax_2d(heatmaps)
        theta_phi = self._convert_to_theta_phi(max_idx)
        r = dists
        theta = theta_phi[:,:,0]
        phi = theta_phi[:,:,1]
        xyz = self._convert_to_cartesian(r, theta, phi)
        return xyz

    def _convert_to_theta_phi(self, max_idx):
        max_idx = max_idx.float()
        img_coord_idx = max_idx - self.img_radius
        u = img_coord_idx[:,:,1]
        v = img_coord_idx[:,:,0]

        phi = torch.atan2(v, u)

        r = (u**2 + v**2).sqrt()

        theta = fisheye.inverse_r_function(self.img_radius, r, self.fisheye_type)

        theta_phi = torch.stack([theta, phi], dim = 2)
        return theta_phi

    def _convert_to_cartesian(self, r, theta, phi):
        z = r * torch.cos(theta)
        xy = r * torch.sin(theta)
        x = xy * torch.cos(phi)
        y = xy * torch.sin(phi)

        xyz = torch.stack([x, y, z], dim = 2)
        return xyz
