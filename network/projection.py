import torch
import torch.nn as nn
import numpy as np
import network.basic_block as bb
import util.fisheye as fisheye
import network.helper as helper
from util.debug import viz_grad, viz_grad_mean
import util.image as image
import util.io as io
import torchvision
import util.math as umath
import util.filter as filter

from abc import ABC, abstractmethod

def create_projection_net(opt, gpu_ids):
    net = FisheyeProjectionNet(opt.num_joints, opt.img_size, opt.gauss_kernel_size, opt.gauss_kernel_sigma)
    return helper.init_net(net, gpu_ids = gpu_ids, initialize_weights = False)

class ProjectionNet(nn.Module, ABC):
    def __init__(self, num_joints, img_size, g_size, g_sigma):
        super().__init__()

        self.num_joints = num_joints
        self.img_size = img_size
        self.img_shape = (img_size, img_size)
        self.gen_heatmap_seed = GenHeatmapSeed.apply
        self.gaussian_filter = filter.GaussianFilter(channels = num_joints, kernel_size = g_size, sigma = g_sigma, peak_to_one = True)

    def forward(self, joint):
        assert len(joint.shape) == 3, "joint shape should be (batch_size, num_joints, 3)"

        mapped_joint = self._convert_to_uv(joint)

        heatmap_seed = self.gen_heatmap_seed(mapped_joint, self.img_size)
        heatmap = self.gaussian_filter(heatmap_seed)

        self.saved_for_backward = heatmap
        return heatmap

    @abstractmethod
    def _convert_to_uv(self, xyz):
        pass

class GenHeatmapSeed(torch.autograd.Function):
    """ A differentiable function that generates a heatmap seed matrix from uv coordinates """
    @staticmethod
    def forward(ctx, uv, img_size):
        # uv: N, n_joints, 2
        n_sample, n_joints = uv.shape[:2]

        int_uv_mat = torch.round(uv).long()
        int_uv_mat = torch.clamp(int_uv_mat, min = 0, max = img_size - 1)

        batch_joint_index = np.arange(0, n_sample * n_joints)

        flatten_idx = int_uv_mat.view(-1, 2)
        x_idx = flatten_idx[:,0]
        y_idx = flatten_idx[:,1]

        heatmap_seeds = torch.zeros(n_sample * n_joints, img_size, img_size)
        heatmap_seeds[batch_joint_index, y_idx, x_idx] = 1
        heatmap_seeds = heatmap_seeds.view(n_sample, n_joints, img_size, img_size)

        heatmap_seeds = heatmap_seeds.to(uv.device)

        ctx.save_for_backward(uv, heatmap_seeds)

        return heatmap_seeds

    @staticmethod
    def backward(ctx, grad_output):
        pred_uv, heatmap_seeds = ctx.saved_tensors

        # Just pass 0. Backward is never called in the project.
        # However, you might want to have a proper backward function.
        # Then the backward function can be implemented in here.
        grad_input = torch.zeros(pred_uv.shape)

        return grad_input, None # None is for img_size input

class DifferentiableRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FisheyeProjectionNet(ProjectionNet):
    def __init__(self, num_joints, img_size, g_size, g_sigma, fisheye_type = 'equidistant'):
        super().__init__(num_joints, img_size, g_size, g_sigma)
        self.name = "Fisheye Projection Network"
        self.radius = int(self.img_size / 2)
        self.center = (self.radius, self.radius)
        self.fisheye_type = fisheye_type
        self.differentiable_round = DifferentiableRound.apply

    def _convert_to_uv(self, xyz):
        #xyz shape: N, num_joint, 3
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

        fish_x = self.differentiable_round(self.center[0] + _x).unsqueeze(1)
        fish_y = self.differentiable_round(self.center[1] + _y).unsqueeze(1)
        out = torch.cat((fish_x, fish_y), dim = 1)
        out = out.view(n_sample, n_joints, -1)

        return out
