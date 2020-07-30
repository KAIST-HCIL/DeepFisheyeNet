import torch.nn as nn
import scipy.ndimage
import torch
import numpy as np

class GaussianFilter(nn.Module):
    def __init__(self, channels, kernel_size, sigma, peak_to_one = False):
        super().__init__()
        padding = int(kernel_size/2)
        self.pad = nn.ZeroPad2d(padding)

        kernel = self._make_gaussian_kernel(kernel_size, sigma, peak_to_one)
        self.kernel_max = kernel.max()
        self.conv = self._define_conv(channels, kernel, kernel_size)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

    def _define_conv(self, channels, kernel, kernel_size):
        conv = nn.Conv2d(channels, channels, groups = channels, kernel_size = kernel_size, padding = 0, stride = 1, bias = False)
        conv.weight.data.copy_(kernel)
        return conv

    def _make_gaussian_kernel(self, kernel_size, sigma, peak_to_one):
        g_kernel = np.zeros((kernel_size, kernel_size)).astype(np.float64)
        center = int(kernel_size / 2)
        g_kernel[center, center] = 1
        g_kernel = scipy.ndimage.gaussian_filter(g_kernel, sigma)
        if peak_to_one:
            g_kernel = g_kernel / g_kernel.max()
        return torch.from_numpy(g_kernel)
