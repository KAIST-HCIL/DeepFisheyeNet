import torch.nn as nn
from functools import partial

import network.basic_block as bb
import network.resnet as resnet
""" Submodules for hand pose estimation networks. """
class HeatmapConv(nn.Module):
    """ Decode heatmaps from encoded features. """
    def __init__(self, input_nc, num_joints, n_deconv, deconv_nc, img_shape, norm_layer):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = num_joints
        self.norm_layer = norm_layer
        self.deconv_nc = deconv_nc
        self.n_deconv = n_deconv
        self.img_shape = img_shape

        self.net = self._make_network()
        self.upsample = nn.Upsample(size = img_shape, mode = 'bilinear')

    def _make_network(self):
        norm_layer = self.norm_layer

        conv1 = bb.down_conv3x3(self.input_nc, int(self.input_nc / 4), norm_layer)

        deconv1 = bb.up_conv4x4(int(self.input_nc / 4), self.deconv_nc, norm_layer, stride = 2)
        blocks = [conv1, deconv1]
        deconv_input_nc = deconv1.output_nc
        for _ in range(self.n_deconv - 1):
            deconv = bb.up_conv4x4(self.deconv_nc, self.deconv_nc, norm_layer, stride = 2)
            blocks.append(deconv)

        final_conv = bb.ConvBlock(nn.Conv2d, self.deconv_nc, self.output_nc,
                            kernel_size = 1, stride = 1, padding = 0, bias = True,
                            norm_layer = None, acti_layer = nn.Sigmoid)

        blocks.append(final_conv)
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.net(x)
        return self.upsample(x)

class IntermHeatmapConv(nn.Module):
    def __init__(self, input_nc, num_joints, img_shape, norm_layer):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = num_joints

        deconv = bb.up_conv4x4(input_nc, num_joints, norm_layer, stride = 2)
        conv1 = bb.ConvBlock(nn.Conv2d, num_joints, num_joints, kernel_size = 1,
                                stride = 1, padding = 0, bias = True,
                                norm_layer = None, acti_layer = nn.Sigmoid)

        upsample = nn.Upsample(size = img_shape, mode = 'bilinear')
        blocks = [deconv, conv1, upsample]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class DistConv(nn.Module):
    """ Decode distance vectors from encoded features. """
    def __init__(self, input_nc, num_joints, norm_layer):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = num_joints
        self.norm_layer = norm_layer

        self.net = self._make_network()

    def _make_network(self):
        norm_layer = self.norm_layer

        conv1 = bb.down_conv3x3(self.input_nc, 128, norm_layer, stride = 2, acti_layer = nn.Sigmoid)
        conv2 = bb.down_conv3x3(128, 256, norm_layer, stride = 2, acti_layer = nn.Sigmoid)
        inner_product = bb.O2OBlock(256, self.output_nc, global_pool = True, acti_layer = nn.Sigmoid)
        # acti layer is sigmoid because of relu's dead neuron problem
        fc1 = bb.FCBlock(self.output_nc, self.output_nc, acti_layer = bb.Sigmoid6, bias = True)
        layers = [conv1, conv2, inner_product, fc1]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class IntermDistConv(nn.Module):
    def __init__(self, input_nc, num_joints, norm_layer):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = num_joints
        self.norm_layer = norm_layer

        self.net = self._make_network()

    def _make_network(self):
        norm_layer = self.norm_layer
        conv1 = bb.down_conv3x3(self.input_nc, 256, norm_layer, stride = 2, acti_layer = nn.Sigmoid)
        one_by_one = bb.O2OBlock(256, self.output_nc, global_pool = True, acti_layer = nn.Sigmoid)

        # acti layer is sigmoid because of relu's dead neuron problem
        small_fc = bb.FCBlock(self.output_nc, self.output_nc, acti_layer = bb.Sigmoid6, bias = True)
        layers = [conv1, one_by_one, small_fc]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def leaky_relu():
    return nn.LeakyReLU(negative_slope = 0.1, inplace = True)

class HandConv(nn.Module):
    """ Conv layers that decodes joint data from encoded features.
        HandConv = HeatmapConv + DistConv + Unprojection Network
    """
    def __init__(self, img_size, heatmap_conv, dist_conv, unproject_net):
        super().__init__()
        self.heatmap_conv = heatmap_conv
        self.dist_conv = dist_conv
        self.unproject_net = unproject_net

    def forward(self, x):
        heatmap = self.heatmap_conv(x)
        dist = self.dist_conv(x)
        joint = self.unproject_net(heatmap, dist)
        return joint, heatmap
