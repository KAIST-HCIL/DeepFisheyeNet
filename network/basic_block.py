import torch
import torch.nn as nn
import numpy as np

def down_conv1x1(inplanes, planes, norm_layer, stride = 1, acti_layer = nn.ReLU):
    return ConvBlock(nn.Conv2d, inplanes, planes, 1, stride, 0, norm_layer = norm_layer, acti_layer = acti_layer)

def down_conv3x3(inplanes, planes, norm_layer, stride = 1, acti_layer = nn.ReLU):
    return ConvBlock(nn.Conv2d, inplanes, planes, 3, stride, 1, norm_layer = norm_layer, acti_layer = acti_layer)

def down_conv7x7(inplanes, planes, norm_layer, stride = 1, acti_layer = nn.ReLU):
    return ConvBlock(nn.Conv2d, inplanes, planes, 7, stride, 3, norm_layer = norm_layer, acti_layer = acti_layer)

def up_conv4x4(inplanes, planes, norm_layer, stride = 1, acti_layer = nn.ReLU):
    return ConvBlock(nn.ConvTranspose2d, inplanes, planes, 4, stride, 1, norm_layer = norm_layer, acti_layer = acti_layer)

def up_conv6x6(inplanes, planes, norm_layer, stride = 1, acti_layer = nn.ReLU):
    return ConvBlock(nn.ConvTranspose2d, inplanes, planes, 6, stride, 2, norm_layer = norm_layer, acti_layer = acti_layer)

class ConvBlock(nn.Module):
    def __init__(self, conv_class, input_nc, output_nc, kernel_size, stride, padding, bias = False, norm_layer = None, acti_layer = None):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        layers = []
        conv = conv_class(input_nc, output_nc, \
                kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        layers.append(conv)

        if norm_layer is not None:
            layers.append(norm_layer(output_nc))

        if acti_layer is not None:
            if acti_layer == nn.ReLU or acti_layer == nn.LeakyReLU:
                layers.append(acti_layer(inplace=True))
            else:
                layers.append(acti_layer())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class O2OBlock(nn.Module):
    # rather than using fully connected, use one-to-one convolution
    def __init__(self, in_channel, out_channel, global_pool, acti_layer = None):
        super().__init__()

        self.conv = ConvBlock(nn.Conv2d, in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True, acti_layer = acti_layer)
        self.global_pool = global_pool

    def forward(self, x):
        # Assuming x as (N, C, H, W)

        x = self.conv(x)
        # avg pooling
        if self.global_pool:
            x = x.mean(dim=(2,3))
        return x

class FCBlock(nn.Module):
    def __init__(self, in_channel, out_channel, acti_layer = None, bias = True):
        super().__init__()
        self.fc = nn.Linear(in_channel, out_channel, bias = bias)

        if acti_layer:
            if acti_layer == nn.ReLU or acti_layer == nn.LeakyReLU:
                self.activation_layer = acti_layer(inplace = True)
            else:
                self.activation_layer = acti_layer()

    def forward(self, x):
        x = self.fc(x)

        if self.activation_layer:
            x = self.activation_layer(x)

        return x

class Sigmoid6(nn.Module):
    value_range = 6
    def __init__(self):
        super().__init__()
        self.acti_layer = nn.Sigmoid()

    def forward(self, x):
        return self.acti_layer(x) * self.value_range
