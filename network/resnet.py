import torch.nn as nn
import network.basic_block as bb
from network.basic_block import ConvBlock

class DownBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=False):
        super().__init__()
        self.stride = stride
        self.input_nc = inplanes
        self.output_nc = planes * self.expansion
        self.conv1 = bb.down_conv1x1(inplanes, planes, norm_layer)
        self.conv2 = bb.down_conv3x3(planes, planes, norm_layer, stride)
        self.conv3 = bb.down_conv1x1(planes, planes * self.expansion, norm_layer, acti_layer = None)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample or (stride > 1)
        if self.downsample:
            self.downsample_block = bb.down_conv1x1(inplanes, planes * self.expansion, norm_layer, stride, acti_layer = None)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample_block(x)

        out += residual
        out = self.relu(out)

        return out

class UniBottleneck(nn.Module):
    def __init__(self, num_features, norm_layer):
        super().__init__()
        self.input_nc = num_features
        self.output_nc = num_features
        self.conv1 = bb.down_conv1x1(num_features, num_features, norm_layer)
        self.conv2 = bb.down_conv3x3(num_features, num_features, norm_layer)
        self.conv3 = bb.down_conv1x1(num_features, num_features, norm_layer, acti_layer = None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.relu(out)

        return out

class UpBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer=nn.BatchNorm2d, stride=1, upsample=False):
        super().__init__()

        self.stride = stride

        self.input_nc = inplanes
        self.output_nc = int(planes / self.expansion)

        if self.stride > 1:
            self.conv1 = bb.up_conv4x4(inplanes, planes, norm_layer, stride)
        else:
            self.conv1 = bb.down_conv1x1(inplanes, planes, norm_layer)

        self.conv2 = bb.down_conv3x3(planes, planes, norm_layer)

        self.conv3 = bb.down_conv1x1(planes, self.output_nc, norm_layer, acti_layer = None)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = upsample or (stride > 1)
        if self.upsample:
            self.upsample_block = bb.up_conv4x4(inplanes, self.output_nc, norm_layer, stride, acti_layer = None)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.upsample:
            residual = self.upsample_block(x)
        out += residual
        out = self.relu(out)

        return out

class DownsampleResnetLayer(nn.Module):
    def __init__(self, input_nc, planes, n_blocks, stride, norm_layer):
        super().__init__()

        self.downsample_block = DownBottleneck(input_nc, planes, norm_layer=norm_layer, stride = stride, downsample = True)
        blocks = []

        in_plane = self.downsample_block.output_nc
        for _ in range(n_blocks - 1):
            b = DownBottleneck(in_plane, planes, norm_layer = norm_layer, stride = 1, downsample = False)
            in_plane = b.output_nc
            blocks.append(b)

        self.identity_blocks = nn.Sequential(*blocks)
        self.input_nc = input_nc
        self.planes = planes
        self.output_nc = blocks[-1].output_nc

    def forward(self, x):
        downsampled = self.downsample_block(x)
        x = self.identity_blocks(downsampled)
        return x, downsampled

class UpsampleRensetLayer(nn.Module):
    def __init__(self, input_nc, planes, n_blocks, stride, norm_layer):
        super().__init__()

        self.upsample_block = UpBottleneck(input_nc, planes, norm_layer=norm_layer, stride = stride, upsample = True)
        blocks = []

        in_plane = self.upsample_block.output_nc
        for _ in range(n_blocks - 1):
            b = UniBottleneck(in_plane, norm_layer)
            in_plane = b.output_nc
            blocks.append(b)

        self.input_nc = input_nc
        self.planes = planes
        self.n_blocks = n_blocks

        if n_blocks > 1:
            self.identity_blocks = nn.Sequential(*blocks)
            self.output_nc = blocks[-1].output_nc
        else:
            self.identity_blocks = None
            self.output_nc = self.upsample_block.output_nc

    def forward(self, x):
        upsampled = self.upsample_block(x)
        if self.n_blocks > 1:
            x = self.identity_blocks(upsampled)
        else:
            x = upsampled
        return x, upsampled

class SimpleResnetLayer(nn.Module):
    def __init__(self, input_nc, n_blocks, norm_layer):
        super().__init__()

        blocks = []
        for _ in range(n_blocks):
            b = UniBottleneck(input_nc, norm_layer)
            input_nc = b.output_nc
            blocks.append(b)

        self.net = nn.Sequential(*blocks)
        self.input_nc = input_nc
        self.output_nc = blocks[-1].output_nc

    def forward(self, x):
        return self.net(x)
