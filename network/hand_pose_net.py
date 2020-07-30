import torch
import torch.nn as nn

import network.helper as helper
import network.norm as norm
import network.resnet as resnet
import network.basic_block as bb
import network.hand_module_net as handnet
import network.unprojection as unproj

from network.base_net import BaseNet

def create_hpe_net(opt, gpu_ids):
    unproject_net_gen = unproj.create_unprojection_net_generator(opt)
    if opt.network == 'basic': # What we used in the paper.
        net = HandPoseNetBasic(opt, unproject_net_gen)
    elif opt.network == 'advance': # Same resnet structure, but deeper.
        net = HandPoseNetAdvance(opt, unproject_net_gen)
    else:
        raise Exception("Hpe network {} is not impelemented.".format(opt.network))

    return helper.init_net(net, opt.init_type, opt.init_gain, gpu_ids)

class HandPoseNetBasic(BaseNet):
    def __init__(self, opt, unproject_net_gen):
        input_nc = opt.input_channel
        output_nc = 0 # multiple output
        mode = opt.mode
        norm_type = opt.norm_type

        super().__init__("HandPoseNetBasic", input_nc, output_nc, mode, norm_type)

        self.num_joints = opt.num_joints
        self.img_size = opt.img_size
        self.img_shape = (opt.img_size, opt.img_size)
        self.norm_type = opt.norm_type

        self.deconv_nc = opt.deconv_channel

        self.unproject_net_gen = unproject_net_gen

        self.run_interm = not self.mode.is_eval()
        self._make_network()

        self.set_mode()

    def _make_network(self):
        base_nc = 64
        n_blocks_list = [3, 3, 4]
        n_stride_list = [2, 2, 2]
        n_heatmap_deconv = 2
        conv4_nc = 512

        self.encoder = HandDataEncoder(self.input_nc, self.num_joints, base_nc, n_blocks_list, n_stride_list, self.img_size, self.mode, self.norm_type, self.unproject_net_gen)

        input_nc = self.encoder.output_nc
        self.conv4e = bb.down_conv3x3(input_nc, conv4_nc, self.norm_layer)

        # create intermediate hand conv part.
        heatmap_interm_conv = handnet.IntermHeatmapConv(conv4_nc, self.num_joints, self.img_shape, self.norm_layer)
        dist_interm_conv = handnet.IntermDistConv(conv4_nc, self.num_joints, self.norm_layer)
        unproject_net = self.unproject_net_gen.create()
        self.hand_interm_conv = handnet.HandConv(self.img_size, heatmap_interm_conv, dist_interm_conv, unproject_net)

        self.conv4f = bb.down_conv3x3(conv4_nc, int(conv4_nc/2), self.norm_layer)

        # create final hand conv part.
        heatmap_conv = handnet.HeatmapConv(int(conv4_nc/2), self.num_joints, n_heatmap_deconv, self.deconv_nc, self.img_shape, self.norm_layer)
        dist_conv = handnet.DistConv(int(conv4_nc/2), self.num_joints, self.norm_layer)
        unproject_net = self.unproject_net_gen.create()
        self.hand_conv = handnet.HandConv(self.img_size, heatmap_conv, dist_conv, unproject_net)

    def forward(self, x):
        result = self.encoder(x)
        x = result['output']

        x = self.conv4e(x)
        if self.run_interm:
            joint_interm, heatmap_interm = self.hand_interm_conv(x)

        x = self.conv4f(x)
        joint, heatmap = self.hand_conv(x)
        result['joint'] = joint
        result['heatmap'] = heatmap

        if self.run_interm:
            result['joint_interms'].append(joint_interm)
            result['heatmap_interms'].append(heatmap_interm)

        return result

class HandPoseNetAdvance(BaseNet):
    def __init__(self, opt, unproject_net_gen):

        input_nc = opt.input_channel
        output_nc = 0 # multiple output
        mode = opt.mode
        norm_type = opt.norm_type

        super().__init__("HandPoseNetAdvance", input_nc, output_nc, mode, norm_type)

        self.num_joints = opt.num_joints
        self.img_size = opt.img_size
        self.img_shape = (opt.img_size, opt.img_size)
        self.norm_type = opt.norm_type
        self.unproject_net_gen = unproject_net_gen

        self.deconv_nc = opt.deconv_channel

        self.run_interm = not self.mode.is_eval()

        self._make_network()

        self.set_mode()

    def _make_network(self):
        base_nc = 64
        n_blocks_list = [3, 4, 4, 3]
        n_stride_list = [2, 2, 2, 2]
        n_heatmap_deconv = 3
        conv6_nc = 1024

        self.encoder = HandDataEncoder(self.input_nc, self.num_joints, base_nc, n_blocks_list, n_stride_list, self.img_size, self.mode, self.norm_type, self.unproject_net_gen)
        input_nc = self.encoder.output_nc

        self.conv6a = bb.down_conv3x3(input_nc, conv6_nc, self.norm_layer)
        heatmap_interm_conv = handnet.IntermHeatmapConv(conv6_nc, self.num_joints, self.img_shape, self.norm_layer)
        dist_interm_conv = handnet.IntermDistConv(conv6_nc, self.num_joints, self.norm_layer)
        unproject_net = self.unproject_net_gen.create()
        self.hand_interm_conv = handnet.HandConv(self.img_size, heatmap_interm_conv, dist_interm_conv, unproject_net)

        self.conv6b = bb.down_conv3x3(conv6_nc, int(conv6_nc/2), self.norm_layer)
        heatmap_conv = handnet.HeatmapConv(int(conv6_nc/2), self.num_joints, n_heatmap_deconv, self.deconv_nc, self.img_shape, self.norm_layer)
        dist_conv = handnet.DistConv(int(conv6_nc/2), self.num_joints, self.norm_layer)
        unproject_net = self.unproject_net_gen.create()
        self.hand_conv = handnet.HandConv(self.img_size, heatmap_conv, dist_conv, unproject_net)

    def forward(self, x):
        result = self.encoder(x)
        x = result['output']

        x = self.conv6a(x)
        joint_interm, heatmap_interm = self.hand_interm_conv(x)
        x = self.conv6b(x)
        joint, heatmap = self.hand_conv(x)
        result['joint'] = joint
        result['heatmap'] = heatmap

        if self.run_interm:
            result['joint_interms'].append(joint_interm)
            result['heatmap_interms'].append(heatmap_interm)

        return result

class HandDataEncoder(BaseNet):
    """ Encodes hand joint information. """
    feature_inc_ratio = 2
    def __init__(self, input_nc, num_joints, base_nc, n_blocks_list, n_stride_list, img_size, mode, norm, unproject_net_gen):
        self.input_nc = input_nc
        self.num_joints = num_joints
        self.base_nc = base_nc
        self.n_blocks_list = n_blocks_list
        self.n_stride_list = n_stride_list
        self.n_layers = len(self.n_blocks_list)
        self.img_size = img_size
        self.img_shape = (img_size, img_size)
        self.unproject_net_gen = unproject_net_gen
        super().__init__("HandDataEncoderPolar", input_nc, 0, mode, norm)

        self._make_network()

        last_layer = self._get_last_layer()
        self.output_nc = last_layer.output_nc
        self.run_interm = not self.mode.is_eval()

        self.set_mode()

    def _make_network(self):
        self.front_conv = bb.ConvBlock(nn.Conv2d, self.input_nc, self.base_nc, kernel_size = 7, norm_layer = self.norm_layer, stride = 1, acti_layer = nn.ReLU, padding = 3)
        self.pooling = nn.MaxPool2d(3, 2, padding=1)
        self._make_resnet_layers()
        self._set_hand_interm_convs()

    def _make_resnet_layers(self):
        input_nc = self.base_nc
        num_channels = self.base_nc
        next_layer_feature_ratio = self.feature_inc_ratio

        for i in range(self.n_layers):
            n_blocks = self.n_blocks_list[i]
            stride = self.n_stride_list[i]
            layer = resnet.DownsampleResnetLayer(input_nc, num_channels, n_blocks, stride, self.norm_layer)
            self._set_layer(i, layer)
            num_channels = int(num_channels * next_layer_feature_ratio)
            input_nc = layer.output_nc

    def _set_layer(self, id, layer):
        setattr(self, "layer_{}".format(id), layer)

    def _get_layer(self, id):
        return getattr(self, "layer_{}".format(id))

    def _get_last_layer(self):
        return self._get_layer(len(self.n_blocks_list)-1)

    def _set_hand_interm_convs(self):
        for i in range(1, self.n_layers):
            layer = self._get_layer(i)
            input_nc = layer.output_nc
            dist_conv = handnet.IntermDistConv(input_nc, self.num_joints, self.norm_layer)
            heatmap_conv = handnet.IntermHeatmapConv(input_nc, self.num_joints, self.img_shape, self.norm_layer)
            unproject_net = self.unproject_net_gen.create()

            hand_conv = handnet.HandConv(self.img_size, heatmap_conv, dist_conv, unproject_net)

            setattr(self, "hand_conv_{}".format(i), hand_conv)

    def _get_hand_interm_convs(self, i):
        return getattr(self, "hand_conv_{}".format(i))

    def forward(self, x):
        x = self.front_conv(x)
        x = self.pooling(x)

        result = {}
        if self.run_interm:
            result['heatmap_interms'] = []
            result['joint_interms'] = []

        first_layer = self._get_layer(0)
        x, downsampeled = first_layer(x)

        for i in range(1, self.n_layers):
            x, joint_interm, heatmap_interm = self.forward_layer(x, i)
            if self.run_interm:
                result['joint_interms'].append(joint_interm)
                result['heatmap_interms'].append(heatmap_interm)

        result['output'] = x

        return result

    def forward_layer(self, x, i):
        layer = self._get_layer(i)
        hand_conv = self._get_hand_interm_convs(i)

        x, downsampeled = layer(x)
        joint_interm = None
        heatmap_interm = None
        if self.run_interm:
            joint_interm, heatmap_interm = hand_conv(downsampeled)

        return x, joint_interm, heatmap_interm
