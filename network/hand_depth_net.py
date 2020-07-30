import torch.nn as nn

import network.resnet as resnet
import network.helper as helper
import network.basic_block as bb
import network.norm as norm
import network.unprojection as unproj

from network.hand_pose_net import HandDataEncoder
from network.base_net import BaseNet

def create_hdg_net(opt, gpu_ids):
    unproject_net_gen = unproj.create_unprojection_net_generator(opt)
    net = HandDepthGenerateNet(opt, unproject_net_gen)
    return helper.init_net(net, opt.init_type, opt.init_gain, gpu_ids)

class HandDepthGenerateNet(BaseNet):
    def __init__(self, opt, unproject_net_gen):

        input_nc = opt.input_nc
        output_nc = 0 # multiple output
        mode = opt.mode
        norm_type = opt.norm

        super().__init__("HandDepthGenerateNet", input_nc, output_nc, mode, norm_type)

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.img_size = opt.img_size
        self.img_shape = (opt.img_size, opt.img_size)
        self.n_joints = opt.num_joints
        self.norm_type = opt.norm
        self.base_nc = opt.base_nc
        self.net_type = opt.net_type
        self.unproject_net_gen = unproject_net_gen

        self.norm_layer = norm.get_norm_layer(norm_type)
        self._make_network()
        self.interm_upscale = nn.Upsample(size = self.img_shape, mode = "bilinear")

        self.set_mode()

        self.run_interm = not self.mode.is_eval()

        if opt.train_only_encoder:
            self.backbone.requires_grad = False
            self.decoder.requires_grad = False

    def _make_network(self):
        encoder_base_nc = 64
        n_blocks_list = [3, 3, 3]
        n_stride_list = [2, 2, 2]
        self.encoder = HandDataEncoder(self.input_nc, self.n_joints, encoder_base_nc, n_blocks_list, n_stride_list, self.img_size, self.mode, self.norm_type, self.unproject_net_gen)
        self.backbone = self._make_resnet_backbone(self.encoder)

        decoder_base_nc = 512
        decoder_n_blocks = [3, 3, 3]
        self.decoder = DepthDecoder(self.backbone.output_nc, self.output_nc, decoder_base_nc, decoder_n_blocks, self.img_shape, self.norm_layer)

    def _make_resnet_backbone(self, encoder):
        if self.net_type == "resnet_6blocks":
            planes = 256
            n_blocks = 6
        elif self.net_type == "resnet_3blocks":
            planes = 256
            n_blocks = 3
        else:
            raise NotImplementedError("{} is not implemented".format(self.netG))

        input_nc = encoder.output_nc
        return resnet.SimpleResnetLayer(input_nc, n_blocks, self.norm_layer)

    def forward(self, x):
        result = self.encoder(x)
        x = result['output']
        x = self.backbone(x)

        decoder_result = self.decoder(x)

        result['fake'] = decoder_result['output']
        result['interms'] = decoder_result['depth_interms']

        return result

class DepthDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, base_nc, n_blocks, img_shape, norm_layer):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.base_nc = base_nc
        self.img_shape = img_shape
        self.num_layers = len(n_blocks)
        self.n_blocks = n_blocks
        self.norm_layer = norm_layer
        self._make_layers()
        self._make_interm_convs()
        self.end_conv = self._make_end_conv()

    def _make_layers(self):
        input_nc = self.input_nc
        num_features = self.base_nc
        for i, nb in enumerate(self.n_blocks):
            layer = resnet.UpsampleRensetLayer(input_nc, num_features, nb, stride = 2, norm_layer = self.norm_layer)
            self._set_layer(i, layer)

            num_features = int(num_features/2)
            input_nc = layer.output_nc

    def _make_interm_convs(self):
        for i in range(1, self.num_layers):
            layer = self._get_layer(i)
            input_nc = layer.output_nc
            interm_conv = DepthIntermConv(input_nc, self.output_nc, self.img_shape, self.norm_layer)
            self._set_interm_conv(i, interm_conv)

    def _set_layer(self, id, layer):
        setattr(self, "layer_{}".format(id), layer)

    def _get_layer(self, id):
        return getattr(self, "layer_{}".format(id))

    def _get_last_layer(self):
        last_id = self.num_layers - 1
        return self._get_layer(last_id)

    def _set_interm_conv(self, id, conv):
        setattr(self, "interm_conv_{}".format(id), conv)

    def _get_interm_conv(self, id):
        return getattr(self, "interm_conv_{}".format(id))

    def _make_end_conv(self):
        last_layer = self._get_last_layer()
        conv = bb.ConvBlock(nn.Conv2d, last_layer.output_nc, self.output_nc, kernel_size = 1, norm_layer = None, stride = 1, padding = 0, acti_layer = nn.Tanh, bias = True)
        upsample = nn.Upsample(size = self.img_shape)
        return nn.Sequential(conv, upsample)

    def forward(self, x):
        interms = []

        first_layer = self._get_layer(0)
        x, upsampeld = first_layer(x)

        for i in range(1, self.num_layers):
            layer = self._get_layer(i)
            x, upsampled = layer(x)
            interm_conv = self._get_interm_conv(i)
            interm = interm_conv(upsampled)
            interms.append(interm)

        result = {}
        result['output'] = self.end_conv(x)
        result['depth_interms'] = interms
        return result

    def _forward_layers(self, i, x):
        layer = getattr(selt, "layer{}".format(i))
        return layer(x)

class DepthIntermConv(nn.Module):
    def __init__(self, input_nc, output_nc, img_shape, norm_layer):
        super().__init__()
        self.upconv = bb.up_conv4x4(input_nc, output_nc, norm_layer)
        self.scale_conv = bb.ConvBlock(nn.Conv2d, output_nc, output_nc, kernel_size = 1, norm_layer = None, stride = 1, padding = 0, acti_layer = nn.Tanh, bias = True)
        self.upscale = nn.Upsample(size = img_shape, mode = "bilinear")

    def forward(self, x):
        x = self.upconv(x)
        x = self.scale_conv(x)
        x = self.upscale(x)
        return x
