from .base import BaseOption, Argument

class HPEOption(BaseOption):

    def set_arguments(self):

        # network params
        self.arguments += [Argument('mode', type=str, default='train', help="mode of the model [train | test | eval]")]
        self.arguments += [Argument('network', type=str, default='basic', help="hand pose estimator network [basic]")]
        self.arguments += [Argument('num_joints', type=int, default=21, help="the number of hand joints.")]
        self.arguments += [Argument('gauss_kernel_size', type=int, default=31, help="the size of gaussian kernel for heatmaps")]
        self.arguments += [Argument('gauss_kernel_sigma', type=int, default=5, help="the sigma of gaussian kernel for heatmaps")]
        self.arguments += [Argument('deconv_channel', type=int, default=64, help="channel for deconv layer of the heatmap network")]
        self.arguments += [Argument('init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')]
        self.arguments += [Argument('init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')]

        self.arguments += [Argument('img_size', type=int, default=256, help='width and height of input and output image')]
        self.arguments += [Argument('input_channel', type=int, default=3, help='the number of channels for the input images. 1 for depth, 3 for rgb image')]
        self.arguments += [Argument('norm_type', type=str, default='group', help='the type of normaliztion layer for the network. [batch | group | instance]')]

        # training
        self.arguments += [Argument('lr', type=float, default=0.1, help='initial learning rate for adam')]
        self.arguments += [Argument('heatmap_loss_weight', type=float, default=1.0, help='weight of heatmap losses')]
        self.arguments += [Argument('heatmap_interm_loss_weight', type=float, default=1.0, help='weight of intermediate heatmap losses')]
        self.arguments += [Argument('joint_loss_weight', type=float, default=1.0, help='weight of joint losses')]
        self.arguments += [Argument('joint_interm_loss_weight', type=float, default=1.0, help='weight of intermediate joint losses')]

        # pretrained
        self.arguments += [Argument('pretrained', type=str, default = "", help="pretrained model weights to load. it should be a checkpoint.")]
