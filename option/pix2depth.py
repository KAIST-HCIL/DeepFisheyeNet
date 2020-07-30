from .base import BaseOption, Argument

class Pix2DepthOption(BaseOption):

    def set_arguments(self):

        # model parameters
        self.arguments += [Argument('mode', type=str, default='train', help="mode of the model [train | test | eval]")]
        self.arguments += [Argument('input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')]
        self.arguments += [Argument('output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')]
        self.arguments += [Argument('num_joints', type=int, default=21, help="the number of hand joints.")]
        self.arguments += [Argument('base_nc', type=int, default=64, help='# of filters for the first conv layer')]
        self.arguments += [Argument('net_type', type=str, default='resnet_3blocks', help='type of the generator backbone')]
        self.arguments += [Argument('norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')]
        self.arguments += [Argument('init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')]
        self.arguments += [Argument('init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')]
        self.arguments += [Argument('img_size', type=int, default=256, help='width and height of input and output image')]
        self.arguments += [Argument('gauss_kernel_size', type=int, default=31, help="the size of gaussian kernel for heatmaps")]
        self.arguments += [Argument('gauss_kernel_sigma', type=int, default=5, help="the sigma of gaussian kernel for heatmaps")]

        # train and test details
        self.arguments += [Argument('niter', type=int, default=1000, help='# of iter(epoch) at starting learning rate')]
        self.arguments += [Argument('niter_decay', type=int, default=1000, help='# of iter(epoch) to linearly decay learning rate to zero')]
        self.arguments += [Argument('beta1', type=float, default=0.5, help='momentum term of adam')]
        self.arguments += [Argument('lr', type=float, default=0.0002, help='initial learning rate for adam')]
        self.arguments += [Argument('train_only_encoder', action='store_true', default = False, help="train only encoder part of pix2depth.")]

        self.arguments += [Argument('depth_loss_weight', type=float, default=1.0, help='weight of depth error')]
        self.arguments += [Argument('heatmap_loss_weight', type=float, default=1.0, help='weight of heatmap losses')]
        self.arguments += [Argument('heatmap_interm_loss_weight', type=float, default=1.0, help='weight of intermediate heatmap losses')]
        self.arguments += [Argument('joint_loss_weight', type=float, default=1.0, help='weight of joint losses')]
        self.arguments += [Argument('joint_interm_loss_weight', type=float, default=1.0, help='weight of intermediate joint losses')]


        # pretrained
        self.arguments += [Argument('pretrained', type=str, default = "", help="pretrained model weights to load. it should be a checkpoint.")]
