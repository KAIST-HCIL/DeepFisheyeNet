import math
import torch
from preset.base_preset import BasePreset

def modify_options(options):
    if options.general.preset:
        preset = find_preset_by(options.general.preset)
        preset.modify_options(options)

def find_preset_by(preset_name):
    class_name = "Preset{}".format(preset_name)
    try:
        preset_class = globals()[class_name]
    except KeyError as e:
        if preset_name:
            print("No preset that as {} name".format(preset_name))
        raise
    return preset_class

###################### Some Common Settings #################
joint_loss_weight = 1.0
joint_interm_loss_weight = 0.5
heatmap_loss_weight = 250.0
heatmap_interm_loss_weight = 125
reprojection_loss_weight = 125

###################### Pix2Joint ###################
class PresetPix2Joint(BasePreset):
    @classmethod
    def modify_options(cls, opt):
        opt.general.dataset = "synth"

        opt.hpe.norm_type = 'instance'
        opt.hpe.img_size = opt.general.img_size
        opt.hpe.input_channel = 3
        opt.hpe.init_gain = 0.2
        opt.hpe.joint_loss_weight = joint_loss_weight
        opt.hpe.joint_interm_loss_weight = joint_interm_loss_weight
        opt.hpe.heatmap_loss_weight = heatmap_loss_weight
        opt.hpe.heatmap_interm_loss_weight = heatmap_interm_loss_weight
        opt.hpe.reprojection_loss_weight = reprojection_loss_weight
        opt.hpe.lr_policy = "step"

class PresetPix2JointTrain(PresetPix2Joint):
    @classmethod
    def modify_options(cls, opt):
        PresetPix2Joint.modify_options(opt)
        opt.hpe.mode = 'train'

class PresetPix2JointTest(PresetPix2Joint):
    @classmethod
    def modify_options(cls, opt):
        PresetPix2Joint.modify_options(opt)
        opt.hpe.mode = 'test'

class PresetRealPix2JointTrain(PresetPix2Joint):
    @classmethod
    def modify_options(cls, opt):
        PresetPix2Joint.modify_options(opt)
        opt.general.dataset = "real"
        opt.hpe.mode = 'train'

class PresetRealPix2JointTest(PresetPix2Joint):
    @classmethod
    def modify_options(cls, opt):
        PresetPix2Joint.modify_options(opt)
        opt.general.dataset = "real"
        opt.hpe.mode = 'test'

###### Pix2depth encoder #####
class PresetPix2Depth(BasePreset):
    @classmethod
    def modify_options(cls, opt):

        opt.general.dataset = 'synth'
        opt.pix2depth.norm = 'instance'
        opt.pix2depth.init_gain = 0.1
        opt.pix2depth.net_type = 'resnet_3blocks'
        opt.pix2depth.depth_loss_weight = heatmap_loss_weight
        opt.pix2depth.output_nc = 1

        opt.pix2depth.joint_loss_weight = joint_loss_weight
        opt.pix2depth.joint_interm_loss_weight = joint_interm_loss_weight
        opt.pix2depth.heatmap_loss_weight = heatmap_loss_weight
        opt.pix2depth.heatmap_interm_loss_weight = heatmap_interm_loss_weight
        opt.pix2depth.reprojection_loss_weight = reprojection_loss_weight


class PresetPix2DepthTrain(PresetPix2Depth):
    @classmethod
    def modify_options(cls, opt):
        PresetPix2Depth.modify_options(opt)
        opt.general.run = 'pix2depth_train'
        opt.pix2depth.mode = 'train'
        return opt

class PresetPix2DepthTest(PresetPix2Depth):
    @classmethod
    def modify_options(cls, opt):
        PresetPix2Depth.modify_options(opt)
        opt.general.run = 'pix2depth_test'
        opt.pix2depth.mode = 'test'
        return opt

class PresetRealPix2DepthTrain(PresetPix2Depth):
    @classmethod
    def modify_options(cls, opt):
        PresetPix2Depth.modify_options(opt)
        opt.general.run = 'pix2depth_train'
        opt.general.dataset = 'real'
        opt.pix2depth.mode = 'train'
        opt.pix2depth.train_only_encoder = True
        opt.pix2depth.depth_loss_weight = 0
        return opt

###################### Pipeline ###################
class PresetPipeline(BasePreset):
    @classmethod
    def modify_options(cls, opt):
        opt.general.dataset = "synth"

        opt.hpe.mode = 'train'
        opt.hpe.network = 'basic'
        opt.hpe.img_size = opt.general.img_size
        opt.hpe.input_channel = 4
        opt.hpe.init_gain = 1.0
        opt.hpe.joint_loss_weight = joint_loss_weight
        opt.hpe.joint_interm_loss_weight = joint_interm_loss_weight
        opt.hpe.heatmap_loss_weight = heatmap_loss_weight
        opt.hpe.heatmap_interm_loss_weight = heatmap_interm_loss_weight
        opt.hpe.lr_policy = "step"
        opt.hpe.norm_type = 'instance'

        opt.pix2depth.model = 'resnet_encoder'
        opt.pix2depth.norm = 'instance'
        opt.pix2depth.n_layers_G = 3
        opt.pix2depth.netG = 'resnet_3blocks'
        opt.pix2depth.netD = 'basic'
        opt.pix2depth.init_gain = 0.1
        opt.pix2depth.depth_loss_weight = heatmap_loss_weight * 10

        opt.pix2depth.joint_loss_weight = joint_loss_weight
        opt.pix2depth.joint_interm_loss_weight = joint_interm_loss_weight
        opt.pix2depth.heatmap_loss_weight = heatmap_loss_weight
        opt.pix2depth.heatmap_interm_loss_weight = heatmap_interm_loss_weight

        opt.pix2depth.output_nc = 1

###################### Pix2Depth2JointLite (pipeline)###################
class PresetPipelineSynthTrain(PresetPipeline):
    @classmethod
    def modify_options(cls, opt):
        PresetPipeline.modify_options(opt)
        opt.general.run = "pipeline_train"

class PresetPipelineSynthTest(PresetPipeline):
    @classmethod
    def modify_options(cls, opt):
        PresetPipeline.modify_options(opt)
        opt.general.run = "pipeline_test"

class PresetPipelineRealTrain(PresetPipelineSynthTrain):
    @classmethod
    def modify_options(cls, opt):
        PresetPipelineSynthTrain.modify_options(opt)
        opt.general.dataset = "real"

class PresetPipelineRealTest(PresetPipelineSynthTest):
    @classmethod
    def modify_options(cls, opt):
        PresetPipelineSynthTest.modify_options(opt)
        opt.general.dataset = "real"
