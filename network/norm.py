import torch.nn as nn
import functools

class InstanceNorm(nn.GroupNorm):
    def __init__(self, channels):
        super().__init__(channels, channels)

class LayerNorm(nn.GroupNorm):
    def __init__(self, channels):
        super().__init__(1, channels)

def get_norm_layer(norm_type, group = 4):
    if norm_type == "batch":
        return nn.BatchNorm2d
    elif norm_type == "group":
        return create_group_norm(group)
    elif norm_type == "instance":
        return InstanceNorm
    elif norm_type == "layer":
        return LayerNorm
    else:
        raise NotImplementedError("norm_type '{}' is not implemented".format(norm_type))

def create_group_norm(self, num_group):
    class GroupNorm(nn.GroupNorm):
        def __init__(self, channels):
            super().__init__(num_group, channels)
