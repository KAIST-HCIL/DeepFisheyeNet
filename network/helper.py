import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import network.norm as norm
import network.basic_block as bb
import util.filter as filter

def is_conv_or_linear(m):
    classname = m.__class__.__name__
    return classname.startswith('Conv') or classname.startswith('Linear')

def is_batch_or_group_norm(m):
    classname = m.__class__.__name__
    return classname.startswith('BatchNorm') or classname.startswith('GroupNorm')

def init_weights(net, init_type='normal', init_gain = 0.1):

    initializer = None
    if init_type == 'normal':
        initializer = functools.partial(init.normal_, mean=0.0, std=init_gain)
    elif init_type == 'xavier':
        initializer = functools.partial(init.xavier_normal_, gain = init_gain)
    elif init_type == 'kaiming':
        initializer = functools.partial(init.kaiming_normal_, gain = init_gain)
    else:
        raise ValueError("init_type with {} is not valid".format(init_type))

    def weights_init(m):
        if hasattr(m, 'weight') and is_conv_or_linear(m):
            initializer(m.weight.data)
        elif is_batch_or_group_norm(m):
            # Instance norm is implemenet with GroupNorm in our case.
            init.normal_(m.weight.data, mean=1.0, std=init_gain)
            init.constant_(m.bias.data, 0.0)

    if isinstance(net, nn.DataParallel):
        name = net.module.name
    else:
        name = net.name

    print("{} is initialized with {}".format(name, init_type))
    net.apply(weights_init)

def send_net_to_device(net, gpu_ids = []):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], initialize_weights = True):
    net = send_net_to_device(net, gpu_ids)
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain)
    return net
