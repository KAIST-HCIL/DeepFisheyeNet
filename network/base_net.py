import torch.nn as nn
import network.norm as norm

class BaseNet(nn.Module):
    def __init__(self, name, input_nc, output_nc, mode, norm_type):

        super().__init__()

        self.name = name
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.mode = mode

        self.norm_layer = norm.get_norm_layer(norm_type)

    def set_mode(self):
        if self.mode.is_train():
            print("{}: in train mode".format(self.name))
            self.train()
        else:
            print("{}: in test mode".format(self.name))
            self.eval()
