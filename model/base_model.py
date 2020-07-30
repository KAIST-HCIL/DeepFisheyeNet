from abc import ABC, abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn

class BaseModel(ABC):
    """ Model has a network, optimizers and losses.
    """
    def __init__(self, opt, gpu_ids):

        self.set_opt_and_mode(opt)

        self.gpu_ids = gpu_ids
        self.loss_names = []
        self.optimizers = []
        self.loss = None
        self.networks = None
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.is_setup = False
        self.metric = 0  # used for learning rate policy 'plateau'

        self.schedulers = []
        self.loss_names = []
        self.visual_names = []

    def set_opt_and_mode(self, opt):
        if opt:
            self.opt = opt
            if hasattr(opt, 'mode'):
                self.mode = opt.mode

    def check_and_load_pretrained(self):
        opt = self.opt

        if opt.pretrained:
            print("Load pretrained weights from {}".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            self.load_from_checkpoint(checkpoint, model_only = True)

    @abstractmethod
    def setup(self):
        """ Basic setup steps for a model. All the initialization should be done in here.
            Here are essential tasks for the setup.
            1. make networks
            2. send the networks to devices
            3. make optimizer
            4. make schedulers
        """

    @abstractmethod
    def set_input(self, data):
        """ Hold data in the model as an input.
        """
        pass

    @abstractmethod
    def forward(self):
        """ Run forward process with input.
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """ Where actual parameters are optimized.
        """
        pass

    @abstractmethod
    def pack_as_checkpoint(self):
        """ Pack the model parameters as a checkpoint.
        """
        pass

    @abstractmethod
    def load_from_checkpoint(self, checkpoint, model_only):
        """ Load the model from a checkpoint.
        """
        pass

    @abstractmethod
    def get_current_results(self):
        """ Return current results.
        """
        pass

    def get_detached_current_results(self):
        results = self.get_current_results()
        detached_results = {}
        for key, value in results.items():
            detached_results[key] = self._detach_value(value)
        return detached_results

    def _detach_value(self, value):
        if value is None:
            return value
        if isinstance(value, list):
            new_value = []
            for v in value:
                _v = v.detach().cpu()
                new_value.append(_v)
            return new_value

        return value.detach().cpu()

    def optimize(self):
        self.check_setup()
        assert self.mode.is_train(), "BaseModel: The model should be in training mode to be optimized."
        self.forward()
        self.optimize_parameters()

    def check_setup(self):
        assert self.is_setup, "BaseModel: call 'setup()' before use the model"

    def set_requires_grad(self, networks, requires_grad):
        if not isinstance(networks, list):
            networks = [networks]

        for network in networks:
            if network is not None:
                for p in network.parameters():
                    p.requires_grad = requires_grad

    def get_current_losses(self):
        losses_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                loss_name = 'loss_' + name
                if hasattr(self, loss_name):
                    losses_ret[name] = float(getattr(self, loss_name))  # float(...) works for both scalar tensor and float number
        return losses_ret

    def extract_weights(self, network):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            weights = network.module.cpu().state_dict()
            network.cuda(self.gpu_ids[0])
            return weights
        else:
            return network.cpu().state_dict()

    def apply_weights(self, network, state_dict):

        # resolve data parallel crush.
        if isinstance(network, torch.nn.DataParallel):
            network = network.module

        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        network.load_state_dict(state_dict)

    def send_tensor_to_device(self, tensor):
        if (not tensor.is_cuda) and self.device.type.startswith('cuda'):
            tensor = tensor.to(self.device)
        return tensor

    @abstractmethod
    def get_grads(self):
        """ Return gradients for visualization
        """
        pass
