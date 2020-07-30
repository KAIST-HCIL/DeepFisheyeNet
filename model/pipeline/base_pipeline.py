from abc import ABC, abstractmethod
from collections import OrderedDict
import torch

from model.base_model import BaseModel

class BasePipeline(BaseModel):
    """ The Pipeline is a special form of the Model.
        It is a collection and a wrapper of models.
        The roles of a pipeline are
        - connect models
        - init models with a correct option
        - (additional) optimize models collectively
        - (additional) manipulate models

        SOME IMPLEMENTATION TIPS :
        - A pipeline should not have its own option. Then it becomes same with a model.
        - A pipeline can have own optimizers and losses.
    """

    def __init__(self, options):
        """
            Override constructor.
        """
        self.options = options
        gpu_ids = options.general.gpu_ids
        self.model_names = []

        super().__init__(None, gpu_ids)

    def check_and_load_pretrained(self):
        """ Override its super class.
            A pretrained weights for a pipeline consists multiple weights for multiple modules.
        """

        opt = self.options.general

        if opt.pipeline_pretrained:
            print("Load pretrained weights from {}".format(opt.pipeline_pretrained))
            checkpoint = torch.load(opt.pipeline_pretrained)
            self.load_from_checkpoint(checkpoint, model_only = True)

    def _get_model_by_name(self, name):
        model = getattr(self, "{}_model".format(name))
        assert isinstance(model, BaseModel), "No module with '{}'".format(name)
        return model

    def _has_model_with_name(self, name):
        return name in self.model_names

    def pack_as_checkpoint(self):
        """ Implement abstract method.
            Basically, a pipeline just packs all the modules as checkpoints and collects them.
            However, it is not a strcit rule and this method can be overridden.
        """
        collected = {}
        for model_name in self.model_names:
            model = self._get_model_by_name(model_name)
            checkpoint = model.pack_as_checkpoint()
            collected[model_name] = checkpoint

        return collected

    def load_from_checkpoint(self, checkpoint, model_only):
        """ Implement abstract method.
            Reverse process of 'pack_as_checkpoint'.
        """
        for model_name, cp in checkpoint.items():
            if not self._has_model_with_name(model_name):
                continue
            model = self._get_model_by_name(model_name)
            model.load_from_checkpoint(cp, model_only = model_only)

    def get_current_losses(self):
        """ Override its super class.
            Usually, a pipeline doesn't have its own losses, so just return losses of each models with a prefix.
            However, it is not a strcit rule and this method can be override.
        """
        losses_ret = OrderedDict()
        for model_name in self.model_names:
            model = self._get_model_by_name(model_name)
            losses = model.get_current_losses()
            for k, v in losses.items():
                new_k = "{}_{}".format(model_name, k)
                losses_ret[new_k] = v
        return losses_ret

    def get_current_results_wo_detach(self):
        raise NotImplementedError("Don't call this for pipeline.")

    def get_grads(self):
        combined_grads = {}
        for model_name in self.model_names:
            model = getattr(self, "{}_model".format(model_name))
            grads = model.get_grads()
            for tag, val in grads.items():
                tag = "{}_{}".format(model_name, tag)
                combined_grads[tag] = val

        return combined_grads
