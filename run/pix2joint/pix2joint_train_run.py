import torch

from model.pipeline import find_pipeline_using_name
from util.io import LossLog
from model.hpe_model import HPEModel
from run.base.hpe_train_base_run import HPETrainBaseRun

class Pix2JointTrainRun(HPETrainBaseRun):

    def make_model(self):
        opt = self.options.hpe
        model = HPEModel(opt, self.gpu_ids)
        model.setup()
        return model

    def arrange_data(self, data):
        fish_data = data['fish']
        _data = {'img': fish_data, \
                'joint': data['joint']}
        return _data
