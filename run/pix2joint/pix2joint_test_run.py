from run.base.hpe_test_base_run import HPETestBaseRun
from model.hpe_model import HPEModel
from util.projector import FisheyeProjector

class Pix2JointTestRun(HPETestBaseRun):

    def make_model(self):
        opt = self.options.hpe
        model = HPEModel(opt, self.gpu_ids)
        model.setup()
        return model

    def arrange_data(self, data):
        _data = {'img': data['fish'], \
                'joint': data['joint']}
        return _data
