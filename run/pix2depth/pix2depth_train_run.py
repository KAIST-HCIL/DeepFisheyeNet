from run.base.train_base_run import TrainBaseRun
import run.pix2depth.helper as helper
from util.io import LossLog
from util import AverageDict
from util.image import convert_to_colormap
from util.projector import FisheyeProjector
from model.pix2depth_model import Pix2DepthModel

import torch

class Pix2DepthTrainRun(TrainBaseRun):

    def setup(self):
        super().setup()

        opt = self.options.pix2depth
        self.img_size = opt.img_size
        self.model = Pix2DepthModel(opt, self.gpu_ids)

        projector = self.get_projector()
        self.model.setup(projector)

        self.last_results = None

    def get_projector(self):
        return FisheyeProjector(self.img_size)

    def iterate(self, data):
        data = helper.arrange_data(data)
        self.model.set_input(data)
        self.model.optimize()
        self.avg_dict.add(self.model.get_current_losses())

        # save the result for visualization
        self.last_results = self.model.get_detached_current_results()

    def log_and_visualize_iteration(self, epoch, current_iter):
        self._log_and_vis_scalar(epoch, current_iter)
        self._visualize_results_as_image(self.last_results, current_iter)

    def _visualize_results_as_image(self, results, cur_iter):
        if results is None:
            return

        results = self._select_first_in_batch(results)
        pix = results['pix']
        fake_depth = convert_to_colormap(results['fake_depth'], 1.0)
        if 'depth' in results:
            depth = convert_to_colormap(results['depth'], 1.0)
        else:
            depth = torch.zeros(fake_depth.shape)

        if 'heatmap_interms' in results:
            last_heatmap = convert_to_colormap(results['heatmap_interms'], 1.0)
        else:
            last_heatmap = torch.zeros(fake_depth.shape)

        img = torch.cat((pix, last_heatmap, fake_depth, depth), 3) # stack horizontally

        self.visualizer.add_image('train sample', img, cur_iter)

    def save_checkpoint(self, epoch):
        checkpoint = self.model.pack_as_checkpoint()
        self.logger.save_checkpoint(checkpoint, epoch)

    def end_epoch(self):
        pass

    def _visualize_network_grad(self, epoch, current_iter):
        grads = self.model.get_grads()
        for tag, val in grads.items():
            self.visualizer.add_histogram(tag, val, epoch)
