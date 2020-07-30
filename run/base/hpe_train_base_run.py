from abc import abstractmethod
from .train_base_run import TrainBaseRun

from util.io import LossLog
from util import Timer

from util.image import *
import torch

import run.base.hpe_base_util as hpe_util
from dataset.data_model import HandDataModel

class HPETrainBaseRun(TrainBaseRun):

    def setup(self):
        super().setup()

        self.img_size = self.options.hpe.img_size
        self.speed_diagnose = self.options.general.speed_diagnose

        self.model = self.make_model()
        self.heatmap_max = 1

        self.last_results = None
        self.timer = Timer()

    @abstractmethod
    def make_model(self):
        pass

    def iterate(self, data):
        if self.speed_diagnose:
            self.timer.start('preprocess')

        data = self.arrange_data(data)

        if self.speed_diagnose:
            self.timer.stop('preprocess')
            self.timer.start('setting input')

        self.model.set_input(data)

        if self.speed_diagnose:
            self.timer.stop('setting input')
            self.timer.start('optimize')

        self.model.optimize()
        if self.speed_diagnose:
            self.timer.stop('optimize')
            self.timer.print_elapsed_times()

        self.avg_dict.add(self.model.get_current_losses())

        # save the result for visualization
        self.last_results = self.model.get_detached_current_results()
        self.last_data = data

    def save_checkpoint(self, epoch):
        checkpoint = self.model.pack_as_checkpoint()
        self.logger.save_checkpoint(checkpoint, epoch)

    def end_epoch(self):
        pass

    @abstractmethod
    def arrange_data(self, data):
        """ reshape the data for the model. """

    def _visualize_results_as_image(self, results, cur_iter):

        if results is None:
            return

        results = self._select_first_in_batch(results)
        img = results['img']
        joint_out, heatmap_out, heatmap_true, heatmap_reprojected = hpe_util.unpack_data(results)

        out_heatmap_img = convert_to_colormap(heatmap_out, 1.0)
        true_heatmap_img = convert_to_colormap(heatmap_true, 1.0)
        reprojected_heatmap_img = convert_to_colormap(heatmap_reprojected, 1.0)
        img = expand_channel(img)

        stacked_img = torch.cat((img, out_heatmap_img, reprojected_heatmap_img, true_heatmap_img), 3) # horizontal_stack
        self.visualizer.add_image('train sample', stacked_img, cur_iter)

    def _visualize_network_grad(self, epoch, current_iter):
        grads = self.model.get_grads()
        for tag, val in grads.items():
            self.visualizer.add_histogram(tag, val, epoch)
