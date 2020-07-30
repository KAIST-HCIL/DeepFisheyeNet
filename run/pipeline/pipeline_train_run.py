import torch

from run.base.train_base_run import TrainBaseRun
import run.pipeline.helper as helper
from model.pipeline.deep_fisheye_pipeline import DeepFisheyePipeline
from util import AverageDict
from util.io import LossLog
from util.image import convert_to_colormap

class PipelineTrainRun(TrainBaseRun):
    def setup(self):
        super().setup()

        self.pipeline = DeepFisheyePipeline(self.options)
        self.pipeline.setup()

        self.avg_dict = AverageDict()

        self.last_results = None

    def iterate(self, data):
        self.pipeline.set_input(data)
        self.pipeline.optimize()
        self.avg_dict.add(self.pipeline.get_current_losses())

        # save the result for visualization
        self.last_results = self.pipeline.get_detached_current_results()

    def _visualize_results_as_image(self, results, cur_iter):
        if results is None:
            return

        results = self._select_first_in_batch(results)
        pix, hand_pix, fake_fish_depth, heatmap, heatmap_true, heatmap_reprojected, joint = helper.unpack_data(results)

        fake_fish_depth_img = convert_to_colormap(fake_fish_depth)
        out_heatmap_img = convert_to_colormap(heatmap)
        true_heatmap_img = convert_to_colormap(heatmap_true)
        reprojected_img = convert_to_colormap(heatmap_reprojected)

        stacked_img = torch.cat((pix, fake_fish_depth_img, hand_pix, out_heatmap_img, reprojected_img, true_heatmap_img), 3)
        self.visualizer.add_image('train sample', stacked_img, cur_iter)

    def _visualize_network_grad(self, epoch, current_iter):
        grads = self.pipeline.get_grads()
        for tag, val in grads.items():
            self.visualizer.add_histogram(tag, val, epoch)

    def save_checkpoint(self, epoch):
        checkpoint = self.pipeline.pack_as_checkpoint()
        self.logger.save_checkpoint(checkpoint, epoch)

    def end_epoch(self):
        pass
