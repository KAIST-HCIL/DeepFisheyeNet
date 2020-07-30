import torch
import numpy as np

from run.base.test_base_run import TestBaseRun
import run.pipeline.helper as helper
from model.pipeline.deep_fisheye_pipeline import DeepFisheyePipeline
from util import StatDict, cal_L1_diff, cal_RMS_diff
from util.image import convert_to_colormap

class PipelineTestRun(TestBaseRun):
    def setup(self):
        super().setup()

        self.model = DeepFisheyePipeline(self.options)
        self.model.setup()

        self.stat_dict = StatDict()

        self.last_results = None

        self.euclidean_errors = []
        self.pck_thresholds = np.linspace(0, 0.5, 11)

    def test(self, data, current_iter):

        self.model.set_input(data)
        self.model.forward()

        results = self.model.get_detached_current_results()

        joint_true = data['joint'].detach().cpu()
        pix, hand_pix, fake_fish_depth, heatmap, heatmap_true, heatmap_reprojected, joint = helper.unpack_data(results)
        fake_fish_depth_img = convert_to_colormap(fake_fish_depth)
        out_heatmap_img = convert_to_colormap(heatmap)
        true_heatmap_img = convert_to_colormap(heatmap_true)

        stacked_img = torch.cat((pix, fake_fish_depth_img, out_heatmap_img, true_heatmap_img), 3)
        stacked_img = stacked_img.squeeze()
        self.save_img(stacked_img)

        losses = {}
        losses['heatmap L1'] = cal_L1_diff(heatmap, heatmap_true, reduction = 'mean')
        losses['heatmap RMS'] = cal_RMS_diff(heatmap, heatmap_true, reduction = 'mean')
        losses['joint L1'] = cal_L1_diff(joint, joint_true, reduction = 'mean')
        losses['joint RMS'] = cal_RMS_diff(joint, joint_true, reduction = 'mean')

        self.stat_dict.add(losses)

        euclidean_error = self._cal_eucliean_error(joint, joint_true)
        self.euclidean_errors.append(euclidean_error)

    def end_test(self):
        result = {}
        result['avg'] = self.stat_dict.get_avg()
        result['std'] = self.stat_dict.get_std()

        print(result)

        self.logger.write_loss(result)
        self.euclidean_errors = np.array(self.euclidean_errors)
        pck_results = self._cal_pck(self.euclidean_errors, self.pck_thresholds)
        self.logger.write_pck(pck_results)

    def _cal_eucliean_error(self, joint1, joint2):
        diff = joint1 - joint2

        diff = diff.numpy()
        diff = diff.reshape((-1, 3))
        error = np.linalg.norm(diff, axis = 1)

        return error

    def _cal_pck(self, error, thresholds):
        results = []
        for thrs in thresholds:
            acc = np.mean(error < thrs)
            results.append((thrs, acc))

        return results
