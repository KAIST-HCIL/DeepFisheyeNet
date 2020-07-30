import numpy as np
from abc import abstractmethod
import torch

from run.base.test_base_run import TestBaseRun
import run.base.hpe_base_util as hpe_util

from util.image import convert_to_colormap
from util import StatDict, cal_L1_diff, cal_RMS_diff

class HPETestBaseRun(TestBaseRun):

    def setup(self):

        # move some paramters from the options
        self.img_size = self.options.hpe.img_size

        self.model = self.make_model()

        self.heatmap_max = 1

        self.stat_dict = StatDict()
        self.no_save_image = self.options.general.no_save_image

        self.euclidean_errors = []
        self.pck_thresholds = np.linspace(0, 0.5, 11)

    @abstractmethod
    def make_model(self):
        pass

    def end_test(self):
        result = {}
        result['avg'] = self.stat_dict.get_avg()
        result['std'] = self.stat_dict.get_std()

        print(result)

        self.logger.write_loss(result)
        pck_results = self._cal_pck(self.euclidean_errors, self.pck_thresholds)
        self.logger.write_pck(pck_results)

    def test(self, data, current_iter):
        data = self.arrange_data(data)

        self.model.set_input(data)
        self.model.forward()
        results = self.model.get_detached_current_results()

        joint = data['joint']
        joint_out, heatmap_out, heatmap_true, heatmap_reprojected = hpe_util.unpack_data(results, self.model.mode.is_eval())

        if (not self.no_save_image) and (heatmap_out is not None):
            img = results['img']
            if img.size(1) == 1:
                img = convert_to_colormap(img, 1)
            out_heatmap_img = convert_to_colormap(heatmap_out, self.heatmap_max)
            true_heatmap_img = convert_to_colormap(heatmap_true, self.heatmap_max)
            reprojected_heatmap_img = convert_to_colormap(heatmap_reprojected, self.heatmap_max)
            stacked_img = torch.cat((img, out_heatmap_img, reprojected_heatmap_img, true_heatmap_img), 3) # horizontal_stack
            stacked_img = stacked_img.squeeze()
            self.save_img(stacked_img)

        losses = {}
        losses['Joint L1'] = cal_L1_diff(joint_out, joint, reduction = 'mean')
        losses['Joint RMS'] = cal_RMS_diff(joint_out, joint, reduction = 'mean')

        if not self.model.mode.is_eval():
            losses['Heatmap L1'] = cal_L1_diff(heatmap_out, heatmap_true, reduction = 'mean')
            losses['Heatmap RMS'] = cal_RMS_diff(heatmap_out, heatmap_true, reduction = 'mean')

        self.stat_dict.add(losses)

        euclidean_error = self._cal_eucliean_error(joint_out, joint)
        self.euclidean_errors.append(euclidean_error)

    @abstractmethod
    def arrange_data(self, data):
        pass

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
