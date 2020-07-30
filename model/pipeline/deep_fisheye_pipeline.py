import torch
import torch.nn as nn

from model.pipeline.base_pipeline import BasePipeline
from model.pix2depth_model import Pix2DepthModel
from model.hpe_model import HPEModel
from util.projector import FisheyeProjector

class DeepFisheyePipeline(BasePipeline):
    def setup(self):

        self.model_names = ['pix2depth', 'hpe']
        self.pix2depth_model = Pix2DepthModel(self.options.pix2depth, self.gpu_ids)
        self.hpe_model = HPEModel(self.options.hpe, self.gpu_ids)

        self.img_size = self.options.general.img_size
        projector = FisheyeProjector(self.img_size)

        self.hpe_model.setup(make_optimizer = False)
        self.pix2depth_model.setup(make_optimizer = False)

        # removes very small depths to create a correct hand segmentation mask.
        self.depth_threshold = nn.Threshold(threshold = self.options.general.min_depth_thrs, value = 0, inplace=True)

        self.is_train = self._is_all_model_train_mode()
        if self.is_train:
            pix2depth_opt = self.options.pix2depth
            hpe_opt = self.options.hpe

            self.pix2depth_optimizer = torch.optim.Adam(self.pix2depth_model.get_net_parameters(), lr=pix2depth_opt.lr, betas = (pix2depth_opt.beta1, 0.999))
            self.hpe_optimizer = torch.optim.Adam(self._get_hpe_parameters(), lr=hpe_opt.lr)

        self.mode = self.pix2depth_model.mode
        self.is_setup = True

        self.check_and_load_pretrained()

    def _is_all_model_train_mode(self):
        p2d_train = self.pix2depth_model.mode.is_train()
        hpe_train = self.hpe_model.mode.is_train()
        return p2d_train
        #return p2d_train and hpe_train

    def _get_hpe_parameters(self):
        return self.hpe_model.get_net_parameters()

    def set_input(self, data):
        assert self.is_setup

        self.pix = self.send_tensor_to_device(data['fish'])

        self.segment = None
        self.fish_depth = None
        self.joint = None

        if 'joint' in data:
            self.joint = self.send_tensor_to_device(data['joint'])

        if 'fish_depth' in data:
            self.fish_depth = self.send_tensor_to_device(data['fish_depth'])

    def forward(self):
        assert self.is_setup

        pix2depth_results = self._forward_pix2depth(self.pix, self.fish_depth)
        self.fake_fish_depth = pix2depth_results['fake_depth']
        self.fake_fish_depth = self.depth_threshold(self.fake_fish_depth)
        self.hand_pix = self._segment_hand(self.pix, self.fake_fish_depth)

        self._forward_hpe(self.hand_pix, self.fake_fish_depth, self.joint)

    def _forward_pix2depth(self, pix, depth):
        pix2depth_input = {"pix" : pix, 'depth': depth, 'joint': self.joint}
        self.pix2depth_model.set_input(pix2depth_input)
        self.pix2depth_model.forward()
        return self.pix2depth_model.get_current_results()

    def _segment_hand(self, pix, depth):
        segment = self._depth_to_segment(depth)
        pix = pix * segment
        return pix

    def _depth_to_segment(self, depth):
        """ Make segment from depth with maintaining autograds """
        segment = depth.clone()
        segment[segment > 0] = 1
        return segment

    def _forward_hpe(self, pix, depth, joint):
        img = self._combine_pix_and_depth(pix, depth)
        if joint is None:
            hpe_input = {'img': img}
        else:
            hpe_input = {'joint' : joint, 'img' : img}
        self.hpe_model.set_input(hpe_input)
        self.hpe_model.forward()

    def _combine_pix_and_depth(self, pix, depth):
        return torch.cat((pix, depth), 1)

    def optimize_parameters(self):
        self.pix2depth_optimizer.zero_grad()
        self.hpe_optimizer.zero_grad()

        self.pix2depth_model.update_loss()
        p2d_loss = self.pix2depth_model.get_total_loss()

        self.hpe_model.update_losses()
        hpe_loss = self.hpe_model.get_total_loss()

        loss = hpe_loss + p2d_loss

        loss.backward()

        self.pix2depth_optimizer.step()
        self.hpe_optimizer.step()

    def _should_optimize_gan_discreminators(self):
        return (self.fish_depth is not None)


    def get_current_results(self):
        results = {}

        results['input'] = self.pix
        results['hand_pix'] = self.hand_pix
        results['fake_fish_depth'] = self.fake_fish_depth

        hpe_results = self.hpe_model.get_current_results()
        results['heatmap'] = hpe_results['heatmap']
        results['joint'] = hpe_results['joint']
        if 'heatmap_true' in hpe_results:
            results['heatmap_true'] = hpe_results['heatmap_true']

        if 'heatmap_reprojected' in hpe_results:
            results['heatmap_reprojected'] = hpe_results['heatmap_reprojected']

        return results
