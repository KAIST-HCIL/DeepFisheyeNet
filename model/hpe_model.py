import torch
import torch.nn as nn

from model.base_model import BaseModel
from network.hand_pose_net import create_hpe_net
from network.projection import create_projection_net
import util.image as image
from util.joint import JointConverter


class HPEModel(BaseModel):
    def setup(self, make_optimizer = True):
        opt = self.opt

        self.make_optimizer = make_optimizer
        self.joint_converter = JointConverter(opt.num_joints)

        self.loss_names = [ 'heatmap', \
                            'joint']

        self.net = create_hpe_net(opt, self.gpu_ids)

        self.criterionL2 = nn.MSELoss()

        self.run_interm = self.mode.is_train() or self.mode.is_test()
        if self.run_interm:
            self.set_requires_grad(self.net, True)

            self.projection_net = create_projection_net(opt, self.gpu_ids)

            self.heatmap_loss_weight = opt.heatmap_loss_weight
            self.heatmap_interm_loss_weight = opt.heatmap_interm_loss_weight
            self.joint_loss_weight = opt.joint_loss_weight
            self.joint_interm_loss_weight = opt.joint_interm_loss_weight

            if make_optimizer:
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr)
                self.optimizers.append(self.optimizer)

        else:
            self.joint_converter.joint_scale = opt.joint_scale

        self.check_and_load_pretrained()

        self.is_setup = True

    def set_input(self, data):
        self.input = self.send_tensor_to_device(data['img'])
        self.input = image.normalize_img(self.input)

        self.joint_true = None
        self.heatmap_seed = None
        self.heatmap_true = None
        if self.run_interm:
            joint_true = self.send_tensor_to_device(data['joint'])
            normalized_joint = self.joint_converter.normalize(joint_true)
            self.heatmap_true = self.projection_net(normalized_joint)
            self.heatmap_true.requires_grad = False
            self.joint_true = self.joint_converter.convert_for_training(joint_true)

    def forward(self):
        assert self.is_setup

        result = self.net(self.input)
        self.joint_out = result['joint']
        self.heatmap_out = None
        self.heatmap_interms = []
        self.heatmap_out = result['heatmap']
        self.reprojected_heatmap = None

        if self.run_interm:
            self.heatmap_interms = result['heatmap_interms']
            self.joint_interms = result['joint_interms']

            unflat_joint_out = self.joint_converter.convert_for_output(self.joint_out, no_unnormalize = True)
            self.reprojected_heatmap = self.projection_net(unflat_joint_out)

    def optimize_parameters(self):
        self.update_losses()
        total_loss = self.get_total_loss()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def update_losses(self):
        self._update_heatmap_losses()
        self._update_joint_losses()

    def _update_heatmap_losses(self):
        self.loss_heatmap = 0
        for i, interm in enumerate(self.heatmap_interms):
            interm_loss = self.criterionL2(self.heatmap_true, interm) * self.heatmap_interm_loss_weight
            loss_name = 'heatmap_interm{}'.format(i+1)
            setattr(self, 'loss_' + loss_name, interm_loss)
            self._add_loss_name_if_not_exists(loss_name)

        self.loss_heatmap = self.criterionL2(self.heatmap_true, self.heatmap_out) * self.heatmap_loss_weight

    def _update_joint_losses(self):
        self.loss_joint = 0
        for i, interm in enumerate(self.joint_interms):
            interm_loss = self.criterionL2(self.joint_true, interm) * self.joint_interm_loss_weight
            loss_name = 'joint_interm{}'.format(i+1)
            setattr(self, "loss_"+loss_name, interm_loss)
            self._add_loss_name_if_not_exists(loss_name)

        self.loss_joint = self.criterionL2(self.joint_true, self.joint_out) * self.joint_loss_weight

    def _add_loss_name_if_not_exists(self, loss_name):
        if loss_name not in self.loss_names:
            self.loss_names.append(loss_name)

    def get_total_loss(self):
        heatmap_loss = self._sum_heatmap_losses()
        joint_loss = self._sum_joint_losses()

        total_loss = heatmap_loss + joint_loss
        return total_loss

    def _sum_heatmap_losses(self):
        return self._sum_losses_by_part_of_name('heatmap')

    def _sum_joint_losses(self):
        return self._sum_losses_by_part_of_name('joint')

    def _sum_losses_by_part_of_name(self, part_of_name):
        losses = []

        for loss_name in self.loss_names:
            if loss_name.startswith(part_of_name):
                losses.append(getattr(self, "loss_" + loss_name))

        return sum(losses)

    def pack_as_checkpoint(self):
        checkpoint = {}
        checkpoint['net'] = self.extract_weights(self.net)
        if self.make_optimizer:
            checkpoint['optim'] = self.optimizer.state_dict()

        return checkpoint

    def load_from_checkpoint(self, checkpoint, model_only):
        self.apply_weights(self.net, checkpoint['net'])
        if not model_only:
            self.optimizer.load_state_dict(checkpoint['optim'])

    def get_current_results(self):
        img = image.unnormalize_as_img(self.input)
        joint_out = self.joint_converter.convert_for_output(self.joint_out)
        heatmap_interms = self.heatmap_interms
        heatmap_out = self.heatmap_out
        heatmap_true = self.heatmap_true
        reprojected_heatmap = self.reprojected_heatmap

        if self.run_interm:
            return {'img': img, \
                    'joint': joint_out, \
                    'heatmap': heatmap_out, \
                    'heatmap_true': heatmap_true, \
                    'heatmap_reprojected': reprojected_heatmap, \
                    'heatmap_interms': heatmap_interms}
        else:
            return {'img': img, \
                    'joint': joint_out, \
                    'heatmap': heatmap_out}

    def get_net_parameters(self):
        return self.net.parameters()

    def get_name_parameters(self):
        return self.net.named_parameters()

    def get_grads(self):
        grads = {}
        for tag, param in self.net.named_parameters():
            grads[tag] = param.grad.data.detach().cpu()

        return grads
