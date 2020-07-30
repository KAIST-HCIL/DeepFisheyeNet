import torch
import torch.nn as nn

from model.base_model import BaseModel
import network.hand_depth_net as depth_net
from network.projection import create_projection_net
import util.image as image
from util.joint import JointConverter

class Pix2DepthModel(BaseModel):
    def setup(self, make_optimizer = True):
        opt = self.opt
        self.make_optimizer = make_optimizer

        self.loss_names = []

        self.network = depth_net.create_hdg_net(opt, self.gpu_ids)
        self.joint_converter = JointConverter(opt.num_joints)

        self.projection_net = create_projection_net(opt, self.gpu_ids)

        if self.mode.is_train():

            self.criterionL2 = nn.MSELoss()
            self.heatmap_loss_weight = opt.heatmap_loss_weight
            self.heatmap_interm_loss_weight = opt.heatmap_interm_loss_weight
            self.joint_loss_weight = opt.joint_loss_weight
            self.joint_interm_loss_weight = opt.joint_interm_loss_weight
            self.depth_loss_weight = opt.depth_loss_weight

            if make_optimizer:
                # initialize optimizers
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer)

        else:
            self.network.eval()
            print("Pix2Depth network set to eval mode.")

        self.check_and_load_pretrained()

        self.run_interm = not self.mode.is_eval()

        self.is_setup = True

    def set_input(self, data):

        pix = data['pix']
        self.real_pix = pix.to(self.device)
        self.real_pix = image.normalize_img(self.real_pix)

        if 'depth' in data and (data['depth'] is not None):
            self.real_depth = data['depth'].to(self.device)
            self.real_depth = image.normalize_img(self.real_depth)
        else:
            self.real_depth = None

        self.joint_true = None
        self.heatmap_true = None

        if self.run_interm:
            joint_true = self.send_tensor_to_device(data['joint'])

            normalized_joint = self.joint_converter.normalize(joint_true)
            self.heatmap_true = self.projection_net(normalized_joint)
            self.heatmap_true.requires_grad = False

            self.joint_true = self.joint_converter.convert_for_training(joint_true)

    def forward(self):
        assert self.is_setup
        result = self.network(self.real_pix)
        self.fake_depth = result['fake']
        if self.run_interm:
            self.fake_interms = result['interms']
            self.joint_interms = result['joint_interms']
            self.heatmap_interms = result['heatmap_interms']

    def optimize_parameters(self):
        self.optimizer.zero_grad()

        self.update_loss()
        self.loss_total.backward()

        self.optimizer.step()

    def update_loss(self):
        self.update_depth_loss()
        self.update_depth_interm_loss()
        self.update_joint_losses()
        self.update_heatmap_losses()
        self.loss_total = self.get_total_loss()

    def get_total_loss(self):
        total_G_loss = self.loss_depth + self.loss_depth_interm + self.loss_joint + self.loss_heatmap
        return total_G_loss

    def update_depth_loss(self):
        self.loss_depth = 0
        if self.real_depth is not None:
            self.loss_depth = self.criterionL2(self.fake_depth, self.real_depth) * self.depth_loss_weight
            self.add_loss_name('depth')

    def update_depth_interm_loss(self):
        self.loss_depth_interm = 0

        if self.real_depth is None:
            return

        for i, interm in enumerate(self.fake_interms):
            loss = self.criterionL2(interm, self.real_depth) * self.depth_loss_weight * 0.5
            loss_name = "depth_interm_{}".format(i)
            setattr(self, "loss_"+loss_name, loss)
            self.add_loss_name(loss_name)

            self.loss_depth_interm += loss

    def update_joint_losses(self):
        self.loss_joint = 0

        for i, interm in enumerate(self.joint_interms):
            interm_loss = self.criterionL2(interm, self.joint_true) * self.joint_loss_weight
            loss_name = "joint_interm_{}".format(i)
            setattr(self, 'loss_'+loss_name, interm_loss)
            self.add_loss_name(loss_name)

            self.loss_joint += interm_loss

    def update_heatmap_losses(self):
        self.loss_heatmap = 0

        for i, interm in enumerate(self.heatmap_interms):
            interm_loss = self.criterionL2(interm, self.heatmap_true) * self.heatmap_loss_weight
            loss_name = "heatmap_interm{}".format(i)
            setattr(self, 'loss_'+loss_name, interm_loss)
            self.add_loss_name(loss_name)

            self.loss_heatmap += interm_loss

    def add_loss_name(self, loss_name):
        if loss_name not in self.loss_names:
            self.loss_names.append(loss_name)

    def pack_as_checkpoint(self):
        checkpoint = {}
        checkpoint['network'] = self.extract_weights(self.network)

        if self.make_optimizer:
            checkpoint['optim'] = self.optimizer.state_dict()

        return checkpoint

    def add_loss_name(self, loss_name):
        if loss_name not in self.loss_names:
            self.loss_names.append(loss_name)

    def load_from_checkpoint(self, checkpoint, model_only):
        self.apply_weights(self.network, checkpoint['network'])

        if not model_only:
            self.optimizer.load_state_dict(checkpoint['optim'])

    def get_fake(self):
        return self.fake_depth

    def get_current_results(self):
        pix = image.unnormalize_as_img(self.real_pix)
        fake_depth = image.unnormalize_as_img(self.fake_depth)
        results = {'pix': pix, 'fake_depth': fake_depth}

        if not self.mode.is_eval():
            depth = image.unnormalize_as_img(self.real_depth)
            results['depth'] = depth
            results['heatmap_interms'] = self.heatmap_interms

        return results

    def get_grads(self):
        grads = {}
        for tag, param in self.network.named_parameters():
            tag = "{}".format(tag)
            grads[tag] = param.grad.data.detach().cpu()
        return grads

    def get_net_parameters(self):
        return self.network.parameters()
