import os
import pathlib
import torch
import torchvision

from datetime import datetime

from .image import blend_to_image

def save_image_hot(filename, tensor):
    img = torchvision.transforms.ToPILImage()(tensor)
    img.save(str(filename))

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint

class Logger:
    """
        Create filenames for saving the results.
    """
    def __init__(self, opt):
        self.opt = opt
        self.create_paths(opt)
        self.loss_file = self.get_loss_file()
        self.pck_file = self.get_pck_file()
        self.cnt = 0

    def create_paths(self, opt):
        this_file = pathlib.Path(os.path.abspath(__file__))
        proj_root = this_file.parents[1]
        results_root = proj_root.joinpath('results')
        results_root.mkdir(exist_ok = True)

        result_path = results_root.joinpath(opt.name)

        dup_cnt = 0
        while result_path.exists():
            dup_cnt += 1
            new_name = "{}{}".format(opt.name, dup_cnt)
            result_path = results_root.joinpath(new_name)

        result_path.mkdir(exist_ok = True)
        self.result_path = result_path

    def get_heatmap_path(self):

        heatmap_path = self.result_path.joinpath('heatmaps')
        heatmap_path.mkdir(exist_ok = True)
        return heatmap_path

    def get_image_path(self):
        image_path = self.result_path.joinpath('images')
        image_path.mkdir(exist_ok = True)
        return image_path

    def get_tensorboard_path(self):
        dir_name = self.result_path.parts[-1]
        return self.result_path.parents[1].joinpath(self.opt.tensorboard_dir, dir_name)

    def get_checkpoint_path(self, epoch):
        checkpoint_path = self.result_path.joinpath('checkpoints')
        checkpoint_path.mkdir(exist_ok = True)
        checkpoint_path = checkpoint_path.joinpath("{}.pth.tar".format(epoch))

        return checkpoint_path

    def get_loss_file(self):
        loss_path = self.result_path.joinpath('losses.txt')
        return open(str(loss_path), 'w')

    def get_pck_file(self):
        pck_path = self.result_path.joinpath('pck.txt')
        return open(str(pck_path), 'w')

    def save_image_tensor(self, tensor):
        img = torchvision.transforms.ToPILImage()(tensor)
        img_path = self.get_image_path()
        img_file_path = img_path.joinpath('{}.png'.format(self.cnt))
        self.cnt += 1
        img.save(str(img_file_path))
        pass

    def save_heatmap(self, img, heatmap):
        blended_img = blend_to_image(img, heatmap)
        heatmap_path = self.get_heatmap_path()
        img_path = heatmap_path.joinpath('{}.png'.format(self.cnt))
        self.cnt += 1

        blended_img.save(str(img_path))

    def save_checkpoint(self, checkpoint, epoch):
        checkpoint['epoch'] = epoch
        cp_path = self.get_checkpoint_path(epoch)
        torch.save(checkpoint, str(cp_path))

    def save_options(self, options):
        options_path = self.result_path.joinpath('options.txt')
        options_path.write_text(options.pretty_str())

    def write_loss(self, loss):
        loss_str = "{}\n".format(str(loss))
        self.loss_file.write(loss_str)

    def write_pck(self, pck_results):
        for thrs, acc in pck_results:
            line = "{}, {}\n".format(thrs, acc)
            self.pck_file.write(line)

    def close(self):
        self.loss_file.close()
        self.pck_file.close()

class LossLog:
    def __init__(self, loss_dict, epoch, total_iter, tag):
        self.loss_dict = self._copy(loss_dict)

        self.loss_dict['timestamp'] = str(datetime.now())
        self.loss_dict['epoch'] = str(epoch)
        self.loss_dict['iter'] = str(total_iter)
        self.loss_dict['tag'] = tag

    def _copy(self, loss_dict):
        new_dict = {}
        for k, v in loss_dict.items():
            new_dict[k] = v

        return new_dict

    def __str__(self):
        log_str_list = []
        for k, v in self.loss_dict.items():
            log_str_list.append("{}:{}".format(k, v))

        log_str = ','.join(log_str_list)
        return log_str
