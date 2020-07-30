from abc import ABC, abstractmethod
from util.io import *
from util.visualizer import Visualizer
from dataset import *

import torch
import torch.nn as nn

class BaseRun(ABC):
    """ This is basic abastract Run class.
        The run class contains core algorthm of train or test running.
    """

    def __init__(self, options):
        self.options = options
        self.logger = self._get_logger(options)
        self.visualizer = self._get_visualizer(self.logger)

        self.gpu_ids = self.options.general.gpu_ids

    def _get_logger(self, options):
        logger = Logger(options.general)
        logger.save_options(options)
        return logger

    def _get_visualizer(self, logger):
        tensorboard_path = logger.get_tensorboard_path()
        return Visualizer(str(tensorboard_path))

    def get_train_loader(self):
        opt = self.options.general
        train_dataset = create_train_dataset(opt)
        print("BaseRun: total number of training data: {}".format(len(train_dataset)))
        train_loader = create_dataloader(train_dataset, batch_size = opt.batch_size, num_workers = opt.num_workers, shuffle=True)
        return train_loader

    def get_test_loader(self, shuffle = False):
        opt = self.options.general
        test_dataset = create_test_dataset(opt)

        print("BaseRun: total number of test data: {}".format(len(test_dataset)))
        test_loader = create_dataloader(test_dataset, batch_size = 1, num_workers = opt.num_workers, shuffle=shuffle)
        return test_loader

    @abstractmethod
    def setup(self):
        pass

    def toss_coin(self):
        return random.random() > 0.5
