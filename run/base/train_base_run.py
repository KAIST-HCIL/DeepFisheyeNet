from abc import ABC, abstractmethod
from .base_run import BaseRun

from util import AverageDict
from util.io import LossLog

class TrainBaseRun(BaseRun):

    def setup(self):
        super().setup()

        self.avg_dict = AverageDict()
        # it is for intermediate visualizations
        self.last_results = None
        self.last_data = None
        self.show_grad = self.options.general.show_grad

    @abstractmethod
    def iterate(self, data):
        """ Runs at every iteration. """
        pass

    @abstractmethod
    def end_epoch(self):
        """ Runs at every end of epoch. """
        pass

    @abstractmethod
    def save_checkpoint(self, epoch):
        pass

    def log_and_visualize_iteration(self, epoch, current_iter):
        self._log_and_vis_scalar(epoch, current_iter)
        self._visualize_results_as_image(self.last_results, current_iter)
        if self.show_grad:
            self._visualize_network_grad(epoch, current_iter)

    def _log_and_vis_scalar(self, epoch, current_iter):
        losses = self.avg_dict.to_dict()
        self.avg_dict.reset()
        loss_log = LossLog(losses, epoch, current_iter, 'train')
        self.logger.write_loss(loss_log)
        self.visualizer.add_losses('train', losses, current_iter)

    @abstractmethod
    def _visualize_results_as_image(self, results, current_iter):
        """ This can be different by a subclasse's purpose. """
        pass

    @abstractmethod
    def _visualize_network_grad(self, epoch, current_iter):
        pass

    def _select_first_in_batch(self, results):
        first_results = {}

        for k, v in results.items():
            first_results[k] = v[0].unsqueeze(0)
            if 'interm' in k:
                # interm results are in list. select the last one.
                first_results[k] = v[-1][0].unsqueeze(0)

        return first_results
