from abc import ABC, abstractmethod
from .base_run import BaseRun

class TestBaseRun(BaseRun):

    @abstractmethod
    def test(self, data, current_iter):
        """ Runs at every iteration. """

    def save_img(self, img):
        if self.options.general.no_save_image:
            return
        self.logger.save_image_tensor(img)

    @abstractmethod
    def end_test(self):
        """ Runs once at the end of the whole test. """
