import argparse
from .general import GeneralOption
from .hpe import HPEOption
from .pix2depth import Pix2DepthOption

class Options():
    """ This class holds all the options.
        You have to modify '_set_options' to add/remove options.
    """
    def __init__(self):
        self.initialized = False
        self._set_options()

    def _set_options(self):
        # Add or remove options in here.
        self.general = GeneralOption(name = 'general', prefix = None)
        self.hpe = HPEOption(name = 'hand_pose_estimator', prefix = 'hpe')
        self.pix2depth = Pix2DepthOption(name = 'pix2depth', prefix = "p2d")

        self.options = []
        self.options.append(self.general)
        self.options.append(self.hpe)
        self.options.append(self.pix2depth)

    def initialize(self):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self._init_parser(parser)
            self.initialized = True

        args = parser.parse_args()

        for opt in self.options:
            opt.initialize(args)

        assert self.general.img_size == self.hpe.img_size and self.general.img_size == self.pix2depth.img_size, "all image size should be same."

    def initialize_with_defaults(self):
        for option in self.options:
            option.initialize_with_defaults()

    def parse(self):
        self.general.gpu_ids = [int(i) for i in self.general.gpu_ids.split(',')]

        for opt in self.options:
            opt.convert_mode_str_to_mode()

    def _init_parser(self, parser):
        for option in self.options:
            option.add_to_parser(parser)

        return parser

    def pretty_str(self):
        message = ""
        message += "-------------- Options --------------\n"
        for opt in self.options:
            message += (str(opt) + "\n")
            message += "----------------------------------\n"

        return message

    def get_gpu_ids(self):
        return self.general.gpu_ids
