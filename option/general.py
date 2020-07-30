from .base import BaseOption, Argument

class GeneralOption(BaseOption):

    def set_arguments(self):

        self._set_basic_params()
        self._set_dataset_params()
        self._set_pipeline_params()
        self._set_train_params()
        self._set_train_visual_params()
        self._set_test_params()

    def _set_basic_params(self):
        self.arguments += [Argument('name', type=str, default='sample', help='name of the run. model and samples will be stored.')]
        self.arguments += [Argument('gpu_ids', type=str, default='0', help="gpu ids to use. -1 for cpu. (e.g. 0  0,1,2).")]
        self.arguments += [Argument('num_workers', type=int, default=0, help="number of data loader workers.")]
        self.arguments += [Argument('max_data', type=int, default = float("inf"), help="number of data to use.")]
        self.arguments += [Argument('preset', type=str, default = '', help="name of preset. you should write Preset(*this_part*) of preset classes in presets.py")]
        self.arguments += [Argument('run', type=str, default = '', help="name of the train or test code to run. please see 'run' module")]

    def _set_dataset_params(self):
        self.arguments += [Argument('dataset', type=str, default='', help="dataset to use.")]
        self.arguments += [Argument('img_size', type=int, default=256, help="the size of images (defaul: 256). all images should be square images")]
        self.arguments += [Argument('no_flip', action='store_true', help='does not flip during a run')]
        self.arguments += [Argument('min_depth_thrs', type=float, default=0.01171875, help=" too small depth to consider (default: 3 / 256)")]

    def _set_pipeline_params(self):
        self.arguments += [Argument('pipeline_pretrained', type=str, default='', help="pretrained_weight for the pipeline")]

    def _set_train_params(self):
        self.arguments += [Argument('batch_size', type=int, default=1, help='batch size.')]
        self.arguments += [Argument('epoch', type=int, default=100, help='number of epochs to train.')]
        self.arguments += [Argument('speed_diagnose', action='store_true', default=False, help='measure time for important steps.')]

        # save
        self.arguments += [Argument('save_epoch', type=int, default=5, help='epoch period to save checkpoints')]

    def _set_train_visual_params(self):
        self.arguments += [Argument('print_iter', type=int, default=100, help='iteration period to print to tensorboard and terminal')]
        self.arguments += [Argument('tensorboard_dir', type=str, default='tensorboard_logs', help='where the tensorboard logs stays')]
        self.arguments += [Argument('show_grad', action='store_true', default = False, help='show gradient histogram while training')]

    def _set_test_params(self):
        self.arguments += [Argument('no_save_image', action='store_true', default=False, help='do not save result images for test')]
