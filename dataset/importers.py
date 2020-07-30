from abc import ABC, abstractmethod
class BaseImporter(ABC):
    """ The main purpose of an Importer is to

        - split test and train dataset
        - unify dataset structure for different datasets.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_file_names(self, args):
        """ This method should generate text files that contains filenames of data in a dataset.
            Also, it should split train and test data by different directories.

            This should return two lists
            - filenames_for_train
            - filenames_for_test

            All the filenames in the lists should be absolute paths.
        """
        pass

    @abstractmethod
    def get_import_root(self):
        """ This method should return the root directory of where the data would be imported to. """
        pass

    def get_num_train_for_even_batch(self, num_total, test_ratio, batch_size):
        assert type(batch_size) == int, "batch_size should be int"

        train_ratio = 1 - test_ratio
        rough_num_train = num_total * train_ratio

        num_iter = int(rough_num_train / batch_size)

        num_train = int(num_iter * batch_size)
        return num_train

import pathlib
import random

class SynthImporter(BaseImporter):

    def get_file_names(self, args):
        data_root = pathlib.Path(args.data_dir)
        fish_root = data_root.joinpath('fish')
        heatmap_root = fish_root
        joints_root = data_root.joinpath('joints')
        fish_depth_root = data_root.joinpath('fish_depth')

        img_paths = fish_root.glob("**/*.png")

        total_filenames = []
        for img_path in img_paths:
            fish_depth_img_path = self.get_depth_img_path(fish_depth_root, img_path)
            joints_path = self.get_joints_path(joints_root, img_path)

            fns = (str(img_path), str(fish_depth_img_path), str(joints_path))
            total_filenames.append(fns)

        random.shuffle(total_filenames)

        chunk_size = 256
        num_train = self.get_num_train_for_even_batch(len(total_filenames), args.test_ratio, chunk_size)

        filenames_for_train = total_filenames[:num_train]
        filenames_for_test = total_filenames[num_train:]

        return filenames_for_train, filenames_for_test

    def get_import_root(self):
        return 'synth'

    def get_matching_path_in_subroot(self, subroot, img_path, fn_pattern):
        id = img_path.stem
        subdir = img_path.parts[-2]

        return subroot.joinpath(subdir, fn_pattern.format(id))
    def get_joints_path(self, joints_root, img_path):
        return self.get_matching_path_in_subroot(joints_root, img_path, '{}_joint_pos.txt')

    def get_depth_img_path(self, depth_root, img_path):
        return self.get_matching_path_in_subroot(depth_root, img_path, '{}.png')

class RealImporter(BaseImporter):
    def get_file_names(self, args):
        data_root = pathlib.Path(args.data_dir)

        img_paths = data_root.glob("**/*.png")
        total_filenames = []
        for img_path in img_paths:
            joints_path = self.get_joints_path(data_root, img_path)
            fns = (str(img_path), str(joints_path))
            total_filenames.append(fns)

        if args.test_ratio > 0:
            num_train = self.get_num_train_for_even_batch(len(total_filenames), args.test_ratio, 256)
            filenames_for_train = total_filenames[:num_train]
            filenames_for_test = total_filenames[num_train:]

            return filenames_for_train, filenames_for_test

        return total_filenames, []

    def get_import_root(self):
        return 'real'

    def get_joints_path(self, data_root, img_path):
        id = img_path.stem
        user = img_path.parts[-3]

        return data_root.joinpath(user, 'joints', '{}_leap.txt'.format(id))
