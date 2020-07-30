from abc import ABC, abstractmethod
import pathlib
import random
import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms

class BaseDataset(ABC, Dataset):
    def __init__(self, opt, is_train):
        self.is_train = is_train
        self.img_size = opt.img_size
        self.no_flip = opt.no_flip

        transform_list = []
        transform_list += [transforms.Resize((self.img_size, self.img_size))]
        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)
        self.color_transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.15)
        self.hand_list = None
        self.set_hand_list()
        if not opt.max_data == float("inf"):
            self.hand_list = self.hand_list[:opt.max_data]

    @abstractmethod
    def set_hand_list(self):
        """ This method should initialize self.hand_list . """
        pass

    def _load_filenames(self, root_name):
        this_file = pathlib.Path(os.path.abspath(__file__))
        this_dir = this_file.parents[0]
        if self.is_train:
            filename_txt = this_dir.joinpath(root_name,'train.txt')
        else:
            filename_txt = this_dir.joinpath(root_name, 'test.txt')

        filenames = []
        with open(str(filename_txt), 'r') as f:
            lines = f.readlines()
            for line in lines:
                names = line.strip().split(',')
                filenames.append(names)
        return filenames

    def __len__(self):
        return len(self.hand_list)

    @abstractmethod
    def __getitem__(self, index):
        pass

    def toss_coin(self):
        return random.random() > 0.5
