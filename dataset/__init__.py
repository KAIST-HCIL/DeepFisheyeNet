import importlib
import inspect
from dataset.base_dataset import BaseDataset
from torch.utils.data import DataLoader

from util.package import find_class_using_name

def find_dataset_by_name(dataset_name):
    dataset_cls = find_class_using_name('dataset', dataset_name, 'dataset')
    if inspect.isclass(dataset_cls) and issubclass(dataset_cls, BaseDataset):
        return dataset_cls

    raise Exception("{} is not correctely implemented as BaseRun class".format(dataset_name))

def create_train_dataset(opt):
    dataset_cls = find_dataset_by_name(opt.dataset)
    return dataset_cls(opt, True)

def create_test_dataset(opt):
    dataset_cls = find_dataset_by_name(opt.dataset)
    return dataset_cls(opt, False)

def create_dataloader(dataset, batch_size, num_workers, shuffle):
    kwargs = {'num_workers':num_workers, 'pin_memory':True}
    dataloader = DataLoader(\
            dataset,\
            batch_size = batch_size, shuffle = shuffle, **kwargs)
    return dataloader
