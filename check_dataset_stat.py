from option.options import Options
from preset import modify_options
from dataset import *
from run import find_run_using_name
from run.empty_run import EmptyRun
import torch

def get_number_of_sum_and_dim(data, data_type):
    if data_type == 'joint':
        b, j, c = data.shape
        nb_data = b * j
        dim_to_reduce = [0, 1]
    else:
        b, c, h, w = data.shape
        nb_data = b * h * w
        dim_to_reduce = [0,2,3]

    return nb_data, dim_to_reduce

def online_stat(loader, data_type):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = None # mean
    snd_moment = None

    _min = float("inf")
    _max = -float("inf")

    for data_packet in loader:
        data = data_packet[data_type]
        nb_data, dim_to_reduce = get_number_of_sum_and_dim(data, data_type)
        sum_ = torch.sum(data, dim=dim_to_reduce)
        sum_of_square = torch.sum(data ** 2, dim=dim_to_reduce)

        if fst_moment is None:
            fst_moment = torch.zeros(sum_.shape)
            snd_moment = torch.zeros(sum_.shape)

        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_data)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_data)

        cnt += nb_data

        data_max = data.max()
        data_min = data.min()

        if data_max > _max:
            _max = data_max

        if data_min < _min:
            _min = data_min

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2), _max, _min

def main():

    dataset_type = 'joint'

    options = Options()
    options.initialize()
    modify_options(options)
    options.parse()
    options.general.dataset = 'synth'
    print(options.pretty_str())
    run = EmptyRun(options)

    train_loader = run.get_train_loader()
    train_mean, train_std, max, min = online_stat(train_loader, dataset_type)
    print("train mean:", train_mean)
    print("train std:", train_std)
    print("train max:", max)
    print("train min:", min)

    test_loader = run.get_test_loader(shuffle = True)
    test_mean, test_std, max, min = online_stat(test_loader, dataset_type)
    print("test mean:", test_mean)
    print("test std:", test_std)
    print("test max:", max)
    print("test min:", min)

if __name__ == '__main__':
    main()
