from collections import defaultdict, OrderedDict
import torch
import numpy as np

def cal_L1_diff(estimated, real, reduction = 'mean', dim = []):
    diff = (estimated - real).abs()
    if reduction == 'mean':
        return diff.mean(dim = dim)
    elif reduction == 'sum':
        return diff.mean(dim = dim)

    raise NotImplementedError("reduction should be either 'mean' or 'sum'")

def cal_RMS_diff(estimated, real, reduction = 'mean', dim = []):
    diff = estimated - real
    if reduction == 'mean':
        return (diff ** 2).mean(dim = dim).sqrt()
    elif reduction == 'sum':
        return (diff ** 2).sum(dim = dim).sqrt()

    raise NotImplementedError("reduction should be either 'mean' or 'sum'")

class StatDict:
    def __init__(self):
        self.data_group = defaultdict(list)

    def add(self, losses):
        for k, v in losses.items():
            v = self._if_single_then_multi_dim(v)
            self.data_group[k].append(v)

    def _if_single_then_multi_dim(self, tensor):
        if len(tensor.shape) == 1:
            return tensor.unsqueeze(0)

        return tensor

    def get_avg(self):
        avg = OrderedDict()
        for k, v in self.data_group.items():
            combined = torch.stack(v, 0)
            avg[k] = torch.mean(combined, dim = [0])
        return avg

    def get_std(self):
        deviation = OrderedDict()
        for k, v in self.data_group.items():
            combined = torch.stack(v, 0)
            deviation[k] = torch.std(combined, dim = [0])

        return deviation


class AverageDict:
    def __init__(self):
        self.meters = {}

    def add(self, losses):
        for k, v in losses.items():
            if k not in self.meters.items():
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)

    def to_dict(self):
        result = {}
        for k, m in self.meters.items():
            result[k] = m.avg
        return result

    def reset(self):
        for k, m in self.meters.items():
            m.reset()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count

class Timer:
    def __init__(self):
        self.stopwatches = {}

    def start(self, key):

        if not key in self.stopwatches:
            self.stopwatches[key] = CudaStopwatch()

        self.stopwatches[key].start()

    def stop(self, key):
        self.stopwatches[key].stop()

    def print_elapsed_times(self):
        for key, sw in self.stopwatches.items():
            print("{}: {} sec".format(key, sw.get_elapsed_time()))

class CudaStopwatch:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing = True)
        self.end_event = torch.cuda.Event(enable_timing = True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()

    def get_elapsed_time(self):
        return self.start_event.elapsed_time(self.end_event)
