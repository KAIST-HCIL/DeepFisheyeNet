import torch

def argmax_2d(tensor):
    assert len(tensor.shape) == 4
    N, C, H, W = tensor.shape
    tensor = tensor.reshape(N, C, H*W)
    _, idx = tensor.max(dim = -1)

    row, col = unravel_index(idx, H, W)
    return torch.stack([row, col], dim = 2)

def unravel_index(idx, H, W):
    row = (idx / W).long()
    col = (idx % W).long()

    return row, col

def argmin_2d(tensor):
    return argmax_2d(-tensor)
