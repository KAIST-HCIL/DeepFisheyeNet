import cv2
import numpy as np
import torchvision
import torch
from PIL import Image

############### Colormap #######################

def convert_to_colormap(heatmap, max_value = 1.0):
    heatmap = scale_to_make_visible(heatmap, max_value)
    heatmap = merge_channels(heatmap)
    heatmap = torch.clamp(heatmap, max = 1.0, min = 0.0)
    heatmap_img = conver_each_channel_to_colormap(heatmap)

    return heatmap_img

def merge_channels(multi_channel_img):
    single_channel_img = multi_channel_img.sum(dim = 1).unsqueeze(1)
    #heatmap_img = heatmap[:,0,:,:].unsqueeze(1)
    return single_channel_img

def scale_to_make_visible(heatmap, max_value):
    return heatmap / max_value

def blend_to_image(img, heatmap, ratio = 0.45):
    img = img.cpu().detach()
    heatmap = heatmap.cpu().detach()
    blended = blend(img, heatmap)
    blended_img = torchvision.transforms.ToPILImage()(blended)
    return blended_img

def blend(images, heatmaps, ratio = 0.45):

    merged_heatmaps = heatmaps.sum(dim = 1).unsqueeze(1)

    colormaps = convert_to_colormap(merged_heatmaps)
    num_sample = images.size(0)

    blended = []
    for i in range(num_sample):
        img = images[i]
        img = torchvision.transforms.ToPILImage()(img)

        cm = colormaps[i].squeeze()
        cm = torchvision.transforms.ToPILImage()(cm)

        b = Image.blend(img, cm, ratio)

        b = torchvision.transforms.ToTensor()(b).unsqueeze(0)

        blended.append(b)
    blended = torch.cat(blended)

    return blended.squeeze()

def conver_each_channel_to_colormap(multi_channel_img):
    colormaps = []

    for img in multi_channel_img:
        img = img.squeeze()
        img *= 255
        img = img.numpy().astype(np.uint8)
        cm = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
        cm = torchvision.transforms.ToTensor()(cm).unsqueeze(0)
        colormaps.append(cm)

    return torch.cat(colormaps)

def gray_to_rgb(img_tensor):
    shape = img_tensor.shape
    assert shape[1] == 1, "input should have a single channel"

    return torch.cat((img_tensor, img_tensor, img_tensor), 1)

############### Normalizations #######################

def normalize_img(img, mean=0.5, std=0.5):
    if img is None:
        return None
    """
        ref : https://github.com/pytorch/vision/issues/528
        normalize: image = (image - mean)/std
        unnormalize: image = (image * std) + mean = (image - (-mean/std)) * (1/std)
        mean = 0.5, std = 0.5
        Please see 'base_dataset'
    """
    return (img - mean) / std # (img + 1) / 2.0

def unnormalize_as_img(img, mean=0.5, std=0.5):
    if img is None:
        return None
    """
        ref : https://github.com/pytorch/vision/issues/528
        normalize: image = (image - mean)/std
        unnormalize: image = (image * std) + mean = (image - (-mean/std)) * (1/std)
        mean = 0.5, std = 0.5
        Please see 'base_dataset'
    """
    unnormalized = (img * std) + mean # (img + 1) / 2.0
    return unnormalized

############### Image Processing ########################
def get_center_circle_mask(img_size, dataformats = 'NCHW'):
    radius = int(np.min(img_size) / 2)

    center = (radius, radius)
    x = np.linspace(-img_size[0]/2, img_size[0]/2, img_size[0])
    y = np.linspace(-img_size[1]/2, img_size[1]/2, img_size[1])

    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    mask = np.zeros(img_size, dtype = int)
    mask[rr <= radius] = 1.0
    if dataformats == 'NCHW':
        return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    elif dataformats == 'CHW':
        return torch.from_numpy(mask).float().unsqueeze(0)
    else:
        raise Exception("dataformats should be either 'NCHW' or 'CHW'")

def expand_channel(img, dataformats = 'NCHW'):
    """ Expand channel to 3
    """
    if dataformats == 'NCHW':
        target_dim = 1
    elif dataformats == 'CHW':
        target_dim = 0
    else:
        raise Exception("dataformats should be either 'NCHW' or 'CHW'")

    if img.size(target_dim) == 3:
        return img

    return torch.cat((img, img, img), target_dim)

def merge_channel(img, dataformats = 'NCHW'):
    """ Merge channel to 1
    """
    if dataformats == 'NCHW':
        target_dim = 1
    elif dataformats == 'CHW':
        target_dim = 0
    else:
        raise Exception("dataformats should be either 'NCHW' or 'CHW'")

    if img.size(target_dim) == 1:
        return img

    return img.mean(dim = target_dim).unsqueeze(target_dim)
