import torch
import math
import numpy as np

def get_focal_length(max_radius, fisheye_type):
    """ Calculcate focal length of a 180 degree fov fisheye camera, which means
        theta is pi/2 at the max radius from the center of the image.
    """
    if fisheye_type == 'orthographic':
        return max_radius
    elif fisheye_type == 'equidistant':
        return (max_radius * 2 / math.pi)
    else:
        raise ValueError("fisheye type {} is not implemented".format(fish_type))

def r_function(max_radius, theta, fisheye_type, use_np = False):
    f = get_focal_length(max_radius, fisheye_type)

    math_module = torch
    if use_np:
        math_module = np

    if fisheye_type == 'orthographic':
        return f * math_module.sin(theta)
    elif fisheye_type == 'equidistant':
        return f * theta
    else:
        raise ValueError("fisheye type {} is not implemented".format(fish_type))

def inverse_r_function(max_radius, r, fish_type, use_np = False):
    f = get_focal_length(max_radius, fish_type)

    math_module = torch
    if use_np:
        math_module = np

    if fish_type == 'orthographic':
        return math_module.arcsin(r/f)
    elif fish_type == 'equidistant':
        return r/f
    else:
        raise ValueError("fisheye type {} is not implemented".format(fish_type))

def make_theta_phi_meshgrid(img_shape, fisheye_type):
    height, width = img_shape
    v = np.linspace(0, height, height)
    u = np.linspace(0, width, width)

    vv, uu = np.meshgrid(v, u)
    vv = vv-height/2
    uu = uu-width/2

    r = np.sqrt(np.power(vv, 2) + np.power(uu, 2))
    max_radius = width / 2
    theta = inverse_r_function(max_radius, r, fisheye_type, use_np = True)
    theta = np.nan_to_num(theta, nan=0)

    phi = np.arctan2(uu, vv)
    return torch.from_numpy(theta), torch.from_numpy(phi)
