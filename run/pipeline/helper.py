import torch

def unpack_data(results):
    pix = results['input']
    hand_pix = results['hand_pix']
    fake_fish_depth = results['fake_fish_depth']
    heatmap = results['heatmap']
    heatmap_true = results['heatmap_true']
    heatmap_reprojected = results['heatmap_reprojected']
    joint = results['joint']

    return pix, hand_pix, fake_fish_depth, heatmap, heatmap_true, heatmap_reprojected, joint
