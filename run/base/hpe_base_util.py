from dataset.data_model import HandDataModel

def unpack_data(results, is_eval = False):
    joint_out = results['joint']
    heatmap = None
    heatmap_true = None
    if not is_eval:
        heatmap = results['heatmap']
        heatmap_true = results['heatmap_true']
        heatmap_reprojected = results['heatmap_reprojected']

    return joint_out, heatmap, heatmap_true, heatmap_reprojected
