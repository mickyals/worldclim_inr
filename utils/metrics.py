import torch
import torch.nn.functional as F


def psnr_pixel(y_true, y_pred, max_val=1.0, **kwargs):

    mse_val = F.mse_loss(y_true, y_pred, reduction='mean')
    return 20 * torch.log10(max_val / torch.sqrt(mse_val))

def mae_mse_pixel(y_true, y_pred, **kwargs):
    mae = F.l1_loss(y_true, y_pred, reduction='mean')
    mse = F.mse_loss(y_true, y_pred, reduction='mean')
    return mae, mse



########################
## Registry
########################

PIXEL_METRICS_REGISTRY = {
    "psnr_pixel": psnr_pixel,
    "mae_mse_pixel": mae_mse_pixel,
}

def compute_pixel_metric(y_true, y_pred, metric_list, mav_val):
    metrics = {}
    for metric in metric_list:
        if metric in PIXEL_METRICS_REGISTRY:
            result = PIXEL_METRICS_REGISTRY[metric](y_true, y_pred, mav_val)
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[metric] = result
    return metrics

