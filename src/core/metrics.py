import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None

##
# 对流性降水等级定义（表 2）
# (min, max, weight)
# ================================================
RAIN_LEVELS_WEIGHT = {(0.0, 0.0): 0.1, (0.0, 0.9): 0.1, (1.0, 1.9): 0.1, (2.0, 4.9): 0.2, (5.0, 7.9): 0.2, (8.0, float('inf')): 0.3}
FORECAST_LEAD_TIMES = {1: 0.0075, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.06, 7: 0.07, 8: 0.08, 9: 0.09, 10: 0.1, 11: 0.09, 12: 0.08, 
                       13: 0.07, 14: 0.06, 15: 0.05, 16: 0.04, 17: 0.03, 18: 0.02, 19: 0.0075, 20: 0.005}


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1

def _threshold(x, y, t):
    t = np.greater_equal(x, t).astype(np.float32)
    p = np.greater_equal(y, t).astype(np.float32)
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    t = np.where(is_nan, np.zeros_like(t, dtype=np.float32), t)
    p = np.where(is_nan, np.zeros_like(p, dtype=np.float32), p)
    return t, p

def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()

def calculate_score(pred, true):
    r_w = np.sqrt(np.exp(calculate_r(pred, true)-1))  #
    weighted_sum = 0.0
    for i, (rain_range, weight) in enumerate(RAIN_LEVELS_WEIGHT.items()):
        TS_ik = calculate_ts(pred, true, rain_range)
        mask = ((true > 0) | (pred > 0)) & (true >= rain_range[0]) & (true <= rain_range[1])
        MAE_ik = np.mean(np.abs(pred[mask] - true[mask])) if mask.any() else 0
        weighted_sum += weight * TS_ik * np.sqrt(np.exp(-MAE_ik))
    return r_w * weighted_sum

def calculate_ts(pred, true, thr):
    p, t = ((thr[0] <= pred) & (pred <= thr[1])), ((thr[0] <= true) & (true <= thr[1]))
    TP = np.sum(p & t); FP = np.sum(p & ~t); FN = np.sum(~p & t)
    s = TP + FP + FN
    return TP / s if s else 1.0

def calculate_r(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    pred_mean = np.mean(pred_flat)
    true_mean = np.mean(true_flat)
    
    covariance = np.sum((pred_flat - pred_mean) * (true_flat - true_mean))
    pred_var = np.sum((pred_flat - pred_mean) ** 2)
    true_var = np.sum((true_flat - true_mean) ** 2)
    
    if pred_var == 0 and true_var == 0:
        return 1.0
    
    denominator = np.sqrt(pred_var * true_var) + 1e-6
    r_value = covariance / denominator
    return r_value


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


def SNR(pred, true):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    signal = ((true)**2).mean()
    noise = ((true - pred)**2).mean()
    return 10. * np.log10(signal / noise)


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def POD(hits, misses, eps=1e-6):
    """
    probability_of_detection
    Inputs:
    Outputs:
        pod = hits / (hits + misses) averaged over the T channels
        
    """
    pod = (hits + eps) / (hits + misses + eps)
    return np.mean(pod)

def SUCR(hits, fas, eps=1e-6):
    """
    success_rate
    Inputs:
    Outputs:
        sucr = hits / (hits + false_alarms) averaged over the D channels
    """
    sucr = (hits + eps) / (hits + fas + eps)
    return np.mean(sucr)

def CSI(hits, fas, misses, eps=1e-6):
    """
    critical_success_index 
    Inputs:
    Outputs:
        csi = hits / (hits + false_alarms + misses) averaged over the D channels
    """
    csi = (hits + eps) / (hits + misses + fas + eps)
    return np.mean(csi)

def sevir_metrics(pred, true, threshold):
    """
    calcaulate t, p, hits, fas, misses
    Inputs:
    pred: [N, T, C, L, L]
    true: [N, T, C, L, L]
    threshold: float
    """
    pred = pred.transpose(1, 0, 2, 3, 4)
    true = true.transpose(1, 0, 2, 3, 4)
    hits, fas, misses = [], [], []
    for i in range(pred.shape[0]):
        t, p = _threshold(pred[i], true[i], threshold)
        hits.append(np.sum(t * p))
        fas.append(np.sum((1 - t) * p))
        misses.append(np.sum(t * (1 - p)))
    return np.array(hits), np.array(fas), np.array(misses)


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def metric(pred, true,region=None,mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        mean (tensor): The mean of the preprocessed video data.
        std (tensor): The std of the preprocessed video data.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        channel_names (list | None): The name of different channels.
        spatial_norm (bool): Weather to normalize the metric by HxW.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean
    elif region is not None:
        for i, (p, t, r) in enumerate(zip(pred, true, region)):
            pred[i] = p * r["std"] + r["mean"]
            true[i] = t * r["std"] + r["mean"]
    else:
        raise ValueError(f'mean and std is not supported.')
        
    pred = np.transpose(pred, (1, 0, 2, 3, 4))
    true = np.transpose(true, (1, 0, 2, 3, 4))
    pred = np.maximum(pred, 0) 
    if threshold == None:
        threshold = 25

    eval_res = {}
    eval_log = ""

    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi', 'ts', 'r','score'] # requested mae ts-scores R Score_k
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group
            
    if 'ts' in metrics:
        if channel_names is None:
            ts_sum = 0.
            for i, (p, t) in enumerate(zip(pred, true)):
                s = 0.
                for j, (range_rain, weight) in enumerate(RAIN_LEVELS_WEIGHT.items()):
                    s += calculate_ts(p,t,range_rain) * weight
            ts_sum += s*FORECAST_LEAD_TIMES[i+1]
            eval_res['ts'] = ts_sum           


    if 'r' in metrics:
        if channel_names is None:
            r_sum = 0.
            for i, (p, t) in enumerate(zip(pred, true)):
                r_sum += calculate_r(p,t) * FORECAST_LEAD_TIMES[i+1]
            eval_res['r'] = r_sum       
              
    if 'score' in metrics:  # 注意：应该是小写的'score'
        if channel_names is None:
            score_sum = 0.
            # 修正：使用zip同时遍历pred和true
            for i, (p, t) in enumerate(zip(pred, true)):
                score_sum += calculate_score(p, t) * FORECAST_LEAD_TIMES[i+1]
            eval_res['score'] = score_sum
            
    # others            
    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                       true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group
    
    if 'pod' in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res['pod'] = POD(hits, misses)
        eval_res['sucr'] = SUCR(hits, fas)
        eval_res['csi'] = CSI(hits, fas, misses) 
        
    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    if 'snr' in metrics:
        snr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                snr += SNR(pred[b, f], true[b, f])
        eval_res['snr'] = snr / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        lpips = 0
        cal_lpips = LPIPS(net='alex', use_gpu=False)
        pred = pred.transpose(0, 1, 3, 4, 2)
        true = true.transpose(0, 1, 3, 4, 2)
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                lpips += cal_lpips(pred[b, f], true[b, f])
        eval_res['lpips'] = lpips / (pred.shape[0] * pred.shape[1])

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str
    return eval_res, eval_log
