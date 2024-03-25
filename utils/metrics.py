# SPDX-License-Identifier: Apache-2.0
import numpy as np
from collections import OrderedDict


def compute_rgb_image_ssim(prediction, ground_truth, mask):
    """Computes SSIM
    Args:
        prediction:  Array with shape (h,w,3), range [0..1], and dtype=np.float32 with the prediction.
        ground_truth: Array with shape (h,w,3), range [0..1], and dtype=np.float32 with the ground truth.
        mask: Array with shape (h,w,1), range [0..1], and dtype=np.float32 with a binary mask.
    Returns:
        Returns the mean SSIM
    """
    from skimage.metrics import structural_similarity as ssim
    pr = prediction * mask
    gt = ground_truth * mask
    assert(pr.dtype == gt.dtype and pr.dtype == np.float32)
    mean, S = ssim(pr, gt, channel_axis=-1, full=True, data_range=1.0)
    return float(S[mask[:,:,0]>0.5].mean())


def compute_rgb_image_psnr(prediction, ground_truth, mask):
    """Computes PSNR
    Args:
        prediction:  Array with shape (h,w,3), range [0..1], and dtype=np.float32 with the prediction.
        ground_truth: Array with shape (h,w,3), range [0..1], and dtype=np.float32 with the ground truth.
        mask: Array with shape (h,w,1), range [0..1], and dtype=np.float32 with a binary mask.
    Returns:
        Returns the PSNR
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr
    pr = prediction[mask[:,:,0]>0.5].ravel()
    gt = ground_truth[mask[:,:,0]>0.5].ravel()
    assert(pr.dtype == gt.dtype and pr.dtype == np.float32)
    ans = psnr(gt, pr)
    return float(ans)


def compute_rgb_image_lpips(prediction, ground_truth, mask):
    """Computes LPIPS
    Args:
        prediction:  Array with shape (h,w,3), range [0..1], and dtype=np.float32 with the prediction.
        ground_truth: Array with shape (h,w,3), range [0..1], and dtype=np.float32 with the ground truth.
        mask: Array with shape (h,w,1), range [0..1], and dtype=np.float32 with a binary mask.
    Returns:
        Returns the LPIPS error
    """
    import torch
    import lpips
    pr = mask*(prediction*2-1)
    gt = mask*(ground_truth*2-1)
    pr = pr.transpose(2,0,1)[None,...]
    gt = gt.transpose(2,0,1)[None,...]

    if 'loss_fn' in compute_rgb_image_lpips.static_vars:
        loss_fn = compute_rgb_image_lpips.static_vars['loss_fn']
    else:
        loss_fn = lpips.LPIPS(net='alex', spatial=True)
        compute_rgb_image_lpips.static_vars['loss_fn'] = loss_fn

    ans = loss_fn.forward(torch.from_numpy(pr), torch.from_numpy(gt), normalize=False).cpu().detach().numpy()
    return float(ans[0,0,...][mask[:,:,0]>0.5].mean())
compute_rgb_image_lpips.static_vars = {}


METRICS = OrderedDict([
    ('PSNR', {'fn': compute_rgb_image_psnr, 'latex':r'PSNR $\uparrow$', 'best': 'largest'}),
    ('SSIM', {'fn': compute_rgb_image_ssim, 'latex':r'SSIM $\uparrow$', 'best': 'largest'}),
    ('LPIPS', {'fn': compute_rgb_image_lpips, 'latex':r'LPIPS $\downarrow$', 'best': 'smallest'}),
])
