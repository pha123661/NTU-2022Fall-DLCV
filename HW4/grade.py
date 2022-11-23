import sys
import os
import glob
from tqdm import tqdm, trange
import scipy.signal
import imageio
import numpy as np

import torch

''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


@torch.no_grad()
def render_viewpoints(test_image_path, gt_image_path):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    device = "cuda"
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    img_paths = sorted(glob.glob(os.path.join(test_image_path,"*.png")))
    gt_paths = sorted(glob.glob(os.path.join(gt_image_path,"*.png")))

    for i in tqdm(range(len(img_paths))):
        rgb = imageio.v2.imread(img_paths[i], pilmode='RGBA')
        rgb = (np.array(rgb)/255.).astype(np.float32)
        rgb = rgb[...,:3]*rgb[...,-1:]

        gt = imageio.v2.imread(gt_paths[i], pilmode='RGBA')
        gt = (np.array(gt)/255.).astype(np.float32)
        gt = gt[...,:3]*gt[...,-1:] + (1.-gt[...,-1:])

        p = -10. * np.log10(np.mean(np.square(rgb - gt)))
        psnrs.append(p)
        ssims.append(rgb_ssim(rgb, gt, max_val=1))
        # lpips_alex.append(rgb_lpips(rgb, gt, net_name='alex', device=device))
        # lpips_vgg.append(rgb_lpips(rgb, gt, net_name='vgg', device=device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing ssim', np.mean(ssims), '(avg)')
        # print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        # print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    return 

if __name__ == '__main__':
    render_viewpoints(sys.argv[1],sys.argv[2])