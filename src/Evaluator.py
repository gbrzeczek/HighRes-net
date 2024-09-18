""" Python script to evaluate super resolved images against ground truth high resolution images """

import itertools
import math

import numpy as np
from tqdm import tqdm
from torch import nn

from DataLoader import get_patch

import torch

def cPSNR(sr, hr, hr_map):
    """
    Clear Peak Signal-to-Noise Ratio. The PSNR score, adjusted for brightness and other volatile features, e.g. clouds.
    Args:
        sr: numpy.ndarray (n, m), super-resolved image
        hr: numpy.ndarray (n, m), high-res ground-truth image
        hr_map: numpy.ndarray (n, m), status map of high-res image, indicating clear pixels by a value of 1
    Returns:
        cPSNR: float, score
    """

    if len(sr.shape) == 2:
        sr = sr[None, ]
        hr = hr[None, ]
        hr_map = hr_map[None, ]

    if sr.dtype.type is np.uint16:  # integer array is in the range [0, 65536]
        sr = sr / np.iinfo(np.uint16).max  # normalize in the range [0, 1]
    else:
        assert 0 <= sr.min() and sr.max() <= 1, 'sr.dtype must be either uint16 (range 0-65536) or float64 in (0, 1).'
    if hr.dtype.type is np.uint16:
        hr = hr / np.iinfo(np.uint16).max

    n_clear = np.sum(hr_map, axis=(1, 2))  # number of clear pixels in the high-res patch
    diff = hr - sr
    bias = np.sum(diff * hr_map, axis=(1, 2)) / n_clear  # brightness bias
    cMSE = np.sum(np.square((diff - bias[:, None, None]) * hr_map), axis=(1, 2)) / n_clear
    cPSNR = -10 * np.log10(cMSE)  # + 1e-10)

    if cPSNR.shape[0] == 1:
        cPSNR = cPSNR[0]

    return cPSNR

def cPSNR_torch(srs, hrs, hr_maps):
    criterion = nn.MSELoss(reduction='none')
    nclear = torch.sum(hr_maps, dim=(1, 2))  # Number of clear pixels in target image
    bright = torch.sum(hr_maps * (hrs - srs), dim=(1, 2)).clone().detach() / nclear  # Correct for brightness
    loss = torch.sum(hr_maps * criterion(srs + bright.view(-1, 1, 1), hrs), dim=(1, 2)) / nclear  # cMSE(A,B) for each point

    return -10 * torch.log10(loss)

def patch_iterator(img, positions, size):
    """Iterator across square patches of `img` located in `positions`."""
    for x, y in positions:
        yield get_patch(img=img, x=x, y=y, size=size)

def get_patch_tensor(img, x, y, size=32):
    """
    Slices out a square patch from `img` tensor starting from the (x, y) top-left corner.
    This function assumes `img` is a tensor and can handle 3D (single image) or 4D (batch of images) tensors.
    Args:
        img: torch.Tensor, input image tensor with shape (C, H, W) or (N, C, H, W)
        x, y: int, top-left corner coordinates of the patch
        size: int, size of the square patch
    Returns:
        patch: torch.Tensor, extracted patch of shape (C, size, size) or (N, C, size, size)
    """
    if img.dim() == 3:  # Single image
        return img[:, x:(x + size), y:(y + size)]
    elif img.dim() == 4:  # Batch of images
        return img[:, :, x:(x + size), y:(y + size)]

def shift_cPSNR(sr, hr, hr_map, border_w=3):
    """
    cPSNR score adjusted for registration errors. Computes the max cPSNR score across shifts of up to `border_w` pixels.
    Args:
        sr: np.ndarray (n, m), super-resolved image
        hr: np.ndarray (n, m), high-res ground-truth image
        hr_map: np.ndarray (n, m), high-res status map
        border_w: int, width of the trimming border around `hr` and `hr_map`
    Returns:
        max_cPSNR: float, score of the super-resolved image
    """

    size = sr.shape[1] - (2 * border_w)  # patch size
    sr = get_patch(img=sr, x=border_w, y=border_w, size=size)
    pos = list(itertools.product(range(2 * border_w + 1), range(2 * border_w + 1)))
    iter_hr = patch_iterator(img=hr, positions=pos, size=size)
    iter_hr_map = patch_iterator(img=hr_map, positions=pos, size=size)
    site_cPSNR = np.array([cPSNR(sr, hr, hr_map) for hr, hr_map in tqdm(zip(iter_hr, iter_hr_map),
                                                                        disable=(len(sr.shape) == 2))
                           ])
    max_cPSNR = np.max(site_cPSNR, axis=0)
    return max_cPSNR

class MultiTaskLossCalculator:
    def __init__(self, lpips_loss, device, writer=None):
        self._lpips_loss = lpips_loss
        self._device = device
        self._counter = 1
        self._writer = writer
    
    def get_lpips(self, hrs, srs):
        srs_normalized = (srs - 0.5) * 2
        hrs_normalized = (hrs - 0.5) * 2

        srs_normalized = srs_normalized.unsqueeze(1).repeat(1, 3, 1, 1)
        hrs_normalized = hrs_normalized.unsqueeze(1).repeat(1, 3, 1, 1)

        lpips_scores = self._lpips_loss(srs_normalized, hrs_normalized)

        return lpips_scores
    
    def get_total_variation_loss(self, img):
        if img.dim() == 3:
            img = img.unsqueeze(1)

        horizontal_tv = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        vertical_tv = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

        losses = horizontal_tv.sum() + vertical_tv.sum()

        return losses
        
    def get_cPSNR(self, hrs, srs, cropped_masks):
        return cPSNR_torch(srs, hrs, cropped_masks)
    
    def get_simple_weighted_loss(self, lpips_values, cpsnr_values):
        lpips_weight = 0.6
        cpsnr_weight = 0.4

        mean_lpips = torch.mean(lpips_values)
        mean_cpsnr = torch.mean(cpsnr_values)

        if self._writer:
            self._writer.add_scalar('lpips', mean_lpips, self._counter)
            self._writer.add_scalar('cpsnr', mean_cpsnr, self._counter)

        lpips_w = mean_lpips * lpips_weight
        cpsnr_w = self._normalize_cpsnr(mean_cpsnr) * cpsnr_weight

        return lpips_w + cpsnr_w

    def update_counter(self):
        self._counter += 1

    def _normalize_tv(self, tv_value):
        top_value = 1500

        return tv_value / top_value

    def _normalize_cpsnr(self, cpsnr_value):
        return -cpsnr_value / 50
