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

        self._metric_count = 2
        self._temperature = 0.1

        self._counter = 1

        self._cPSNR_metric_name = 'cPSNR'
        self._lpips_metric_name = 'lpips'
        self._losses = {self._cPSNR_metric_name: [], self._lpips_metric_name: []}

        self._writer = writer
    
    def get_lpips(self, hrs, srs):
        srs_normalized = (srs - 0.5) * 2
        hrs_normalized = (hrs - 0.5) * 2

        srs_normalized = srs_normalized.unsqueeze(1).repeat(1, 3, 1, 1)
        hrs_normalized = hrs_normalized.unsqueeze(1).repeat(1, 3, 1, 1)

        lpips_scores = self._lpips_loss(srs_normalized, hrs_normalized)

        return lpips_scores
    
    def get_cPSNR(self, hrs, srs, cropped_masks):
        return cPSNR_torch(srs, hrs, cropped_masks)
    
    def get_weighted_loss(self, lpips_values, cpsnr_values):
        mean_lpips = torch.mean(lpips_values).item()
        mean_cpsnr = torch.mean(cpsnr_values).item()

        if self._writer:
            self._writer.add_scalar('lpips', mean_lpips, self._counter)
            self._writer.add_scalar('cPSNR', mean_cpsnr, self._counter)

        normalized_cpsnr = self._normalize_cpsnr(mean_cpsnr)
        return self._get_weighted_loss(mean_lpips, normalized_cpsnr)
    
    def update(self, lpips_values, cpsnr_values):
        self._losses[self._lpips_metric_name].append(torch.mean(lpips_values).item())
        self._losses[self._cPSNR_metric_name].append(torch.mean(cpsnr_values).item())

        self._counter += 1
    
    def _normalize_cpsnr(self, mean_cpsnr):
        return 1 / mean_cpsnr # TODO evaluate, it may not be good
    
    def _get_weighted_loss(self, lpips_value, cPSNR_value):
        weight_lpips, weight_cPSNR = self._get_weights()

        if self._writer:
            self._writer.add_scalar('weight_lpips', weight_lpips, self._counter)
            self._writer.add_scalar('weight_cPSNR', weight_cPSNR, self._counter)

        return weight_lpips * lpips_value + weight_cPSNR * cPSNR_value

    def _get_weights(self):
        if len(self._losses['lpips']) < 2 or len(self._losses['cPSNR']) < 2:
            return 0.5, 0.5

        rn_lpips = self._losses['lpips'][-1] / self._losses['lpips'][-2]
        rn_cPSNR = self._losses['cPSNR'][-1] / self._losses['cPSNR'][-2]

        numerator_lpips = self._metric_count * math.exp(rn_lpips * self._temperature)
        numerator_cPSNR = self._metric_count * math.exp(rn_cPSNR * self._temperature)

        denominator = numerator_lpips / 2 + numerator_cPSNR / 2

        weight_lpips = numerator_lpips / denominator
        weight_cPSNR = numerator_cPSNR / denominator

        return weight_lpips, weight_cPSNR
