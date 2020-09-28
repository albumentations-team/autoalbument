import random

import torch

from . import kornia_compat as KC
from .utils import MAX_VALUES_BY_DTYPE, TorchPadding, clipped


def solarize(img_batch, threshold):
    dtype = img_batch.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]
    return torch.where(img_batch >= threshold, max_val - img_batch, img_batch)


@clipped
def shift_rgb(img_batch, r_shift, g_shift, b_shift):
    result_img_batch = img_batch.clone()
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img_batch[:, i] = result_img_batch[:, i] + shift
    return result_img_batch


@clipped
def brightness_adjust(img_batch, beta):
    return img_batch + beta


@clipped
def contrast_adjust(img_batch, alpha):
    return img_batch * alpha


def vflip(img_batch):
    return torch.flip(img_batch, [-2]).contiguous()


def hflip(img_batch):
    return torch.flip(img_batch, [-1]).contiguous()


def shift_x(img_batch, dx, padding_mode=TorchPadding.REFLECTION):
    batch_size, _, _, width = img_batch.size()
    v = torch.zeros(batch_size, 2).to(img_batch.device)
    v[:, 0] = dx * width
    return KC.translate(img_batch, v, align_corners=True, padding_mode=padding_mode)


def shift_y(img_batch, dy, padding_mode=TorchPadding.REFLECTION):
    batch_size, _, height, _ = img_batch.size()
    v = torch.zeros(batch_size, 2).to(img_batch.device)
    v[:, 1] = dy * height
    return KC.translate(img_batch, v, align_corners=True, padding_mode=padding_mode)


def rotate(img_batch, angle, padding_mode=TorchPadding.REFLECTION):
    return KC.rotate(img_batch, angle, align_corners=True, padding_mode=padding_mode)


def scale(img_batch, scale, padding_mode=TorchPadding.REFLECTION):
    return KC.scale(img_batch, scale, align_corners=True, padding_mode=padding_mode)


def cutout(img_batch, num_holes, hole_size, fill_value=0):
    img_batch = img_batch.clone()
    height, width = img_batch.shape[-2:]
    for i in range(len(img_batch)):
        for _n in range(num_holes):
            y1 = random.randint(0, height - hole_size)
            x1 = random.randint(0, width - hole_size)
            y2 = y1 + hole_size
            x2 = x1 + hole_size
            img_batch[i, :, y1:y2, x1:x2] = fill_value
    return img_batch
