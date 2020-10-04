import random

import torch

from autoalbument.faster_autoaugment.albumentations_pytorch.affine import (
    get_scaling_matrix,
    get_rotation_matrix,
    warp_affine,
)
from autoalbument.faster_autoaugment.albumentations_pytorch.utils import MAX_VALUES_BY_DTYPE, TorchPadding, clipped


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
    scaling_matrix = get_scaling_matrix(img_batch, dx=dx)
    return warp_affine(img_batch, scaling_matrix, padding_mode)


def shift_y(img_batch, dy, padding_mode=TorchPadding.REFLECTION):
    scaling_matrix = get_scaling_matrix(img_batch, dy=dy)
    return warp_affine(img_batch, scaling_matrix, padding_mode)


def rotate(img_batch, angle, padding_mode=TorchPadding.REFLECTION):
    rotation_matrix = get_rotation_matrix(img_batch, angle=angle)
    return warp_affine(img_batch, rotation_matrix, padding_mode)


def scale(img_batch, scale, padding_mode=TorchPadding.REFLECTION):
    rotation_matrix = get_rotation_matrix(img_batch, scale=scale)
    return warp_affine(img_batch, rotation_matrix, padding_mode)


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
