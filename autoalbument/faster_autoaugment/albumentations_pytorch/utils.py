from enum import Enum
from functools import wraps

import cv2
import torch

MAX_VALUES_BY_DTYPE = {torch.float32: 1.0, torch.float64: 1.0}


class TorchPadding(str, Enum):
    REFLECTION = "reflection"
    BORDER = "border"
    ZEROS = "zeros"


TORCH_TO_OPENCV_BORDER = {
    TorchPadding.REFLECTION: cv2.BORDER_REFLECT,
    TorchPadding.BORDER: cv2.BORDER_REPLICATE,
    TorchPadding.ZEROS: cv2.BORDER_CONSTANT,
}


def clip(img, dtype, maxval):
    return torch.clamp(img, 0, maxval).type(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function
