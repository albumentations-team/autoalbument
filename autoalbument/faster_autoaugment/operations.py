"""
Based on the official implementation of Faster AutoAugment
https://github.com/moskomule/dda/blob/master/dda/operations.py
"""

import warnings

import albumentations as A
import torch
from torch import nn
from torch.autograd import Function
from torch.distributions import RelaxedBernoulli

from autoalbument.faster_autoaugment.albumentations_pytorch import functional as F
from autoalbument.faster_autoaugment.utils import target_requires_grad


class _STE(Function):
    @staticmethod
    def forward(ctx, input_forward, input_backward):
        ctx.shape = input_backward.shape
        return input_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


def ste(input_forward: torch.Tensor, input_backward: torch.Tensor):
    return _STE.apply(input_forward, input_backward).clone()


class Operation(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        value_range=(0.0, 1.0),
        has_magnitude=True,
        is_spatial_level=False,
        ste=False,
        requires_uint8_scaling=False,
    ):

        super().__init__()
        self.is_spatial_level = is_spatial_level
        self.ste = ste
        self.value_range = value_range
        self.requires_uint8_scaling = requires_uint8_scaling
        self._magnitude = nn.Parameter(torch.empty(1)) if has_magnitude else None
        self._probability = nn.Parameter(torch.empty(1))
        self.register_buffer("temperature", torch.empty(1).fill_(temperature))

    def forward(self, input):
        mask = self.get_probability_mask(input["image_batch"].size(0))
        operation_output = self.operation(input)

        targets = input.keys()
        output = {}
        for target in targets:
            with torch.set_grad_enabled(target_requires_grad(target)):
                output[target] = (mask * operation_output[target] + (1 - mask) * input[target]).clamp_(0, 1)

        return output

    def get_probability_mask(self, batch_size):
        size = (batch_size, 1, 1)
        return RelaxedBernoulli(self.temperature, self.probability).rsample(size)

    @property
    def magnitude(self):
        if self._magnitude is None:
            return None
        return self._magnitude.clamp(0.0, 1.0)

    @property
    def targets(self):
        return {
            "image_batch": self.apply_operation,
            "mask_batch": self.apply_operation_to_mask,
        }

    @property
    def probability(self):
        return self._probability.clamp(0.0, 1.0)

    def __repr__(self):
        probability = self.probability.item()
        magnitude = self.magnitude
        if magnitude is not None:
            magnitude = magnitude.item()
        temperature = self.temperature.item()  # type: ignore

        repr_str = self.__class__.__name__
        repr_str += f"(probability={probability:.3f}, "
        if magnitude is not None:
            repr_str += f"magnitude={magnitude:.3f}, "
        repr_str += f"temperature={temperature:.3f})"
        return repr_str

    def apply_operation_to_mask(self, input, value):
        with torch.no_grad():
            if self.is_spatial_level:
                return self.apply_operation(input, value)
            return input

    def operation(self, input):
        magnitude = self.magnitude
        value = convert_value_range(magnitude, self.value_range) if magnitude is not None else None

        output = {}
        targets = input.keys()
        for target in targets:
            operation_fn = self.targets[target]
            operation_output = operation_fn(input[target], value)
            if target_requires_grad(target) and self.ste:
                operation_output = ste(operation_output, magnitude)
            output[target] = operation_output
        return output

    def create_transform(self, input_dtype, p):
        magnitude = self.magnitude
        if magnitude is not None:
            value = convert_value_range(magnitude, self.value_range)
            transform_param = value_to_transform_param(
                value, input_dtype, requires_uint8_scaling=self.requires_uint8_scaling
            )
        else:
            transform_param = None
        return self.as_transform(transform_param, p)


def value_to_transform_param(value, input_dtype="float32", requires_uint8_scaling=False):
    value = value.item()
    if input_dtype == "uint8" and requires_uint8_scaling:
        value = int(value * 255.0)
    return value


def convert_value_range(value, new_range):
    lower, upper = new_range
    total = upper - lower
    return value * total + lower


class ShiftRGB(Operation):
    def __init__(self, temperature, shift_r=False, shift_g=False, shift_b=False):
        super().__init__(temperature, value_range=(-1.0, 1.0), requires_uint8_scaling=True)
        self.shift_r = shift_r
        self.shift_g = shift_g
        self.shift_b = shift_b

    def apply_operation(self, input, value):
        return F.shift_rgb(
            input,
            r_shift=value if self.shift_r else 0.0,
            g_shift=value if self.shift_g else 0.0,
            b_shift=value if self.shift_b else 0.0,
        )

    def as_transform(self, value, p):
        limit = (value, value)
        return A.RGBShift(
            r_shift_limit=limit if self.shift_r else 0.0,
            g_shift_limit=limit if self.shift_g else 0.0,
            b_shift_limit=limit if self.shift_b else 0.0,
            p=p,
        )


class RandomBrightness(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, value_range=(-1.0, 1.0))

    def apply_operation(self, input, value):
        return F.brightness_adjust(input, beta=value)

    def as_transform(self, value, p):
        return A.RandomBrightnessContrast(brightness_limit=(value, value), contrast_limit=0, p=p)


class RandomContrast(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, value_range=(0.0, 10.0))

    def apply_operation(self, input, value):
        return F.contrast_adjust(input, alpha=value)

    def as_transform(self, value, p):
        return A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(value, value), p=p)


class Solarize(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, requires_uint8_scaling=True)

    def apply_operation(self, input, value):
        return F.solarize(input, threshold=value)

    def as_transform(self, value, p):
        return A.Solarize(threshold=value, p=p)


class HorizontalFlip(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, has_magnitude=False, is_spatial_level=True)

    def apply_operation(self, input, value):
        return F.hflip(input)

    def as_transform(self, value, p):
        return A.HorizontalFlip(p=p)


class VerticalFlip(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, has_magnitude=False, is_spatial_level=True)

    def apply_operation(self, input, value):
        return F.vflip(input)

    def as_transform(self, value, p):
        return A.VerticalFlip(p=p)


class ShiftX(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, value_range=(-1.0, 1.0), is_spatial_level=True)

    def apply_operation(self, input, value):
        return F.shift_x(input, dx=value)

    def as_transform(self, value, p):
        return A.ShiftScaleRotate(
            shift_limit_x=(value, value),
            shift_limit_y=(0, 0),
            rotate_limit=(0, 0),
            scale_limit=(0, 0),
            p=p,
        )


class ShiftY(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, value_range=(-1.0, 1.0), is_spatial_level=True)

    def apply_operation(self, input, value):
        return F.shift_y(input, dy=value)

    def as_transform(self, value, p):
        return A.ShiftScaleRotate(
            shift_limit_x=(0, 0),
            shift_limit_y=(value, value),
            rotate_limit=(0, 0),
            scale_limit=(0, 0),
            p=p,
        )


class Scale(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, value_range=(0 + 1e-8, 10.0), is_spatial_level=True)

    def apply_operation(self, input, value):
        return F.scale(input, scale=value)

    def as_transform(self, value, p):
        return A.ShiftScaleRotate(
            shift_limit_x=(0, 0),
            shift_limit_y=(0, 0),
            rotate_limit=(0, 0),
            scale_limit=(value, value),
            p=p,
        )


class Rotate(Operation):
    def __init__(self, temperature):
        super().__init__(temperature, value_range=(-180, 180), is_spatial_level=True)

    def apply_operation(self, input, value):
        return F.rotate(input, angle=value)

    def as_transform(self, value, p):
        return A.ShiftScaleRotate(
            shift_limit_x=(0, 0),
            shift_limit_y=(0, 0),
            rotate_limit=(value, value),
            scale_limit=(0, 0),
            p=p,
        )


class Cutout(Operation):
    def __init__(self, temperature, value_range=(0.0, 1.0)):
        super().__init__(temperature, value_range=value_range)
        self.register_buffer("saved_image_shape", torch.Tensor([0, 0]).type(torch.int64))
        self.is_image_shape_saved = False

    def _save_image_shape(self, image_shape):
        if not torch.equal(self.saved_image_shape, image_shape):
            if self.is_image_shape_saved:
                warnings.warn(
                    f"Shape of images in a batch changed between iterations "
                    f"from {self.saved_image_shape} to {image_shape}. "
                    f"This will affect the created Albumentations transform. "
                    f"The transform will use the shape {image_shape} to initialize its parameters",
                    RuntimeWarning,
                )
            self.is_image_shape_saved = True
            self.saved_image_shape = image_shape

    def apply_operation(self, input, value):
        image_shape = input.shape[-2:]
        self._save_image_shape(torch.tensor(image_shape).to(input.device))
        return self._apply_cutout(input, value, image_shape)

    def as_transform(self, value, p):
        image_shape = self.saved_image_shape
        return self._as_cutout_transform(value, p, image_shape)

    def _apply_cutout(self, input, value, image_shape):
        raise NotImplementedError

    def _as_cutout_transform(self, value, p, image_shape):
        raise NotImplementedError


class CutoutFixedNumerOfHoles(Cutout):
    def __init__(self, temperature, num_holes=16):
        super().__init__(temperature)
        self.num_holes = num_holes

    def _calculate_hole_size(self, value, image_shape):
        height, width = image_shape
        min_size = min(height, width)
        return max(int(min_size * value), 1)

    def _apply_cutout(self, input, value, image_shape):
        hole_size = self._calculate_hole_size(value, image_shape)
        return F.cutout(input, num_holes=self.num_holes, hole_size=hole_size)

    def _as_cutout_transform(self, value, p, image_shape):
        hole_size = self._calculate_hole_size(value, image_shape)
        return A.CoarseDropout(
            min_holes=self.num_holes,
            max_holes=self.num_holes,
            max_height=hole_size,
            max_width=hole_size,
            p=p,
        )


class CutoutFixedSize(Cutout):
    def __init__(self, temperature, hole_size_divider=16, max_holes=16):
        super().__init__(temperature, value_range=(0, max_holes))
        self.hole_size_divider = hole_size_divider

    def _calculate_hole_size(self, value, image_shape):
        height, width = image_shape
        min_size = min(height, width)
        hole_size = int(min_size // self.hole_size_divider)
        return max(hole_size, 1)

    def _apply_cutout(self, input, value, image_shape):
        hole_size = self._calculate_hole_size(value, image_shape)
        return F.cutout(input, num_holes=int(value), hole_size=hole_size)

    def _as_cutout_transform(self, value, p, image_shape):
        hole_size = self._calculate_hole_size(value, image_shape)
        num_holes = max(int(value), 1)
        return A.CoarseDropout(
            min_holes=num_holes,
            max_holes=num_holes,
            max_height=hole_size,
            max_width=hole_size,
            p=p,
        )
