from typing import Union

import torch
from kornia import warp_affine
from kornia.geometry.transform.affwarp import (
    _compute_rotation_matrix,
    _compute_scaling_matrix,
    _compute_tensor_center,
    _compute_translation_matrix,
)

from .utils import TorchPadding


def affine(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool = False,
    padding_mode=TorchPadding.REFLECTION,
) -> torch.Tensor:
    r"""Apply an affine transformation to the image.

    Args:
        tensor (torch.Tensor): The image tensor to be warped.
        matrix (torch.Tensor): The 2x3 affine transformation matrix.
        mode (str): 'bilinear' | 'nearest'
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail

    Returns:
        torch.Tensor: The warped image.
    """
    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: torch.Tensor = warp_affine(
        tensor,
        matrix,
        (height, width),
        mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


def translate(
    tensor: torch.Tensor,
    translation: torch.Tensor,
    align_corners: bool = False,
    padding_mode=TorchPadding.REFLECTION,
) -> torch.Tensor:
    r"""Translate the tensor in pixel units.

    See :class:`~kornia.Translate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))
    if not torch.is_tensor(translation):
        raise TypeError("Input translation type is not a torch.Tensor. Got {}".format(type(translation)))
    if len(tensor.shape) not in (
        3,
        4,
    ):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix: torch.Tensor = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(
        tensor,
        translation_matrix[..., :2, :3],
        align_corners=align_corners,
        padding_mode=padding_mode,
    )


def rotate(
    tensor: torch.Tensor,
    angle: torch.Tensor,
    center: Union[None, torch.Tensor] = None,
    mode: str = "bilinear",
    align_corners: bool = False,
    padding_mode=TorchPadding.REFLECTION,
) -> torch.Tensor:
    r"""Rotate the image anti-clockwise about the centre.

    See :class:`~kornia.Rotate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}".format(type(angle)))
    if center is not None and not torch.is_tensor(angle):
        raise TypeError("Input center type is not a torch.Tensor. Got {}".format(type(center)))
    if len(tensor.shape) not in (
        3,
        4,
    ):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(
        tensor,
        rotation_matrix[..., :2, :3],
        mode,
        align_corners,
        padding_mode=padding_mode,
    )


def scale(
    tensor: torch.Tensor,
    scale_factor: torch.Tensor,
    center: Union[None, torch.Tensor] = None,
    align_corners: bool = False,
    padding_mode=TorchPadding.REFLECTION,
) -> torch.Tensor:
    r"""Scales the input image.

    See :class:`~kornia.Scale` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}".format(type(tensor)))
    if not torch.is_tensor(scale_factor):
        raise TypeError("Input scale_factor type is not a torch.Tensor. Got {}".format(type(scale_factor)))

    # compute the tensor center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    center = center.expand(tensor.shape[0], -1)
    scale_factor = scale_factor.expand(tensor.shape[0])
    scaling_matrix: torch.Tensor = _compute_scaling_matrix(scale_factor, center)

    # warp using the affine transform
    return affine(
        tensor,
        scaling_matrix[..., :2, :3],
        align_corners=align_corners,
        padding_mode=padding_mode,
    )
