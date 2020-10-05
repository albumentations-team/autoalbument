import albumentations.augmentations.functional as F
import pytest
import torch
from torch.autograd import gradcheck

import autoalbument.faster_autoaugment.albumentations_pytorch.functional as PF
from tests.utils import assert_batches_match


class Base:
    def scalar_to_tensor(self, arg, requires_grad=False, dtype=torch.float32):
        if arg is None:
            return None
        return torch.tensor(arg, requires_grad=requires_grad, dtype=dtype)

    def test_albumentations_match(self, image_batches, arg):
        np_images, pytorch_batch = image_batches
        tensor_arg = self.scalar_to_tensor(arg)
        augmented_np_images = [self.albumentations_fn(image, arg) for image in np_images]
        augmented_pytorch_batch = self.albumentations_pytorch_fn(pytorch_batch, tensor_arg)
        assert_batches_match(augmented_np_images, augmented_pytorch_batch)

    def test_gradients(self, gradcheck_batch, arg):
        tensor_arg = self.scalar_to_tensor(arg, requires_grad=True, dtype=torch.float64)
        gradcheck(self.albumentations_pytorch_fn, (gradcheck_batch, tensor_arg))

    def albumentations_fn(self, image, arg):
        raise NotImplementedError

    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        raise NotImplementedError


@pytest.mark.parametrize("arg", [0.2, 0.4, 0.8])
class TestSolarize(Base):
    def albumentations_fn(self, image, arg):
        return F.solarize(image, threshold=arg)

    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.solarize(pytorch_batch, threshold=arg)

    def test_gradients(self, gradcheck_batch, arg):
        pass


@pytest.mark.parametrize("arg", [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [0.0, 0.7, -0.2]])
class TestShiftRgb(Base):
    def albumentations_fn(self, image, arg):
        return F.shift_rgb(image, r_shift=arg[0], g_shift=arg[1], b_shift=arg[2])

    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.shift_rgb(pytorch_batch, r_shift=arg[0], g_shift=arg[1], b_shift=arg[2])


@pytest.mark.parametrize("arg", [-1.0, 0.1, 0.5, 1.0])
class TestBrightnessAdjust(Base):
    def albumentations_fn(self, image, arg):
        return F.brightness_contrast_adjust(image, beta=arg, beta_by_max=True)

    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.brightness_adjust(pytorch_batch, beta=arg)


@pytest.mark.parametrize("arg", [-1.0, 0.1, 0.5, 1.0])
class TestContrastAdjust(Base):
    def albumentations_fn(self, image, arg):
        return F.brightness_contrast_adjust(image, alpha=arg)

    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.contrast_adjust(pytorch_batch, alpha=arg)


@pytest.mark.parametrize("arg", [None])
class TestVflip(Base):
    def albumentations_fn(self, image, arg):
        return F.vflip(image)

    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.vflip(pytorch_batch)

    def test_gradients(self, gradcheck_batch, arg):
        pass


@pytest.mark.parametrize("arg", [None])
class TestHflip(Base):
    def albumentations_fn(self, image, arg):
        return F.hflip(image)

    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.hflip(pytorch_batch)

    def test_gradients(self, gradcheck_batch, arg):
        pass


@pytest.mark.parametrize(
    "arg",
    [
        [0.01],
        [-0.5],
        [0.5],
        [1.0 - 1e-6],
        [-1.0 + 1e-6],
    ],
)
class TestShiftX(Base):
    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.shift_x(pytorch_batch, dx=arg)

    def test_albumentations_match(self, image_batches, arg):
        pass


@pytest.mark.parametrize(
    "arg",
    [
        [0.01],
        [-0.5],
        [0.5],
        [1.0 - 1e-6],
        [-1.0 + 1e-6],
    ],
)
class TestShiftY(Base):
    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.shift_y(pytorch_batch, dy=arg)

    def test_albumentations_match(self, image_batches, arg):
        pass


@pytest.mark.parametrize(
    "arg",
    [
        [0.1],
        [-0.5],
        [0.5],
        [1.0 - 1e-6],
        [-1.0 + 1e-6],
    ],
)
class TestScale(Base):
    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.scale(pytorch_batch, scale=arg)

    def test_albumentations_match(self, image_batches, arg):
        pass


@pytest.mark.parametrize(
    "arg",
    [
        [0.1],
        [-0.5],
        [0.5],
        [1.0 - 1e-6],
        [-1.0 + 1e-6],
    ],
)
class TestRotate(Base):
    def albumentations_pytorch_fn(self, pytorch_batch, arg):
        return PF.rotate(pytorch_batch, angle=arg)

    def test_albumentations_match(self, image_batches, arg):
        pass
