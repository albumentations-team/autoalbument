import numpy as np
import pytest
import torch

from tests.utils import np_to_pytorch


@pytest.fixture
def image_batches(batch_size=4):
    np_images = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 128, 128, 3)).astype("float32")
    pytorch_batch = np_to_pytorch(np_images)
    return np_images, pytorch_batch


@pytest.fixture
def gradcheck_batch(batch_size=4, num_channels=3, height=4, width=4):
    return torch.rand(batch_size, num_channels, height, width).type(torch.float64)
