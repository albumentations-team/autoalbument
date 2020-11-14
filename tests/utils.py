import random

import torch
import numpy as np


def np_to_pytorch(np_array):
    return torch.from_numpy(np_array.transpose(0, 3, 1, 2))


def torch_to_np(torch_tensor):
    return torch_tensor.numpy().transpose(0, 2, 3, 1)


def assert_batches_match(np_images, pytorch_batch, rtol=1.0e-4, atol=1.0e-8):
    np_batch = torch_to_np(pytorch_batch)
    for np_image, batch_image in zip(np_images, np_batch):
        assert np.allclose(np_image, batch_image, rtol=rtol, atol=atol)
