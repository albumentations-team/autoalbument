import torch
import torch.nn.functional as F


def get_scaling_matrix(img_batch, dx=0, dy=0):
    height, width = img_batch.shape[-2:]
    matrix = torch.eye(2, 3, dtype=img_batch.dtype, device=img_batch.device)
    matrix[0, 2] = dx * width
    matrix[1, 2] = dy * height
    return matrix


def get_normalization_matrix(img_batch):
    height, width = img_batch.shape[-2:]
    matrix = torch.zeros(3, 3, dtype=img_batch.dtype, device=img_batch.device)
    matrix[0, 0] = 2.0 / width
    matrix[0, 1] = 0
    matrix[1, 1] = 2.0 / height
    matrix[1, 0] = 0
    matrix[0, -1] = -1.0
    matrix[1, -1] = -1.0
    matrix[-1, -1] = 1.0
    return matrix


def get_rotation_matrix(img_batch, angle=None, scale=None):
    matrix = torch.zeros(2, 3, dtype=img_batch.dtype, device=img_batch.device)
    height, width = img_batch.shape[-2:]
    if angle is None:
        angle = torch.tensor([0.0], dtype=img_batch.dtype, device=img_batch.device)
    if scale is None:
        scale = torch.tensor([1.0], dtype=img_batch.dtype, device=img_batch.device)

    center_x = width / 2
    center_y = height / 2
    angle_in_radians = torch.deg2rad(angle)
    alpha = torch.cos(angle_in_radians) * scale
    beta = torch.sin(angle_in_radians) * scale
    matrix[0, 0] = alpha
    matrix[0, 1] = beta
    matrix[0, 2] = (1.0 - alpha) * center_x - beta * center_y
    matrix[1, 0] = -beta
    matrix[1, 1] = alpha
    matrix[1, 2] = beta * center_x + (1.0 - alpha) * center_y
    return matrix


def convert_2x3_affine_matrix_to_3x3(matrix):
    pad = torch.tensor([[0, 0, 1]], dtype=matrix.dtype, device=matrix.device)
    return torch.cat([matrix, pad])


def expand_matrix_at_batch_dimension(matrix, batch_size):
    return matrix.expand(batch_size, -1, -1)


def warp_affine(img_batch, affine_matrix, padding_mode, align_corners=True):
    matrix = convert_2x3_affine_matrix_to_3x3(affine_matrix)
    normalization_matrix = get_normalization_matrix(img_batch)

    batch_size = img_batch.shape[0]
    matrix = expand_matrix_at_batch_dimension(matrix, batch_size)
    normalization_matrix = expand_matrix_at_batch_dimension(normalization_matrix, batch_size)

    inverse_normalization_matrix = torch.inverse(normalization_matrix)
    theta_3x3 = torch.inverse(normalization_matrix @ matrix @ inverse_normalization_matrix)
    theta = theta_3x3[:, :2, :]

    grid = F.affine_grid(theta, img_batch.shape, align_corners=align_corners)
    return F.grid_sample(img_batch, grid, padding_mode=padding_mode, align_corners=align_corners)
