import random
from copy import deepcopy

import albumentations as A
import torch
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn

from autoalbument.faster_autoaugment.operations import (
    CutoutFixedNumerOfHoles,
    CutoutFixedSize,
    HorizontalFlip,
    RandomBrightness,
    RandomContrast,
    Rotate,
    Scale,
    ShiftRGB,
    ShiftX,
    ShiftY,
    Solarize,
    VerticalFlip,
)


class SubPolicyStage(nn.Module):
    def __init__(
        self,
        operations: nn.ModuleList,
        temperature: float,
    ):
        super(SubPolicyStage, self).__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature

    def forward(self, input: Tensor) -> Tensor:
        return (torch.stack([op(input) for op in self.operations]) * self.weights.view(-1, 1, 1, 1, 1)).sum(0)

    @property
    def weights(self):
        return self._weights.div(self.temperature).softmax(0)

    def create_transform(self, input_dtype):
        weights = self.weights.detach().cpu().numpy().tolist()
        probabilities = [op.probability.item() for op in self.operations]
        true_probabilities = [w * p for (w, p) in zip(weights, probabilities)]
        assert sum(true_probabilities) <= 1.0
        transforms = []
        p_sum = 0
        for operation, p in zip(self.operations, true_probabilities):
            transforms.append(operation.create_transform(input_dtype, p))
            p_sum += p
        transforms.append(A.NoOp(p=1.0 - p_sum))
        return OneOf(transforms, p=1)


class SubPolicy(nn.Module):
    def __init__(
        self,
        sub_policy_stage: SubPolicyStage,
        operation_count: int,
    ):
        super(SubPolicy, self).__init__()
        self.stages = nn.ModuleList([deepcopy(sub_policy_stage) for _ in range(operation_count)])

    def forward(self, input: Tensor) -> Tensor:
        for stage in self.stages:
            input = stage(input)
        return input

    def create_transform(self, input_dtype, p):
        return A.Sequential([stage.create_transform(input_dtype) for stage in self.stages], p=p)


class Policy(nn.Module):
    def __init__(
        self,
        operations: nn.ModuleList,
        num_sub_policies: int,
        temperature: float,
        operation_count: int,
        num_chunks: int,
        mean: Tensor,
        std: Tensor,
    ):
        super(Policy, self).__init__()
        self.sub_policies = nn.ModuleList(
            [SubPolicy(SubPolicyStage(operations, temperature), operation_count) for _ in range(num_sub_policies)]
        )
        self.num_sub_policies = num_sub_policies
        self.temperature = temperature
        self.operation_count = operation_count
        self.num_chunks = num_chunks
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

        for p in self.parameters():
            nn.init.uniform_(p, 0, 1)

    def forward(self, input: Tensor) -> Tensor:
        # [0, 1] -> [-1, 1]
        if self.num_chunks > 1:
            out = [self._forward(inp) for inp in input.chunk(self.num_chunks)]
            x = torch.cat(out, dim=0)
        else:
            x = self._forward(input)

        return self.normalize_(x)

    def _forward(self, input: Tensor) -> Tensor:
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](input)

    def normalize_(self, input):
        # [0, 1] -> [-1, 1]
        return input.add_(-self._mean[:, None, None]).div_(self._std[:, None, None])

    def denormalize_(self, input):
        # [-1, 1] -> [0, 1]
        return input.mul_(self._std[:, None, None]).add_(self._mean[:, None, None])

    @staticmethod
    def dda_operations(temperature=0.05):
        return [
            ShiftRGB(shift_r=True, temperature=temperature),
            ShiftRGB(shift_g=True, temperature=temperature),
            ShiftRGB(shift_b=True, temperature=temperature),
            RandomBrightness(temperature=temperature),
            RandomContrast(temperature=temperature),
            Solarize(temperature=temperature),
            HorizontalFlip(temperature=temperature),
            VerticalFlip(temperature=temperature),
            Rotate(temperature=temperature),
            ShiftX(temperature=temperature),
            ShiftY(temperature=temperature),
            Scale(temperature=temperature),
            CutoutFixedNumerOfHoles(temperature=temperature),
            CutoutFixedSize(temperature=temperature),
        ]

    @staticmethod
    def faster_auto_augment_policy(
        num_sub_policies: int,
        temperature: float,
        operation_count: int,
        num_chunks: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        return Policy(
            nn.ModuleList(Policy.dda_operations(temperature)),
            num_sub_policies,
            temperature,
            operation_count,
            num_chunks,
            mean=mean,
            std=std,
        )

    def create_transform(self, input_dtype="float32"):
        return Compose(
            [
                OneOf([sp.create_transform(input_dtype, p=1 / len(self.sub_policies)) for sp in self.sub_policies], p=1),
                A.Normalize(mean=self._mean.tolist(), std=self._std.tolist()),
                ToTensorV2(),
            ]
        )
