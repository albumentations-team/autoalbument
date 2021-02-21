"""
Based on the official implementation of Faster AutoAugment
https://github.com/moskomule/dda/blob/master/faster_autoaugment/policy.py
"""

import random
import warnings
from copy import deepcopy
from typing import Dict

import albumentations as A
import torch
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn

from autoalbument.faster_autoaugment.utils import (
    MAX_VALUES_BY_INPUT_DTYPE,
    target_requires_grad,
)

PROBABILITY_EPS = 0.01


class SubPolicyStage(nn.Module):
    def __init__(
        self,
        operations: nn.ModuleList,
        temperature: float,
    ):
        super().__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        targets = input.keys()
        operation_outputs = [op(input) for op in self.operations]
        output = {}
        for target in targets:
            with torch.set_grad_enabled(target_requires_grad(target)):
                output[target] = (
                    torch.stack([op[target] for op in operation_outputs]) * self.weights.view(-1, 1, 1, 1, 1)
                ).sum(0)
        return output

    @property
    def weights(self):
        return self._weights.div(self.temperature).softmax(0)

    def create_transform(self, input_dtype):
        weights = self.weights.detach().cpu().numpy().tolist()
        probabilities = [op.probability.item() for op in self.operations]
        true_probabilities = [w * p for (w, p) in zip(weights, probabilities)]
        p_sum = sum(true_probabilities)
        if p_sum > 1.0 + PROBABILITY_EPS:
            warnings.warn(
                f"Sum of all augmentation probabilities exceeds 1.0 and equals {p_sum}. "
                "This may indicate an error in AutoAlbument. "
                "Please report an issue at https://github.com/albumentations-team/autoalbument/issues.",
                RuntimeWarning,
            )
        transforms = []
        for operation, p in zip(self.operations, true_probabilities):
            transforms.append(operation.create_transform(input_dtype, p))
        if p_sum < 1.0:
            transforms.append(A.NoOp(p=1.0 - p_sum))
        return OneOf(transforms, p=1.0)


class SubPolicy(nn.Module):
    def __init__(
        self,
        sub_policy_stage: SubPolicyStage,
        operation_count: int,
    ):
        super().__init__()
        self.stages = nn.ModuleList([deepcopy(sub_policy_stage) for _ in range(operation_count)])

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
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
        super().__init__()
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

    def forward(self, input):
        targets = input.keys()

        chunked_input = {}
        num_chunks = None
        for target in targets:
            chunked_input[target] = input[target].chunk(self.num_chunks)
            num_chunks = len(chunked_input[target])

        prepared_chunked_input = []
        for i in range(num_chunks):
            prepared_chunked_input.append({target: chunked_input[target][i] for target in targets})

        if self.num_chunks > 1:
            output = {}
            out_chunks = [self._forward(inp) for inp in prepared_chunked_input]
            for target in targets:
                output[target] = torch.cat([out_chunk[target] for out_chunk in out_chunks], dim=0)
        else:
            output = self._forward(input)

        output["image_batch"] = self.normalize_(output["image_batch"])
        return output

    def _forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](input)

    def normalize_(self, input):
        # [0, 1] -> [-1, 1]
        return input.add_(-self._mean[:, None, None]).div_(self._std[:, None, None])

    def denormalize_(self, input):
        # [-1, 1] -> [0, 1]
        return input.mul_(self._std[:, None, None]).add_(self._mean[:, None, None])

    @staticmethod
    def faster_auto_augment_policy(
        operations,
        num_sub_policies: int,
        temperature: float,
        operation_count: int,
        num_chunks: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        return Policy(
            nn.ModuleList(operations),
            num_sub_policies,
            temperature,
            operation_count,
            num_chunks,
            mean=mean,
            std=std,
        )

    def create_transform(self, input_dtype="float32", preprocessing_transforms=()):
        sub_policy_p = 1 / len(self.sub_policies)
        return Compose(
            [
                *preprocessing_transforms,
                OneOf([sp.create_transform(input_dtype, p=sub_policy_p) for sp in self.sub_policies], p=1),
                A.Normalize(
                    mean=self._mean.tolist(),
                    std=self._std.tolist(),
                    max_pixel_value=MAX_VALUES_BY_INPUT_DTYPE[input_dtype],
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
