"""
Based on the official implementation of Faster AutoAugment
https://github.com/moskomule/dda/blob/master/faster_autoaugment/search.py
"""

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.nn import functional as F

from autoalbument.faster_autoaugment.models.policy_model import Policy


class FAABaseModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.cfg = cfg
        self.main_model = self.create_main_model()
        self.policy_model = self.create_policy_model()

    def configure_optimizers(self):
        optimizer_config = self.cfg.optim
        main_optimizer = instantiate(optimizer_config.main, params=self.main_model.parameters())
        policy_optimizer = instantiate(optimizer_config.policy, params=self.policy_model.parameters())
        return main_optimizer, policy_optimizer

    def create_main_model(self):
        model_cfg = self.get_main_model_cfg()
        main_model = instantiate(model_cfg)
        return main_model

    def create_policy_model(self):
        policy_model_cfg = self.cfg.policy_model
        normalization_cfg = self.cfg.data.normalization
        policy_operations = [
            instantiate(operation, temperature=policy_model_cfg.temperature)
            for operation in policy_model_cfg.operations
        ]

        policy_model = Policy.faster_auto_augment_policy(
            policy_operations,
            policy_model_cfg.num_sub_policies,
            policy_model_cfg.temperature,
            policy_model_cfg.operation_count,
            policy_model_cfg.num_chunks,
            mean=torch.tensor(normalization_cfg.mean),
            std=torch.tensor(normalization_cfg.std),
        )
        return policy_model

    def gradient_penalty(self, real: Tensor, fake: Tensor) -> Tensor:
        alpha = real.new_empty(real.size(0), 1, 1, 1).uniform_(0, 1)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_()
        _, output = self.main_model(interpolated)
        grad = torch.autograd.grad(
            outputs=output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return (grad.norm(2, dim=1) - 1).pow(2).mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        input, target = batch
        b = input.size(0) // 2
        a_input, a_target = input[:b], target[:b]
        n_input, n_target = input[b:], target[b:]

        main_optimizer, policy_optimizer = self.optimizers(use_pl_optimizer=True)

        ones = n_input.new_tensor(1.0)
        self.main_model.requires_grad_(True)
        self.main_model.zero_grad()
        output, n_output = self.main_model(n_input)
        loss = self.cfg.policy_model.task_factor * self.criterion(output, n_target)
        self.manual_backward(loss, main_optimizer, retain_graph=True)

        d_n_loss = n_output.mean()

        self.manual_backward(d_n_loss.unsqueeze(0), main_optimizer, -ones.unsqueeze(0))

        with torch.no_grad():
            a_input = self.policy_model.denormalize_(a_input)
            augmented = self.policy_model({"image_batch": a_input})["image_batch"]

        _, a_output = self.main_model(augmented)
        d_a_loss = a_output.mean()

        self.manual_backward(d_a_loss.unsqueeze(0), main_optimizer, ones.unsqueeze(0))

        gp = self.cfg.policy_model.gp_factor * self.gradient_penalty(n_input, augmented)
        self.manual_backward(gp, main_optimizer)
        main_optimizer.step()

        self.main_model.requires_grad_(False)
        self.policy_model.zero_grad()
        augmented_input, maybe_augmented_target = self.policy_forward_for_policy_train(a_input, a_target)
        _output, a_output = self.main_model(augmented_input)

        _loss = self.cfg.policy_model.task_factor * self.criterion(_output, maybe_augmented_target)
        self.manual_backward(_loss, policy_optimizer, retain_graph=True)

        a_loss = a_output.mean()
        self.manual_backward(a_loss.unsqueeze(0), policy_optimizer, -ones.unsqueeze(0))

        policy_optimizer.step()
        with torch.no_grad():
            metrics = {
                "loss": loss + _loss,
                "d_loss": -d_n_loss + d_a_loss + gp,
                "a_loss": -a_loss,
            }
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def get_main_model_cfg(self):
        raise NotImplementedError

    def criterion(self, input, target):
        raise NotImplementedError

    def policy_forward_for_policy_train(self, a_input, a_target):
        raise NotImplementedError


class FAAClassificationModel(FAABaseModel):
    def get_main_model_cfg(self):
        return self.cfg.classification_model

    def criterion(self, input, target):
        return F.cross_entropy(input, target)

    def policy_forward_for_policy_train(self, a_input, a_target):
        output = self.policy_model({"image_batch": a_input})["image_batch"]
        return output, a_target


class FAASemanticSegmentationModel(FAABaseModel):
    def get_main_model_cfg(self):
        return self.cfg.semantic_segmentation_model

    def criterion(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target)

    def policy_forward_for_policy_train(self, a_input, a_target):
        output = self.policy_model({"image_batch": a_input, "mask_batch": a_target})
        return output["image_batch"], output["mask_batch"]
