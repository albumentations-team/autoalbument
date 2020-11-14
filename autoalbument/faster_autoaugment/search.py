"""
Based on the official implementation of Faster AutoAugment
https://github.com/moskomule/dda/blob/master/faster_autoaugment/search.py
"""

import copy
import logging
import os
from pathlib import Path
from typing import Tuple

import albumentations as A
import timm
import torch
from albumentations.pytorch import ToTensorV2
from hydra.utils import instantiate
from torch import Tensor, nn
from torch.nn import Flatten
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoalbument.faster_autoaugment.metrics import get_average_parameter_change
from autoalbument.faster_autoaugment.utils import MAX_VALUES_BY_INPUT_DTYPE, get_dataset_cls, MetricTracker, set_seed
from autoalbument.utils.files import symlink
from autoalbument.faster_autoaugment.policy import Policy
import segmentation_models_pytorch as smp

log = logging.getLogger(__name__)


class ClassificationDiscriminator(nn.Module):
    def __init__(self, architecture, pretrained, num_classes):
        super().__init__()
        self.base_model = timm.create_model(architecture, pretrained=pretrained)
        self.base_model.reset_classifier(num_classes)
        self.classifier = self.base_model.get_classifier()
        num_features = self.classifier.in_features
        self.discriminator = nn.Sequential(
            nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, 1)
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.base_model.forward_features(input)
        x = self.base_model.global_pool(x).flatten(1)
        return self.classifier(x), self.discriminator(x).view(-1)


class SegmentationDiscriminator(nn.Module):
    def __init__(self, architecture, encoder_architecture, num_classes, pretrained):
        super().__init__()
        model = getattr(smp, architecture)

        self.base_model = model(
            encoder_architecture, encoder_weights=self._get_encoder_weights(pretrained), classes=num_classes
        )
        num_features = self.base_model.encoder.out_channels[-1]
        self.base_model.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, 1),
        )

    @staticmethod
    def _get_encoder_weights(pretrained):
        if isinstance(pretrained, bool):
            return "imagenet" if pretrained else None
        return pretrained

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        mask, discriminator_output = self.base_model(input)
        return mask, discriminator_output.view(-1)


class FasterAutoAugmentBase:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.backends.cudnn.benchmark = self.cfg.cudnn_benchmark
        self.start_epoch = 1
        self.set_seed()
        self.dataloader = self.create_dataloader()
        self.models = self.create_models()
        self.optimizers = self.create_optimizers()
        self.loss = self.create_loss()
        self.metric_tracker = self.create_metric_tracker()
        self.paths = self.create_directories()
        self.tensorboard_writer = self.create_tensorboard_writer()
        self.epoch = None
        if self.cfg.checkpoint_path:
            self.load_checkpoint()
        self.saved_policy_state_dict = self.get_policy_state_dict()

    def set_seed(self):
        seed = getattr(self.cfg, "seed", None)
        if seed is not None:
            set_seed(seed)

    def create_tensorboard_writer(self):
        if self.cfg.tensorboard_logs_dir:
            return SummaryWriter(os.path.join(self.cfg.tensorboard_logs_dir, os.getcwd().replace(os.sep, ".")))
        return None

    def get_policy_state_dict(self):
        state_dict = copy.deepcopy(self.models["policy"].state_dict())
        for k, v in state_dict.items():
            state_dict[k] = v.to("cpu")
        return state_dict

    def create_directories(self):
        policy_dir = Path.cwd() / "policy"
        policy_dir.mkdir(exist_ok=True)
        if self.cfg.save_checkpoints:
            checkpoints_dir = Path.cwd() / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)
        else:
            checkpoints_dir = None
        return {
            "policy_dir": policy_dir,
            "checkpoints_dir": checkpoints_dir,
            "latest_policy_path": policy_dir / "latest.json",
            "latest_checkpoint_path": checkpoints_dir / "latest.pth" if checkpoints_dir is not None else None,
        }

    def get_latest_policy_save_path(self):
        return

    def create_metric_tracker(self):
        return MetricTracker()

    def get_preprocessing_transforms(self):
        preprocessing_config = self.cfg.data.preprocessing
        preprocessing_transforms = []
        if preprocessing_config:
            for preprocessing_transform in preprocessing_config:
                for transform_name, transform_args in preprocessing_transform.items():
                    transform = A.from_dict(
                        {
                            "transform": {
                                "__class_fullname__": "albumentations.augmentations.transforms." + transform_name,
                                **transform_args,
                            }
                        }
                    )
                    preprocessing_transforms.append(transform)
        return preprocessing_transforms

    def create_preprocessing_transform(self):
        normalization_config = self.cfg.data.normalization
        input_dtype = self.cfg.data.input_dtype
        transform = A.Compose(
            [
                *self.get_preprocessing_transforms(),
                A.Normalize(
                    mean=normalization_config.mean,
                    std=normalization_config.std,
                    max_pixel_value=MAX_VALUES_BY_INPUT_DTYPE[input_dtype],
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
        log.info(f"Preprocessing transform:\n{transform}")
        return transform

    def create_dataloader(self):
        transform = self.create_preprocessing_transform()

        data_config = self.cfg.data

        if getattr(data_config, "dataset", None):
            dataset = instantiate(data_config.dataset, transform=transform)
        elif getattr(data_config, "dataset_file", None):
            dataset_cls = get_dataset_cls(self.cfg.data.dataset_file)
            dataset = dataset_cls(transform=transform)
        else:
            raise ValueError(f"You should provide a correct dataset in data.dataset, got {data_config.dataset}")

        dataloader = instantiate(self.cfg.data.dataloader, dataset=dataset)
        return dataloader

    def create_optimizers(self):
        optimizer_config = self.cfg.optim
        main_optimizer = instantiate(optimizer_config.main, params=self.models["main"].parameters())
        policy_optimizer = instantiate(optimizer_config.policy, params=self.models["policy"].parameters())
        return {
            "main": main_optimizer,
            "policy": policy_optimizer,
        }

    def create_main_model(self):
        raise NotImplementedError

    def create_policy_model(self):
        policy_model_cfg = self.cfg.policy_model
        normalization_cfg = self.cfg.data.normalization
        policy_model = Policy.faster_auto_augment_policy(
            policy_model_cfg.num_sub_policies,
            policy_model_cfg.temperature,
            policy_model_cfg.operation_count,
            policy_model_cfg.num_chunks,
            mean=torch.tensor(normalization_cfg.mean),
            std=torch.tensor(normalization_cfg.std),
        ).to(self.cfg.device)
        return policy_model

    def create_models(self):
        main_model = self.create_main_model()
        policy_model = self.create_policy_model()
        return {"main": main_model, "policy": policy_model}

    def policy_forward_for_policy_train(self, a_input, a_target):
        raise NotImplementedError

    def gradient_penalty(self, real: Tensor, fake: Tensor) -> Tensor:
        alpha = real.new_empty(real.size(0), 1, 1, 1).uniform_(0, 1)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_()
        _, output = self.models["main"](interpolated)
        grad = torch.autograd.grad(
            outputs=output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return (grad.norm(2, dim=1) - 1).pow(2).mean()

    def create_loss(self):
        return nn.CrossEntropyLoss().to(self.cfg.device)

    def wgan_loss(
        self, n_input: Tensor, n_target: Tensor, a_input: Tensor, a_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ones = n_input.new_tensor(1.0)
        self.models["main"].requires_grad_(True)
        self.models["main"].zero_grad()
        output, n_output = self.models["main"](n_input)
        loss = self.cfg.policy_model.task_factor * self.loss(output, n_target)
        loss.backward(retain_graph=True)
        d_n_loss = n_output.mean()
        d_n_loss.backward(-ones)

        with torch.no_grad():
            a_input = self.models["policy"].denormalize_(a_input)
            augmented = self.models["policy"]({"image_batch": a_input})["image_batch"]

        _, a_output = self.models["main"](augmented)
        d_a_loss = a_output.mean()
        d_a_loss.backward(ones)
        gp = self.cfg.policy_model.gp_factor * self.gradient_penalty(n_input, augmented)
        gp.backward()
        self.optimizers["main"].step()

        self.models["main"].requires_grad_(False)
        self.models["policy"].zero_grad()

        augmented_input, maybe_augmented_target = self.policy_forward_for_policy_train(a_input, a_target)

        _output, a_output = self.models["main"](augmented_input)
        _loss = self.cfg.policy_model.task_factor * self.loss(_output, maybe_augmented_target)
        _loss.backward(retain_graph=True)
        a_loss = a_output.mean()
        a_loss.backward(-ones)
        self.optimizers["policy"].step()
        return loss + _loss, -d_n_loss + d_a_loss + gp, -a_loss

    def train_step(self, input, target):
        b = input.size(0) // 2
        a_input, a_target = input[:b], target[:b]
        n_input, n_target = input[b:], target[b:]
        loss, d_loss, a_loss = self.wgan_loss(n_input, n_target, a_input, a_target)
        return {
            "loss": loss,
            "d_loss": d_loss,
            "a_loss": a_loss,
        }

    def get_progress_bar_description(self):
        return f"Epoch: {self.epoch}. {self.metric_tracker}"

    def train_epoch(self):
        self.metric_tracker.reset()
        pbar = tqdm(total=len(self.dataloader))
        for input, target in self.dataloader:
            input = input.to(self.cfg.device)
            target = target.to(self.cfg.device)
            loss_dict = self.train_step(input, target)
            self.metric_tracker.update(loss_dict)
            pbar.set_description(self.get_progress_bar_description())
            pbar.update()

        average_parameter_change = get_average_parameter_change(
            self.saved_policy_state_dict, self.get_policy_state_dict()
        )
        self.saved_policy_state_dict = self.get_policy_state_dict()
        self.metric_tracker.update({"Average Parameter change": average_parameter_change})
        pbar.set_description(self.get_progress_bar_description())
        pbar.close()

    def write_tensorboard_logs(self):
        if not self.tensorboard_writer:
            return
        for metric, avg_value in self.metric_tracker.get_avg_values():
            self.tensorboard_writer.add_scalar(metric, avg_value, self.epoch)

    def save_policy(self):

        transform = self.models["policy"].create_transform(
            input_dtype=self.cfg.data.input_dtype,
            preprocessing_transforms=self.get_preprocessing_transforms(),
        )
        policy_save_path = self.paths["policy_dir"] / f"epoch_{self.epoch}.json"
        A.save(transform, str(policy_save_path))
        symlink(policy_save_path, self.paths["latest_policy_path"])
        log.info(
            f"Policy is saved to {policy_save_path}. "
            f"{self.paths['latest_policy_path']} now also points to this policy file."
        )

    def save_checkpoint(self):
        checkpoint = {
            "epoch": self.epoch,
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
        }
        checkpoint_save_path = self.paths["checkpoints_dir"] / f"epoch_{self.epoch}.pth"
        torch.save(checkpoint, checkpoint_save_path)
        symlink(checkpoint_save_path, self.paths["latest_checkpoint_path"])
        log.info(
            f"Checkpoint is saved to {checkpoint_save_path}. "
            f"{self.paths['latest_checkpoint_path']} now also points to this checkpoint file."
        )

    def load_checkpoint(self):
        checkpoint_path = self.cfg.checkpoint_path
        log.info(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.cfg.device)
        for name, model in self.models.items():
            model.load_state_dict(checkpoint["models"][name])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][name])
        self.start_epoch = checkpoint["epoch"] + 1
        self.saved_policy_state_dict = self.get_policy_state_dict()

    def save(self):
        self.save_policy()
        if self.cfg.save_checkpoints:
            self.save_checkpoint()

    def get_search_summary(self):
        summary = [
            f"\n\nSearch is finished.\n"
            f"- Configuration files for policies found at each epoch are located in {self.paths['policy_dir']}."
        ]
        if self.cfg.save_checkpoints:
            summary.append(
                f"- Checkpoint files for models and optimizers at each epoch are located in "
                f"{self.paths['checkpoints_dir']}."
            )

        latest_policy_path = self.paths["latest_policy_path"]
        load_command = f'transform = A.load("{latest_policy_path}")\n'

        separator = "-" * len(load_command)
        if self.cfg.task == "classification":
            transform_text = "transformed = transform(image=image)\n" 'transformed_image = transformed["image"]\n'
        else:
            transform_text = (
                "transformed = transform(image=image, mask=mask)\n"
                'transformed_image = transformed["image"]\n'
                'transformed_mask = transformed["mask"]\n'
            )

        summary.append(
            f"\nUse Albumentations to load the found policies and transform images:\n"
            f"{separator}\n"
            f"import albumentations as A\n\n"
            f'transform = A.load("{latest_policy_path}")\n' + transform_text + f"{separator}\n"
        )

        return "\n".join(summary)

    def search(self):
        for epoch in range(self.start_epoch, self.cfg.optim.epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            self.write_tensorboard_logs()
            self.save()

        log.info(self.get_search_summary())


class FAAClassification(FasterAutoAugmentBase):
    def create_main_model(self):
        model_cfg = self.cfg.classification_model
        main_model = ClassificationDiscriminator(
            model_cfg.architecture, num_classes=model_cfg.num_classes, pretrained=model_cfg.pretrained
        ).to(self.cfg.device)
        return main_model

    def create_loss(self):
        return nn.CrossEntropyLoss().to(self.cfg.device)

    def policy_forward_for_policy_train(self, a_input, a_target):
        output = self.models["policy"]({"image_batch": a_input})["image_batch"]
        return output, a_target


class FAASemanticSegmentation(FasterAutoAugmentBase):
    def create_main_model(self):
        model_cfg = self.cfg.semantic_segmentation_model
        main_model = SegmentationDiscriminator(
            model_cfg.architecture,
            encoder_architecture=model_cfg.encoder_architecture,
            num_classes=model_cfg.num_classes,
            pretrained=model_cfg.pretrained,
        ).to(self.cfg.device)
        return main_model

    def create_loss(self):
        return nn.BCEWithLogitsLoss().to(self.cfg.device)

    def policy_forward_for_policy_train(self, a_input, a_target):
        output = self.models["policy"]({"image_batch": a_input, "mask_batch": a_target})
        return output["image_batch"], output["mask_batch"]


def get_faa_seacher(cfg):
    task = cfg.task
    if task == "semantic_segmentation":
        return FAASemanticSegmentation(cfg)
    elif task == "classification":
        return FAAClassification(cfg)
    raise ValueError(f"Unsupported task: {task}. Supported tasks: classification, semantic_segmentation.")
