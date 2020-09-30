import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import albumentations as A
import timm
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm

from autoalbument.faster_autoaugment.utils import MAX_VALUES_BY_INPUT_DTYPE
from autoalbument.utils.hydra import get_dataset_filepath
from autoalbument.faster_autoaugment.policy import Policy

torch.backends.cudnn.benchmark = True


class Discriminator(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        num_features = self.base_model.feature_info[-1]["num_chs"]
        self.classifier = nn.Linear(num_features, num_classes)
        self.discriminator = nn.Sequential(
            nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, 1)
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.base_model(input)
        return self.classifier(x), self.discriminator(x).view(-1)


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"value": 0, "batch_count": 0, "avg": 0})

    def update(self, batch_metrics):
        for name, batch_metric in batch_metrics.items():
            self.metrics[name]["value"] += batch_metric.item()
            self.metrics[name]["batch_count"] += 1
            self.metrics[name]["avg"] = self.metrics[name]["value"] / self.metrics[name]["batch_count"]

    def __repr__(self):
        return ", ".join(f'{name}={value["avg"]:.3f}' for name, value in self.metrics.items())


def get_dataset_cls(dataset_file, dataset_cls_name="SearchDataset"):
    dataset_filepath = get_dataset_filepath(dataset_file)
    spec = importlib.util.spec_from_file_location("dataset", dataset_filepath)
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)
    return getattr(dataset, dataset_cls_name)


class FasterAutoAugment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataloader = self.create_dataloader()
        self.models = self.create_models()
        self.optimizers = self.create_optimizers()
        self.loss_tracker = self.create_loss_tracker()
        self.directories = self.create_directories()
        self.epoch = None

    def create_directories(self):
        policy_dir = Path.cwd() / "policy"
        policy_dir.mkdir(exist_ok=True)
        return {"policy": policy_dir}

    def create_loss_tracker(self):
        return MetricTracker()

    def create_dataloader(self):
        dataset_cls = get_dataset_cls(self.cfg.data.dataset_file)
        preprocessing_cfg = self.cfg.data.preprocessing
        input_dtype = self.cfg.data.input_dtype
        transform = A.Compose(
            [
                A.Normalize(
                    mean=preprocessing_cfg.mean,
                    std=preprocessing_cfg.std,
                    max_pixel_value=MAX_VALUES_BY_INPUT_DTYPE[input_dtype],
                ),
                ToTensorV2(),
            ]
        )
        dataset = dataset_cls(transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, **self.cfg.data.dataloader)
        return dataloader

    def create_optimizers(self):
        optimizer_config = self.cfg.optim
        main_optimizer = Adam(
            self.models["main"].parameters(),
            lr=optimizer_config.main_lr,
            betas=(0, 0.999),
        )
        policy_optimizer = Adam(
            self.models["policy"].parameters(),
            lr=optimizer_config.policy_lr,
            betas=(0, 0.999),
        )
        return {
            "main": main_optimizer,
            "policy": policy_optimizer,
        }

    def create_models(self):
        model_cfg = self.cfg.model
        preprocessing_cfg = self.cfg.data.preprocessing
        base_model = timm.create_model(model_cfg.architecture, pretrained=model_cfg.pretrained, num_classes=0)
        main_model = Discriminator(base_model, num_classes=model_cfg.num_classes).to(self.cfg.device)
        policy_model = Policy.faster_auto_augment_policy(
            model_cfg.num_sub_policies,
            model_cfg.temperature,
            model_cfg.operation_count,
            model_cfg.num_chunks,
            mean=torch.tensor(preprocessing_cfg.mean),
            std=torch.tensor(preprocessing_cfg.std),
        ).to(self.cfg.device)
        return {"main": main_model, "policy": policy_model}

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

    def wgan_loss(
        self, n_input: Tensor, n_target: Tensor, a_input: Tensor, a_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ones = n_input.new_tensor(1.0)
        self.models["main"].requires_grad_(True)
        self.models["main"].zero_grad()
        output, n_output = self.models["main"](n_input)
        loss = self.cfg.model.cls_factor * F.cross_entropy(output, n_target)
        loss.backward(retain_graph=True)
        d_n_loss = n_output.mean()
        d_n_loss.backward(-ones)

        with torch.no_grad():
            a_input = self.models["policy"].denormalize_(a_input)
            augmented = self.models["policy"](a_input)

        _, a_output = self.models["main"](augmented)
        d_a_loss = a_output.mean()
        d_a_loss.backward(ones)
        gp = self.cfg.model.gp_factor * self.gradient_penalty(n_input, augmented)
        gp.backward()
        self.optimizers["main"].step()

        self.models["main"].requires_grad_(False)
        self.models["policy"].zero_grad()
        _output, a_output = self.models["main"](self.models["policy"](a_input))
        _loss = self.cfg.model.cls_factor * F.cross_entropy(_output, a_target)
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
        return dict(loss=loss, d_loss=d_loss, a_loss=a_loss)

    def train_epoch(self):
        self.loss_tracker.reset()
        pbar = tqdm(total=len(self.dataloader))
        for input, target in self.dataloader:
            input = input.to(self.cfg.device)
            target = target.to(self.cfg.device)
            loss_dict = self.train_step(input, target)
            self.loss_tracker.update(loss_dict)
            pbar.set_description(f"Epoch: {self.epoch}. {self.loss_tracker}")
            pbar.update()
        pbar.close()

    def save(self):
        t = self.models["policy"].create_transform(input_dtype=self.cfg.data.input_dtype)
        policy_save_path = self.directories["policy"] / f"epoch_{self.epoch}.json"
        A.save(t, str(policy_save_path))

    def search(self):
        for epoch in range(1, self.cfg.optim.epochs + 1):
            self.epoch = epoch
            self.train_epoch()
            self.save()
