import logging

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autoalbument.faster_autoaugment.utils import MAX_VALUES_BY_INPUT_DTYPE
from autoalbument.utils.hydra import get_dataset_cls

log = logging.getLogger(__name__)


class FasterAutoAugmentDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.transform = self.create_transform()
        self.dataset = None

    def prepare_data(self):
        self._instantiate_dataset()

    def setup(self, stage=None):
        self.dataset = self._instantiate_dataset()

    def train_dataloader(self):
        dataloader = instantiate(self.data_cfg.dataloader, dataset=self.dataset)
        return dataloader

    def create_transform(self):
        data_cfg = self.data_cfg
        normalization_config = data_cfg.normalization
        input_dtype = data_cfg.input_dtype
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

    def get_preprocessing_transforms(self):
        preprocessing_config = self.data_cfg.preprocessing
        if not preprocessing_config:
            return []
        preprocessing_config = OmegaConf.to_container(preprocessing_config, resolve=True)
        preprocessing_transforms = []
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

    def _instantiate_dataset(self):
        data_cfg = self.data_cfg
        transform = self.transform
        if getattr(data_cfg, "dataset", None):
            dataset = instantiate(data_cfg.dataset, transform=transform)
        elif getattr(data_cfg, "dataset_file", None):
            dataset_cls = get_dataset_cls(data_cfg.dataset_file)
            dataset = dataset_cls(transform=transform)
        else:
            raise ValueError(f"You should provide a correct dataset in data.dataset, got {data_cfg.dataset}")
        return dataset
