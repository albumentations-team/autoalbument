from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class PreprocessingConfig:
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class DataloaderConfig:
    batch_size: int = 128
    shuffle: bool = True
    num_workers: int = 4


@dataclass
class DataConfig:
    preprocessing: PreprocessingConfig
    dataloader: DataloaderConfig
    dataset_file: str = "dataset.py"
    input_dtype: str = "uint8"


@dataclass
class OptimConfig:
    epochs: int = 20
    main_lr: float = 1e-3
    policy_lr: float = 1e-3


@dataclass
class ModelConfig:
    num_classes: int = MISSING
    architecture: str = "resnet18"
    pretrained: bool = False
    cls_factor: float = 0.1
    gp_factor: float = 10
    temperature: float = 0.05
    num_sub_policies: int = 150
    num_chunks: int = 8
    operation_count: int = 4


@dataclass
class FasterAutoAugmentSearchConfig:
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    device: str = "cuda"
    cudnn_benchmark: bool = True
    save_checkpoints: bool = False
    checkpoint_path: Optional[str] = None
