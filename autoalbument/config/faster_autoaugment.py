from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from omegaconf import MISSING


@dataclass
class NormalizationConfig:
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class DataloaderConfig:
    batch_size: int = 128
    shuffle: bool = True
    num_workers: int = 4


@dataclass
class DataConfig:
    normalization: NormalizationConfig = NormalizationConfig()
    dataloader: Any = MISSING
    preprocessing: Optional[Any] = None
    dataset_file: Optional[Any] = None
    dataset: Optional[Any] = None
    input_dtype: str = "uint8"


@dataclass
class OptimConfig:
    epochs: int = 20
    main: Any = MISSING
    policy: Any = MISSING


@dataclass
class PolicyModelConfig:
    task_factor: float = 0.1
    gp_factor: float = 10
    temperature: float = 0.05
    num_sub_policies: int = 150
    num_chunks: int = 8
    operation_count: int = 4


@dataclass
class ClassificationModelConfig:
    num_classes: int = MISSING
    architecture: str = "resnet18"
    pretrained: bool = False


@dataclass
class SemanticSegmentationModelConfig:
    num_classes: int = MISSING
    architecture: str = "Unet"
    encoder_architecture: str = "resnet18"
    pretrained: bool = False


@dataclass
class FasterAutoAugmentSearchConfig:
    policy_model: PolicyModelConfig = PolicyModelConfig()
    optim: OptimConfig = OptimConfig(main=MISSING, policy=MISSING)
    device: str = "cuda"
    task: str = MISSING
    cudnn_benchmark: bool = True
    save_checkpoints: bool = False
    checkpoint_path: Optional[str] = None
    tensorboard_logs_dir: Optional[str] = None
    classification_model: Optional[ClassificationModelConfig] = None
    semantic_segmentation_model: Optional[SemanticSegmentationModelConfig] = None
    data: DataConfig = DataConfig(dataloader=MISSING)
    seed: Optional[int] = None
