from hydra.utils import instantiate
from pytorch_lightning import seed_everything

from autoalbument.faster_autoaugment.datamodule import FasterAutoAugmentDataModule
from autoalbument.faster_autoaugment.models.faa_model import (
    FAAClassificationModel,
    FAASemanticSegmentationModel,
)
from autoalbument.search_interface import SearcherBase


class FasterAutoAugmentSearcher(SearcherBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.set_seed()
        self.model = self.create_model()
        self.datamodule = self.create_datamodule()
        self.trainer = self.create_trainer()

    def set_seed(self):
        seed = getattr(self.cfg, "seed", None)
        if seed is None:
            return
        seed_everything(seed)

    def create_model(self):
        cfg = self.cfg
        task = cfg.task
        if task == "semantic_segmentation":
            return FAASemanticSegmentationModel(cfg)
        elif task == "classification":
            return FAAClassificationModel(cfg)
        raise ValueError(f"Unsupported task: {task}. Supported tasks: classification, semantic_segmentation.")

    def create_datamodule(self):
        datamodule = FasterAutoAugmentDataModule(self.cfg.data)
        return datamodule

    def create_logger(self):
        logger = instantiate(self.cfg.logger)
        return logger

    def create_callbacks(self):
        callbacks = self.cfg.callbacks
        if not callbacks:
            return []
        return [instantiate(callback) for callback in callbacks]

    def create_trainer_additional_params(self):
        logger = self.create_logger()
        callbacks = self.create_callbacks()
        return {
            "logger": logger,
            "callbacks": callbacks,
        }

    def create_trainer(self):
        cfg = self.cfg
        additional_params = self.create_trainer_additional_params()
        trainer = instantiate(cfg.trainer, **additional_params)
        return trainer

    def search(self):
        self.trainer.fit(self.model, datamodule=self.datamodule)
