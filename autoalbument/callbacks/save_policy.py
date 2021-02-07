import logging
import os
import shutil

import albumentations as A
from pytorch_lightning import Callback

__all__ = [
    "SavePolicy",
]


log = logging.getLogger(__name__)


class SavePolicy(Callback):
    def __init__(self, dirpath=None, latest_policy_filename="latest.json"):
        self.dirpath = dirpath or os.path.join(os.getcwd(), "policy")
        self.latest_policy_filepath = os.path.join(self.dirpath, latest_policy_filename)
        os.makedirs(self.dirpath, exist_ok=True)

    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        datamodule = trainer.datamodule
        cfg = pl_module.cfg
        transform = pl_module.policy_model.create_transform(
            input_dtype=cfg.data.input_dtype,
            preprocessing_transforms=datamodule.get_preprocessing_transforms(),
        )
        policy_file_filepath = os.path.join(self.dirpath, f"epoch_{epoch}.json")
        A.save(transform, policy_file_filepath)
        shutil.copy2(policy_file_filepath, self.latest_policy_filepath)
        log.info(
            f"Policy is saved to {policy_file_filepath}. "
            f"{self.latest_policy_filepath} now also contains this policy."
        )
