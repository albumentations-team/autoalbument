import os
import shutil

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from autoalbument.config.faster_autoaugment import FasterAutoAugmentSearchConfig
from autoalbument.faster_autoaugment.search import FasterAutoAugment
from autoalbument.utils.hydra import get_hydra_config_dir, get_dataset_filepath

OmegaConf.register_resolver("config_dir", get_hydra_config_dir)
cs = ConfigStore.instance()
cs.store(name="search", node=FasterAutoAugmentSearchConfig)


@hydra.main(config_name="search")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    cwd = os.getcwd()
    print(f"Working directory : {cwd}")
    dataset_filepath = get_dataset_filepath(cfg.data.dataset_file)
    shutil.copy2(dataset_filepath, cwd)
    faa = FasterAutoAugment(cfg)
    faa.search()
