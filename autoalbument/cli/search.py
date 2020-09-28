import os
import shutil

import hydra
from omegaconf import OmegaConf

from autoalbument.faster_autoaugment.search import FasterAutoAugment
from autoalbument.utils.hydra import get_hydra_config_dir, get_dataset_filepath

OmegaConf.register_resolver("config_dir", get_hydra_config_dir)


@hydra.main(config_name="search")
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    cwd = os.getcwd()
    print(f"Working directory : {cwd}")
    dataset_filepath = get_dataset_filepath(cfg.data.dataset_file)
    shutil.copy2(dataset_filepath, cwd)
    faa = FasterAutoAugment(cfg)
    faa.search()
