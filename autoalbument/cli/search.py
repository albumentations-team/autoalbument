import copy
import os
import sys

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from autoalbument.config.faster_autoaugment import FasterAutoAugmentSearchConfig
from autoalbument.config.validation import validate_cfg
from autoalbument.faster_autoaugment.search import get_faa_seacher
from autoalbument.utils.hydra import get_config_dir

OmegaConf.register_resolver("config_dir", get_config_dir)
cs = ConfigStore.instance()
cs.store(name="config", node=FasterAutoAugmentSearchConfig)


def get_prettified_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg["task"] == "classification":
        del cfg["semantic_segmentation_model"]
    else:
        del cfg["classification_model"]

    return OmegaConf.to_yaml(cfg)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    validate_cfg(cfg)
    config_dir = get_config_dir()
    if config_dir is not None:
        sys.path.append(config_dir)
    print(get_prettified_cfg(cfg))
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")
    faa_searcher = get_faa_seacher(cfg)
    faa_searcher.search()
