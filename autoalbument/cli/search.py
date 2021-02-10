import copy
import os
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autoalbument.config.validation import validate_cfg
from autoalbument.utils.hydra import get_config_dir

OmegaConf.register_resolver("config_dir", get_config_dir)


def get_prettified_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg["task"] == "classification":
        del cfg["semantic_segmentation_model"]
    else:
        del cfg["classification_model"]

    return OmegaConf.to_yaml(cfg)


def check_config_version(cfg, config_dir):
    version = cfg.get("version", 1)
    if version == 1:
        config_path = config_dir / "search.yaml"
        raise ValueError(
            f"{config_path} file uses the old configuration format. Please update its format by running"
            f"`autoalbument-migrate --config-dir {config_dir}`. Then run autoalbument-search again."
        )


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    config_dir = get_config_dir()
    check_config_version(cfg, config_dir)
    validate_cfg(cfg)
    if config_dir is not None:
        sys.path.append(config_dir)
    print(get_prettified_cfg(cfg))
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")
    searcher = instantiate(cfg.searcher, cfg=cfg)
    searcher.search()
