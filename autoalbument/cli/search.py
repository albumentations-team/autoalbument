import copy
import os
import sys
from pathlib import Path

import colorama
import hydra
from colorama import Style
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autoalbument.config.validation import validate_cfg
from autoalbument.utils.hydra import get_config_dir

OmegaConf.register_resolver("config_dir", get_config_dir)
colorama.init()


def get_prettified_cfg(cfg):
    cfg = copy.deepcopy(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg["task"] == "classification":
        del cfg["semantic_segmentation_model"]
    else:
        del cfg["classification_model"]

    return OmegaConf.to_yaml(cfg)


def check_config_version(cfg, config_dir):
    version = cfg.get("_version", 1)
    if version == 1:
        config_path = Path(config_dir) / "search.yaml"
        raise ValueError(
            f"\n\n{Style.BRIGHT}{config_path}{Style.RESET_ALL} file uses the old configuration format. "
            f"Please do the following steps:\n"
            f"1. Run {Style.BRIGHT}autoalbument-migrate --config-dir {config_dir}{Style.RESET_ALL} to automatically "
            f"update the configuration file to the new format.\n"
            f"2. Run {Style.BRIGHT}autoalbument-search{Style.RESET_ALL} again."
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
