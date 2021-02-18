import os
from pathlib import Path

from hydra.experimental import compose, initialize
from hydra.utils import instantiate


def test_dataloader_drops_last(tmpdir) -> None:
    config_path = "./configs/classification"
    with initialize(config_path=str(config_path)):
        os.environ["AUTOALBUMENT_TEST_DATASET_LENGTH"] = "17"
        os.environ["AUTOALBUMENT_CONFIG_DIR"] = str((Path(__file__).parent / config_path).resolve())
        os.chdir(tmpdir)
        cfg = compose(config_name="search", overrides=["data.dataloader.batch_size=12"])
        faa_searcher = instantiate(cfg.searcher, cfg=cfg)
        faa_searcher.search()
