import os
from pathlib import Path

from hydra.experimental import initialize, compose, initialize_config_dir

from autoalbument.faster_autoaugment.search import get_faa_seacher


def test_dataloader_drops_last(tmpdir) -> None:
    config_path = "./configs/classification"
    with initialize(config_path=str(config_path)):
        os.environ["AUTOALBUMENT_TEST_DATASET_LENGTH"] = "17"
        os.environ["AUTOALBUMENT_CONFIG_DIR"] = str((Path(__file__).parent / config_path).resolve())
        os.chdir(tmpdir)
        cfg = compose(config_name="search", overrides=["data.dataloader.batch_size=12"])
        faa_searcher = get_faa_seacher(cfg)
        faa_searcher.search()
