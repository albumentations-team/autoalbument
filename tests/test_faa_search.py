import os

from hydra.experimental import initialize, compose

from autoalbument.faster_autoaugment.search import get_faa_seacher


def test_dataloader_drops_last(tmpdir) -> None:
    with initialize(config_path="./configs/classification"):
        os.environ["AUTOALBUMENT_TEST_DATASET_LENGTH"] = "17"
        os.chdir(tmpdir)
        cfg = compose(config_name="search", overrides=["data.dataloader.batch_size=12"])
        faa_searcher = get_faa_seacher(cfg)
        faa_searcher.search()
