import os

import pytest
from hydra.experimental import initialize, compose

from autoalbument.faster_autoaugment.search import get_faa_seacher
from tests.utils import set_seed, calculate_sha256


@pytest.mark.parametrize(
    ["task", "policy_file_hash"],
    [
        ["classification", "404eb2372cc0a044286abdaf2e458f2ce2370836b01cd1513842955f58b528a8"],
        ["semantic_segmentation", "2236c8f58e203291d6b95fd5d9946202ce436bc79dc56207b7d79d052bc07c92"],
    ],
)
def test_e2e(tmpdir, task, policy_file_hash) -> None:
    with initialize(config_path=f"./configs/{task}"):
        os.chdir(tmpdir)
        cfg = compose(config_name="search")
        set_seed(42)
        faa_searcher = get_faa_seacher(cfg)
        faa_searcher.search()
        policy_file = tmpdir / "policy" / "latest.json"
        assert calculate_sha256(policy_file) == policy_file_hash
