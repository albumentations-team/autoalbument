import json
import os
from pathlib import Path

import pytest
from hydra.experimental import initialize, compose

from autoalbument.faster_autoaugment.search import get_faa_seacher
from tests.utils import set_seed, calculate_sha256


@pytest.mark.parametrize(
    ["task", "policy_file_hash"],
    [
        ["classification", "159b06d6b0310bd4dfb427429b95b96ca71eacc8ed84b8b680c77b22c6a204af"],
        ["semantic_segmentation", "b3f7efde4ab63b45555577f3d4e00daf66f1fb30cd84db19c4a28ba812c34cba"],
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
        expected_policy_file = Path(__file__).parent.absolute() / "configs" / task / "expected_policy.json"
        with policy_file.open() as f:
            policy = json.load(f)["transform"]
        with expected_policy_file.open() as f:
            expected_policy = json.load(f)["transform"]
        assert policy == expected_policy
