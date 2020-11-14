import argparse
import os


def get_config_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=False)
    args, unknown = parser.parse_known_args()
    config_dir = args.config_dir
    if config_dir is None:
        config_dir = os.environ.get("AUTOALBUMENT_CONFIG_DIR")
    return config_dir


def get_dataset_filepath(dataset_file):
    if os.path.isabs(dataset_file):
        return dataset_file

    base_path = get_config_dir()
    if base_path is None:
        raise ValueError(
            f"Couldn't deduce the value for `config_dir` (a directory that contains the search.yaml configuration "
            f"file) to locate the {dataset_file} dataset file. Please use one of the following alternatives to fix "
            f"the error: "
            f"\n- Run `autoalbument-search` with the `--config-dir` parameter."
            f"\n- Use the absolute filepath for the `data.dataset_file` configuration parameter."
            f"\n- Provide the value for `config_dir` through the `AUTOALBUMENT_CONFIG_DIR` environment variable, e.g. "
            f"`export AUTOALBUMENT_CONFIG_DIR=/path/to/config/dir"
        )

    return os.path.join(base_path, dataset_file)
