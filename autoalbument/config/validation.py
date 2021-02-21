import warnings


def validate_cfg(cfg):
    data_cfg = cfg.data
    dataset = getattr(data_cfg, "dataset", None)
    dataset_file = getattr(data_cfg, "dataset_file", None)

    if dataset is None and dataset_file is None:
        raise ValueError("You should provide either data.dataset or data.dataset_file")
    elif dataset is not None and dataset_file is not None:
        warnings.warn(f"Using value {dataset_file} from `dataset_file` to load a dataset", RuntimeWarning)
