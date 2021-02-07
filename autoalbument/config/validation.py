def validate_cfg(cfg):
    data_cfg = cfg.data
    dataset = getattr(data_cfg, "dataset", None)
    dataset_file = getattr(data_cfg, "dataset_file", None)
    if (dataset is None) == (dataset_file is None):
        raise ValueError(
            f"You should provide either data.dataset or data.dataset_file but not both. "
            f"Got {dataset} for data.dataset and {dataset_file} for data.dataset_file."
        )
