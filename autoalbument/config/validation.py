def validate_cfg(cfg):
    data_cfg = cfg.data
    if (data_cfg.dataset is None) == (data_cfg.dataset_file is None):
        raise ValueError(
            f"You should provide either data.dataset or data.dataset_file but not both. "
            f"Got {data_cfg.dataset} for data.dataset and {data_cfg.dataset_file} for data.dataset_file."
        )
