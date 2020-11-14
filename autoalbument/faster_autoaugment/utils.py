import importlib.util
from collections import defaultdict
import random

import numpy as np
import torch

from autoalbument.utils.hydra import get_dataset_filepath

MAX_VALUES_BY_INPUT_DTYPE = {
    "uint8": 255,
    "float32": 1.0,
}


def get_dataset_cls(dataset_file, dataset_cls_name="SearchDataset"):
    dataset_filepath = get_dataset_filepath(dataset_file)
    spec = importlib.util.spec_from_file_location("dataset", dataset_filepath)
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)
    return getattr(dataset, dataset_cls_name)


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"value": 0, "batch_count": 0, "avg": 0})

    def update(self, batch_metrics):
        for name, batch_metric in batch_metrics.items():
            value = batch_metric.item() if isinstance(batch_metric, torch.Tensor) else batch_metric
            self.metrics[name]["value"] += value
            self.metrics[name]["batch_count"] += 1
            self.metrics[name]["avg"] = self.metrics[name]["value"] / self.metrics[name]["batch_count"]

    def get_avg_values(self):
        return [(name, value["avg"]) for name, value in self.metrics.items()]

    def __repr__(self):
        return ", ".join(f"{name}={avg_value:.6f}" for name, avg_value in self.get_avg_values())


def target_requires_grad(target):
    return target == "image_batch"


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
