import copy
import logging

import torch
from pytorch_lightning import Callback

__all__ = [
    "MonitorAverageParameterChange",
]


log = logging.getLogger(__name__)


class MonitorAverageParameterChange(Callback):
    def __init__(self):
        self.policy_model_state_dict_epoch_start = None

    @classmethod
    def get_combined_parameter_values(cls, state_dict, prefixes):
        combined_parameter_values = []
        for prefix in prefixes:
            weights = state_dict[f"{prefix}._weights"]
            temperature = torch.zeros(weights.size(), dtype=weights.dtype).to(weights.device)
            for i in range(len(weights)):
                temperature_key = f"{prefix}.operations.{i}.temperature"
                temperature[i] = state_dict[temperature_key]
            weights = torch.div(weights, temperature).softmax(0)
            for i, weight in enumerate(weights):
                probability_key = f"{prefix}.operations.{i}._probability"
                magnitude_key = f"{prefix}.operations.{i}._magnitude"
                probability = (weight * state_dict[probability_key].clamp(0.0, 1.0)).item()
                magnitude = state_dict.get(magnitude_key)
                if magnitude is not None:
                    magnitude = magnitude.clamp(0.0, 1.0).item()
                value = probability * magnitude if magnitude is not None else probability
                combined_parameter_values.append(value)
        return combined_parameter_values

    @classmethod
    def get_average_parameter_change(cls, state_dict_1, state_dict_2):
        prefixes = {".".join(key.split(".")[:4]) for key in state_dict_1.keys() if key.startswith("sub_policies")}
        values_1 = cls.get_combined_parameter_values(state_dict_1, prefixes)
        values_2 = cls.get_combined_parameter_values(state_dict_2, prefixes)
        values_change = [abs(v2 - v1) for v1, v2 in zip(values_1, values_2)]
        return sum(values_change) / len(values_change)

    def on_epoch_start(self, trainer, pl_module):
        self.policy_model_state_dict_epoch_start = copy.deepcopy(pl_module.policy_model.state_dict())

    def on_epoch_end(self, trainer, pl_module):
        policy_model_state_dict_epoch_end = pl_module.policy_model.state_dict()
        average_parameter_change = self.get_average_parameter_change(
            self.policy_model_state_dict_epoch_start,
            policy_model_state_dict_epoch_end,
        )
        pl_module.log(
            "average_parameter_change",
            average_parameter_change,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        log.info(f"Average Parameter change at epoch {trainer.current_epoch}: {average_parameter_change:.6f}.")
