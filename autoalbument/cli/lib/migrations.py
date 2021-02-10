import copy
import logging
from abc import abstractmethod
from pathlib import Path

from ruamel.yaml import YAML

log = logging.getLogger(__name__)


yaml = YAML(typ="rt", pure=True)


def pop_key(dct, key):
    first, *next_ = key

    if not next_:
        value = dct[first]
        del dct[first]
        return value

    dct = dct[first]
    return pop_key(dct, next_)


def set_key(config, key, value):
    first, *next_ = key
    if not next_:
        config[first] = value
        return config

    if first not in config:
        config[first] = {}
    return set_key(config[first], next_, value)


class MigrationOperation:
    def __init__(self, old_key, new_key=None):
        self.old_key = old_key.split(".")
        self.new_key = new_key.split(".") if new_key is not None else None

    def __call__(self, config):
        config = copy.deepcopy(config)
        try:
            old_value = pop_key(config, self.old_key)
        except KeyError:
            return config
        log.info("Migrating {}")

        return self.apply(config, self.old_key, old_value, self.new_key)

    @staticmethod
    @abstractmethod
    def apply(config, old_key, old_value, new_key):
        ...


class MoveOperation(MigrationOperation):
    @staticmethod
    def apply(config, old_key, old_value, new_key):
        set_key(config, new_key, old_value)
        return config


class MigrateSaveCheckpoints(MigrationOperation):
    @staticmethod
    def apply(config, old_key, old_value, new_key):
        if not old_value:
            callbacks_config_path = Path(__file__).parent / "conf" / "callbacks" / "default.yaml"
            callbacks_config = yaml.load(callbacks_config_path)
            callbacks_config = [
                callback
                for callback in callbacks_config
                if callback["_target_"] != "pytorch_lightning.callbacks.ModelCheckpoint"
            ]
            config["callbacks"] = callbacks_config
        return config


class MigrateTensorboardLogsDir(MigrationOperation):
    @staticmethod
    def apply(config, old_key, old_value, new_key):
        if old_value:
            set_key(config, new_key, old_value)
        else:
            set_key(config, new_key[:1], None)
        return config


class MigrateDevice(MigrationOperation):
    @staticmethod
    def apply(config, old_key, old_value, new_key):
        new_value = 1 if old_value == "cuda" else 0
        set_key(config, new_key, new_value)
        return config


class MigrateCheckpointPath(MigrationOperation):
    @staticmethod
    def apply(config, old_key, old_value, new_key):
        if old_value:
            raise ValueError(
                "The current version of AutoAlbument doesn't support the checkpoint format defined "
                "in `checkpoint_path`.Use AutoAlbument==0.3.0 to load this checkpoint or manually delete "
                "this configuration parameterto migrate the config file."
            )
        return config


def migrate_v1_to_v2(config):

    operations = [
        MoveOperation("optim.epochs", "trainer.max_epochs"),
        MoveOperation("cudnn_benchmark", "trainer.benchmark"),
        MigrateSaveCheckpoints("save_checkpoints"),
        MigrateTensorboardLogsDir("tensorboard_logs_dir", "logger.save_dir"),
        MigrateDevice("device", "trainer.gpus"),
        MigrateCheckpointPath("checkpoint_path"),
    ]
    for operation in operations:
        config = operation(config)

    config["_version"] = 2
    return config
