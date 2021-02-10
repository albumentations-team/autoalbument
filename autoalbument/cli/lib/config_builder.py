from enum import Enum

from ruamel.yaml import YAML


class PackageType(str, Enum):
    group_ = "_group_"
    global_ = "_global_"


MISSING_VALUE_PLACEHOLDER = "???"


class SearchConfigBuilder:
    def __init__(self, base_config_path, short_config_keys_path, task, num_classes, generate_full_config):
        self.base_config_path = base_config_path
        self.short_config_keys_path = short_config_keys_path
        self.task = task
        self.missing_values = {
            "task": task,
            "num_classes": num_classes,
        }
        self.generate_full_config = generate_full_config
        self.yaml = YAML(typ="rt", pure=True)
        self.add_representers()

    def add_representers(self):
        def null_representer(self_, data):
            return self_.represent_scalar("tag:yaml.org,2002:null", "null")

        self.yaml.representer.add_representer(type(None), null_representer)

    @staticmethod
    def get_package_type(config_text):
        if "_group_" in config_text:
            return PackageType.group_
        return PackageType.global_

    def parse_config(self, config_path):
        try:
            with open(config_path) as f:
                lines = f.readlines()
                package_type = self.get_package_type(lines[0])
                first_config_line = 1
                return self.yaml.load("".join(lines[first_config_line:]) + "\n\n"), package_type

        except FileNotFoundError:
            return None, None

    def build_full_config(self):
        base_config = self.yaml.load(self.base_config_path)
        parent_dir = self.base_config_path.parent

        config = self.yaml.load("{}")
        for included_config_name in base_config["defaults"]:
            if isinstance(included_config_name, dict):
                group, filename = next(iter(included_config_name.items()))
                included_config_path = parent_dir / group / f"{filename}.yaml"
            else:
                group = None
                included_config_path = parent_dir / f"{included_config_name}.yaml"

            inc_config, package_type = self.parse_config(included_config_path)
            if inc_config is None:
                continue
            if package_type == PackageType.group_:
                config[group] = inc_config
            else:
                config.update(inc_config)

        return config

    def fill_missing_values(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                self.fill_missing_values(v)
            elif k in self.missing_values and v == MISSING_VALUE_PLACEHOLDER:
                config[k] = self.missing_values[k]
        return config

    @staticmethod
    def delete_not_required_keys(config, required):
        required = required or {}
        keys_to_delete = set(config.keys()) - set(required.keys())
        for key in keys_to_delete:
            del config[key]
        for k, v in config.items():
            if not isinstance(v, dict):
                continue
            SearchConfigBuilder.delete_not_required_keys(config[k], required.get(k, {}))
        return config

    def delete_unused_task_model_key(self, config):
        unused_task = "classification" if self.task == "semantic_segmentation" else "semantic_segmentation"
        del config[f"{unused_task}_model"]
        return config

    def write_config(self, config_file_destination):
        config = self.build_full_config()
        if not self.generate_full_config:
            short_config_keys = self.yaml.load(self.short_config_keys_path)
            config = self.delete_not_required_keys(config, short_config_keys)
        config = self.delete_unused_task_model_key(config)
        config = self.fill_missing_values(config)
        with open(config_file_destination, "w") as f:
            self.yaml.dump(config, f)
