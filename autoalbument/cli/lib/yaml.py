from ruamel.yaml import YAML

yaml = YAML(typ="rt", pure=True)


def null_representer(self_, data):
    return self_.represent_scalar("tag:yaml.org,2002:null", "null")


yaml.representer.add_representer(type(None), null_representer)
