import argparse

from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file")
parser.add_argument("yaml_changest")
args = parser.parse_args()


conf = OmegaConf.load(args.yaml_file)
changeset = OmegaConf.load(args.yaml_changest)
conf = OmegaConf.merge(conf, changeset)
OmegaConf.save(conf, args.yaml_file)
