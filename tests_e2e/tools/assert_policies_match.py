import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("policy_1_file")
parser.add_argument("policy_2_file")
args = parser.parse_args()


with open(args.policy_1_file) as f:
    policy_1 = json.load(f)
with open(args.policy_2_file) as f:
    policy_2 = json.load(f)

assert policy_1 == policy_2
