#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
CONFIG_BASE_DIR=$SCRIPTPATH/configs
TOOLS_DIR=$SCRIPTPATH/tools

test_simple_search () {
  CONFIG_DIR=$CONFIG_BASE_DIR/$1
  echo $CONFIG_DIR
  autoalbument-search --config-dir $CONFIG_DIR
  python $TOOLS_DIR/assert_policies_match.py $CONFIG_DIR/expected_policy.json $CONFIG_DIR/outputs/policy/latest.json
}

test_search_from_create () {
  CONFIG_DIR=$CONFIG_BASE_DIR/$1
  TASK=$2
  echo $CONFIG_DIR $TASK
  autoalbument-create --config-dir $CONFIG_DIR --task $TASK --num-classes 10
  cp $CONFIG_DIR/dataset_implementation.py $CONFIG_DIR/dataset.py
  autoalbument-search --config-dir $CONFIG_DIR device=cpu semantic_segmentation_model.pretrained=False optim.epochs=1
}

test_search_from_create_relative_directory () {
  CONFIG_DIR=./tests_e2e/configs/$1
  TASK=$2
  echo $CONFIG_DIR $TASK
  autoalbument-create --config-dir $CONFIG_DIR --task $TASK --num-classes 10
  cp $CONFIG_DIR/dataset_implementation.py $CONFIG_DIR/dataset.py
  autoalbument-search --config-dir $CONFIG_DIR device=cpu semantic_segmentation_model.pretrained=False optim.epochs=1
}


test_search_from_create_full_config () {
  CONFIG_DIR=$CONFIG_BASE_DIR/$1
  TASK=$2
  echo $CONFIG_DIR $TASK
  autoalbument-create --config-dir $CONFIG_DIR --task $TASK --num-classes 10 --generate-full-config
  cp $CONFIG_DIR/dataset_implementation.py $CONFIG_DIR/dataset.py
  autoalbument-search --config-dir $CONFIG_DIR device=cpu
}


test_search_from_merged_create () {
  CONFIG_DIR=$CONFIG_BASE_DIR/$1
  TASK=$2
  echo $CONFIG_DIR $TASK
  autoalbument-create --config-dir $CONFIG_DIR --task $TASK --num-classes 10
  python $TOOLS_DIR/merge_changeset.py $CONFIG_DIR/search.yaml $CONFIG_DIR/changeset.yaml
  cp $CONFIG_DIR/dataset_implementation.py $CONFIG_DIR/dataset.py
  autoalbument-search --config-dir $CONFIG_DIR
  python $TOOLS_DIR/assert_policies_match.py $CONFIG_DIR/expected_policy.json $CONFIG_DIR/outputs/policy/latest.json
}

test_simple_search classification_1
test_simple_search classification_2
test_simple_search classification_3
test_simple_search classification_1_1
test_simple_search classification_2_1
test_simple_search classification_3_1

test_simple_search semantic_segmentation_1
test_simple_search semantic_segmentation_2
test_simple_search semantic_segmentation_3
test_simple_search semantic_segmentation_1_1
test_simple_search semantic_segmentation_2_1
test_simple_search semantic_segmentation_3_1

test_search_from_merged_create classification_from_create_1 classification
test_search_from_merged_create semantic_segmentation_from_create_1 semantic_segmentation

test_search_from_create classification_from_create_2 classification
test_search_from_create semantic_segmentation_from_create_2 semantic_segmentation

test_search_from_create_relative_directory classification_from_create_3 classification
test_search_from_create_relative_directory semantic_segmentation_from_create_3 semantic_segmentation

test_search_from_create_full_config classification_from_create_2_full_config classification
test_search_from_create_full_config semantic_segmentation_from_create_2_full_config semantic_segmentation
