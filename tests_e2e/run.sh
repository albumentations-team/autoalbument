#!/usr/bin/env bash

set -e

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

autoalbument-search --config-dir $SCRIPTPATH/configs/classification_1
python $SCRIPTPATH/tools/assert_policies_match.py $SCRIPTPATH/configs/classification_1/expected_policy.json $SCRIPTPATH/configs/classification_1/outputs/policy/latest.json

autoalbument-search --config-dir $SCRIPTPATH/configs/classification_2
python $SCRIPTPATH/tools/assert_policies_match.py $SCRIPTPATH/configs/classification_2/expected_policy.json $SCRIPTPATH/configs/classification_2/outputs/policy/latest.json

autoalbument-search --config-dir $SCRIPTPATH/configs/semantic_segmentation_1
python $SCRIPTPATH/tools/assert_policies_match.py $SCRIPTPATH/configs/semantic_segmentation_1/expected_policy.json $SCRIPTPATH/configs/semantic_segmentation_1/outputs/policy/latest.json

autoalbument-search --config-dir $SCRIPTPATH/configs/semantic_segmentation_2
python $SCRIPTPATH/tools/assert_policies_match.py $SCRIPTPATH/configs/semantic_segmentation_2/expected_policy.json $SCRIPTPATH/configs/semantic_segmentation_2/outputs/policy/latest.json

autoalbument-create --config-dir $SCRIPTPATH/configs/classification_from_create_1 --task classification --num-classes 10
python $SCRIPTPATH/tools/merge_changeset.py $SCRIPTPATH/configs/classification_from_create_1/search.yaml $SCRIPTPATH/configs/classification_from_create_1/changeset.yaml
cp $SCRIPTPATH/configs/classification_from_create_1/dataset_implementation.py $SCRIPTPATH/configs/classification_from_create_1/dataset.py
autoalbument-search --config-dir $SCRIPTPATH/configs/classification_from_create_1
python $SCRIPTPATH/tools/assert_policies_match.py $SCRIPTPATH/configs/classification_from_create_1/expected_policy.json $SCRIPTPATH/configs/classification_from_create_1/outputs/policy/latest.json

autoalbument-create --config-dir $SCRIPTPATH/configs/semantic_segmentation_from_create_1 --task semantic_segmentation --num-classes 10
python $SCRIPTPATH/tools/merge_changeset.py $SCRIPTPATH/configs/semantic_segmentation_from_create_1/search.yaml $SCRIPTPATH/configs/semantic_segmentation_from_create_1/changeset.yaml
cp $SCRIPTPATH/configs/semantic_segmentation_from_create_1/dataset_implementation.py $SCRIPTPATH/configs/semantic_segmentation_from_create_1/dataset.py
autoalbument-search --config-dir $SCRIPTPATH/configs/semantic_segmentation_from_create_1
python $SCRIPTPATH/tools/assert_policies_match.py $SCRIPTPATH/configs/semantic_segmentation_from_create_1/expected_policy.json $SCRIPTPATH/configs/semantic_segmentation_from_create_1/outputs/policy/latest.json

autoalbument-create --config-dir $SCRIPTPATH/configs/classification_from_create_2 --task classification --num-classes 10
cp $SCRIPTPATH/configs/classification_from_create_2/dataset_implementation.py $SCRIPTPATH/configs/classification_from_create_2/dataset.py
autoalbument-search --config-dir $SCRIPTPATH/configs/classification_from_create_2 device=cpu

autoalbument-create --config-dir $SCRIPTPATH/configs/semantic_segmentation_from_create_2 --task semantic_segmentation --num-classes 10
cp $SCRIPTPATH/configs/semantic_segmentation_from_create_2/dataset_implementation.py $SCRIPTPATH/configs/semantic_segmentation_from_create_2/dataset.py
autoalbument-search --config-dir $SCRIPTPATH/configs/semantic_segmentation_from_create_2 device=cpu
