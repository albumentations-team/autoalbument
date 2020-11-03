# AutoAlbument

AutoAlbument is an AutoML tool that learns image augmentation policies from data using the [Faster AutoAugment algorithm](https://arxiv.org/abs/1911.06987). It relieves the user from manually selecting augmentations and tuning their parameters. AutoAlbument provides a complete ready-to-use configuration for an augmentation pipeline.

AutoAlbument supports image classification and semantic segmentation tasks.

## Table of contents
- [Installation](#installation)
- [Usage](#usage)
- [Search algorithms](#search-algorithms)
- [Tuning the search parameters](#tuning-the-search-parameters)
- [Examples](#examples)
- [FAQ](#faq)

## Installation
AutoAlbument requires Python 3.6 or higher.

#### PyPI
To install the latest stable version from PyPI:

`pip install -U autoalbument`

#### GitHub
To install the latest version from GitHub:

`pip install -U git+https://github.com/albumentations-team/autoalbument`

## Usage
### 1. Create a directory with configuration files.
 Run `autoalbument-create --config-dir </path/to/directory> --task <deep_learning_task> --num-classes <num_classes>`, e.g. `autoalbument-create --config-dir ~/experiments/autoalbument-search-cifar10 --task classification --num-classes 10`.
 - A value for the `--config-dir` option should contain a path to the directory. AutoAlbument will create this directory and put two files into it: `dataset.py` and `search.yaml` (more on them later).
  - A value for the `--task` option should contain the name of a deep learning task. Supported values are `classification` and `semantic_segmentation`.
 - A value for the `--num-classes` option should contain the number of distinct classes in the classification or segmentation dataset.

### 2. Add implementation for `__len__` and `__getitem__` methods in `dataset.py`.

The `dataset.py` file created at step 1 by `autoalbument-create` contains stubs for implementing a PyTorch dataset (you can read more about creating custom PyTorch datasets [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)). You need to add implementation for for `__len__` and `__getitem__` methods (and optionally add the initialization logic if required).

A dataset for a classification task should return an image and a class label. A dataset for a segmentation task should return an image and an associated mask.

### 3. \[Optional\] Adjust search parameters in `search.yaml`.
You may want to change parameters that AutoAlbument will use to search for augmentation policies. To do this, you need to edit the `search.yaml` file created by `autoalbument-create` at step 1. Each configuration parameter contains a comment that describes the meaning of the setting. Please refer to the  "Tuning the search parameters" section that includes a description of the most critical parameters.

`search.yaml` is a [Hydra](https://hydra.cc/) config file. You can use all Hydra features inside it.

### 4. Run a search for augmentation policies.

To search for augmentation policies, run `autoalbument-search --config-dir </path/to/directory>`, e.g. `autoalbument-search --config-dir ~/experiments/autoalbument-search-cifar10`. The value of `--config-dir` should be the same value that was passed to `autoalbument-create` at step 1.

`autoalbument-search` will create a directory with output files (by default the path of the directory will be `<config_dir>/outputs/<current_date>/<current_time>`, but you can customize it in search.yaml).  The `policy` subdirectory will contain JSON files with policies found at each search phase's epoch.

`autoalbument-search` is a command wrapped with the `@hydra.main` decorator from [Hydra](https://hydra.cc/). You can use all Hydra features when calling this command.

AutoAlbument uses PyTorch to search for augmentation policies. You can speed up the search by using a CUDA-capable GPU.

### 5. Use found augmentation policies in your training pipeline.
AutoAlbument produces a JSON file that contains a configuration for an augmentation pipeline. You can load that JSON file with [Albumentations](https://github.com/albumentations-team/albumentations):

```
import albumentations as A
transform = A.load("/path/to/policy.json")
```

Then you can use the created augmentation pipeline to augment the input data.

For example, to augment an image for a classification task:

```
transformed = transform(image=image)
transformed_image = transformed["image"]
```

To augment an image and a mask for a semantic segmentation task:
```
transformed = transform(image=image, mask=mask)
transformed_image = transformed["image"]
transformed_mask = transformed["mask"]
```

You can read more about using Albumentations for augmentation in those articles [Image augmentation for classification](https://albumentations.ai/docs/getting_started/image_augmentation/),
[Mask augmentation for segmentation](https://albumentations.ai/docs/getting_started/mask_augmentation/).

Refer to [this section of the documentation](https://albumentations.ai/docs/#examples-of-how-to-use-albumentations-with-different-deep-learning-frameworks) to get examples of how to use Albumentations with PyTorch and TensorFlow 2.


## Search algorithms

AutoAlbument uses the following algorithms to search for augmentation policies.

### Faster AutoAugment
"Faster AutoAugment: Learning Augmentation Strategies using Backpropagation" by Ryuichiro Hataya, Jan Zdenek, Kazuki Yoshizoe, and Hideki Nakayama. [Paper](https://arxiv.org/abs/1911.06987) | [Original implementation](https://github.com/moskomule/dda/tree/master/faster_autoaugment)

## Tuning the search parameters

`search.yaml` contains parameters for the search of augmentation policies. Here is an [example `search.yaml`](examples/cifar10/search.yaml) for image classification on the CIFAR-10 dataset, and here is an [example `search.yaml`](examples/pascal_voc/search.yaml) for semantic segmentation on the Pascal VOC dataset.

#### Task-specific model
A task-specific model is a model that classifies images for a classification task or outputs masks for a semantic segmentation task. Settings for a task-specific model are defined by either `classification_model` or `semantic_segmentation_model` depending on a selected task. Ideally, you should select the same model (the same architecture and the same pretrained weights) that you will use in an actual task. AutoAlbument uses models from [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models/) and [Segmentation models](https://github.com/qubvel/segmentation_models.pytorch) packages for classification and semantic segmentation respectively.


#### Base PyTorch parameters.

You may want to adjust the following parameters for a PyTorch pipeline:
- `data.dataloader` parameters such as batch_size and `num_workers`
- Number of epochs to search for best augmentation policies in `optim.epochs`.
- Learning rate for optimizers in `optim.main.lr` and `optim.policy.lr`.

#### Parameters for the augmentations search.
Those parameters are defined in `policy_model`. You may want to tune the following ones:
- `num_sub_policies` - number of distinct augmentation sub-policies. A random sub-policy is selected in each iteration, and that sub-policy is applied to input data. The larger number of sub-policies will produce a more diverse set of augmentations. On the other side, the more sub-policies you have, the more time and data you need to tune those sub-policies correctly.
- `num_chunks` controls the balance between speed and diversity of augmentations in a search phase. Each batch is split-up into `num_chunks` chunks, and then a random sub-policy is applied to each chunk separately. The larger the value of `num_chunks` helps to learn augmentation policies better but simultaneously increases the searching time. Authors of FasterAutoAugment used such values for `num_chunks` that each chunk consisted of 8 to 16 images.
- `operation_count` - the number of augmentation operations that will be applied to each input data instance. For example, `operation_count: 1` means that only one operation will be applied to an input image/mask, and `operation_count: 4` means that four sequential operations will be applied to each input image/mask. The larger number of operations produces a more diverse set of augmentations but simultaneously increases the searching time.

#### Preprocessing transforms
If images have different sizes or you want to train a model on image patches, you could define preprocessing transforms (such as Resizing, Cropping, and Padding) in `data.preprocessing`. Those transforms will always be applied to all input data. Found augmentation policies will also contain those preprocessing transforms.

Note that it is crucial for Policy Model (a model that searches for augmentation parameters) to receive images of the same size that will be used during the training of an actual model. For some augmentations, parameters depend on input data's height and width (for example, hole sizes for the Cutout augmentation).

## Examples
The [`examples`](examples/) directory contains example configs for different tasks and datasets.

## FAQ

#### Search takes a lot of time. How can I speed it up?
Instead of a full training dataset, you can use a reduced version to search for augmentation policies. For example, the authors of Faster AutoAugment used 6000 images from the 120 selected classes to find augmentation policies for ImageNet (while the full dataset for ILSVRC contains 1.2 million images and 1000 classes).
