# AutoAlbument

AutoAlbument is an AutoML tool that learns augmentation policies from data. You can apply discovered policies with [Albumentations](https://github.com/albumentations-team/albumentations), an image augmentation library. For now, classification and semantic segmentation tasks are supported.


## Table of contents
- [Installation](#installation)
- [Usage](#usage)
- [Search algorithms](#search-algorithms)
- [Examples](#examples)
- [FAQ](#faq)


## Installation
AutoAlbument requires Python 3.6 or higher. To install the library:
- Clone the repository: `git clone git@github.com:albumentations-team/autoalbument.git`.
- Install the library: `pip install -e autoalbument/`.

Note: for now, AutoAlbument uses features that are available only in this branch of Albumentations: [https://github.com/albumentations-team/albumentations/tree/updated_transforms](https://github.com/albumentations-team/albumentations/tree/updated_transforms)


## Usage
### 1. Create a directory with configuration files.
 Run `autoalbument-create --config-dir </path/to/directory> --task <deep_learning_task> --num-classes <num_classes>`, e.g. `autoalbument-create --config-dir ~/experiments/autoalbument-search-cifar10 --task classification --num-classes 10`.
 - A value for the `--config-dir` option should contain a path to the directory. AutoAlbument will create this directory and put two files into it: `dataset.py` and `search.yaml` (more on them later).
  - A value for the `--task` option should contain the name of a deep learning task. Supported values are `classification` and `semantic_segmentation`.
 - A value for the `--num-classes` option should contain the number of distinct classes in the classification or segmentation dataset.

### 2. Add implementation for `__len__` and `__getitem__` methods in dataset.py.

The `dataset.py` file created at step 1 by `autoalbument-create` contains stubs for implementing a PyTorch dataset (you can read more about creating custom PyTorch datasets [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)).

#### An example dataset stub for a classification task:

```
class SearchDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # Implement additional initialization logic if needed

    def __len__(self):
        # Replace `...` with the actual implementation
        ...

    def __getitem__(self, index):
        # Implement logic to get an image and its label using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `label` should be an integer in the range [0, model.num_classes - 1] where `model.num_classes`
        # is a value set in the `search.yaml` file.

        image = ...
        label = ...

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
```

#### An example dataset stub for a semantic segmentation task:

```
import torch.utils.data


class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # Implement additional initialization logic if needed

    def __len__(self):
        # Replace `...` with the actual implementation
        ...

    def __getitem__(self, index):
        # Implement logic to get an image and its mask using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `mask` should be a NumPy array with the shape [height, width, num_classes] where `num_classes`
        # is a value set in the `search.yaml` file. Each mask channel should encode values for a single class (usually
        # pixel in that channel has a value of 1.0 if the corresponding pixel from the image belongs to this class and
        # 0.0 otherwise). During augmentation search, `nn.BCEWithLogitsLoss` is used as a segmentation loss.

        image = ...
        mask = ...

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
```

### 3. \[Optional\] Adjust search parameters in `search.yaml`.
You may want to change parameters that AutoAlbument will use to search for augmentation policies. To do this, you need to edit the `search.yaml` file created by `autoalbument-create` at step 1. Each configuration parameter contains a comment that describes the meaning of the setting.

`search.yaml` is a [Hydra](https://hydra.cc/) config file. You can use all Hydra features inside it.

### 4. Run the search for augmentation policies.

To search for augmentation policies, run `autoalbument-search --config-dir </path/to/directory>`, e.g. `autoalbument-search --config-dir ~/experiments/autoalbument-search-cifar10`. The value of `--config-dir` should be the same value that was passed to `autoalbument-create` at step 1.

`autoalbument-search` will create a directory with output files (by default the path of the directory will be `<config_dir>/outputs/<current_date>/<current_time>`, but you can customize it in search.yaml).  The `policy` subdirectory will contain JSON files with policies found at each search phase's epoch.

`autoalbument-search` is a command wrapped with the `@hydra.main` decorator from [Hydra](https://hydra.cc/). You can use all Hydra features when calling this command.

AutoAlbument uses PyTorch to search for augmentation policies. You can speed up the search by using a CUDA-capable GPU.

### 5. Use found augmentation policies in your training pipeline.
You can use a JSON file with a policy to create an augmentation pipeline with [Albumentations](https://github.com/albumentations-team/albumentations):

```
import albumentations as A

transform = A.load("/path/to/policy.json")
```

You can read more about using Albumentation for augmentation in [this article](https://albumentations.ai/docs/getting_started/image_augmentation/).


## Search algorithms

### Faster AutoAugment
"Faster AutoAugment: Learning Augmentation Strategies using Backpropagation"  by Ryuichiro Hataya, Jan Zdenek, Kazuki Yoshizoe, and Hideki Nakayama. [Paper](https://arxiv.org/abs/1911.06987) | [Original implementation](https://github.com/moskomule/dda/tree/master/faster_autoaugment)


## Examples
The [`examples`](examples/) directory contains example configs for different tasks and datasets.


## FAQ

#### Search takes a lot of time. How can I speed it up?
Instead of a full training dataset, you can use a reduced version to search for augmentation policies. For example, the authors of Faster AutoAugment used 6000 images from the 120 selected classes to find augmentation policies for ImageNet (while the full dataset for ILSVRC contains 1.2 million images and 1000 classes).
