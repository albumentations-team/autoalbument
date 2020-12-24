import os

import numpy as np
import torch.utils.data


class SearchDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return int(os.environ.get("AUTOALBUMENT_TEST_DATASET_LENGTH", 16))

    def __getitem__(self, index):
        np.random.seed(index)
        image = np.random.randint(low=0, high=256, size=(32, 32, 3), dtype=np.uint8)
        label = np.random.randint(low=0, high=10, dtype=np.int64)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
