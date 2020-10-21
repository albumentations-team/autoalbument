import numpy as np
import torch.utils.data


class SearchDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 16

    def __getitem__(self, index):
        np.random.seed(index)
        image = np.random.randint(low=0, high=256, size=(32, 32, 3), dtype=np.uint8)
        mask = np.random.uniform(low=0.0, high=1.0, size=(32, 32, 10)).astype(np.float32)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask
