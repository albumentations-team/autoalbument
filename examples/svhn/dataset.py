import cv2
import numpy as np
import torch
import torchvision

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class SVHNSearchDataset(torchvision.datasets.SVHN):
    def __getitem__(self, index):
        image, label = self.data[index], int(self.labels[index])
        image = np.transpose(image, (1, 2, 0))

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class ConcatSVHNSearchDataset(torch.utils.data.ConcatDataset):
    def __init__(self, root, download, transform=None):
        datasets = [
            SVHNSearchDataset(root=root, split="train", download=download, transform=transform),
            SVHNSearchDataset(root=root, split="extra", download=download, transform=transform),
        ]
        super().__init__(datasets)
