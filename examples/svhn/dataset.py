import torchvision
import numpy as np

import cv2

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
