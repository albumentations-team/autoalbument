import torchvision

import cv2
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class CityscapesSearchDataset(torchvision.datasets.Cityscapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, target_type="semantic")
        self.semantic_target_type_index = [i for i, t in enumerate(self.target_type) if t == "semantic"][0]
        self.colormap = self._generate_colormap()

    def _generate_colormap(self):
        colormap = {}
        for class_ in self.classes:
            if class_.train_id in (-1, 255):
                continue
            colormap[class_.train_id] = class_.id
        return colormap

    def _convert_to_segmentation_mask(self, mask):
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(self.colormap)), dtype=np.float32)
        for label_index, label in self.colormap.items():
            segmentation_mask[:, :, label_index] = (mask == label).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.targets[index][self.semantic_target_type_index], cv2.IMREAD_UNCHANGED)

        mask = self._convert_to_segmentation_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask
