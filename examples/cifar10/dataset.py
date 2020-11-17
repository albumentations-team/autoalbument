import torchvision


class SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, transform=None):
        super().__init__(root="~/data/cifar10", train=True, download=True, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
