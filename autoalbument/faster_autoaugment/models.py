from typing import Tuple

import segmentation_models_pytorch as smp
import timm
from torch import nn, Tensor
from torch.nn import Flatten


class BaseDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class ClassificationModel(BaseDiscriminator):
    def __init__(self, architecture, pretrained, num_classes):
        super().__init__()
        self.base_model = timm.create_model(architecture, pretrained=pretrained)
        self.base_model.reset_classifier(num_classes)
        self.classifier = self.base_model.get_classifier()
        num_features = self.classifier.in_features
        self.discriminator = nn.Sequential(
            nn.Linear(num_features, num_features), nn.ReLU(), nn.Linear(num_features, 1)
        )

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.base_model.forward_features(input)
        x = self.base_model.global_pool(x).flatten(1)
        return self.classifier(x), self.discriminator(x).view(-1)


class SemanticSegmentationModel(BaseDiscriminator):
    def __init__(self, architecture, encoder_architecture, num_classes, pretrained):
        super().__init__()
        model = getattr(smp, architecture)

        self.base_model = model(
            encoder_architecture, encoder_weights=self._get_encoder_weights(pretrained), classes=num_classes
        )
        num_features = self.base_model.encoder.out_channels[-1]
        self.base_model.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, 1),
        )

    @staticmethod
    def _get_encoder_weights(pretrained):
        if isinstance(pretrained, bool):
            return "imagenet" if pretrained else None
        return pretrained

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        mask, discriminator_output = self.base_model(input)
        return mask, discriminator_output.view(-1)
