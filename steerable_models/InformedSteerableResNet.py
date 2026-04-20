import torch
import torch.nn as nn
from torchvision import models

from steerable_models.SteerableResNet import SteeringController


class InformedSteerableResNet(nn.Module):
    def __init__(self, num_classes=20, pretrained=True, bottleneck_dim=64):
        super().__init__()

        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.gap = backbone.avgpool

        self.steering = SteeringController(in_channels=3, bottleneck_dim=bottleneck_dim)

        self.classifier = nn.Linear(256, num_classes)

    def _backbone(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return torch.flatten(self.gap(x), 1)

    def _invariant_features(self, x):
        B = x.size(0)
        # Create the 8 transforms (4 rotations * 2 flips)
        transforms = []
        for k in range(4):
            rot = torch.rot90(x, k=k, dims=[-2, -1])
            transforms.append(rot)
            transforms.append(torch.flip(rot, dims=[-1]))

        combined = torch.cat(transforms, dim=0)  # [8B, 3, H, W]
        group_feats = self._backbone(combined)  # [8B, 256]
        return group_feats.view(8, B, -1).mean(dim=0)  # [B, 256]

    def forward(self, x, labels=None):
        """
        TRAINING: Pass labels to force correct routing.
        TESTING: Labels are None; SteeringController predicts alpha.
        """
        if labels is not None:
            # We know the answer: Classes 0-9 (MNIST) -> 0, Classes 10-19 (Birds) -> 1
            alpha = (labels >= 10).float().unsqueeze(1)
        else:
            # We don't know the answer: Let the model look at 'x' and decide
            alpha = self.steering(x)

        f_steered = torch.zeros((x.size(0), 256), device=x.device)

        inv_mask = alpha.squeeze(1) > 0.5
        std_mask = ~inv_mask

        if std_mask.any():
            f_steered[std_mask] = self._backbone(x[std_mask])
        if inv_mask.any():
            f_steered[inv_mask] = self._invariant_features(x[inv_mask])

        return self.classifier(f_steered)