import torch
import torch.nn as nn
from torchvision import models


class StandardResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # 1. Load the official ResNet backbone
        # We use resnet34 because it matches your multi-layer structure
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        # 2. Re-map the layers to your existing naming convention
        # (This keeps the rest of your code working)
        self.stem = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        # Note: We omit layer4 if you want to keep your 256-channel limit

        self.gap = self.backbone.avgpool

        # 3. Match your custom classifier head
        # ResNet34 layer3 outputs 256 channels
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)