import torch
import torch.nn as nn
from torchvision import models


class D4InvariantResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # 1. Load the ResNet backbone
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        # 2. Extract layers (preserving your custom architecture)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.gap = backbone.avgpool

        # 3. Classifier
        self.classifier = nn.Linear(256, num_classes)

    def get_d4_group(self, x):
        """Generates the 8 symmetries of the D4 group."""
        out = []
        for i in range(4):
            # Rotations: 0, 90, 180, 270
            rotated = torch.rot90(x, k=i, dims=[-2, -1])
            out.append(rotated)
            # Reflections of each rotation
            out.append(torch.flip(rotated, dims=[-1]))
        return out

    def forward_single(self, x):
        """Passes a single view through the backbone."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        # Generate all 8 transformed versions of the input
        # Shape of each: [Batch, C, H, W]
        group_elements = self.get_d4_group(x)

        # Stack into a single batch to process efficiently
        # New shape: [8 * Batch, C, H, W]
        combined_input = torch.cat(group_elements, dim=0)

        # Features shape: [8 * Batch, 256]
        combined_features = self.forward_single(combined_input)

        # Reshape to separate the Group dimension: [8, Batch, 256]
        group_features = combined_features.view(8, x.size(0), -1)

        # Reynolds Operator: Mean pooling over the group (D4 invariance)
        # This is the step that makes it mathematically invariant.
        invariant_features = torch.mean(group_features, dim=0)

        return self.classifier(invariant_features)