import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class StandardResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Stem: matches the 64 channels of the V4 model's first layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Layers: Using the full 64, 128, 256 channel widths
        self.layer1 = StandardResBlock(64, 64)
        self.layer2 = StandardResBlock(64, 128, stride=2)
        self.layer3 = StandardResBlock(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # The V4 model pooled 256 channels down to 64 before the linear layer.
        # To keep this "standard" version comparable, we use the full 256.
        self.classifier = nn.Linear(256, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
