import torch.nn as nn
import torch.nn.functional as F

from .V4DigitsCNN import V4GroupConv


class V4BirdsCNN(nn.Module):
    """
    Reflection-equivariant CNN for iNaturalist Birds dataset. Uses V4GroupConv for lifting and group convolution.
    """
    def __init__(self, num_classes):
        super().__init__()

        # LIFTING (RGB input)
        self.lift = V4GroupConv(3, 16, 3, 1, lifting=True)  # → 16*4 channels

        # Downsampling + depth
        self.block1 = V4Block(16, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = V4Block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = V4Block(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.lift(x))
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class V4Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = V4GroupConv(in_ch, out_ch, 3, 1, lifting=False)
        self.conv2 = V4GroupConv(out_ch, out_ch, 3, 1, lifting=False)
        self.skip = nn.Identity() if in_ch == out_ch else V4GroupConv(in_ch, out_ch, 1, 0, lifting=False)

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + identity)
