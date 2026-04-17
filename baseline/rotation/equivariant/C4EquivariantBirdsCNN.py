import torch.nn as nn
import torch.nn.functional as F

from ..C4Block import C4Block
from ..C4GroupConv import C4GroupConv


class C4EquivariantBirdsCNN(nn.Module):
    """
    Rotation-equivariant CNN for iNaturalist Birds dataset. Uses C4GroupConv for lifting and group convolution.
    """
    def __init__(self, num_classes):
        super().__init__()

        # LIFTING (RGB input)
        self.lift = C4GroupConv(3, 16, 3, 1, lifting=True)  # → 16*4 channels

        # Downsampling + depth
        self.block1 = C4Block(16, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = C4Block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = C4Block(64, 128)

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
