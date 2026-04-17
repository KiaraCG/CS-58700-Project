import torch
import torch.nn as nn
import torch.nn.functional as F

from ..C4GroupConv import C4GroupConv

class C4InvariantDigitsCNN(nn.Module):
    """
    Rotation-invariant CNN for MNIST/ColorMNIST/SVHN.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = C4GroupConv(in_channels, 8,  3, 1, lifting=True)   # → 32ch
        self.conv2 = C4GroupConv(8,  16, 3, 1, lifting=False)           # → 64ch
        self.conv3 = C4GroupConv(16, 32, 3, 1, lifting=False)           # → 128ch

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.pool(x).flatten(1)             # [B, 128]
        # print(x.shape)                        # should be [B, 128]
        x = x.view(x.size(0), 4, 32).mean(1)
        # print(x.shape)                        # should be [B, 32]
        return self.fc(x)