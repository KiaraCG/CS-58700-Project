import torch
import torch.nn as nn
import torch.nn.functional as F

from ..C4GroupConv import C4GroupConv

class C4EquivariantDigitsCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = C4GroupConv(in_channels, 8,  3, 1, lifting=True)
        self.conv2 = C4GroupConv(8,  16, 3, 1, lifting=False)
        self.conv3 = C4GroupConv(16, 32, 3, 1, lifting=False)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # ← keep some spatial info
        self.fc1 = nn.Linear(2048, 128)
        self.fc   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).flatten(1)   # [B, 2048]
        x = F.relu(self.fc1(x))
        return self.fc(x)