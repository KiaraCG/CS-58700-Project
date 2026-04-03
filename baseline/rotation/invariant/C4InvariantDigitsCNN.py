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
        # First layer "lifts" the image to the group (output channels = 8 * 4 = 32)
        self.conv1 = C4GroupConv(in_channels, 8, 3, 1, lifting=True)
        # Subsequent layers process group-data (input 32, output 16 * 4 = 64)
        self.conv2 = C4GroupConv(8, 16, 3, 1, lifting=False)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # To get an invariant classification, we must pool over spatial AND group dims
        # x shape is (B, 64, H, W). We reshaped to pool over the 4 transformations.
        x = self.pool(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), 128, 4)          # [B, 128, 4]  ← split group dim
        x = x.mean(dim=2)                       # [B, 128]     ← pool over group
        return self.fc(x)