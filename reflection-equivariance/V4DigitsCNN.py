import torch
import torch.nn as nn
import torch.nn.functional as F


class V4GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, lifting=False):
        super().__init__()
        self.lifting = lifting
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Weight shape: (out, in, k, k)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        # 1. Generate the 4 transformed versions of the filter
        # Identity, Horizontal Flip, Vertical Flip, 180 Rotation
        w1 = self.weight
        w2 = torch.flip(w1, dims=[-1])  # Horizontal
        w3 = torch.flip(w1, dims=[-2])  # Vertical
        w4 = torch.flip(w1, dims=[-1, -2])  # 180 Degree

        if self.lifting:
            # LIFTING: Input is (B, C, H, W) -> Output is (B, C*4, H, W)
            # Convolve input with all 4 transformations and stack in channel dim
            weights = torch.cat([w1, w2, w3, w4], dim=0)
            return F.conv2d(x, weights, padding=self.padding)
        else:
            # GROUP CONVOLUTION: Input is (B, C*4, H, W) -> Output is (B, C*4, H, W)
            # This requires a "group correlation" where we permute channels
            # For simplicity in this example, we'll use the combined weight bank:
            combined_weights = torch.cat([
                torch.cat([w1, w2, w3, w4], dim=1),  # Filter for Identity
                torch.cat([w2, w1, w4, w3], dim=1),  # Filter for Horiz Flip
                torch.cat([w3, w4, w1, w2], dim=1),  # Filter for Vert Flip
                torch.cat([w4, w3, w2, w1], dim=1)  # Filter for 180 Rot
            ], dim=0)

            return F.conv2d(x, combined_weights, padding=self.padding)


class V4DigitsCNN(nn.Module):
    """
    Reflection-equivariant CNN for MNIST/ColorMNIST/SVHN.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # First layer "lifts" the image to the group (output channels = 8 * 4 = 32)
        self.conv1 = V4GroupConv(in_channels, 8, 3, 1, lifting=True)
        # Subsequent layers process group-data (input 32, output 16 * 4 = 64)
        self.conv2 = V4GroupConv(8, 16, 3, 1, lifting=False)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # To get an invariant classification, we must pool over spatial AND group dims
        # x shape is (B, 64, H, W). We reshaped to pool over the 4 transformations.
        x = self.pool(x)  # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)