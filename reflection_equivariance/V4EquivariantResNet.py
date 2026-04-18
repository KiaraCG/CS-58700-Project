import torch
import torch.nn as nn
import torch.nn.functional as F

# Very similar to the one in V4DigitsCNN, but adapted for ResNet-style blocks and more channels.
class V4GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, lifting=False):
        super().__init__()
        self.lifting = lifting
        self.stride = stride
        self.padding = padding

        # Base weight: (out, in, k, k)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Generate the 4 transformed versions of the filter
        w1 = self.weight
        w2 = torch.flip(w1, dims=[-1])  # Horizontal Flip
        w3 = torch.flip(w1, dims=[-2])  # Vertical Flip
        w4 = torch.flip(w1, dims=[-1, -2])  # 180 Degree Rotation

        if self.lifting:
            # Input: (B, 3, H, W) -> Output: (B, C_out*4, H, W)
            weights = torch.cat([w1, w2, w3, w4], dim=0)
            return F.conv2d(x, weights, padding=self.padding, stride=self.stride)
        else:
            # Input: (B, C_in*4, H, W) -> Output: (B, C_out*4, H, W)
            # We permute the filter-input pairings to satisfy the V4 group law
            combined_weights = torch.cat([
                torch.cat([w1, w2, w3, w4], dim=1),  # Identity path
                torch.cat([w2, w1, w4, w3], dim=1),  # H-Flip path
                torch.cat([w3, w4, w1, w2], dim=1),  # V-Flip path
                torch.cat([w4, w3, w2, w1], dim=1)  # 180-Rot path
            ], dim=0)
            return F.conv2d(x, combined_weights, padding=self.padding, stride=self.stride)


class V4ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # Internal channels are base_channels; actual tensor channels are base_channels * 4
        self.conv1 = V4GroupConv(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c * 4)
        self.conv2 = V4GroupConv(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c * 4)

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            # Use a 1x1 V4GroupConv to match dimensions in the skip connection
            self.shortcut = nn.Sequential(
                V4GroupConv(in_c, out_c, kernel_size=1, padding=0, stride=stride),
                nn.BatchNorm2d(out_c * 4)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class V4EquivariantResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial Lifting: 3 RGB -> 16 base channels (64 total)
        self.stem = nn.Sequential(
            V4GroupConv(3, 16, kernel_size=7, padding=3, stride=2, lifting=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet Layers (using base channel counts)
        self.layer1 = V4ResBlock(16, 16)
        self.layer2 = V4ResBlock(16, 32, stride=2)
        self.layer3 = V4ResBlock(32, 64, stride=2)

        # Invariant Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 1. Spatial Pooling
        x = self.gap(x)  # Shape: (B, 64*4, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (B, 256)

        # 2. Symmetry Group Pooling (The Invariance Step)
        # Reshape to (Batch, Group_Size, Base_Channels)
        x = x.view(x.size(0), 4, -1)
        x = x.mean(dim=1)  # Average over the 4 orientations

        return self.classifier(x)

