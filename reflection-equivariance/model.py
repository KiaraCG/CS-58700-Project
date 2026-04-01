import torch
import torch.nn as nn
import torch.nn.functional as F

class V4Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding

    def forward(self, x):
        # Project weights to equivariant subspace
        w_equiv = project_to_equivariant_subspace(self.weight)

        # Standard convolution
        y = F.conv2d(x, w_equiv, bias=self.bias, padding=self.padding)
        return y


class V4CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = V4Conv(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = V4Conv(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def project_to_equivariant_subspace(weight):
    """
    weight: (C_out, C_in, K, K) tensor
    returns: projected weight in equivariant subspace
    """
    w = weight

    # Apply group elements
    w_h = torch.flip(w, dims=[-1])  # horizontal
    w_v = torch.flip(w, dims=[-2])  # vertical
    w_hv = torch.flip(w, dims=[-1, -2]) # 180 rotation

    w_equiv = (w + w_h + w_v + w_hv) / 4.0

    return w_equiv