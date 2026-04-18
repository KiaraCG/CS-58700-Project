import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalD4Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        # Standard kernel weights
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight)

    def get_d4_kernels(self, w):
        """Generates all 8 D4 transformations of the kernel."""
        w0 = w
        w1 = torch.rot90(w, 1, [2, 3])
        w2 = torch.rot90(w, 2, [2, 3])
        w3 = torch.rot90(w, 3, [2, 3])
        w4 = torch.flip(w0, [3])
        w5 = torch.flip(w1, [3])
        w6 = torch.flip(w2, [3])
        w7 = torch.flip(w3, [3])
        return [w0, w1, w2, w3, w4, w5, w6, w7]

    def forward(self, x, alpha):
        batch_size = x.size(0)

        # Create the D4-Symmetric kernel by averaging all 8 transformations
        kernels = self.get_d4_kernels(self.weight)
        w_inv = torch.stack(kernels).mean(dim=0)

        # Interpolate based on alpha: 0 = Standard Conv, 1 = Fully D4-Invariant
        alpha = alpha.view(batch_size, 1, 1, 1, 1)
        dynamic_weight = (1 - alpha) * self.weight.unsqueeze(0) + alpha * w_inv.unsqueeze(0)

        x_reshaped = x.view(1, -1, x.size(2), x.size(3))
        w_reshaped = dynamic_weight.view(-1, self.weight.size(1), self.weight.size(2), self.weight.size(3))

        out = F.conv2d(x_reshaped, w_reshaped, stride=self.stride,
                       padding=self.padding, groups=batch_size)

        return out.view(batch_size, self.weight.size(0), out.size(2), out.size(3))


class ConditionalD4CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.conv1 = ConditionalD4Conv(3, 32, 3)
        self.conv2 = ConditionalD4Conv(32, 64, 3)
        self.conv3 = ConditionalD4Conv(64, 128, 3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. Determine D4-invariance requirement for this specific input
        alpha = self.controller(x)

        # 2. Main pipeline with conditional kernels
        x = F.relu(self.conv1(x, alpha))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x, alpha))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x, alpha))

        # 3. Global Spatial Invariance
        x = self.pool(x).view(x.size(0), -1)

        # Note: True G-Invariance usually involves pooling over the Group dimension.
        # Since we are "symmetrizing" the filter itself, the feature map produced
        # by a symmetric filter is inherently invariant to the input's D4 transformations.
        return self.fc(x)