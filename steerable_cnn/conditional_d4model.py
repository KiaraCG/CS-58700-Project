import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalD4Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.stride, self.padding = stride, padding
        self.weight = nn.Parameter(torch.Tensor(out_ch, in_ch, k, k))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x, alpha):
        B = x.size(0)
        # 1. D4 Symmetrization (Average of 8 versions)
        w = self.weight
        w_rots = [torch.rot90(w, i, [2, 3]) for i in range(4)]
        w_all = w_rots + [torch.flip(r, [3]) for r in w_rots]
        w_inv = torch.stack(w_all).mean(0)

        # 2. Interpolate weights per sample in batch
        # alpha shape: (B, 1) -> (B, out_ch, in_ch, k, k)
        alpha = alpha.view(B, 1, 1, 1, 1)
        # Broadcast standard weight and invariant weight across the batch
        dynamic_w = (1 - alpha) * w.unsqueeze(0) + alpha * w_inv.unsqueeze(0)

        # 3. Optimized Batched Conv
        # We reshape input to (1, B*C, H, W) and use group convolution
        x_reshaped = x.view(1, -1, x.size(2), x.size(3))
        w_reshaped = dynamic_w.reshape(-1, self.in_ch, w.size(2), w.size(3))

        out = F.conv2d(x_reshaped, w_reshaped, stride=self.stride,
                       padding=self.padding, groups=B)
        return out.view(B, self.out_ch, out.size(2), out.size(3))


class Controller(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            # No Sigmoid here, we apply it in forward to allow bias control
        )

        # --- CRITICAL: Initialize to "Standard CNN" mode ---
        # A bias of -3.0 results in sigmoid(-3) approx 0.04 (mostly standard conv)
        nn.init.constant_(self.net[-1].bias, -3.0)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class ConditionalResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConditionalD4Conv(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = ConditionalD4Conv(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Handle skip connection dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x, alpha):
        out = F.relu(self.bn1(self.conv1(x, alpha)))
        out = self.bn2(self.conv2(out, alpha))
        out += self.shortcut(x)  # The Skip Connection
        return F.relu(out)


class ConditionalD4CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.controller = Controller(in_channels=3)

        self.layer1 = ConditionalResBlock(3, 32, stride=2)
        self.layer2 = ConditionalResBlock(32, 64, stride=2)
        self.layer3 = ConditionalResBlock(64, 128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        alpha = self.controller(x)

        x = self.layer1(x, alpha)
        x = self.layer2(x, alpha)
        x = self.layer3(x, alpha)

        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x)
