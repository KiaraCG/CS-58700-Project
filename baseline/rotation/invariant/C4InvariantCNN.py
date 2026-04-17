import torch
import torch.nn as nn
import torch.nn.functional as F

from ..C4Block import C4Block
from ..C4GroupConv import C4GroupConv


class C4InvariantCNN(nn.Module):
    """
    Rotation-invariant CNN for joint bird species and digit classification.
    """
    def __init__(self, num_bird_classes,in_channels=3):
        super().__init__()

        self.lift = C4GroupConv(in_channels, 16, 3, 1, lifting=True)

        self.block1 = C4Block(16, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = C4Block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = C4Block(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Separate heads
        self.bird_head = nn.Linear(128, num_bird_classes)
        self.digit_head = nn.Linear(128, 10)

    def forward(self, x, tasks):
        x = F.relu(self.lift(x))
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = x.view(x.size(0), 4, 128).mean(dim=1)   # [B, 128]

        # Route each sample to the correct head
        out = torch.zeros(x.size(0), max(self.bird_head.out_features,
                                        self.digit_head.out_features),
                        device=x.device)

        bird_mask  = torch.tensor([t == "bird"  for t in tasks], device=x.device)
        digit_mask = torch.tensor([t == "digit" for t in tasks], device=x.device)

        if bird_mask.any():
            out[bird_mask, :self.bird_head.out_features] = self.bird_head(x[bird_mask])
        if digit_mask.any():
            out[digit_mask, :self.digit_head.out_features] = self.digit_head(x[digit_mask])

        return out, bird_mask, digit_mask