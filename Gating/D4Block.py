import torch
import torch.nn as nn
import torch.nn.functional as F

from D4GroupConv import D4GroupConv

class D4Block(nn.Module):
    """
    Two-layer D4 group conv block with skip connection.
    Mirrors C4Block exactly — no BatchNorm, residual path instead.

    in/out_channels are per-group-element; actual tensor depth is ×8.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = D4GroupConv(in_ch,  out_ch, 3, padding=1, lifting=False)
        self.conv2 = D4GroupConv(out_ch, out_ch, 3, padding=1, lifting=False)

        # Skip: project channels if in_ch != out_ch, using groups=8 to stay D4-equivariant
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch * 8, out_ch * 8, kernel_size=1, groups=8)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + identity)