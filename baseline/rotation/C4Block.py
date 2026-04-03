import torch.nn as nn
import torch.nn.functional as F

from C4GroupConv import C4GroupConv

class C4Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = C4GroupConv(in_ch, out_ch, 3, 1, lifting=False)
        self.conv2 = C4GroupConv(out_ch, out_ch, 3, 1, lifting=False)
        self.skip = nn.Identity() if in_ch == out_ch else C4GroupConv(in_ch, out_ch, 1, 0, lifting=False)

    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + identity)
