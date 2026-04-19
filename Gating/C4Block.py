import torch.nn as nn
import torch.nn.functional as F

from C4GroupConv import C4GroupConv

class C4Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = C4GroupConv(in_ch, out_ch, 3, 1, lifting=False)
        # self.bn1   = nn.BatchNorm2d(out_channels * 4)
        self.conv2 = C4GroupConv(out_ch, out_ch, 3, 1, lifting=False)
        #  self.bn2   = nn.BatchNorm2d(out_channels * 4)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch * 4, out_ch * 4, kernel_size=1, groups=4)
        else:
            self.skip = nn.Identity()
    def forward(self, x):
        identity = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + identity)
    



