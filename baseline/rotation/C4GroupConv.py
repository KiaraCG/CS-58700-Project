import torch
import torch.nn as nn
import torch.nn.functional as F


class C4GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, lifting=False):
        super().__init__()
        self.lifting = lifting
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        w1 = self.weight
        w2 = torch.rot90(self.weight, 1, [-2, -1])
        w3 = torch.rot90(self.weight, 2, [-2, -1])
        w4 = torch.rot90(self.weight, 3, [-2, -1])

        if self.lifting:
            weights = torch.cat([w1, w2, w3, w4], dim=0).to(x.device)
            return F.conv2d(x, weights, padding=self.padding)
        else:
            combined_weights = torch.cat([
                torch.cat([w1, w2, w3, w4], dim=1),
                torch.cat([w4, w1, w2, w3], dim=1),
                torch.cat([w3, w4, w1, w2], dim=1),
                torch.cat([w2, w3, w4, w1], dim=1),
            ], dim=0).to(x.device)
            return F.conv2d(x, combined_weights, padding=self.padding)