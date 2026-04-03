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
        # Generate the 4 rotated versions of the filter
        w1 = self.weight                               # 0°
        w2 = torch.rot90(self.weight, 1, [-2, -1])    # 90°
        w3 = torch.rot90(self.weight, 2, [-2, -1])    # 180°
        w4 = torch.rot90(self.weight, 3, [-2, -1])    # 270°

        if self.lifting:
            # Input: (B, C, H, W) → Output: (B, C*4, H, W)
            # Apply each rotated filter independently — same as V4
            weights = torch.cat([w1, w2, w3, w4], dim=0)
            return F.conv2d(x, weights, padding=self.padding)
        else:
            # Group conv: permutation follows C4 Cayley table
            # Entry (g_out, g_in) = w[ (g_in - g_out) mod 4 ]
            #   V4 was: [w2,w1,w4,w3] for g_out=1  ← swap pairs
            #   C4 is:  [w4,w1,w2,w3] for g_out=1  ← cyclic shift
            combined_weights = torch.cat([
                torch.cat([w1, w2, w3, w4], dim=1),  # g_out=0°
                torch.cat([w4, w1, w2, w3], dim=1),  # g_out=90°  ← differs from V4
                torch.cat([w3, w4, w1, w2], dim=1),  # g_out=180°
                torch.cat([w2, w3, w4, w1], dim=1),  # g_out=270° ← differs from V4
            ], dim=0)
            return F.conv2d(x, combined_weights, padding=self.padding)
