import torch
import torch.nn as nn
import torch.nn.functional as F

# Ignoring color for the time being
# ── D4 group element index convention ────────────────────────────────────────
#
#  Index  Element  Meaning
#  ─────  ───────  ───────────────────────────────
#    0      e      identity
#    1      r      rotate 90° CCW
#    2      r²     rotate 180°
#    3      r³     rotate 270° CCW
#    4      s      horizontal flip
#    5      sr     horizontal flip + rotate 90°
#    6      sr²    horizontal flip + rotate 180°
#    7      sr³    horizontal flip + rotate 270°
#
#  Group law: r⁴ = e,  s² = e,  s r s⁻¹ = r⁻¹
# ─────────────────────────────────────────────────────────────────────────────

_D4_PERM = [
    [0, 1, 2, 3, 4, 5, 6, 7],   # g=0  (inv = e)
    [3, 0, 1, 2, 5, 6, 7, 4],   # g=1  (inv = r³)
    [2, 3, 0, 1, 6, 7, 4, 5],   # g=2  (inv = r²)
    [1, 2, 3, 0, 7, 4, 5, 6],   # g=3  (inv = r)
    [4, 5, 6, 7, 0, 1, 2, 3],   # g=4  (inv = s)
    [5, 6, 7, 4, 3, 0, 1, 2],   # g=5  (inv = sr)
    [6, 7, 4, 5, 2, 3, 0, 1],   # g=6  (inv = sr²)
    [7, 4, 5, 6, 1, 2, 3, 0],   # g=7  (inv = sr³)
]


def _d4_transform_weight(w: torch.Tensor, g: int) -> torch.Tensor:
    if   g == 0: return w
    elif g == 1: return torch.rot90(w, 1, [2, 3])
    elif g == 2: return torch.rot90(w, 2, [2, 3])
    elif g == 3: return torch.rot90(w, 3, [2, 3])
    elif g == 4: return torch.flip(w, [3])
    elif g == 5: return torch.rot90(torch.flip(w, [3]), 1, [2, 3])
    elif g == 6: return torch.rot90(torch.flip(w, [3]), 2, [2, 3])
    else:        return torch.rot90(torch.flip(w, [3]), 3, [2, 3])


class D4GroupConv(nn.Module):
    """
    D4 Group Convolution (dihedral group of order 8: 4 rotations × 2 reflections).

    'in_channels' and 'out_channels' refer to channels PER GROUP ELEMENT.
    Actual tensor channel depth = channels × 8.

    lifting=True  : Z2 → D4  — first layer (takes a regular image)
    lifting=False : D4 → D4  — subsequent layers
    """
    GROUP_SIZE = 8

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, lifting=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.padding      = padding
        self.lifting      = lifting

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight)

    def _transformed_weights(self):
        return [_d4_transform_weight(self.weight, g) for g in range(8)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        w = self.weight  # [out_ch, in_ch, K, K]

        if self.lifting:
            weights = torch.cat(
                [_d4_transform_weight(w, g) for g in range(8)], dim=0
            ).to(x.device)                                  # [8*out_ch, in_ch, K, K]
            return F.conv2d(x, weights, padding=self.padding)

        else:
            # Build [8*out_ch, 8*in_ch, K, K] combined weight
            # Row block g uses g-transformed filter, columns permuted by _D4_PERM[g]
            O, I, K, _ = w.shape
            combined = torch.zeros(8*O, 8*I, K, K, device=x.device)
            for g in range(8):
                w_g  = _d4_transform_weight(w, g)           # [O, I, K, K]
                perm = _D4_PERM[g]
                for h, ph in enumerate(perm):
                    combined[g*O:(g+1)*O, ph*I:(ph+1)*I] = w_g
            return F.conv2d(x, combined, padding=self.padding)
