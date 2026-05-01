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
    """
    FIXED: Rotates FIRST, then flips. Because D4 is non-commutative,
    applying the flip first results in incorrect mappings for sr and sr³.
    """
    if   g == 0: return w
    elif g == 1: return torch.rot90(w, 1, [2, 3])
    elif g == 2: return torch.rot90(w, 2, [2, 3])
    elif g == 3: return torch.rot90(w, 3, [2, 3])
    elif g == 4: return torch.flip(w, [3])
    elif g == 5: return torch.flip(torch.rot90(w, 1, [2, 3]), [3])
    elif g == 6: return torch.flip(torch.rot90(w, 2, [2, 3]), [3])
    else:        return torch.flip(torch.rot90(w, 3, [2, 3]), [3])


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

        # FIXED: Group-to-Group convs (lifting=False) must accept 8 input group orientations
        if self.lifting:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, self.GROUP_SIZE, in_channels, kernel_size, kernel_size)
            )
            
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        if self.lifting:
            weights = torch.cat(
                [_d4_transform_weight(w, g) for g in range(8)], dim=0
            )                                                   # [8*out_ch, in_ch, K, K]
            return F.conv2d(x, weights, padding=self.padding)

        else:
            # FIXED: Properly index and transform the 8 input orientations for the group elements
            O, _, I, K, _ = w.shape
            combined = torch.zeros(8*O, 8*I, K, K, device=x.device)
            
            for g in range(8):
                perm = _D4_PERM[g]
                for h in range(8):
                    ph = perm[h]  # ph represents g^{-1} * h
                    
                    w_slice = w[:, ph, :, :, :]                   # [O, I, K, K]
                    w_g = _d4_transform_weight(w_slice, g)        # Transform spatially by g
                    
                    # Connect input orientation h to output orientation g
                    combined[g*O:(g+1)*O, h*I:(h+1)*I] = w_g
                    
            return F.conv2d(x, combined, padding=self.padding)


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

        # FIXED: To keep the skip connection strictly equivariant, we use a 1x1 Group Conv 
        # instead of a grouped standard Conv2D (which would learn independent filters per orientation).
        if in_ch != out_ch:
            self.skip = D4GroupConv(in_ch, out_ch, kernel_size=1, padding=0, lifting=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + identity)
    

class D4Backbone(nn.Module):
    """
    D4-equivariant backbone: lift → 3 group conv blocks.

    out_channels = 128 per group element
    group_size   = 8  (4 rotations × 2 reflections)
    tensor depth = 1024 channels  ← 2× C4Backbone memory

    Memory tip: if OOM, set out_channels=64 here and in MultiGatedCNN heads.
    """
    out_channels = 128
    group_size   = 8

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.lift   = D4GroupConv(in_channels, 16, 3, padding=1, lifting=True)
        self.block1 = D4Block(16,  32)
        self.pool1  = nn.MaxPool2d(2)
        self.block2 = D4Block(32,  64)
        self.pool2  = nn.MaxPool2d(2)
        self.block3 = D4Block(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lift(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)          # [B, 1024, H, W]
        return x


class D4CNNDualHead(nn.Module):
    """
    D4-equivariant backbone with two hard-gated heads.
    Mirrors the interface of StandardCNNDualHead exactly.

    For inaturalist_mnist:
        task = "bird"  → bird_head  (10 bird species classes)
        task = "digit" → digit_head (10 digit classes)

    For mnist_svhn:
        task = "bird"  → bird_head  (mnist — rotation invariant task)
        task = "digit" → digit_head (svhn  — rotation sensitive task)
    """

    DATASET_CFG = {
        'inaturalist_mnist': (3, (3, 64, 64)),  # RGB 64×64 — in_channels and dummy shape must agree
        'mnist_svhn':        (3, (3, 32, 32)),  # RGB 32×32
    }
    
    def __init__(self, dataset: str, num_bird_classes: int = 10, num_digit_classes: int = 10):
        super().__init__()

        if dataset not in self.DATASET_CFG:
            raise ValueError(f"D4CNNDualHead only supports: {list(self.DATASET_CFG)}")

        in_channels, input_size = self.DATASET_CFG[dataset]

        self.backbone  = D4Backbone(in_channels)
        self.dropout   = nn.Dropout(0.5)

        self._to_linear = self._get_conv_output(input_size)
        self.fc_shared  = nn.Linear(self._to_linear, 256)

        self.bird_head  = nn.Linear(256, num_bird_classes)
        self.digit_head = nn.Linear(256, num_digit_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            feat = self.backbone(x)           # [1, 1024, H, W]
            feat = feat.mean(dim=[-2, -1])    # Spatial Pool -> [1, 1024]
            # FIXED: Group Pool for output sizing -> [1, 128]
            feat = feat.view(1, 8, -1).mean(dim=1) 
            return feat.shape[1]

    def _backbone_features(self, x):
        feat = self.backbone(x)           # [B, 1024, H, W]
        feat = feat.mean(dim=[-2, -1])    # [B, 1024]
        
        # FIXED: Reynolds Operator (Group Pooling) to force rotational/reflection invariance
        B = feat.size(0)
        feat = feat.view(B, 8, -1).mean(dim=1)  # [B, 128]
        
        feat = self.dropout(feat)
        return F.relu(self.fc_shared(feat))

    def forward(self, x: torch.Tensor, tasks):
        feats = self._backbone_features(x)

        bird_mask  = torch.tensor([t == "bird"  for t in tasks], device=x.device)
        digit_mask = torch.tensor([t == "digit" for t in tasks], device=x.device)

        out = torch.zeros(
            x.size(0),
            max(self.bird_head.out_features, self.digit_head.out_features),
            device=x.device
        )

        if bird_mask.any():
            out[bird_mask,  :self.bird_head.out_features]  = self.bird_head(feats[bird_mask])
        if digit_mask.any():
            out[digit_mask, :self.digit_head.out_features] = self.digit_head(feats[digit_mask])

        return out, bird_mask, digit_mask