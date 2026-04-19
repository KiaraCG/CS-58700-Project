"""
Multi-Gate Group CNN
====================
Soft-gated against Hard-gating CNN that blends rotation-invariant and rotation-equivariant paths.
Supports both C4 and D4 backbones as drop-in swaps.

Gates (sigmoid → 0…1):
    g_rot  →  1 = trust rotation-invariant path   (C4 and D4)
    g_ref  →  1 = trust reflection-invariant path  (meaningful only with D4Backbone)
    # g_col →  1 = trust color-invariant path      - not baseline has been implemented yet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from C4Block import C4Block
from C4GroupConv import C4GroupConv
from D4Block import D4Block
from D4GroupConv import D4GroupConv

# ── Backbones ─────────────────────────────────────────────────────────────────

class C4Backbone(nn.Module):
    """
    C4-equivariant backbone: lift → 3 group conv blocks.

    out_channels = 128 per group element
    group_size   = 4
    tensor depth = 512 channels
    """
    out_channels = 128
    group_size   = 4

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.lift   = C4GroupConv(in_channels, 16, 3, padding=1, lifting=True)
        self.block1 = C4Block(16,  32)
        self.pool1  = nn.MaxPool2d(2)
        self.block2 = C4Block(32,  64)
        self.pool2  = nn.MaxPool2d(2)
        self.block3 = C4Block(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lift(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        return self.block3(x)                   # [B, 512, H, W]


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
        return self.block3(x)                   # [B, 1024, H, W]


# ── Multi-Gate CNN ─────────────────────────────────────────────────────────────

class MultiGatedCNN(nn.Module):
    """
    Soft multi-gate group CNN for joint bird/digit classification.

    The key experiment: does g_rot diverge between bird and digit samples?
        birds  → expect g_rot → 1  (network learns rotation invariance)
        digits → expect g_rot → 0  (network learns rotation sensitivity: 6 ≠ 9)

    With D4Backbone, g_ref adds a second axis:
        birds  → g_ref may → 1  (flipped bird is still a bird)
        digits → g_ref → 0      (flipped digit may change class)
    """

    def __init__(
        self,
        num_bird_classes: int,
        in_channels:      int  = 3,
        backbone_cls             = None,    # C4Backbone (default) or D4Backbone
    ):
        super().__init__()

        backbone_cls     = backbone_cls or C4Backbone
        self.backbone    = backbone_cls(in_channels)

        C  = self.backbone.out_channels     # channels per group element  (128)
        G  = self.backbone.group_size       # 4 (C4) or 8 (D4)
        CG = C * G                          # total feature channels (512 or 1024)

        self._C = C
        self._G = G

        # ── Gate ──────────────────────────────────────────────────────────────
        # Output: [g_rot, g_ref]
        # To add color gate: change Linear(64, 2) → Linear(64, 3)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(CG, 64),
            nn.ReLU(),
            nn.Linear(64, 2),               # ← change to 3 for +g_col
            nn.Sigmoid(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Classification heads ───────────────────────────────────────────────
        self.bird_head_inv   = nn.Linear(C,  num_bird_classes)
        self.digit_head_inv  = nn.Linear(C,  10)

        self.bird_head_equi  = nn.Linear(CG, num_bird_classes)
        self.digit_head_equi = nn.Linear(CG, 10)

   
    def _apply_rotation_invariance(self, f: torch.Tensor) -> torch.Tensor:
        return self.global_pool(f).expand_as(f)

    @staticmethod
    def _apply_reflection_invariance(f: torch.Tensor) -> torch.Tensor:
        return (f + torch.flip(f, dims=[3])) / 2.0

    # @staticmethod
    # def _apply_color_invariance(f: torch.Tensor) -> torch.Tensor:
    #     Not Implemented Error

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(self, x, tasks=None):
        # 1. Backbone
        feat = self.backbone(x)                             # [B, CG, H, W]

        # 2. Gate
        gates = self.gate(feat)                             # [B, 2]
        g_rot = gates[:, 0].view(-1, 1, 1, 1)
        g_ref = gates[:, 1].view(-1, 1, 1, 1)

        # 3. Blend features
        f = g_ref * self._apply_reflection_invariance(feat) + (1 - g_ref) * feat
        f = g_rot * self._apply_rotation_invariance(f)      + (1 - g_rot) * f

        # 4. Two paths
        pooled = self.global_pool(f).flatten(1)             # [B, CG]
        inv    = pooled.view(pooled.size(0), self._G, self._C).mean(dim=1)  # [B, C]
        equi   = pooled                                     # [B, CG]

        g_blend = gates[:, 0].unsqueeze(1)                  # [B, 1]

        # 5. Route by task
        if tasks is None:
            # Binary mode — use bird head for both classes
            out = g_blend * self.bird_head_inv(inv) + (1 - g_blend) * self.bird_head_equi(equi)
            return out, gates

        # Multi-class mode — route to correct head per sample
        bird_mask  = torch.tensor([t == "bird"  for t in tasks], device=x.device)
        digit_mask = torch.tensor([t == "digit" for t in tasks], device=x.device)

        n_bird  = self.bird_head_inv.out_features
        n_digit = self.digit_head_inv.out_features
        out     = torch.zeros(x.size(0), max(n_bird, n_digit), device=x.device)

        if bird_mask.any():
            g  = g_blend[bird_mask]
            oi = self.bird_head_inv(inv[bird_mask])
            oe = self.bird_head_equi(equi[bird_mask])
            out[bird_mask, :n_bird] = g * oi + (1 - g) * oe

        if digit_mask.any():
            g  = g_blend[digit_mask]
            oi = self.digit_head_inv(inv[digit_mask])
            oe = self.digit_head_equi(equi[digit_mask])
            out[digit_mask, :n_digit] = g * oi + (1 - g) * oe

        return out, gates, bird_mask, digit_mask