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
from torchvision import models

from backbone import C4Backbone, D4Backbone, C4GroupConv, D4GroupConv 

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
        num_classes: int,
        in_channels:      int  = 3,
        backbone_cls             = None,    # C4Backbone (default)
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
        self.svhn_head_inv   = nn.Linear(C,  num_classes) # dataset 1 - svhn | inaturalist
        self.mnist_head_inv  = nn.Linear(C,  10) # dataset 2

        self.svhn_head_equi  = nn.Linear(CG, num_classes)
        self.mnist_head_equi = nn.Linear(CG, 10)

   
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
        raw_feat = self.backbone(x)                             # [B, CG, H, W]

        if hasattr(raw_feat, 'tensor'):
            feat = raw_feat.tensor
        else:
            feat = raw_feat

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
            out = g_blend * self.svhn_head_inv(inv) + (1 - g_blend) * self.svhn_head_equi(equi)
            return out, gates

        # Multi-class mode — route to correct head per sample
        # bird_mask  = torch.tensor([t == "bird"  for t in tasks], device=x.device)
        # digit_mask = torch.tensor([t == "digit" for t in tasks], device=x.device)

        svhn_mask  = torch.tensor([t == "svhn"  for t in tasks], device=x.device)
        mnist_mask = torch.tensor([t == "mnist" for t in tasks], device=x.device)
        n_svhn  = self.svhn_head_inv.out_features
        n_mnist = self.mnist_head_inv.out_features
        out     = torch.zeros(x.size(0), max(n_svhn, n_mnist), device=x.device)

        if svhn_mask.any():
            g  = g_blend[svhn_mask]
            oi = self.svhn_head_inv(inv[svhn_mask])
            oe = self.svhn_head_equi(equi[svhn_mask])
            out[svhn_mask, :n_svhn] = g * oi + (1 - g) * oe

        if mnist_mask.any():
            g  = g_blend[mnist_mask]
            oi = self.mnist_head_inv(inv[mnist_mask])
            oe = self.mnist_head_equi(equi[mnist_mask])
            out[mnist_mask, :n_mnist] = g * oi + (1 - g) * oe

        return out, gates, svhn_mask, mnist_mask