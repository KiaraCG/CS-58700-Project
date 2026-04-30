"""
MultiGatedCNN.py
================
Soft multi-gate group CNN for joint classification.

Backbones (from backbone.py):
    PlainResNetBackbone      — no group averaging (baseline)
    C4ResNetBackbone         — 4 rotations
    C4ResNetBackbone_TSBN    — C4 + frozen early layers + task-specific BN
    D4ResNetBackbone         — 4 rotations + reflections

Gates:
    NoGate          — fixed equal blend (ablation baseline)
    LearnableGate   — learnable temperature + Gumbel noise
"""

import torch
import torch.nn as nn

from Gating.backbone import C4ResNetBackbone_TSBN

# ── Gates ──────────────────────────────────────────────────────────────────────

class NoGate(nn.Module):
    """Fixed equal blend — gate contributes nothing. Baseline for gate ablation."""
    def __init__(self, in_features: int, n_choices: int = 3):
        super().__init__()
        self.n = n_choices

    def forward(self, x: torch.Tensor):
        B       = x.size(0)
        gates   = torch.full((B, self.n), 1.0 / self.n, device=x.device)
        entropy = torch.zeros(B, device=x.device)
        return gates, entropy


class LearnableGate(nn.Module):
    """
    Softmax gate with learnable temperature + Gumbel noise.
    - log_temp: starts at init_temp, sharpens toward discrete over training
    - Gumbel noise: encourages commitment (training only)
    - Returns entropy for regularization loss
    """
    def __init__(self, in_features: int, n_choices: int = 3, init_temp: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, n_choices),
        )
        self.log_temp = nn.Parameter(torch.tensor(init_temp).log())

    def forward(self, x: torch.Tensor):
        logits = self.net(x)                                        # [B, n_choices]
        temp   = self.log_temp.exp().clamp(min=0.01)

        if self.training:
            gumbel = -torch.empty_like(logits).exponential_().log()
            logits = logits + gumbel

        gates   = torch.softmax(logits / temp, dim=-1)             # [B, n_choices]
        entropy = -(gates * (gates + 1e-8).log()).sum(dim=-1)      # [B]
        return gates, entropy


# ── Model ──────────────────────────────────────────────────────────────────────

class MultiGatedCNN(nn.Module):
    """
    Args:
        num_classes : number of classes for task_a (svhn=10, inaturalist=10)
        in_channels : input image channels (3 for RGB)
        backbone_cls: one of the backbone classes above
        gate_cls    : NoGate or LearnableGate
    """



    def __init__(
        self,
        num_classes  : int,
        in_channels  : int  = 3,
        backbone_cls        = None,
        gate_cls            = None,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────

        backbone_cls = backbone_cls 
        gate_cls     = gate_cls     or LearnableGate

        self.backbone = backbone_cls(in_channels=in_channels)
        self.gate     = gate_cls(in_features=self.backbone.out_channels * self.backbone.group_size)
        C  = self.backbone.out_channels     # channels per group element
        G  = self.backbone.group_size       # 1 (plain), 4 (C4), 8 (D4)
        CG = C * G                          # total feature channels

        self._C = C
        self._G = G

        # ── Gate ──────────────────────────────────────────────────────────────

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Classification heads ───────────────────────────────────────────────
        # task_a: svhn or inaturalist
        # task_b: mnist (always 10 classes)
        self.task_a_head_inv  = nn.Linear(C,  num_classes)
        self.task_b_head_inv  = nn.Linear(C,  10)

        self.task_a_head_equi = nn.Linear(CG, num_classes)
        self.task_b_head_equi = nn.Linear(CG, 10)

    # ── Symmetry ops ──────────────────────────────────────────────────────────

    def _apply_rotation_invariance(self, f: torch.Tensor) -> torch.Tensor:
        return self.global_pool(f).expand_as(f)

    @staticmethod
    def _apply_reflection_invariance(f: torch.Tensor) -> torch.Tensor:
        return (f + torch.flip(f, dims=[3])) / 2.0

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, tasks=None):
        # 1. Backbone — C4ResNetBackbone_TSBN needs task_idx for its BN layers
        if isinstance(self.backbone, C4ResNetBackbone_TSBN) and tasks is not None:
            task_a_count = sum(t in ("svhn", "bird") for t in tasks)
            task_idx     = 0 if task_a_count >= len(tasks) // 2 else 1
            raw_feat     = self.backbone(x, task_idx)
        else:
            raw_feat = self.backbone(x)

        feat = raw_feat.tensor if hasattr(raw_feat, 'tensor') else raw_feat
        # feat: [B, CG, H, W]

        # 2. Gate — operates on spatially pooled features
        pooled_for_gate         = self.global_pool(feat).flatten(1)    # [B, CG]
        gates, gate_entropy     = self.gate(pooled_for_gate)           # [B, 3], [B]

        # 3. Blend: identity / rotation-invariant / reflection-invariant
        g   = gates.view(gates.size(0), 3, 1, 1, 1)
        f   = (g[:, 0] * feat
             + g[:, 1] * self._apply_rotation_invariance(feat)
             + g[:, 2] * self._apply_reflection_invariance(feat))      # [B, CG, H, W]

        # 4. Two paths from blended features
        pooled  = self.global_pool(f).flatten(1)                       # [B, CG]
        inv     = pooled.view(pooled.size(0), self._G, self._C).mean(dim=1)  # [B, C]
        equi    = pooled                                               # [B, CG]

        # rotation gate weight drives inv/equi blend
        g_blend = gates[:, 1].unsqueeze(1)                            # [B, 1]

        # 5. Binary mode (no task labels)
        if tasks is None:
            out = (g_blend       * self.task_a_head_inv(inv)
                 + (1 - g_blend) * self.task_a_head_equi(equi))
            return out, gates, gate_entropy

        # 6. Multi-task mode — route each sample to its head
        task_a_mask = torch.tensor([t in ("svhn", "bird")  for t in tasks], device=x.device)
        task_b_mask = torch.tensor([t in ("digit", "mnist") for t in tasks], device=x.device)

        n_a = self.task_a_head_inv.out_features
        n_b = self.task_b_head_inv.out_features
        out = torch.zeros(x.size(0), max(n_a, n_b), device=x.device)

        if task_a_mask.any():
            g_ = g_blend[task_a_mask]
            out[task_a_mask, :n_a] = (g_       * self.task_a_head_inv(inv[task_a_mask])
                                    + (1 - g_) * self.task_a_head_equi(equi[task_a_mask]))

        if task_b_mask.any():
            g_ = g_blend[task_b_mask]
            out[task_b_mask, :n_b] = (g_       * self.task_b_head_inv(inv[task_b_mask])
                                    + (1 - g_) * self.task_b_head_equi(equi[task_b_mask]))

        return out, gates, gate_entropy, task_a_mask, task_b_mask