import torch
import torch.nn as nn
from torchvision import models


class _STEBinaryGate(torch.autograd.Function):
    """
    Straight-Through Estimator for a hard {0, 1} gate.

    Forward  :  α_hard = round(α_soft)   →  exactly 0 or 1
    Backward :  gradient passes through as if round() were the identity,
                i.e. ∂L/∂α_soft  ←  ∂L/∂α_hard   (no clipping needed
                because α_soft is already in (0,1) from sigmoid).

    This lets the SteeringController receive proper gradients even though
    its output is binarised before being used.
    """

    @staticmethod
    def forward(ctx, alpha_soft: torch.Tensor) -> torch.Tensor:
        return alpha_soft.round()  # 0.0 or 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output  # identity straight-through


def ste_binarize(alpha_soft: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper around _STEBinaryGate."""
    return _STEBinaryGate.apply(alpha_soft)


class SteeringController(nn.Module):
    """
    Produces a per-sample hard binary gate α ∈ {0, 1} from a low-resolution
    bottleneck representation of the input.

    α = 0  :  fully equivariant  (standard ResNet – pose-sensitive)
    α = 1  :  fully invariant    (D4-averaged    – pose-insensitive)

    The forward pass binarises the sigmoid output with a Straight-Through
    Estimator so the controller is still trained end-to-end.

    Args:
        in_channels (int): input image channels (default 3).
        bottleneck_dim (int): hidden width of the MLP.
        init_bias (float): initial logit bias fed into sigmoid.
                           0.0  → α_soft ≈ 0.5, so ~50 % of samples start
                           in each regime and both paths receive gradients.
    """

    def __init__(self,
                 in_channels: int = 3,
                 bottleneck_dim: int = 64,
                 init_bias: float = 0.0):
        super().__init__()

        self.encoder = nn.Sequential(
            # Aggressive spatial downsampling – only coarse statistics needed.
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # → [B, 32, 1, 1]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 1),  # scalar logit per sample
        )

        nn.init.constant_(self.fc[-1].bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images  [B, C, H, W]
        Returns:
            alpha_hard: [B, 1]  – exactly 0.0 or 1.0 in the forward pass;
                        gradients flow back via STE.
        """
        z = self.encoder(x)
        alpha_soft = torch.sigmoid(self.fc(z))  # [B, 1], values in (0, 1)
        alpha_hard = ste_binarize(alpha_soft)  # [B, 1], values in {0, 1}
        return alpha_hard


class SteerableResNet(nn.Module):
    """
    Steerable ResNet with a **hard binary gate** that routes each sample to
    either the equivariant path (α=0) or the D4-invariant path (α=1).

    Architecture overview
    ─────────────────────
                         ┌─────────────────────────────────┐
    x ──► SteeringCtrl ──►  α ∈ {0,1}  (hard, per-sample)  │
    │                    └─────────────────────────────────┘
    │                                  │
    │    ┌─────────────────────────────┼──────────────────────┐
    │    │  Standard path  (α=0)       │  D4-inv path  (α=1)  │
    │    │  backbone(x)                │  mean_k backbone(gₖx)│
    └────►  f_std                      │  f_inv               │
           └──── α·f_inv + (1-α)·f_std ──────────────────────►classifier

    Because α is binary the blending reduces to a hard selection:
        α=0  →  f_std     (standard, pose-sensitive)
        α=1  →  f_inv     (D4-invariant, pose-insensitive)

    Gradients for the SteeringController are supplied by the Straight-Through
    Estimator, so end-to-end training is still possible.

    The D4-invariant path is *only computed* for samples where α=1, saving
    ~8× backbone FLOPs for the equivariant subset.

    Args:
        num_classes (int): number of output classes.
        pretrained (bool): initialise backbone from ImageNet weights.
        bottleneck_dim (int): hidden size of the steering controller.
        inv_path_always (bool): if True, always compute the invariant path
            for all samples (simpler code, useful for debugging).  Default
            False uses the efficient selective-compute version.
    """

    def __init__(self,
                 num_classes: int = 10,
                 pretrained: bool = True,
                 bottleneck_dim: int = 64,
                 inv_path_always: bool = False):
        super().__init__()

        self.inv_path_always = inv_path_always

        # ── Shared backbone (ResNet-34 up to layer3) ──────────────────────
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # out: 64  ch
        self.layer2 = backbone.layer2  # out: 128 ch
        self.layer3 = backbone.layer3  # out: 256 ch
        self.gap = backbone.avgpool  # → [B, 256, 1, 1]

        # ── Steering controller ───────────────────────────────────────────
        self.steering = SteeringController(
            in_channels=3,
            bottleneck_dim=bottleneck_dim,
            init_bias=0.0,
        )

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Linear(256, num_classes)

    def _d4_group(self, x: torch.Tensor):
        """Returns the 8 D4 transforms of x as a list of [B,C,H,W] tensors."""
        transforms = []
        for k in range(4):
            rot = torch.rot90(x, k=k, dims=[-2, -1])
            transforms.append(rot)
            transforms.append(torch.flip(rot, dims=[-1]))
        return transforms  # 8 elements, each [B, C, H, W]

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Shared forward pass through backbone → flat feature vector [B,256]."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return torch.flatten(x, 1)

    def _invariant_features(self, x: torch.Tensor) -> torch.Tensor:
        """Reynolds operator: mean over D4 orbit of backbone features."""
        B = x.size(0)
        combined = torch.cat(self._d4_group(x), dim=0)  # [8B, C, H, W]
        group_feats = self._backbone(combined)  # [8B, 256]
        return group_feats.view(8, B, -1).mean(dim=0)  # [B,  256]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images  [B, C, H, W]
        Returns:
            logits  [B, num_classes]
        """
        B = x.size(0)

        # ── 1. Hard binary gate ──────────────────────────────────────────
        # alpha_hard ∈ {0.0, 1.0},  shape [B, 1]
        # Gradients flow back to the controller via STE.
        alpha = self.steering(x)  # [B, 1]

        # ── 2. Standard (equivariant) features ───────────────────────────
        f_std = self._backbone(x)  # [B, 256]

        # ── 3. D4-invariant features ──────────────────────────────────────
        if self.inv_path_always or not self.training:
            # Simple branch: always compute for all samples.
            # In eval mode we always do this so get_steering_weights is clean.
            f_inv = self._invariant_features(x)  # [B, 256]
        else:
            # Efficient branch: only run the costly D4 average for the subset
            # of samples where alpha=1, then scatter back into a [B,256] buffer.
            inv_mask = alpha.squeeze(1).bool()  # [B]
            f_inv = torch.zeros_like(f_std)
            if inv_mask.any():
                f_inv[inv_mask] = self._invariant_features(x[inv_mask])

        # ── 4. Hard selection via the binary alpha ────────────────────────
        # alpha broadcasts: [B,1] × [B,256]
        # When α=0  →  f_steered = f_std   (equivariant, pure)
        # When α=1  →  f_steered = f_inv   (invariant,   pure)
        f_steered = alpha * f_inv + (1.0 - alpha) * f_std  # [B, 256]

        # ── 5. Classify ───────────────────────────────────────────────────
        return self.classifier(f_steered)

    @torch.no_grad()
    def get_steering_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the hard binary α for each sample.

        Args:
            x: input images  [B, C, H, W]
        Returns:
            alpha: [B]  – each value is exactly 0 or 1.
        """
        return self.steering(x).squeeze(1)

    def steering_summary(self, x: torch.Tensor) -> dict:
        """
        Returns a dict with steering statistics for a batch.
        Because α ∈ {0,1} the mean equals the fraction routed to invariance.
        """
        alpha = self.get_steering_weights(x)
        return {
            "alpha_mean": alpha.mean().item(),
            "pct_invariant": alpha.mean().item(),  # α=1
            "pct_equivariant": (1.0 - alpha.mean()).item(),  # α=0
            "n_invariant": int(alpha.sum().item()),
            "n_equivariant": int((1 - alpha).sum().item()),
        }


def build_steerable_resnet(num_classes: int = 10,
                           pretrained: bool = True,
                           bottleneck_dim: int = 64,
                           inv_path_always: bool = False,
                           ) -> SteerableResNet:
    """Drop-in factory matching the interface of StandardResNet / D4InvariantResNet."""
    return SteerableResNet(
        num_classes=num_classes,
        pretrained=pretrained,
        bottleneck_dim=bottleneck_dim,
        inv_path_always=inv_path_always,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_steerable_resnet(num_classes=10, pretrained=False).to(device)

    x = torch.randn(8, 3, 224, 224, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    print("=" * 60)
    print("SteerableResNet (Hard Binary Gate) – smoke test")
    print("=" * 60)
    print(f"  Input  shape : {tuple(x.shape)}")
    print(f"  Output shape : {tuple(logits.shape)}")
    print()

    stats = model.steering_summary(x)
    print("  Steering statistics (random init):")
    for k, v in stats.items():
        print(f"    {k:25s}: {v}")
    print()

    # ── Verify hard binarisation ──────────────────────────────────────────
    alpha = model.get_steering_weights(x)
    unique_vals = alpha.unique().tolist()
    print(f"  Unique α values : {unique_vals}  (must be subset of {{0.0, 1.0}})")
    assert all(v in (0.0, 1.0) for v in unique_vals), "α is NOT binary!"
    print("  ✓ Hard gate confirmed – no intermediate values.")
    print()

    # ── Verify gradients flow to controller ──────────────────────────────
    model.train()
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    ctrl_grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.steering.named_parameters()
        if p.grad is not None
    }
    print("  Controller gradient norms (STE check):")
    for name, norm in ctrl_grad_norms.items():
        print(f"    {name:40s}: {norm:.6f}")
    all_nonzero = all(n > 0 for n in ctrl_grad_norms.values())
    print(f"  ✓ All gradients non-zero: {all_nonzero}")
    print()

    total_params = sum(p.numel() for p in model.parameters())
    ctrl_params = sum(p.numel() for p in model.steering.parameters())
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Controller params   : {ctrl_params:,}  ({100 * ctrl_params / total_params:.1f}%)")
    print("=" * 60)
