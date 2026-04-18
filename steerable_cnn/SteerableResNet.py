import torch
import torch.nn as nn
from torchvision import models


class SteeringController(nn.Module):
    """
    Produces a per-sample scalar α ∈ (0, 1) from a low-resolution bottleneck
    representation of the input.

    α → 0  :  fully equivariant (standard ResNet – pose-sensitive)
    α → 1  :  fully invariant   (D4-averaged  – pose-insensitive)

    The controller is deliberately lightweight so it does not dominate the
    compute budget.  A small conv stem compresses the image to a 64-d vector,
    then two FC layers produce the steering logit.

    Args:
        in_channels (int): input image channels (default 3).
        bottleneck_dim (int): width of the intermediate representation.
        init_bias (float): initial logit bias.  0.0 → α ≈ 0.5 at the start
                           of training so gradients flow through both paths.
    """

    def __init__(self, in_channels: int = 3,
                 bottleneck_dim: int = 64,
                 init_bias: float = 0.0):
        super().__init__()

        self.encoder = nn.Sequential(
            # Aggressive spatial downsampling; we only need coarse statistics.
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

        # Initialise output bias so α starts near init_bias after sigmoid.
        nn.init.constant_(self.fc[-1].bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images  [B, C, H, W]
        Returns:
            alpha: steering weights  [B, 1]  in (0, 1)
        """
        z = self.encoder(x)
        alpha = torch.sigmoid(self.fc(z))  # [B, 1]
        return alpha


# ---------------------------------------------------------------------------
# Steerable ResNet
# ---------------------------------------------------------------------------

class SteerableResNet(nn.Module):
    """
    Steerable ResNet that smoothly interpolates between equivariant and
    invariant behaviour on a *per-sample* basis.

    Architecture overview
    ─────────────────────
                         ┌──────────────────────────────┐
    x ──► SteeringCtrl ──►  α  (scalar per image)        │
    │                    └──────────────────────────────┘
    │                                │
    │    ┌───────────────────────────┼──────────────────────┐
    │    │ Standard path             │  D4-invariant path   │
    │    │  (equivariant)            │  (Reynolds operator) │
    └────► backbone(x)              └► mean_k backbone(gₖx) │
           ↓ f_std                         ↓ f_inv           │
           └─────── α·f_inv + (1-α)·f_std ─────────────────►classifier
    
    The two paths share *all* backbone weights.  The D4-invariant path
    applies the same backbone to all 8 group transforms of x, then averages
    (Reynolds operator).  The steering controller decides, per sample, how
    much to blend toward invariance.

    Args:
        num_classes (int): number of output classes.
        pretrained (bool): initialise backbone from ImageNet weights.
        bottleneck_dim (int): hidden size of the steering controller.
        learnable_temperature (bool): if True, add a learned temperature
            parameter τ that sharpens/flattens α after sigmoid, allowing the
            network to learn more decisive steering.
    """

    def __init__(self,
                 num_classes: int = 10,
                 pretrained: bool = True,
                 bottleneck_dim: int = 64,
                 learnable_temperature: bool = True):
        super().__init__()

        # ── Shared backbone (ResNet-34 minus layer4) ──────────────────────
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

        # Optional learned temperature τ > 0 (log-parameterised for stability)
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            # initialised to τ = 1  (no sharpening at start)
            self.log_temperature = nn.Parameter(torch.zeros(1))

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
        """Shared forward pass through backbone → flat feature vector."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return torch.flatten(x, 1)  # [B, 256]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images  [B, C, H, W]
        Returns:
            logits  [B, num_classes]
        """
        B = x.size(0)

        # ── 1. Steering weights ──────────────────────────────────────────
        alpha = self.steering(x)  # [B, 1],  values in (0,1)

        if self.learnable_temperature:
            tau = torch.exp(self.log_temperature).clamp(min=1e-2)
            # Re-centre at 0.5 before applying temperature, then re-sigmoid
            logit = torch.log(alpha / (1.0 - alpha + 1e-8))
            alpha = torch.sigmoid(logit * tau)

        # ── 2. Standard (equivariant) features ───────────────────────────
        f_std = self._backbone(x)  # [B, 256]

        # ── 3. D4-invariant features  (Reynolds operator) ────────────────
        group_imgs = self._d4_group(x)  # 8 × [B,C,H,W]
        combined = torch.cat(group_imgs, dim=0)  # [8B, C, H, W]
        group_feats = self._backbone(combined)  # [8B, 256]
        group_feats = group_feats.view(8, B, -1)  # [8, B, 256]
        f_inv = group_feats.mean(dim=0)  # [B, 256]

        # ── 4. Smooth interpolation ───────────────────────────────────────
        #   alpha [B,1] broadcasts over feature dim 256
        f_steered = alpha * f_inv + (1.0 - alpha) * f_std  # [B, 256]

        # ── 5. Classify ───────────────────────────────────────────────────
        return self.classifier(f_steered)

    @torch.no_grad()
    def get_steering_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-sample α values for interpretability / analysis.

        Args:
            x: input images  [B, C, H, W]
        Returns:
            alpha: [B]  in (0, 1)
        """
        alpha = self.steering(x)
        if self.learnable_temperature:
            tau = torch.exp(self.log_temperature).clamp(min=1e-2)
            logit = torch.log(alpha / (1.0 - alpha + 1e-8))
            alpha = torch.sigmoid(logit * tau)
        return alpha.squeeze(1)

    def steering_summary(self, x: torch.Tensor) -> dict:
        """
        Returns a dict with steering statistics for a batch.
        Useful during training to monitor whether the network is learning
        to steer toward one regime or staying in the middle.
        """
        alpha = self.get_steering_weights(x)
        return {
            "alpha_mean": alpha.mean().item(),
            "alpha_std": alpha.std().item(),
            "alpha_min": alpha.min().item(),
            "alpha_max": alpha.max().item(),
            "pct_invariant": (alpha > 0.75).float().mean().item(),
            "pct_equivariant": (alpha < 0.25).float().mean().item(),
        }


def build_steerable_resnet(num_classes: int = 10,
                           pretrained: bool = True,
                           bottleneck_dim: int = 64,
                           learnable_temperature: bool = True
                           ) -> SteerableResNet:
    """Drop-in factory matching the interface of StandardResNet / D4InvariantResNet."""
    return SteerableResNet(
        num_classes=num_classes,
        pretrained=pretrained,
        bottleneck_dim=bottleneck_dim,
        learnable_temperature=learnable_temperature,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_steerable_resnet(num_classes=10, pretrained=False).to(device)
    model.eval()

    x = torch.randn(4, 3, 224, 224, device=device)
    logits = model(x)

    print("=" * 60)
    print("SteerableResNet – smoke test")
    print("=" * 60)
    print(f"  Input  shape : {tuple(x.shape)}")
    print(f"  Output shape : {tuple(logits.shape)}")
    print()
    stats = model.steering_summary(x)
    print("  Steering statistics (random init):")
    for k, v in stats.items():
        print(f"    {k:25s}: {v:.4f}")
    print()

    # Verify invariance of the D4 path
    with torch.no_grad():
        rot_x = torch.rot90(x, k=1, dims=[-2, -1])
        flip_x = torch.flip(x, dims=[-1])

        alpha_x = model.get_steering_weights(x)
        alpha_rot = model.get_steering_weights(rot_x)
        alpha_flip = model.get_steering_weights(flip_x)

        print("  Alpha sensitivity to D4 transforms (should differ – controller")
        print("  is NOT constrained to be equivariant, by design):")
        print(f"    mean |α(x) – α(rot x)|  : {(alpha_x - alpha_rot).abs().mean():.4f}")
        print(f"    mean |α(x) – α(flip x)| : {(alpha_x - alpha_flip).abs().mean():.4f}")

    print()
    total_params = sum(p.numel() for p in model.parameters())
    ctrl_params = sum(p.numel() for p in model.steering.parameters())
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Controller params   : {ctrl_params:,}  ({100 * ctrl_params / total_params:.1f}%)")
    print("=" * 60)
