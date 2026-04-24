import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from C4Block import C4Block
from C4GroupConv import C4GroupConv
from D4Block import D4Block
from D4GroupConv import D4GroupConv

# Removing classifier has it is implemented in MultiGatedCNN and not needed in the backbone. 
# Also, it is not used in the D4InvariantResNet backbone, so for consistency we remove it from both backbones. \
# The classifier will be implemented in the MultiGatedCNN class instead, 
# which will allow us to easily swap out different backbones without needing to modify the classifier code.

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


class C4ResNetBackbone(nn.Module):
    """
    ResNet-34 feature extractor + C4 Reynolds averaging.
 
    Reynolds operator averages features over 4 rotations (0°, 90°, 180°, 270°),
    producing rotation-invariant features.
 
    out_channels = 256  (ResNet-34 layer3 output)
    group_size   = 4    (C4: 4 rotations)
    """
    out_channels = 256
    group_size   = 4
 
    def __init__(self, in_channels: int = 3):
        super().__init__()
        backbone = models.resnet34(weights=None)
 
        # Replace 7×7 stride-2 stem — too aggressive for 32×32 / 64×64 inputs
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = backbone.layer1   # 64  ch
        self.layer2 = backbone.layer2   # 128 ch
        self.layer3 = backbone.layer3   # 256 ch
        # self.gap    = nn.AdaptiveAvgPool2d(1) 
 
    def get_c4_group(self, x: torch.Tensor):
        """4 rotations: 0°, 90°, 180°, 270°."""
        return [torch.rot90(x, k=k, dims=[-2, -1]) for k in range(4)]
 
    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
        # x = self.gap(x)
        # return x
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        group_elements = self.get_c4_group(x)           # 4 × [B, C, H, W]
 
        combined = torch.cat(group_elements, dim=0)     # [4B, C, H, W]
        feats    = self._forward_single(combined)       # [4B, 256]

        _, C_out, H, W = feats.size()

        group_feats = feats.view(self.group_size, B, C_out, H, W)   # [4, B, 256, H, W]
        group_feats = group_feats.permute(1, 0, 2, 3, 4)            # [B, 4, 256, H, W]
        
        return group_feats.reshape(B, self.group_size * C_out, H, W)

 
 
# ── D4 ResNet Backbone ─────────────────────────────────────────────────────────
 
class D4ResNetBackbone(nn.Module):
    """
    ResNet-34 feature extractor + D4 Reynolds averaging.
 
    D4 = 4 rotations × 2 reflections = 8 group elements.
    Produces features invariant to both rotation and reflection.
 
    out_channels = 256  (ResNet-34 layer3 output)
    group_size   = 8    (D4: 4 rotations × 2 reflections)
    """
    out_channels = 256
    group_size   = 8
 
    def __init__(self, in_channels: int = 3):
        super().__init__()
        backbone = models.resnet34(weights=None)
 
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        # self.gap    = nn.AdaptiveAvgPool2d(1)
 
    def get_d4_group(self, x: torch.Tensor):
        """8 D4 transforms: 4 rotations × flip/no-flip."""
        out = []
        for k in range(4):
            rot = torch.rot90(x, k=k, dims=[-2, -1])
            out.append(rot)                             # rotation only
            out.append(torch.flip(rot, dims=[-1]))      # rotation + horizontal flip
        return out                                      # 8 × [B, C, H, W]
 
    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
        # x = self.gap(x)
        # return torch.flatten(x, 1)     # [B, 256]
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        group_elements = self.get_d4_group(x)           # 8 × [B, C, H, W]
 
        combined = torch.cat(group_elements, dim=0)     # [8B, C, H, W]
        feats    = self._forward_single(combined)       # [8B, 256]
 
        # Reynolds operator — mean over D4 orbit
        # group_feats = feats.view(8, B, -1)              # [8, B, 256]
        # return group_feats.mean(dim=0)                  # [B,  256]
    
    
        _, C_out, H, W = feats.size()

        group_feats = feats.view(self.group_size, B, C_out, H, W)   # [8, B, 256, H, W]
        group_feats = group_feats.permute(1, 0, 2, 3, 4)            # [B, 8, 256, H, W]
        
        return group_feats.reshape(B, self.group_size * C_out, H, W)

 
 
# ── Steerable Backbone ─────────────────────────────────────────────────────────
 
class SteerableBackbone(nn.Module):
    """
    Wraps SteerableResNet's feature extractor (stem + layer1-3) as a backbone.
 
    The SteerableResNet already implements D4 Reynolds averaging internally,
    so this is functionally equivalent to D4ResNetBackbone but uses the
    existing SteerableResNet code path.
 
    Pass your SteerableResNet instance at construction time:
        steerable_resnet = build_steerable_resnet(num_classes=10, pretrained=False)
        backbone = SteerableBackbone(steerable_resnet)
        model = MultiGatedCNN(num_classes=10, backbone_cls=lambda ic: backbone)
 
    out_channels = 256
    group_size   = 8   (D4)
    """
    out_channels = 256
    group_size   = 8
 
    def __init__(self, steerable_resnet: nn.Module):
        super().__init__()
        # Borrow only the feature extractor — no classifier, no steering controller
        self.stem   = steerable_resnet.stem
        self.layer1 = steerable_resnet.layer1
        self.layer2 = steerable_resnet.layer2
        self.layer3 = steerable_resnet.layer3
        # self.gap    = steerable_resnet.gap
 
    def get_d4_group(self, x: torch.Tensor):
        """8 D4 transforms — same as D4ResNetBackbone."""
        out = []
        for k in range(4):
            rot = torch.rot90(x, k=k, dims=[-2, -1])
            out.append(rot)
            out.append(torch.flip(rot, dims=[-1]))
        return out
 
    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
        # x = self.gap(x)
        # return torch.flatten(x, 1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        group_elements = self.get_d4_group(x)
        combined  = torch.cat(group_elements, dim=0)    # [8B, C, H, W]
        feats     = self._forward_single(combined)      # [8B, 256]
            
        _, C_out, H, W = feats.size()

        group_feats = feats.view(self.group_size, B, C_out, H, W)   # [8, B, 256, H, W]
        group_feats = group_feats.permute(1, 0, 2, 3, 4)            # [B, 8, 256, H, W]
        
        return group_feats.reshape(B, self.group_size * C_out, H, W)
        # return feats.view(8, B, -1).mean(dim=0)         # [B,  256]
 