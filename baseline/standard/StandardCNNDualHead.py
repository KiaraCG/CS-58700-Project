import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCNNDualHead(nn.Module):
    """
    Baseline CNN with two hard-gated heads — no symmetry constraints.
    Routes samples to the correct head based on task string.

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
            raise ValueError(f"StandardCNNDualHead only supports: {list(self.DATASET_CFG)}")

        in_channels, input_size = self.DATASET_CFG[dataset]

        # ── Shared backbone ───────────────────────────────────────────────────
        self.conv1   = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64,       kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(64, 128,      kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self._to_linear = self._get_conv_output(input_size)

        self.fc_shared = nn.Linear(self._to_linear, 256)

        # ── Dual heads ────────────────────────────────────────────────────────
        self.bird_head  = nn.Linear(256, num_bird_classes)
        self.digit_head = nn.Linear(256, num_digit_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.numel()

    def _backbone(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(torch.flatten(x, 1))
        return F.relu(self.fc_shared(x))

    def forward(self, x, tasks):
        feats = self._backbone(x)           # [B, 256]

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
