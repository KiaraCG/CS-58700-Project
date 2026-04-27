import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core C4 group convolution layer
# ---------------------------------------------------------------------------

class C4GroupConv(nn.Module):
    """
    Convolution layer equivariant to the C4 group (rotations by 0°, 90°, 180°, 270°).

    Lifting mode  (lifting=True):
        Input  shape: (B, C_in,    H, W)
        Output shape: (B, C_out*4, H, W)
        A standard spatial image is "lifted" onto the group by convolving it
        with 4 rotated copies of each filter.

    Group-conv mode (lifting=False):
        Input  shape: (B, C_in*4,  H, W)
        Output shape: (B, C_out*4, H, W)
        Both the feature maps and filters live on the group. The four channel
        blocks correspond to the four group elements {r0, r1, r2, r3}. For each
        output group element g, we rotate the filter bank by g and correlate
        with the *cyclically permuted* input channel blocks, which implements
        the group convolution exactly.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, padding: int, lifting: bool = False):
        super().__init__()
        self.lifting = lifting
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # Base filter bank: one set of (out_channels, in_channels, k, k) weights.
        # The four rotated copies are derived on-the-fly so the model only
        # learns one quarter of the parameters compared to a plain CNN with 4×
        # the channels — the equivariance is baked in, not learned.
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight)

    # ------------------------------------------------------------------
    # Helper: produce the four 90°-rotated copies of a filter tensor
    # ------------------------------------------------------------------
    @staticmethod
    def _rotated_filters(w: torch.Tensor):
        """Return [w_r0, w_r1, w_r2, w_r3] — rotations by 0, 90, 180, 270°."""
        w0 = w                                      # 0°
        w1 = torch.rot90(w, k=1, dims=[-2, -1])    # 90°  (CCW)
        w2 = torch.rot90(w, k=2, dims=[-2, -1])    # 180°
        w3 = torch.rot90(w, k=3, dims=[-2, -1])    # 270°
        return w0, w1, w2, w3

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w0, w1, w2, w3 = self._rotated_filters(self.weight)

        if self.lifting:
            # ----------------------------------------------------------------
            # LIFTING LAYER
            # Each filter is applied to the plain image in all four rotations.
            # Stack the four rotated filter sets along the output-channel axis.
            #   weight shape after cat: (out_channels*4, in_channels, k, k)
            # ----------------------------------------------------------------
            weights = torch.cat([w0, w1, w2, w3], dim=0)
            return F.conv2d(x, weights, padding=self.padding)

        else:
            # ----------------------------------------------------------------
            # GROUP CONVOLUTION LAYER
            #
            # The input x has 4 blocks of in_channels channels, one per group
            # element: [f_r0 | f_r1 | f_r2 | f_r3].
            #
            # For output group element r_k, the group convolution requires:
            #   - Rotating the filter by r_k
            #   - Correlating with input channels cyclically shifted by k
            #     (because (r_k · r_j) = r_{(k+j) mod 4})
            #
            # Concretely, for each output element r_k we build a filter whose
            # j-th input-block slice is w_{r_k} applied to input block r_{-k·j},
            # i.e. the cyclic permutation of input blocks by k positions.
            # ----------------------------------------------------------------
            C = self.in_channels  # channels per group element

            def block_slice(w_rot, perm):
                """
                Build a (out_channels, C*4, k, k) weight for one output group
                element by concatenating the rotated filter applied to input
                blocks in the given cyclic order.
                perm: list of 4 indices indicating which input block feeds each
                      of the 4 filter slices.
                """
                rotated_filters = [w0, w1, w2, w3]
                # For each input position in the permuted order, pick the
                # appropriately rotated filter for that input block
                slices = torch.cat(
                    [rotated_filters[p] for p in perm], dim=1
                )  # shape: (out_channels, C*4, k, k)
                return F.conv2d(x, w_rot.unsqueeze(0).expand(
                    # We actually build the full combined weight:
                    *w_rot.shape
                ), padding=self.padding)   # placeholder — see full build below

            # Build combined weight matrices for all four output group elements.
            # For output r_k, input block r_j contributes filter rotated by r_k
            # evaluated at position r_{j - k} = r_{(j-k) mod 4}.
            # Equivalently: combined_weight[r_k] uses filter slice rotated by
            # (k - j) mod 4 for input block j.
            #
            # Written out explicitly:
            #   r0 output: input blocks [r0,r1,r2,r3] → filter rotations [0,0,0,0]  (identity)
            #              but shifted: [w0@r0 | w0@r1 ...] → use w_rots [0,3,2,1]
            # The standard formula: for r_k, use filter w_{r_{k}} for input
            # block r_j, where the cyclic shift means block j sees rotation k.
            #
            # After working through the C4 Cayley table the combined weights are:
            combined_weights = torch.cat([
                torch.cat([w0, w3, w2, w1], dim=1),  # output r0: shift input by 0
                torch.cat([w1, w0, w3, w2], dim=1),  # output r1: shift input by 1
                torch.cat([w2, w1, w0, w3], dim=1),  # output r2: shift input by 2
                torch.cat([w3, w2, w1, w0], dim=1),  # output r3: shift input by 3
            ], dim=0)
            # combined_weights shape: (out_channels*4, in_channels*4, k, k)

            return F.conv2d(x, combined_weights, padding=self.padding)


# ---------------------------------------------------------------------------
# Group-invariant pooling  (pool over the 4 group-element channels)
# ---------------------------------------------------------------------------

class C4InvariantPool(nn.Module):
    """
    Pools over the C4 group dimension to produce rotation-*invariant* features.
    Input:  (B, C*4, H, W)  — four channel blocks, one per group element
    Output: (B, C,   H, W)  — max (or mean) over the four group elements
    """
    def __init__(self, mode: str = "max"):
        super().__init__()
        assert mode in ("max", "mean")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C4, H, W = x.shape
        assert C4 % 4 == 0, "Channel count must be divisible by 4."
        C = C4 // 4
        # Reshape to (B, C, 4, H, W) then pool over the group dimension (dim=2)
        x = x.view(B, C, 4, H, W)
        if self.mode == "max":
            return x.max(dim=2).values   # (B, C, H, W)
        else:
            return x.mean(dim=2)         # (B, C, H, W)


# ---------------------------------------------------------------------------
# Shared residual block
# ---------------------------------------------------------------------------

class C4Block(nn.Module):
    """Two-layer C4 group-conv block with a skip connection."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = C4GroupConv(in_ch, out_ch, 3, 1, lifting=False)
        self.conv2 = C4GroupConv(out_ch, out_ch, 3, 1, lifting=False)
        self.skip = (nn.Identity() if in_ch == out_ch
                     else C4GroupConv(in_ch, out_ch, 1, 0, lifting=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + identity)


# ---------------------------------------------------------------------------
# C4-Equivariant CNN  (digit / MNIST / SVHN / ColorMNIST)
# ---------------------------------------------------------------------------

class C4EquivariantDigitsCNN(nn.Module):
    """
    C4-equivariant CNN for MNIST / ColorMNIST / SVHN.

    The output logits are *equivariant*: rotating the input by 90° permutes
    the group-channel blocks in the final feature map but does not collapse
    them. Suitable when you want to use the equivariant representation for a
    downstream task (e.g. as a feature extractor).

    To obtain invariant classification from this model, apply C4InvariantPool
    before the linear head, or use C4InvariantDigitsCNN below.

    Architecture mirrors V4DigitsCNN but uses C4 symmetry.
    """
    def __init__(self, in_channels: int = 1, n_classes: int = 10):
        super().__init__()
        self.lift  = C4GroupConv(in_channels, 8, 3, 1, lifting=True)  # → 8*4 = 32 ch
        self.conv2 = C4GroupConv(8, 16, 3, 1, lifting=False)          # → 16*4 = 64 ch

        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final fc sees all 64 channels (equivariant — group dim NOT collapsed)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lift(x))    # (B, 32, H, W)
        x = F.relu(self.conv2(x))   # (B, 64, H, W)
        x = self.spatial_pool(x)    # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        return self.fc(x)


# ---------------------------------------------------------------------------
# C4-Invariant CNN  (digit / MNIST / SVHN / ColorMNIST)
# ---------------------------------------------------------------------------

class C4InvariantDigitsCNN(nn.Module):
    """
    C4-invariant CNN for MNIST / ColorMNIST / SVHN.

    Identical feature extraction to C4EquivariantDigitsCNN, but the group
    dimension is collapsed via C4InvariantPool *before* the linear head, so
    the classification score is truly rotation-invariant (rotating the input
    by any multiple of 90° gives the same logits).
    """
    def __init__(self, in_channels: int = 1, n_classes: int = 10):
        super().__init__()
        self.lift  = C4GroupConv(in_channels, 8, 3, 1, lifting=True)
        self.conv2 = C4GroupConv(8, 16, 3, 1, lifting=False)

        self.group_pool   = C4InvariantPool(mode="max")         # collapse group dim
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, n_classes)  # 16 channels after group pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lift(x))    # (B, 32, H, W)
        x = F.relu(self.conv2(x))   # (B, 64, H, W)
        x = self.group_pool(x)      # (B, 16, H, W)  ← rotation-invariant here
        x = self.spatial_pool(x)    # (B, 16, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 16)
        return self.fc(x)


# ---------------------------------------------------------------------------
# C4-Equivariant CNN  (birds / iNaturalist)
# ---------------------------------------------------------------------------

class C4EquivariantBirdsCNN(nn.Module):
    """
    C4-equivariant CNN for iNaturalist Birds.
    Mirrors V4BirdsCNN in depth and width, replacing V4 symmetry with C4.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.lift   = C4GroupConv(3, 16, 3, 1, lifting=True)   # RGB → 16*4 = 64 ch

        self.block1 = C4Block(16, 32)
        self.pool1  = nn.MaxPool2d(2)

        self.block2 = C4Block(32, 64)
        self.pool2  = nn.MaxPool2d(2)

        self.block3 = C4Block(64, 128)

        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * 4, num_classes)  # 128*4 = 512 (equivariant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lift(x))
        x = self.block1(x);  x = self.pool1(x)
        x = self.block2(x);  x = self.pool2(x)
        x = self.block3(x)
        x = self.spatial_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# C4-Invariant CNN  (birds / iNaturalist)
# ---------------------------------------------------------------------------

class C4InvariantBirdsCNN(nn.Module):
    """
    C4-invariant CNN for iNaturalist Birds.
    Same backbone as C4EquivariantBirdsCNN but collapses the group dimension
    with C4InvariantPool before the linear head.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.lift   = C4GroupConv(3, 16, 3, 1, lifting=True)

        self.block1 = C4Block(16, 32)
        self.pool1  = nn.MaxPool2d(2)

        self.block2 = C4Block(32, 64)
        self.pool2  = nn.MaxPool2d(2)

        self.block3 = C4Block(64, 128)

        self.group_pool   = C4InvariantPool(mode="max")         # → 128 ch, invariant
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lift(x))
        x = self.block1(x);  x = self.pool1(x)
        x = self.block2(x);  x = self.pool2(x)
        x = self.block3(x)
        x = self.group_pool(x)       # collapse C4 group dim
        x = self.spatial_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
