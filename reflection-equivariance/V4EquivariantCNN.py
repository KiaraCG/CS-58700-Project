import torch.nn as nn
import torch.nn.functional as F

from V4BirdsCNN import V4Block
from V4DigitsCNN import V4GroupConv


class V4EquivariantCNN(nn.Module):
    def __init__(self, num_bird_classes):
        super().__init__()

        self.lift = V4GroupConv(3, 16, 3, 1, lifting=True)

        self.block1 = V4Block(16, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = V4Block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = V4Block(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Separate heads
        self.bird_head = nn.Linear(128 * 4, num_bird_classes)
        self.digit_head = nn.Linear(128 * 4, 10)

    def forward(self, x, task):
        x = F.relu(self.lift(x))
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if task == "bird":
            return self.bird_head(x)
        else:
            return self.digit_head(x)
