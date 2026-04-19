import torch.nn as nn
import torchvision.models as models


class ResNetBirdsCNN(nn.Module):
    """
    ResNet18 pretrained on ImageNet, fine-tuned for bird classification.
    Use this as a strong baseline against C4 models.

    Stage 1 (epochs 1-10):  only the head trains (frozen backbone)
    Stage 2 (epoch 11+):    full model fine-tunes at a lower lr

    Call model.unfreeze() after epoch 10 and reset optimizer lr to 1e-5.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace head — only this trains initially
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes),
        )

    def unfreeze(self):
        """Call after warm-up epochs to fine-tune the full network."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)