import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCNN(nn.Module):
    def __init__(self, dataset: str):
        super(StandardCNN, self).__init__()
        # Use 3 input channels to support ColorMNIST
        # For MNIST, simply repeat the 1 channel 3 times in your transform

        if dataset == "mnist":
            in_channels = 1
            num_classes = 10
            input_size = (1, 28, 28)
        elif dataset == 'colormnist':
            in_channels = 3
            num_classes = 10
            input_size = (3, 28, 28)
        elif dataset == 'svhn':
            in_channels = 3
            num_classes = 10
            input_size = (3, 32, 32)
        elif dataset == 'inaturalist':
            in_channels = 3
            num_classes = 10
            input_size  = (3, 64, 64)    # ← 224 → 64

        elif dataset == 'inaturalist_mnist':
            in_channels = 3
            num_classes = 20
            input_size  = (3, 64, 64)    # ← (1, 224) → (3, 64)
        elif dataset == 'mnist_svhn':
            in_channels = 3
            num_classes = 20
            input_size = (3, 32, 32)
        else:
            raise ValueError("Invalid dataset. Should be one of ['mnist', 'colormnist', 'svhn'].")

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self._to_linear = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten: (Batch, 64, 6, 6) -> (Batch, 1152)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()
