import torch
import torch.nn as nn
import torch.nn.functional as F

def flip_tensor_horizaontal(x):
    # horizontal reflection (flip width dimension)
    return torch.flip(x, dims=[-1])

def flip_tensor_vertical(x):
    # vertical reflection (flip height dimension)
    return torch.flip(x, dims=[-2])

class Z2Conv(nn.Module):
    """
    Z2-equivariant convolution layer.
    Input: (B, C_in, H, W)
    Output: (B, 2*C_out, H, W)  # one channel per group element
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, reflection_type):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding

        if reflection_type == 'horizontal':
            self.flip_tensor = flip_tensor_horizaontal
        elif reflection_type == 'vertical':
            self.flip_tensor = flip_tensor_vertical
        else:
            raise ValueError("reflection_type must be 'horizontal' or 'vertical'")

    def forward(self, x):
        # Standard convolution
        y = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding)

        # Reflected filters
        flipped_weight = self.flip_tensor(self.weight)

        # Convolution with flipped filters
        y_reflect = F.conv2d(x, weight=flipped_weight, bias=self.bias, padding=self.padding)

        # Stack along channel dimension (group dimension)
        out = torch.cat([y, y_reflect], dim=1)
        return out

class Z2CNN(nn.Module):
    def __init__(self, reflection_type):
        super().__init__()
        self.conv1 = Z2Conv(1, 8, kernel_size=3, padding=1, reflection_type=reflection_type)
        self.conv2 = Z2Conv(16, 16, kernel_size=3, padding=1, reflection_type=reflection_type)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)  # 2 * 16 channels

        self.bn1 = nn.BatchNorm2d(16)  # 2 * 8
        self.bn2 = nn.BatchNorm2d(32)  # 2 * 16

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples)
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss
        """
        X_train = X
        y_train = y

        # forward
        y_pred = self.model(X_train)
        loss = self.loss_fn(y_pred, y_train.squeeze())

        # backward
        self.model.zero_grad()
        loss.backward()

        # update weights and biases
        for param in self.model.parameters():
            param.data -= learning_rate * param.grad.data
        return loss.item()