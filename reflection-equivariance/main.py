import argparse
import sys

import torch

from data_loaders import get_loaders
from V4EquivariantCNN import V4EquivariantCNN
from StandardCNN import StandardCNN

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training V4CNN on MNIST / SVHN / ColorMNIST')

    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'svhn', 'colormnist'],
                        help='Dataset to train/evaluate on (DEFAULT: mnist)')
    parser.add_argument('-e', '--n_epochs', type=int, default=50,
                        help='Number of epochs (DEFAULT: 50)')
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help='Batch size (DEFAULT: 256)')
    parser.add_argument("--reflect_images", action="store_true",
                        help="Apply random reflections at test time.")

    args = parser.parse_args(argv)
    return args


def evaluate(model, loader, device, n_classes=10):
    model.eval()
    correct = 0
    total = 0

    correct_per_class = [0] * n_classes
    total_per_class = [0] * n_classes

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            for i in range(len(y)):
                label = y[i].item()
                pred = preds[i].item()
                total_per_class[label] += 1
                if pred == label:
                    correct_per_class[label] += 1

    return 100 * correct / total, correct_per_class, total_per_class

def main():
    train_loader, test_loader, in_channels, n_classes = get_loaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset : {args.dataset.upper()}")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.n_epochs}  |  Batch size: {args.batch_size}")
    print(f"Test reflections  : {args.reflect_images}")
    print("-" * 60)

    model = V4EquivariantCNN(in_channels=in_channels).to(device)
    # model = StandardCNN(in_channels=in_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Training accuracy on the fly to avoid double-passing the data
            _, predicted = outputs.max(1)
            total_train += y.size(0)
            correct_train += predicted.eq(y).sum().item()

        train_acc = 100. * correct_train / total_train
        test_acc, correct_pc, total_pc = evaluate(model, test_loader, device, n_classes)

        print(f"Epoch {epoch + 1:>3}: Loss={total_loss:.4f}  "
              f"Train={train_acc:.2f}%  Test={test_acc:.2f}%")

        print("  Per-class accuracy:")
        for i in range(n_classes):
            if total_pc[i] > 0:
                acc = 100 * correct_pc[i] / total_pc[i]
                print(f"    Digit {i}: {acc:.2f}%  ({correct_pc[i]}/{total_pc[i]})")
            else:
                print(f"    Digit {i}: N/A")


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])
    main()
