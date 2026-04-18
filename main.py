import argparse
import sys

import torch

from steerable_cnn.conditional_d4model import ConditionalD4CNN
from data_loaders import get_loaders


def get_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Training model on Steerable CNN.')
    parser.add_argument('-m', '--model', type=str, default='cnn',
                        choices=['cnn', 'resnet'],
                        help='Dataset to train/evaluate on (DEFAULT: cnn)')
    parser.add_argument('-d', '--dataset', type=str, default='inaturalist_mnist',
                        choices=['mnist', 'svhn', 'colormnist', 'inaturalist', 'inaturalist_mnist'],
                        help='Dataset to train/evaluate on (DEFAULT: inaturalist_mnist)')
    parser.add_argument('-e', '--n_epochs', type=int, default=50,
                        help='Number of epochs (DEFAULT: 50)')
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help='Batch size (DEFAULT: 256)')

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
    print(f"Model   : {args.model}")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.n_epochs}  |  Batch size: {args.batch_size}")
    print("-" * 60)

    if args.model == 'cnn':
        model = ConditionalD4CNN(n_classes).to(device)
    elif args.model == 'resnet':
        raise NotImplementedError
    else:
        raise ValueError(f"Model {args.model} is not supported. Should be one of ['v4cnn', 'standard_cnn'].")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print("Starting training...")
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
                print(f"    Class {i}: {acc:.2f}%  ({correct_pc[i]}/{total_pc[i]})")
            else:
                print(f"    Class {i}: N/A")


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])
    main()
