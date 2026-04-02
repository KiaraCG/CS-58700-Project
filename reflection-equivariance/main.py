import argparse
import sys

import torch

from data_loaders import get_loaders
from V4DigitsCNN import V4DigitsCNN
from StandardCNN import StandardCNN
from V4EquivariantCNN import V4EquivariantCNN
from V4BirdsCNN import V4BirdsCNN

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training model on MNIST / SVHN / ColorMNIST / iNaturalist / iNaturalist + MNIST.')
    parser.add_argument('-m', '--model', type=str, default='v4cnn',
                        choices=['v4cnn', 'standard_cnn'],
                        help='Dataset to train/evaluate on (DEFAULT: v4cnn)')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'svhn', 'colormnist', 'inaturalist', 'inaturalist_mnist'],
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
    print(f"Model   : {args.model}")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.n_epochs}  |  Batch size: {args.batch_size}")
    print(f"Test reflections  : {args.reflect_images}")
    print("-" * 60)

    if args.model == 'v4cnn':
        if args.dataset in ['mnist', 'colormnist', 'svhn']:
            model = V4DigitsCNN(in_channels=in_channels).to(device)
        elif args.dataset == 'inaturalist':
            model = V4BirdsCNN(n_classes).to(device)
        elif args.dataset == 'inaturalist_mnist':
            model = V4EquivariantCNN(n_classes).to(device)
        else:
            raise ValueError(f"Dataset {args.model} is not supported.")
    elif args.model == 'standard_cnn':
        model = StandardCNN(args.dataset).to(device)
    else:
        raise ValueError(f"Model {args.model} is not supported. Should be one of ['v4cnn', 'standard_cnn'].")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            print("start")
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            print("lol")
            total_loss += loss.item()

            # Training accuracy on the fly to avoid double-passing the data
            _, predicted = outputs.max(1)
            total_train += y.size(0)
            correct_train += predicted.eq(y).sum().item()
            print("done")

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
