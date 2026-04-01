import argparse
import random
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import V4CNN


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for MNIST')
    parser.add_argument('-e', '--n_epochs', type=int, default=50,
                        help='number of epochs (DEFAULT: 20)')
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help='(DEFAULT: 256)')
    parser.add_argument('-ri', '--reflect_images', type=bool, default=False,
                        help='(DEFAULT: 256)')
    args = parser.parse_args(argv)
    return args


def main():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = train_transform

    if args.reflect_images:
        def random_flip(x):
            a = random.random()
            if a < 0.33:
                return torch.flip(x, dims=[-1])
            if a < 0.67:
                return torch.flip(x, dims=[-2])
            return x

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(random_flip),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=train_transform),
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=test_transform),
        batch_size=args.batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = V4CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0

        correct_per_class = [0] * 10
        total_per_class = [0] * 10
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                outputs = model(x)
                preds = outputs.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                # per-class stats
                for i in range(len(y)):
                    label = y[i].item()
                    pred = preds[i].item()

                    total_per_class[label] += 1
                    if pred == label:
                        correct_per_class[label] += 1

        return 100 * correct / total, correct_per_class, total_per_class

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_acc, _, _ = evaluate(model, train_loader)
        test_acc, correct_pc, total_pc = evaluate(model, test_loader)

        print(f"Epoch {epoch + 1}: Train Loss={total_loss:.4f}, Train={train_acc:.2f}%, Test={test_acc:.2f}%")

        print("Per-class accuracy:")
        for i in range(10):
            acc = 100 * correct_pc[i] / total_pc[i]
            print(f"  Digit {i}: {acc:.2f}%")


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])
    main()
