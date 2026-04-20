import argparse
import sys

import torch
import torch.nn as nn

from data_loaders import get_loaders
from steerable_models.InformedSteerableResNet import InformedSteerableResNet


def get_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Training model on Steerable ResNet.')
    parser.add_argument('-d', '--dataset', type=str, default='inaturalist_mnist',
                        choices=['mnist', 'inaturalist', 'inaturalist_mnist'],
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
    print(f"Dataset : {args.dataset.upper()}")
    print(f"Model   : informed_steerable_resnet")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.n_epochs}  |  Batch size: {args.batch_size}")
    print("-" * 60)

    model = InformedSteerableResNet(n_classes).to(device)

    print("Stage 1: Training Steering Controller...")
    model.steering.train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.steering.parameters():
        param.requires_grad = True
    steer_optimizer = torch.optim.Adam(model.steering.parameters(), lr=1e-3)

    correct_controller = 0
    total_controller = 0
    for epoch in range(2):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            steer_optimizer.zero_grad()

            # Binary target: 0 for MNIST, 1 for Birds
            target = (y >= 10).float().unsqueeze(1)
            logits = model.steering.fc(model.steering.encoder(x))

            loss = nn.BCEWithLogitsLoss()(logits, target)
            loss.backward()
            steer_optimizer.step()

            alpha_soft = torch.sigmoid(model.steering.fc(model.steering.encoder(x)))
            alpha_target = (y >= 10).float().unsqueeze(1)
            controller_preds = (alpha_soft > 0.5).float()
            correct_controller += controller_preds.eq(alpha_target).sum().item()
            total_controller += y.size(0)

        controller_acc = 100. * correct_controller / total_controller
        print(f"Epoch {epoch + 1:>3}: Controller Accuracy = {controller_acc:.2f}%")

    print("Stage 2: Training Backbone and Classifier...")
    for param in model.parameters():
        param.requires_grad = True
    for param in model.steering.parameters():
        param.requires_grad = False
    full_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.n_epochs-5):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            full_optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            full_optimizer.step()

            total_loss += loss.item()

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

    print("Stage 3: Full Model Co-adaptation (Fine-tuning)...")
    for param in model.parameters():
        param.requires_grad = True

    # Use a MUCH lower learning rate (10x smaller) to prevent destroying the experts we just built.
    optimizer_s3 = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer_s3.zero_grad()
            outputs = model(x, labels=y)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer_s3.step()
        evaluate(model, test_loader, device)

if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])
    main()
