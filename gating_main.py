import argparse
import sys
import os
import random

import torch
import torch.nn as nn

from data_loaders import get_loaders
from Gating.MultiGatedCNN import MultiGatedCNN, C4Backbone, D4Backbone


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Gated C4/D4 CNN — iNaturalist + MNIST')

    parser.add_argument('-m', '--model', type=str, default='multi_gated_c4',
                        choices=['multi_gated_c4', 'multi_gated_d4'],
                        help='Model to train (DEFAULT: multi_gated_c4)')
    parser.add_argument('-d', '--dataset', type=str, default='inaturalist_mnist',
                        choices=['inaturalist_mnist'],
                        help='Dataset (DEFAULT: inaturalist_mnist)')
    parser.add_argument('-e', '--n_epochs', type=int, default=50,
                        help='Number of epochs (DEFAULT: 50)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate (DEFAULT: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch size (DEFAULT: 64)')
    parser.add_argument('--reflect_images', action='store_true',
                        help='Apply horizontal flip at test time')
    parser.add_argument('--rotate_images', action='store_true',
                        help='Apply 90° rotation at test time')

    return parser.parse_args(argv)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    correct_per_class = [0, 0]
    total_per_class   = [0, 0]

    # For gate monitoring
    gate_sum   = torch.zeros(2, device=device)  # [g_rot, g_ref]
    gate_count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out, gates = model(x)           # binary: no task routing needed
            preds = out.argmax(dim=1)

            correct += (preds == y).sum().item()
            total   += y.size(0)

            gate_sum   += gates.sum(dim=0)
            gate_count += x.size(0)

            for i in range(len(y)):
                label = y[i].item()
                total_per_class[label] += 1
                if preds[i].item() == label:
                    correct_per_class[label] += 1

    gate_avg = gate_sum / gate_count
    return (
        100 * correct / total,
        correct_per_class,
        total_per_class,
        gate_avg,
    )


def main():
    args   = get_arguments(sys.argv[1:])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, in_channels, n_classes = get_loaders(args)

    # ── Model ────────────────────────────────────────────────────────────────
    backbone_cls = C4Backbone if args.model == 'multi_gated_c4' else D4Backbone
    model = MultiGatedCNN(
        num_bird_classes=2,          # binary: bird=1, digit=0
        in_channels=in_channels,
        backbone_cls=backbone_cls,
    ).to(device)

    print(f"Dataset : {args.dataset.upper()}")
    print(f"Model   : {args.model}  ({backbone_cls.__name__})")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.n_epochs}  |  Batch size: {args.batch_size}")
    print(f"Reflect : {args.reflect_images}  |  Rotate: {args.rotate_images}")
    print("-" * 60)

    if args.model == 'multi_gated_c4':
        model = MultiGatedCNN(num_bird_classes=2, backbone_cls=C4Backbone).to(device)

    elif args.model == 'multi_gated_d4':
        model = MultiGatedCNN(num_bird_classes=2, backbone_cls=D4Backbone).to(device)

    # ── Training setup ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR = '/scratch/scholar/shams3/checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_acc    = 0.0
    start_epoch = 0

    ckpt_path = f'{CHECKPOINT_DIR}/{args.model}_latest.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt['test_acc']
        print(f"Resumed from epoch {start_epoch}, best acc={best_acc:.2f}%")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Starting training...")
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        total_loss = correct_train = total_train = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if args.model in ('multi_gated_c4', 'multi_gated_d4'):
                out, gates = model(x)
                if epoch % 10 == 0 and total_train == 0:
                    print(f"  [gates] rot={gates[:,0].mean():.3f} ref={gates[:,1].mean():.3f}")
            else:
                out = model(x)

            loss      = criterion(out, y)
            predicted = out.argmax(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss    += loss.item()
            total_train   += y.size(0)
            correct_train += out.argmax(1).eq(y).sum().item()

        scheduler.step()

        train_acc = 100. * correct_train / total_train
        test_acc, correct_pc, total_pc, gate_avg = evaluate(
            model, test_loader, device
        )

        print(f"Epoch {epoch+1:>3}: Loss={total_loss:.4f}  "
              f"Train={train_acc:.2f}%  Test={test_acc:.2f}%  "
              f"gates=[rot={gate_avg[0]:.3f} ref={gate_avg[1]:.3f}]")
        print(f"  Class 0 (digit): {100*correct_pc[0]/max(total_pc[0],1):.2f}%  "
              f"({correct_pc[0]}/{total_pc[0]})")
        print(f"  Class 1 (bird):  {100*correct_pc[1]/max(total_pc[1],1):.2f}%  "
              f"({correct_pc[1]}/{total_pc[1]})")

        # Checkpointing
        ckpt = {
            'epoch':           epoch,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_acc':       train_acc,
            'test_acc':        test_acc,
        }
        torch.save(ckpt, ckpt_path)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(ckpt, f'{CHECKPOINT_DIR}/{args.model}_best.pt')
            print(f"  ✓ New best: {best_acc:.2f}%")


if __name__ == '__main__':
    args = get_arguments(sys.argv[1:])
    main()