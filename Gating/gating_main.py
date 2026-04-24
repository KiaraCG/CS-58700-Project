"""
gating_main_svhn_mnist.py
=========================
Trains MultiGatedCNN on SVHN + MNIST joint classification.

Gate hypothesis:
    SVHN  → g_rot should → 0  (rotation sensitive — street numbers have orientation)
    MNIST → g_rot should → 1  (rotation invariant — handwritten digits rotated during training)

Run:
    # Normal training
    python gating_main.py -m multi_gated_c4 -e 50 -bs 64 -lr 1e-4

    # Test with 90° rotation at test time
    python gating_main.py -m multi_gated_c4 -e 50 -bs 64 -lr 1e-4 --rotate_images
"""

import argparse
import sys
import os


import torch
import torch.nn as nn

# sys.path.append('/home/shams3/CS-58700-Project')
# from mulitclass_loader import get_multiclass_loaders
from Gating.digits_mutliclass_loader import get_svhn_mnist_loaders
from Gating.mulitclass_loader import get_multiclass_loaders

from Gating.MultiGatedCNN import MultiGatedCNN
from Gating.backbone import C4Backbone, D4Backbone, C4ResNetBackbone, D4ResNetBackbone, SteerableBackbone

from steerable_cnn.SteerableResNet import SteerableResNet

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Gated C4/D4 — SVHN + MNIST')
    parser.add_argument('-m', '--model', type=str, default='multi_gated_c4',
                        choices=['multi_gated_c4', 'multi_gated_d4', 'multi_gated_c4_resnet', 'multi_gated_d4_resnet', 'multi_gated_steerable'])
    parser.add_argument('-d', '--dataset', type=str, default='svhn',
                        choices=['bird', 'svhn'])
    parser.add_argument('-e', '--n_epochs',       type=int,   default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-bs', '--batch_size',    type=int,   default=64)
    parser.add_argument('--reflect_images', action='store_true')
    parser.add_argument('--rotate_images',  action='store_true')
    return parser.parse_args(argv)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0

    gate_sum   = {'svhn':  torch.zeros(2, device=device),
                  'mnist': torch.zeros(2, device=device)}
    gate_count = {'svhn': 0, 'mnist': 0}

    correct_per_task = {'svhn': 0, 'mnist': 0}
    total_per_task   = {'svhn': 0, 'mnist': 0}

    with torch.no_grad():
        for x, y, tasks in loader:
            x, y = x.to(device), y.to(device)

            out, gates, svhn_mask, mnist_mask = model(x, tasks)

            preds = torch.empty(x.size(0), dtype=torch.long, device=device)
            if svhn_mask.any():
                preds[svhn_mask]  = out[svhn_mask,  :10].max(1).indices
            if mnist_mask.any():
                preds[mnist_mask] = out[mnist_mask, :10].max(1).indices

            correct += (preds == y).sum().item()
            total   += y.size(0)

            for task, mask in [('svhn', svhn_mask), ('mnist', mnist_mask)]:
                if mask.any():
                    gate_sum[task]   += gates[mask].sum(dim=0)
                    gate_count[task] += mask.sum().item()
                    correct_per_task[task] += (preds[mask] == y[mask]).sum().item()
                    total_per_task[task]   += mask.sum().item()

    gate_avg = {
        t: gate_sum[t] / max(gate_count[t], 1)
        for t in ('svhn', 'mnist')
    }

    return (
        100 * correct / total,
        correct_per_task,
        total_per_task,
        gate_avg,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_arguments(sys.argv[1:])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, n_classes = get_svhn_mnist_loaders(args)
    if args.dataset == 'bird':
        train_loader, test_loader, n_classes = get_multiclass_loaders(args)
    else:
        train_loader, test_loader, n_classes = get_svhn_mnist_loaders(args)

    # Both tasks have 10 classes — use same head size for both
    if args.model == 'multi_gated_c4':
        backbone_cls = C4Backbone
    elif args.model == 'multi_gated_d4':
        backbone_cls = D4Backbone
    elif args.model == 'multi_gated_c4_resnet':
        backbone_cls = C4ResNetBackbone
    elif args.model == 'multi_gated_d4_resnet':
        backbone_cls = D4ResNetBackbone
    elif args.model == 'multi_gated_steerable':
        steerable_resnet = SteerableResNet()
        backbone_cls = SteerableBackbone(steerable_resnet)

    # backbone_cls = C4Backbone if args.model == 'multi_gated_c4' else D4Backbone
    model = MultiGatedCNN(
        num_classes=10,   # repurposed: svhn head
        in_channels=3,
        backbone_cls=backbone_cls,
    ).to(device)

    # digit_head also has 10 outputs — both heads identical size, perfect for this

    print(f"Model  : {args.model}  ({backbone_cls.__name__})")
    print(f"Device : {device}")
    print(f"Epochs : {args.n_epochs}  |  Batch: {args.batch_size}  |  LR: {args.learning_rate}")
    print(f"Rotate test: {args.rotate_images}  |  Reflect test: {args.reflect_images}")
    print("-" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(0, args.n_epochs):
        model.train()
        total_loss = correct_train = total_train = 0

        for x, y, tasks in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out, gates, svhn_mask, mnist_mask = model(x, tasks)

            loss = torch.tensor(0.0, device=device)
            if svhn_mask.any():
                loss += criterion(out[svhn_mask,  :10], y[svhn_mask])
            if mnist_mask.any():
                loss += criterion(out[mnist_mask, :10], y[mnist_mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            predicted = torch.empty(x.size(0), dtype=torch.long, device=device)
            if svhn_mask.any():
                predicted[svhn_mask]  = out[svhn_mask,  :10].max(1).indices
            if mnist_mask.any():
                predicted[mnist_mask] = out[mnist_mask, :10].max(1).indices

            total_loss    += loss.item()
            total_train   += y.size(0)
            correct_train += predicted.eq(y).sum().item()

            # Print gates once per epoch
            if total_train <= args.batch_size and epoch % 5 == 0:
                sg = gates[svhn_mask].mean(0)  if svhn_mask.any()  else torch.zeros(2)
                mg = gates[mnist_mask].mean(0) if mnist_mask.any() else torch.zeros(2)
                print(f"  [gates] svhn:  rot={sg[0]:.3f} ref={sg[1]:.3f} | "
                      f"mnist: rot={mg[0]:.3f} ref={mg[1]:.3f}")

        scheduler.step()
        train_acc = 100. * correct_train / total_train
        test_acc, correct_pt, total_pt, gate_avg = evaluate(
            model, test_loader, device
        )

        svhn_acc  = 100 * correct_pt['svhn']  / max(total_pt['svhn'],  1)
        mnist_acc = 100 * correct_pt['mnist'] / max(total_pt['mnist'], 1)

        print(f"Epoch {epoch+1:>3}: Loss={total_loss:.4f}  "
              f"Train={train_acc:.2f}%  Test={test_acc:.2f}%")
        print(f"  SVHN  acc={svhn_acc:.2f}%  "
              f"gates: rot={gate_avg['svhn'][0]:.3f}  ref={gate_avg['svhn'][1]:.3f}")
        print(f"  MNIST acc={mnist_acc:.2f}%  "
              f"gates: rot={gate_avg['mnist'][0]:.3f}  ref={gate_avg['mnist'][1]:.3f}")

        

if __name__ == '__main__':
    main()