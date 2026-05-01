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

from Gating.MultiGatedCNN import MultiGatedCNN , LearnableGate, NoGate
from Gating.backbone import C4Backbone, D4Backbone
from Gating.backbone import PlainResNetBackbone, C4ResNetBackbone, C4ResNetBackbone_TSBN, D4ResNetBackbone, SteerableBackbone
from steerable_models.SteerableResNet import SteerableResNet

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Gated C4/D4 — SVHN + MNIST')
    parser.add_argument('-m', '--model', type=str, default='multi_gated_c4',
                        choices=['multi_gated_c4', 'multi_gated_d4', 'multi_gated_c4_resnet', 'multi_gated_d4_resnet', 'multi_gated_steerable',
                                 'c4_tsbn', 'plain_resnet'])
    parser.add_argument('-d', '--dataset', type=str, default='svhn',
                        choices=['bird', 'svhn'])
    parser.add_argument('-g','--gate_cls', type=str, default='learnable',
                        choices=['none', 'learnable'])
    parser.add_argument('-e', '--n_epochs',       type=int,   default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-bs', '--batch_size',    type=int,   default=64)
    parser.add_argument('--reflect_images', action='store_true')
    parser.add_argument('--rotate_images',  action='store_true')
    return parser.parse_args(argv)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, task_a_label, device):
    model.eval()
    correct = total = 0

    gate_sum   = {task_a_label:  torch.zeros(3, device=device),
                  'mnist': torch.zeros(3, device=device)}
    gate_count = {task_a_label: 0, 'mnist': 0}

    correct_per_task = {task_a_label: 0, 'mnist': 0}
    total_per_task   = {task_a_label: 0, 'mnist': 0}

    with torch.no_grad():
        for x, y, tasks in loader:
            x, y = x.to(device), y.to(device)

              # ── 3. Unpack all 5 return values ──────────────────────────────────
            out, gates, gate_entropy, task_a_mask, task_b_mask = model(x, tasks)

            preds = torch.empty(x.size(0), dtype=torch.long, device=device)
            if task_a_mask.any():
                preds[task_a_mask]  = out[task_a_mask,  :10].max(1).indices
            if task_b_mask.any():
                preds[task_b_mask] = out[task_b_mask, :10].max(1).indices

            correct += (preds == y).sum().item()
            total   += y.size(0)

            for task, mask in [(task_a_label, task_a_mask), ('mnist', task_b_mask)]:
                if mask.any():
                    gate_sum[task]   += gates[mask].sum(dim=0)
                    gate_count[task] += mask.sum().item()
                    correct_per_task[task] += (preds[mask] == y[mask]).sum().item()
                    total_per_task[task]   += mask.sum().item()

    gate_avg = {
        t: gate_sum[t] / max(gate_count[t], 1)
        for t in (task_a_label, 'mnist')
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

    # train_loader, test_loader, n_classes = get_svhn_mnist_loaders(args)
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
        backbone_cls = lambda in_channels=None: SteerableBackbone(steerable_resnet)
    elif args.model == 'c4_tsbn':
        backbone_cls = C4ResNetBackbone_TSBN
    elif args.model == 'plain_resnet':
        backbone_cls = PlainResNetBackbone

    # mnist_svhn ablations
    if args.gate_cls == 'none':
        model = MultiGatedCNN(
            num_classes=10,   # repurposed: svhn head
            in_channels=3,
            backbone_cls=backbone_cls,
            gate_cls=NoGate,
        ).to(device)
    elif args.gate_cls == 'learnable':
        model = MultiGatedCNN(
            num_classes=10,   # repurposed: svhn head
            in_channels=3,
            backbone_cls=backbone_cls,
            gate_cls=LearnableGate,
        ).to(device)

    # digit_head also has 10 outputs — both heads identical size, perfect for this
    model = model.to(device)

    print(f"Model  : {args.model}")
    print (f"Gate   : {args.gate_cls}")
    print (f"Dataset : {args.dataset}")
    print(f"Device : {device}")
    print(f"Epochs : {args.n_epochs}  |  Batch: {args.batch_size}  |  LR: {args.learning_rate}")
    print(f"Rotate test: {args.rotate_images}  |  Reflect test: {args.reflect_images}")

    print("-" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Detect task label once before training (not per batch)
    task_a_label = "bird" if args.dataset == "bird" else "svhn"

    print("Starting training...")
    for epoch in range(args.n_epochs):
        model.train()
        total_loss = correct_train = total_train = 0

        for x, y, tasks in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ── 3. Unpack all 5 return values ──────────────────────────────────
            out, gates, gate_entropy, task_a_mask, task_b_mask = model(x, tasks)

            losses = []
            if task_a_mask.any():
                losses.append(criterion(out[task_a_mask, :10], y[task_a_mask]))
            if task_b_mask.any():
                losses.append(criterion(out[task_b_mask, :10], y[task_b_mask]))

            if not losses:
                continue

            loss = sum(losses) + 0.01 * gate_entropy.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            predicted = torch.empty(x.size(0), dtype=torch.long, device=device)
            if task_a_mask.any():
                predicted[task_a_mask] = out[task_a_mask, :10].max(1).indices
            if task_b_mask.any():
                predicted[task_b_mask] = out[task_b_mask, :10].max(1).indices

            total_loss    += loss.item()
            total_train   += y.size(0)
            correct_train += predicted.eq(y).sum().item()

            # ── 4. Gate print — 3 values now (id, rot, ref) ────────────────────
            if total_train <= args.batch_size and epoch % 5 == 0:
                sg = gates[task_a_mask].mean(0) if task_a_mask.any() else torch.zeros(3)
                mg = gates[task_b_mask].mean(0) if task_b_mask.any() else torch.zeros(3)
                print(f"  [gates] {task_a_label}: "
                      f"id={sg[0]:.3f} rot={sg[1]:.3f} ref={sg[2]:.3f} | "
                      f"mnist: "
                      f"id={mg[0]:.3f} rot={mg[1]:.3f} ref={mg[2]:.3f}")

        scheduler.step()
        train_acc = 100. * correct_train / total_train
        test_acc, correct_pt, total_pt, gate_avg = evaluate(
            model, test_loader, task_a_label, device
        )

        task_a_acc = 100 * correct_pt[task_a_label] / max(total_pt[task_a_label], 1)
        mnist_acc  = 100 * correct_pt['mnist']       / max(total_pt['mnist'],      1)

        print(f"Epoch {epoch+1:>3}: Loss={total_loss:.4f}  "
              f"Train={train_acc:.2f}%  Test={test_acc:.2f}%")
        print(f"  {task_a_label:5s} acc={task_a_acc:.2f}%  "
              f"gates: id={gate_avg[task_a_label][0]:.3f}  "
              f"rot={gate_avg[task_a_label][1]:.3f}  "
              f"ref={gate_avg[task_a_label][2]:.3f}")
        print(f"  mnist acc={mnist_acc:.2f}%  "
              f"gates: id={gate_avg['mnist'][0]:.3f}  "
              f"rot={gate_avg['mnist'][1]:.3f}  "
              f"ref={gate_avg['mnist'][2]:.3f}")        

if __name__ == '__main__':
    main()

