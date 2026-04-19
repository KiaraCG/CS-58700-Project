import argparse
import sys
import os
import random

import torch
import torch.nn as nn

# sys.path.append('/home/shams3/CS-58700-Project')

from mulitclass_loader import get_multiclass_loaders
from MultiGatedCNN import MultiGatedCNN, C4Backbone, D4Backbone


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Gated C4/D4 CNN — iNaturalist + MNIST (multi-class)')

    parser.add_argument('-m', '--model', type=str, default='multi_gated_c4',
                        choices=['multi_gated_c4', 'multi_gated_d4'])
    parser.add_argument('-e', '--n_epochs',      type=int,   default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-bs', '--batch_size',    type=int,   default=64)
    parser.add_argument('--reflect_images', action='store_true')
    parser.add_argument('--rotate_images',  action='store_true')

    return parser.parse_args(argv)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, n_bird_classes):
    model.eval()
    correct = total = 0

    # Track gates separately per task
    gate_sum   = {'bird': torch.zeros(2, device=device),
                  'digit': torch.zeros(2, device=device)}
    gate_count = {'bird': 0, 'digit': 0}

    # Per-class accuracy: birds (0..n_bird-1), digits offset by n_bird (n_bird..n_bird+9)
    total_classes     = n_bird_classes + 10
    correct_per_class = [0] * total_classes
    total_per_class   = [0] * total_classes

    with torch.no_grad():
        for x, y, tasks in loader:
            x, y = x.to(device), y.to(device)

            out, gates, bird_mask, digit_mask = model(x, tasks)

            # Predictions
            preds = torch.empty(x.size(0), dtype=torch.long, device=device)
            if bird_mask.any():
                preds[bird_mask]  = out[bird_mask, :n_bird_classes].max(1).indices
            if digit_mask.any():
                preds[digit_mask] = out[digit_mask, :10].max(1).indices

            # Labels with digit offset for per-class tracking
            y_offset = y.clone()
            y_offset[digit_mask] += n_bird_classes
            preds_offset = preds.clone()
            preds_offset[digit_mask] += n_bird_classes

            correct += (preds == y).sum().item()   # unshifted for overall acc
            total   += y.size(0)

            # Gate averages per task
            if bird_mask.any():
                gate_sum['bird']   += gates[bird_mask].sum(dim=0)
                gate_count['bird'] += bird_mask.sum().item()
            if digit_mask.any():
                gate_sum['digit']   += gates[digit_mask].sum(dim=0)
                gate_count['digit'] += digit_mask.sum().item()

            # Per-class accuracy
            for i in range(len(y)):
                label = y_offset[i].item()
                total_per_class[label] += 1
                if preds_offset[i].item() == label:
                    correct_per_class[label] += 1

    gate_avg = {
        task: gate_sum[task] / max(gate_count[task], 1)
        for task in ('bird', 'digit')
    }

    return 100 * correct / total, correct_per_class, total_per_class, gate_avg


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = get_arguments(sys.argv[1:])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, n_bird_classes = get_multiclass_loaders(args)

    # ── Model ────────────────────────────────────────────────────────────────
    backbone_cls = C4Backbone if args.model == 'multi_gated_c4' else D4Backbone
    model = MultiGatedCNN(
        num_bird_classes=n_bird_classes,
        in_channels=3,
        backbone_cls=backbone_cls,
    ).to(device)

    print(f"Model   : {args.model}  ({backbone_cls.__name__})")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.n_epochs}  |  Batch: {args.batch_size}  |  LR: {args.learning_rate}")
    print(f"Bird classes: {n_bird_classes}  |  Digit classes: 10")
    print(f"Reflect: {args.reflect_images}  |  Rotate: {args.rotate_images}")
    print("-" * 60)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR = '/scratch/scholar/shams3/checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = f'{CHECKPOINT_DIR}/{args.model}_mc_latest.pt'
    best_acc  = 0.0
    start_epoch = 0

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

        for x, y, tasks in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out, gates, bird_mask, digit_mask = model(x, tasks)

            # Loss per task head
            loss = torch.tensor(0.0, device=device)
            if bird_mask.any():
                loss += criterion(out[bird_mask, :n_bird_classes], y[bird_mask])
            if digit_mask.any():
                loss += criterion(out[digit_mask, :10], y[digit_mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accuracy
            predicted = torch.empty(x.size(0), dtype=torch.long, device=device)
            if bird_mask.any():
                predicted[bird_mask]  = out[bird_mask, :n_bird_classes].max(1).indices
            if digit_mask.any():
                predicted[digit_mask] = out[digit_mask, :10].max(1).indices

            total_loss    += loss.item()
            total_train   += y.size(0)
            correct_train += predicted.eq(y).sum().item()

            # Print gates once per epoch at first batch
            if total_train <= args.batch_size and epoch % 5 == 0:
                b_gates = gates[bird_mask].mean(0)  if bird_mask.any()  else torch.zeros(2)
                d_gates = gates[digit_mask].mean(0) if digit_mask.any() else torch.zeros(2)
                print(f"  [gates] bird: rot={b_gates[0]:.3f} ref={b_gates[1]:.3f} | "
                      f"digit: rot={d_gates[0]:.3f} ref={d_gates[1]:.3f}")

        scheduler.step()
        train_acc = 100. * correct_train / total_train
        test_acc, correct_pc, total_pc, gate_avg = evaluate(
            model, test_loader, device, n_bird_classes
        )

        print(f"Epoch {epoch+1:>3}: Loss={total_loss:.4f}  "
              f"Train={train_acc:.2f}%  Test={test_acc:.2f}%")
        print(f"  [bird  gates] rot={gate_avg['bird'][0]:.3f}  ref={gate_avg['bird'][1]:.3f}")
        print(f"  [digit gates] rot={gate_avg['digit'][0]:.3f}  ref={gate_avg['digit'][1]:.3f}")

        # Overall per-task accuracy
        bird_correct  = sum(correct_pc[:n_bird_classes])
        bird_total    = sum(total_pc[:n_bird_classes])
        digit_correct = sum(correct_pc[n_bird_classes:])
        digit_total   = sum(total_pc[n_bird_classes:])
        print(f"  Bird  acc: {100*bird_correct/max(bird_total,1):.2f}%  "
              f"({bird_correct}/{bird_total})")
        print(f"  Digit acc: {100*digit_correct/max(digit_total,1):.2f}%  "
              f"({digit_correct}/{digit_total})")

        # Checkpoint
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
            torch.save(ckpt, f'{CHECKPOINT_DIR}/{args.model}_mc_best.pt')
            print(f"  ✓ New best: {best_acc:.2f}%")


if __name__ == '__main__':
    main()


# import argparse
# import sys
# import os
# import random

# import torch
# import torch.nn as nn

# from binary_loader import get_loaders
# from MultiGatedCNN import MultiGatedCNN, C4Backbone, D4Backbone


# def get_arguments(argv):
#     parser = argparse.ArgumentParser(description='Gated C4/D4 CNN — iNaturalist + MNIST')

#     parser.add_argument('-m', '--model', type=str, default='multi_gated_c4',
#                         choices=['multi_gated_c4', 'multi_gated_d4'],
#                         help='Model to train (DEFAULT: multi_gated_c4)')
#     parser.add_argument('-d', '--dataset', type=str, default='inaturalist_mnist',
#                         choices=['inaturalist_mnist'],
#                         help='Dataset (DEFAULT: inaturalist_mnist)')
#     parser.add_argument('-e', '--n_epochs', type=int, default=50,
#                         help='Number of epochs (DEFAULT: 50)')
#     parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
#                         help='Learning rate (DEFAULT: 1e-4)')
#     parser.add_argument('-bs', '--batch_size', type=int, default=64,
#                         help='Batch size (DEFAULT: 64)')
#     parser.add_argument('--reflect_images', action='store_true',
#                         help='Apply horizontal flip at test time')
#     parser.add_argument('--rotate_images', action='store_true',
#                         help='Apply 90° rotation at test time')

#     return parser.parse_args(argv)


# def evaluate(model, loader, device):
#     model.eval()
#     correct = total = 0
#     correct_per_class = [0, 0]
#     total_per_class   = [0, 0]

#     # For gate monitoring
#     gate_sum   = torch.zeros(2, device=device)  # [g_rot, g_ref]
#     gate_count = 0

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)

#             out, gates = model(x)           # binary: no task routing needed
#             preds = out.argmax(dim=1)

#             correct += (preds == y).sum().item()
#             total   += y.size(0)

#             gate_sum   += gates.sum(dim=0)
#             gate_count += x.size(0)

#             for i in range(len(y)):
#                 label = y[i].item()
#                 total_per_class[label] += 1
#                 if preds[i].item() == label:
#                     correct_per_class[label] += 1

#     gate_avg = gate_sum / gate_count
#     return (
#         100 * correct / total,
#         correct_per_class,
#         total_per_class,
#         gate_avg,
#     )


# def main():
#     args   = get_arguments(sys.argv[1:])
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     train_loader, test_loader = get_loaders(args)

#     # ── Model ────────────────────────────────────────────────────────────────
#     backbone_cls = C4Backbone if args.model == 'multi_gated_c4' else D4Backbone
#     model = MultiGatedCNN(
#         num_bird_classes=2,          # binary: bird=1, digit=0
#         in_channels=3,
#         backbone_cls=backbone_cls,
#     ).to(device)

#     print(f"Dataset : {args.dataset.upper()}")
#     print(f"Model   : {args.model}  ({backbone_cls.__name__})")
#     print(f"Device  : {device}")
#     print(f"Epochs  : {args.n_epochs}  |  Batch size: {args.batch_size}")
#     print(f"Reflect : {args.reflect_images}  |  Rotate: {args.rotate_images}")
#     print("-" * 60)

#     if args.model == 'multi_gated_c4':
#         model = MultiGatedCNN(num_bird_classes=2, backbone_cls=C4Backbone).to(device)

#     elif args.model == 'multi_gated_d4':
#         model = MultiGatedCNN(num_bird_classes=2, backbone_cls=D4Backbone).to(device)

#     # ── Training setup ────────────────────────────────────────────────────────
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#     criterion = nn.CrossEntropyLoss()

#     # CHECKPOINT_DIR = '/scratch/scholar/shams3/checkpoints'
#     # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#     best_acc    = 0.0
#     start_epoch = 0

#     # ckpt_path = f'{CHECKPOINT_DIR}/{args.model}_latest.pt'
#     # if os.path.exists(ckpt_path):
#     #     ckpt = torch.load(ckpt_path, weights_only=True)
#     #     model.load_state_dict(ckpt['model_state'])
#     #     optimizer.load_state_dict(ckpt['optimizer_state'])
#     #     start_epoch = ckpt['epoch'] + 1
#     #     best_acc    = ckpt['test_acc']
#     #     print(f"Resumed from epoch {start_epoch}, best acc={best_acc:.2f}%")

#     # ── Training loop ─────────────────────────────────────────────────────────
#     print("Starting training...")
#     for epoch in range(start_epoch, args.n_epochs):
#         model.train()
#         total_loss = correct_train = total_train = 0

#         for x, y in train_loader:
#             x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             if args.model in ('multi_gated_c4', 'multi_gated_d4'):
#                 out, gates = model(x)
#                 if epoch % 10 == 0 and total_train == 0:
#                     print(f"  [gates] rot={gates[:,0].mean():.3f} ref={gates[:,1].mean():.3f}")
#             else:
#                 out = model(x)

#             loss      = criterion(out, y)
#             predicted = out.argmax(1)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             total_loss    += loss.item()
#             total_train   += y.size(0)
#             correct_train += out.argmax(1).eq(y).sum().item()

#         scheduler.step()

#         train_acc = 100. * correct_train / total_train
#         test_acc, correct_pc, total_pc, gate_avg = evaluate(
#             model, test_loader, device
#         )

#         print(f"Epoch {epoch+1:>3}: Loss={total_loss:.4f}  "
#               f"Train={train_acc:.2f}%  Test={test_acc:.2f}%  "
#               f"gates=[rot={gate_avg[0]:.3f} ref={gate_avg[1]:.3f}]")
#         print(f"  Class 0 (digit): {100*correct_pc[0]/max(total_pc[0],1):.2f}%  "
#               f"({correct_pc[0]}/{total_pc[0]})")
#         print(f"  Class 1 (bird):  {100*correct_pc[1]/max(total_pc[1],1):.2f}%  "
#               f"({correct_pc[1]}/{total_pc[1]})")

#         # Checkpointing
#         ckpt = {
#             'epoch':           epoch,
#             'model_state':     model.state_dict(),
#             'optimizer_state': optimizer.state_dict(),
#             'train_acc':       train_acc,
#             'test_acc':        test_acc,
#         }
#         # torch.save(ckpt, ckpt_path)
#         # if test_acc > best_acc:
#         #     best_acc = test_acc
#         #     torch.save(ckpt, f'{CHECKPOINT_DIR}/{args.model}_best.pt')
#         #     print(f"  ✓ New best: {best_acc:.2f}%")


# if __name__ == '__main__':
#     args = get_arguments(sys.argv[1:])
#     main()