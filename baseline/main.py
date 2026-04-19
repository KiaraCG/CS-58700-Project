import argparse
import sys

import torch
import torch.nn.functional as F
import os

from data.data_loaders import get_loaders
from standard.StandardCNN import StandardCNN
# from rotation.equivariant.C4EquivariantCNN_HWK import GEquivariantCNN
from rotation.invariant.C4InvariantResNetBird import ResNetBirdsCNN
from rotation.invariant.C4InvariantDigitsCNN import C4InvariantDigitsCNN
from rotation.invariant.C4InvariantBirdsCNN import C4InvariantBirdsCNN
from rotation.invariant.C4InvariantCNN import C4InvariantCNN
from rotation.equivariant.C4EquivariantDigitsCNN import C4EquivariantDigitsCNN
from rotation.equivariant.C4EquivariantBirdsCNN import C4EquivariantBirdsCNN
from rotation.equivariant.C4EquivariantCNN import C4EquivariantCNN
# from rotation.invariant.V4InvariantDigitsCNN import V4InvariantDigitsCNN
# from rotation.invariant.V4InvariantBirdsCNN import V4InvariantBirdsCNN
# from rotation.invariant.V4EquivariantCNN import V4EquivariantCNN

from standard import StandardCNN


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training model on MNIST / SVHN / ColorMNIST / iNaturalist / iNaturalist + MNIST.')
    parser.add_argument('-m', '--model', type=str, default='c4_invariant',
                        choices=[
                            'v4_equivariant',
                            'v4_invariant', 
                            'c4_equivariant',
                            'c4_invariant',
                            'standard_cnn',
                            'resnet'
                        ])
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'svhn', 'colormnist', 'inaturalist', 'inaturalist_mnist'],
                        help='Dataset to train/evaluate on (DEFAULT: mnist)')
    parser.add_argument('-e', '--n_epochs', type=int, default=50,
                        help='Number of epochs (DEFAULT: 50)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        help='Learning rate (DEFAULT: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help='Batch size (DEFAULT: 256)')
    parser.add_argument("--reflect_images", action="store_true",
                        help="Apply random reflections at test time.")
    parser.add_argument("--rotate_images", action="store_true",
                    help="Apply random rotations at test time.")

    args = parser.parse_args(argv)
    return args


def evaluate(model, loader, device, n_classes=10, dataset=None):
    model.eval()
    correct = 0
    total = 0

    if isinstance(n_classes, tuple):
        n_bird_classes, n_digit_classes = n_classes
        total_classes = n_bird_classes + n_digit_classes
    else:
        n_bird_classes = None
        total_classes = n_classes

    correct_per_class = [0] * total_classes
    total_per_class   = [0] * total_classes

    with torch.no_grad():
        for batch in loader:
            if dataset == 'inaturalist_mnist':
                x, y, tasks = batch
            else:
                x, y = batch
                tasks = None

            x, y = x.to(device), y.to(device)

            if tasks is not None:
                out, bird_mask, digit_mask = model(x, tasks)

                preds = torch.empty(x.size(0), dtype=torch.long, device=device)
                if bird_mask.any():
                    preds[bird_mask]  = out[bird_mask, :n_bird_classes].max(1).indices
                if digit_mask.any():
                    preds[digit_mask] = out[digit_mask, :10].max(1).indices

                # Offset digit labels so they don't collide with bird class indices
                y_offset = y.clone()
                y_offset[digit_mask] += n_bird_classes
                preds[digit_mask]    += n_bird_classes
            else:
                out = model(x)
                preds    = out.argmax(dim=1)
                y_offset = y

            correct += (preds == y_offset).sum().item()
            total   += y.size(0)

            for i in range(len(y)):
                label = y_offset[i].item()
                total_per_class[label] += 1
                if preds[i].item() == label:
                    correct_per_class[label] += 1

    return 100 * correct / total, correct_per_class, total_per_class

def test_equivariance(model, device):
    model.eval()
    x = torch.randn(1, 1, 28, 28).to(device)
    
    with torch.no_grad():
        out_0   = model(x)
        out_90  = model(torch.rot90(x, 1, [-2, -1]))
        out_180 = model(torch.rot90(x, 2, [-2, -1]))
        out_270 = model(torch.rot90(x, 3, [-2, -1]))
    
    print("out_0  :", out_0)
    print("out_90 :", out_90)
    print("out_180:", out_180)
    print("out_270:", out_270)
    print("max diff 0 vs 90 :", (out_0 - out_90).abs().max().item())
    print("max diff 0 vs 180:", (out_0 - out_180).abs().max().item())
    print("max diff 0 vs 270:", (out_0 - out_270).abs().max().item())

def test_equivariance_per_layer(model, device):
    model.eval()
    x = torch.randn(1, 1, 28, 28).to(device)
    x_90 = torch.rot90(x, 1, [-2, -1])

    def correct_for_group(tensor, n_rotations=1):
        """Rotate spatially and cycle group channels."""
        t = torch.rot90(tensor, -n_rotations, [-2, -1])
        B, C, H, W = t.shape
        n_semantic = C // 4
        t = t.view(B, n_semantic, 4, H, W)
        t = torch.roll(t, n_rotations, dims=2)
        t = t.view(B, C, H, W)
        return t

    with torch.no_grad():
        # after conv1
        out1    = F.relu(model.conv1(x))
        out1_90 = F.relu(model.conv1(x_90))
        print("conv1 diff:", (out1 - correct_for_group(out1_90)).abs().max().item())

        # after conv2
        out2    = F.relu(model.conv2(out1))
        out2_90 = F.relu(model.conv2(out1_90))
        print("conv2 diff:", (out2 - correct_for_group(out2_90)).abs().max().item())

        # after conv3
        out3    = F.relu(model.conv3(out2))
        out3_90 = F.relu(model.conv3(out2_90))
        print("conv3 diff:", (out3 - correct_for_group(out3_90)).abs().max().item())

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 1. define device first
    
    train_loader, test_loader, in_channels, n_classes = get_loaders(args)

    print(f"Dataset : {args.dataset.upper()}")
    print(f"Model   : {args.model}")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.n_epochs}  |  Batch size: {args.batch_size}")
    print(f"Test reflections  : {args.reflect_images}")
    print(f"Test rotations    : {args.rotate_images}")
    print("-" * 60)

    if args.model == 'c4_invariant':
        if args.dataset in ['mnist', 'colormnist', 'svhn']:
            model = C4InvariantDigitsCNN(in_channels=in_channels).to(device)
        elif args.dataset == 'inaturalist':
            model = C4InvariantBirdsCNN(n_classes).to(device)
        elif args.model == 'resnet':
            model = ResNetBirdsCNN(n_classes).to(device)
        elif args.dataset == 'inaturalist_mnist':
            n_bird_classes, _ = n_classes
            model = C4InvariantCNN(n_bird_classes).to(device)
        else:
            raise ValueError(f"Dataset {args.model} is not supported.")
    elif args.model == 'c4_equivariant':
        if args.dataset in ['mnist', 'colormnist', 'svhn']:
            model = C4EquivariantDigitsCNN(in_channels=in_channels).to(device)
        elif args.dataset == 'inaturalist':
            model = C4EquivariantBirdsCNN(n_classes).to(device)
        elif args.dataset == 'inaturalist_mnist':
            n_bird_classes, _ = n_classes
            model = C4EquivariantCNN(n_bird_classes, args.task).to(device)
        else:
            raise ValueError(f"Dataset {args.model} is not supported.")
    # elif args.model == 'v4_invariant':
    #     if args.dataset in ['mnist', 'colormnist', 'svhn']:
    #         model = V4InvariantDigitsCNN(in_channels=in_channels)
    #     elif args.dataset == 'inaturalist':
    #         model = V4InvariantBirdsCNN(n_classes)
    #     elif args.dataset == 'inaturalist_mnist':
    #         model = V4EquivariantCNN(n_classes).to(device)
    #     else:
    #         raise ValueError(f"Dataset {args.model} is not supported.")
    # elif args.model == 'v4_equivariant':
    #     if args.dataset in ['mnist', 'colormnist', 'svhn']:
    #         model = V4EquivariantDigitsCNN(in_channels=in_channels)
    #     elif args.dataset == 'inaturalist':
    #         model = V4EquivariantBirdsCNN(n_classes)
    #     elif args.dataset == 'inaturalist_mnist':
    #         model = V4EquivariantCNN(n_classes).to(device)
    #     else:
    #         raise ValueError(f"Dataset {args.model} is not supported.")
    elif args.model == 'standard_cnn':
        model = StandardCNN(args.dataset).to(device)
    else:
        raise ValueError(f"Model {args.model} is not supported. Should be one of ['v4cnn', 'standard_cnn'].")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    # call before training
    # test_equivariance(model, device)
    # test_equivariance_per_layer(model, device) 

    # CHECKPOINT_DIR = '/scratch/scholar/shams3/checkpoints'
    # os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_acc  = 0.0

    # Resume from checkpoint if one exists
    start_epoch = 0
    # if os.path.exists(f'{CHECKPOINT_DIR}/latest.pt'):
    #     ckpt = torch.load(f'{CHECKPOINT_DIR}/latest.pt')
    #     model.load_state_dict(ckpt['model_state'])
    #     optimizer.load_state_dict(ckpt['optimizer_state'])
    #     start_epoch = ckpt['epoch'] + 1
    #     best_acc    = ckpt['test_acc']
    #     print(f"Resumed from epoch {start_epoch}, best acc={best_acc:.2f}%")

    print("Starting training...")
    for epoch in range(start_epoch, args.n_epochs):   
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            if args.dataset == 'inaturalist_mnist':
                x, y, tasks = batch
            else:
                x, y = batch
                tasks = None

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if tasks is not None:
                out, bird_mask, digit_mask = model(x, tasks)

                loss = torch.tensor(0.0, device=device)
                if bird_mask.any():
                    loss += criterion(out[bird_mask, :n_bird_classes], y[bird_mask])
                if digit_mask.any():
                    loss += criterion(out[digit_mask, :10], y[digit_mask])

                # Accuracy: pick the right slice per sample
                predicted = torch.empty(x.size(0), dtype=torch.long, device=device)
                if bird_mask.any():
                    predicted[bird_mask]  = out[bird_mask, :n_bird_classes].max(1).indices
                if digit_mask.any():
                    predicted[digit_mask] = out[digit_mask, :10].max(1).indices
            else:
                out = model(x)
                loss = criterion(out, y)
                predicted = out.max(1).indices

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()

            total_loss    += loss.item()
            total_train   += y.size(0)
            correct_train += predicted.eq(y).sum().item()

        train_acc = 100. * correct_train / total_train
        test_acc, correct_pc, total_pc = evaluate(model, test_loader, device, n_classes, dataset=args.dataset)
        print(f"Epoch {epoch + 1:>3}: Loss={total_loss:.4f}  "
              f"Train={train_acc:.2f}%  Test={test_acc:.2f}%")

        print("  Per-class accuracy:")
        for i in range(n_classes):
            if total_pc[i] > 0:
                acc = 100 * correct_pc[i] / total_pc[i]
                print(f"    Class {i}: {acc:.2f}%  ({correct_pc[i]}/{total_pc[i]})")
            else:
                print(f"    Class {i}: N/A")
        scheduler.step()
                # ── Checkpointing ────────────────────────────────────────────────
        # ckpt = {
        #     'epoch':           epoch,
        #     'model_state':     model.state_dict(),
        #     'optimizer_state': optimizer.state_dict(),
        #     'train_acc':       train_acc,
        #     'test_acc':        test_acc,
        # }
        # torch.save(ckpt, f'{CHECKPOINT_DIR}/latest.pt')
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(ckpt, f'{CHECKPOINT_DIR}/best.pt')
        #     print(f"  ✓ New best: {best_acc:.2f}%")


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])
    main()
