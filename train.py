#!/usr/bin/env python3
"""
train.py - Training script for HandShaperr hand shape classification
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import prepare_data
from model import HandShapeClassifier, count_parameters
from utils import set_seed, save_json, create_run_dir


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Get all predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    save_path: Path,
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Show labels
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix',
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {save_path}")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================


def train(args):
    """Main training function."""
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")

    # Create run directory
    out_dir = Path(args.out_dir)
    run_dir = create_run_dir(out_dir)
    print(f"[INFO] Run directory: {run_dir}")

    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['timestamp'] = datetime.now().isoformat()
    save_json(config, run_dir / "config.json")

    # Prepare data
    train_loader, val_loader, test_loader, label_map, feature_stats, num_classes = prepare_data(
        data_dir=Path(args.data_dir),
        out_dir=run_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        split_by_session=args.split_by_session,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    # Create model
    model = HandShapeClassifier(
        input_dim=128,  # 21*3*2 + 2 mask flags
        hidden_dim=args.hidden,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    print(f"\n[INFO] Model parameters: {count_parameters(model):,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [],
    }

    best_val_acc = 0.0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - start_time

        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_map': label_map,
                'feature_stats': {k: v.tolist() for k, v in feature_stats.items()},
                'config': config,
            }
            torch.save(checkpoint, run_dir / "best.pt")
            print(f"  -> Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

    # Save training history
    save_json(history, run_dir / "history.json")

    # ==========================================================================
    # EVALUATION ON TEST SET
    # ==========================================================================

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])

    # Get predictions
    y_pred, y_true = get_predictions(model, test_loader, device)

    # Metrics
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Classification report
    idx_to_label = {v: k for k, v in label_map.items()}
    target_names = [idx_to_label[i] for i in range(num_classes)]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, target_names, run_dir / "confusion_matrix.png")

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'classification_report': classification_report(y_true, y_pred, target_names=target_names, output_dict=True),
    }
    save_json(test_results, run_dir / "test_results.json")

    print("\n" + "=" * 60)
    print(f"Training complete! Results saved to: {run_dir}")
    print("=" * 60)

    return run_dir


# =============================================================================
# MAIN
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HandShaperr hand shape classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="./dataset",
                        help="Directory containing dataset files")
    parser.add_argument("--out_dir", type=str, default="./runs",
                        help="Directory for output runs")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience")

    # Model
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")

    # Misc
    parser.add_argument("--split_by_session", action="store_true",
                        help="Split data by session_id to avoid leakage")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
