#!/usr/bin/env python3
"""Training script for the SpatialForensicClassifier.

Trains an EfficientNet-B0 binary classifier on preprocessed face crops
(224×224) using mixed precision (FP16) training optimized for RTX 4050
(6 GB VRAM).

Usage::

    python scripts/train_spatial.py \\
        --data-dir data_cache/processed_faces \\
        --epochs 20 \\
        --batch-size 32 \\
        --lr 1e-4 \\
        --device cuda

Expected data layout::

    data_cache/processed_faces/
    ├── real/           # Directory of real face crops
    │   ├── *.png
    │   └── ...
    └── fake/           # Directory of fake face crops
        ├── *.png
        └── ...
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from deepfake_core.models.spatial_efficientnet import SpatialForensicClassifier
from deepfake_core.utils.vram_manager import clear_gpu_cache, get_vram_summary

logger = logging.getLogger(__name__)


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders.

    Expects directory structure with 'real/' and 'fake/' subdirectories.
    Automatically maps: real → 0, fake → 1 (alphabetical order).

    Args:
        data_dir: Path to preprocessed face crops directory.
        batch_size: Training batch size (tune for VRAM).
        num_workers: DataLoader worker processes.
        val_split: Fraction of data to use for validation.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Split into train/val
    total = len(full_dataset)
    val_count = int(total * val_split)
    train_count = total - val_count

    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_count, val_count]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "Dataset loaded: %d total (%d train, %d val), classes=%s",
        total,
        train_count,
        val_count,
        full_dataset.classes,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch with mixed precision.

    Args:
        model: The classifier model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for FP16.
        device: Target device.
        epoch: Current epoch number (for logging).

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}", unit="batch", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation pass.

    Args:
        model: The classifier model.
        loader: Validation data loader.
        criterion: Loss function.
        device: Target device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predictions = (outputs >= 0.5).long()
        correct += (predictions == labels.long()).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main(argv: list[str] | None = None) -> int:
    """Main training entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Train SpatialForensicClassifier on preprocessed face crops.",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Path to face crops directory.")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory for saved models.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (tune for 6GB VRAM).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze EfficientNet backbone.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    # Log GPU info
    if device.type == "cuda":
        summary = get_vram_summary()
        if summary:
            logger.info(
                "GPU: %.0f MB total, %.0f MB allocated",
                summary["total_mb"],
                summary["allocated_mb"],
            )

    # Build data pipeline
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return 1

    train_loader, val_loader = build_dataloaders(
        str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build model
    model = SpatialForensicClassifier(
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    start_time = time.time()

    logger.info("Starting training: %d epochs, batch_size=%d, lr=%.1e", args.epochs, args.batch_size, args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | lr=%.2e",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_acc,
            optimizer.param_groups[0]["lr"],
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_spatial_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                str(checkpoint_path),
            )
            logger.info("  ✓ New best model saved (val_acc=%.4f) → %s", val_acc, checkpoint_path.name)

        # Clear GPU cache between epochs
        clear_gpu_cache()

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f min. Best val_acc=%.4f", elapsed / 60, best_val_acc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
