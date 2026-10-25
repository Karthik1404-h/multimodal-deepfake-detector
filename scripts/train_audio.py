#!/usr/bin/env python3
"""Training script for AudioForensicCNN on real vs. synthetic speech spectrograms.

Trains a 2D CNN on (1, 128, 128) Mel-spectrograms extracted from the audio deepfake dataset
using BCEWithLogitsLoss and Adam optimizer.

Usage::

    python scripts/train_audio.py \\
        --data-dir data/raw/audio_deepfake \\
        --epochs 15 \\
        --batch-size 16 \\
        --lr 1e-4 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from deepfake_core.data.audio_dataset import DeepfakeAudioDataset
from deepfake_core.models.audio_cnn import AudioForensicCNN
from deepfake_core.utils.vram_manager import clear_gpu_cache, get_vram_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_audio")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Execute one training epoch.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", unit="batch")

    for x_batch, y_batch in progress:
        x_batch = x_batch.to(device)  # (B, 1, 128, 128)
        y_batch = y_batch.to(device)  # (B,)

        optimizer.zero_grad()

        use_amp = (device.type == "cuda")
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x_batch)  # (B,)
            loss = criterion(logits, y_batch)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size

        preds = (logits >= 0.0).float()
        correct += (preds == y_batch).sum().item()
        total += batch_size

        running_loss = total_loss / max(total, 1)
        running_acc = (correct / max(total, 1)) * 100
        progress.set_postfix({"loss": f"{running_loss:.4f}", "acc": f"{running_acc:.1f}%"})

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Evaluate model on the validation split.

    Returns:
        Tuple of (val_loss, val_acc).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Val]  ", unit="batch")

    with torch.no_grad():
        for x_batch, y_batch in progress:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            use_amp = (device.type == "cuda")
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size

            preds = (logits >= 0.0).float()
            correct += (preds == y_batch).sum().item()
            total += batch_size

            running_loss = total_loss / max(total, 1)
            running_acc = (correct / max(total, 1)) * 100
            progress.set_postfix({"val_loss": f"{running_loss:.4f}", "val_acc": f"{running_acc:.1f}%"})

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train AudioForensicCNN on deepfake speech dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/audio_deepfake",
        help="Path to audio deepfake dataset root (default: data/raw/audio_deepfake).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to save the best model checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Total training epochs (default: 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and validation (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20,
        help="Validation split ratio (default: 0.20).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on (default: cuda).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0).",
    )

    args = parser.parse_args(argv)

    # 1. Device resolution
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    logger.info("Initializing Audio Forensic Training Pipeline")
    logger.info("Device:      %s", device)
    if device.type == "cuda":
        vram = get_vram_summary()
        if vram:
            logger.info("VRAM Total:  %.0f MB (Allocated: %.0f MB)", vram["total_mb"], vram["allocated_mb"])

    # 2. Dataset and Counts
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Audio dataset directory not found: %s", data_dir)
        return 1

    dataset = DeepfakeAudioDataset(root_dir=data_dir)
    total_samples = len(dataset)
    if total_samples == 0:
        logger.error("No valid audio samples found in %s", data_dir)
        return 1

    real_count = sum(1 for _, label in dataset.samples if label == DeepfakeAudioDataset.LABEL_REAL)
    fake_count = sum(1 for _, label in dataset.samples if label == DeepfakeAudioDataset.LABEL_FAKE)

    logger.info("Dataset Composition: %d Real, %d Fake (%d total samples)", real_count, fake_count, total_samples)

    # 3. 80/20 Train/Validation Split
    val_size = max(1, int(total_samples * args.val_split))
    train_size = total_samples - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    logger.info("Partition: %d Train | %d Validation", train_size, val_size)

    # 3. Stratified and Balanced Training Setup
    train_labels = [dataset.samples[i][1] for i in train_dataset.indices]
    num_real = sum(1 for y in train_labels if y == 0.0)
    num_fake = sum(1 for y in train_labels if y == 1.0)
    logger.info("Train set composition: %d REAL, %d FAKE", num_real, num_fake)

    # Compute inverse class frequencies for balanced sampling
    weight_real = 1.0 / max(num_real, 1)
    weight_fake = 1.0 / max(num_fake, 1)
    sample_weights = [weight_fake if y == 1.0 else weight_real for y in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    batch_size = min(args.batch_size, train_size) if train_size > 0 else args.batch_size

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    # 4. Model, Loss, Optimizer with Cosine Annealing Schedule
    model = AudioForensicCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    # 5. Output and Log Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_audio_model.pth"

    log_dir = Path("models/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    history_file = log_dir / "audio_training_history.json"
    history: list[dict] = []

    best_val_score = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    logger.info("Starting Audio CNN Training: %d epochs, batch_size=%d, lr=%.1e", args.epochs, batch_size, args.lr)

    # 6. Training Loop with Live Console Streaming
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        scheduler.step()

        # Print Epoch Summary directly to terminal with immediate flush
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}%",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)

        # Checkpoint prioritizing validation accuracy then validation loss
        is_best = (val_acc > best_val_score) or (val_acc == best_val_score and val_loss < best_val_loss)
        if is_best:
            best_val_score = val_acc
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_checkpoint_path,
            )
            logger.info("--> Best audio model saved to '%s' (val_loss=%.4f, val_acc=%.1f%%)", best_checkpoint_path, val_loss, val_acc * 100)

        clear_gpu_cache()

    elapsed = time.time() - start_time
    logger.info(
        "\nTraining complete in %.1f minutes. Best Epoch: %d (Val Loss: %.4f, Val Acc: %.1f%%). Checkpoint: %s",
        elapsed / 60.0,
        best_epoch,
        best_val_loss,
        best_val_score * 100,
        best_checkpoint_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
