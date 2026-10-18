#!/usr/bin/env python3
"""Training script for the SpatialForensicClassifier.

Trains an EfficientNet-B0 binary classifier on preprocessed face crops
(224×224) using mixed precision (FP16) training optimized for RTX 4050
(6 GB VRAM). Prevents data leakage by splitting train/val at the video level,
using stratified splitting to guarantee both classes are represented.

Usage::

    python scripts/train_spatial.py \
        --data-dir data/processed/train \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-4 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from deepfake_core.models.spatial_efficientnet import SpatialForensicClassifier
from deepfake_core.utils.vram_manager import clear_gpu_cache, get_vram_summary

logger = logging.getLogger(__name__)

class LabelMapper:
    """Picklable target transform for mapping dataset classes."""
    def __init__(self, fake_class_idx: int):
        self.fake_class_idx = fake_class_idx
        
    def __call__(self, x: int) -> float:
        return 1.0 if x == self.fake_class_idx else 0.0


class AddGaussianNoise:
    """Inject zero-mean Gaussian noise to simulate sensor noise and compression block artifacts."""

    def __init__(self, mean: float = 0.0, std: float = 0.025, p: float = 0.4) -> None:
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation DataLoaders with stratified video-level split.

    Args:
        data_dir: Path to preprocessed face crops directory.
        batch_size: Training batch size (tune for VRAM).
        num_workers: DataLoader worker processes.
        val_split: Fraction of data to use for validation.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Aggressive augmentations to simulate H.264 video compression, blur, and lighting variance
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Video compression & lighting variations
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        # Codec quantization and contrast shifts
        transforms.RandomAutocontrast(p=0.3),
        # Compression blurring and motion blur simulation
        transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 2.0)),
        ]),
        # Softening and codec sharpness ringing
        transforms.RandomAdjustSharpness(sharpness_factor=0.6, p=0.4),
        transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.3),
        # Subtle translation/scale
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        # Sensor & compression block noise
        AddGaussianNoise(mean=0.0, std=0.02, p=0.35),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value="random"),
    ])

    # Validation set transforms remain pure and unaugmented
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Default ImageFolder loads alphabetically (fake=0, real=1).
    # We will use target_transform to strictly enforce REAL=0.0, FAKE=1.0.
    temp_dataset = datasets.ImageFolder(data_dir)
    real_class_idx = temp_dataset.class_to_idx['real']
    fake_class_idx = temp_dataset.class_to_idx['fake']
    
    # Explicit mapping using a picklable class instead of lambda
    target_transform = LabelMapper(fake_class_idx)

    full_dataset_train = datasets.ImageFolder(data_dir, transform=transform_train, target_transform=target_transform)
    full_dataset_val = datasets.ImageFolder(data_dir, transform=transform_val, target_transform=target_transform)

    # --- Stratified Video-Level Split ---
    video_to_indices = defaultdict(list)
    video_to_class = {}
    
    for idx, (path, class_idx) in enumerate(temp_dataset.samples):
        filename = Path(path).stem
        if "_f" in filename:
            video_id = filename.rsplit("_f", 1)[0]
        else:
            video_id = filename
            
        video_to_indices[video_id].append(idx)
        video_to_class[video_id] = class_idx

    real_videos = [vid for vid, c_idx in video_to_class.items() if c_idx == real_class_idx]
    fake_videos = [vid for vid, c_idx in video_to_class.items() if c_idx == fake_class_idx]
    
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    val_real_count = max(1, int(len(real_videos) * val_split)) if real_videos else 0
    val_fake_count = max(1, int(len(fake_videos) * val_split)) if fake_videos else 0
    
    val_vids = set(real_videos[:val_real_count] + fake_videos[:val_fake_count])
    train_vids = set(real_videos[val_real_count:] + fake_videos[val_fake_count:])

    train_indices = []
    val_indices = []
    for vid, indices in video_to_indices.items():
        if vid in val_vids:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    train_set = Subset(full_dataset_train, train_indices)
    val_set = Subset(full_dataset_val, val_indices)
    
    print("Train Class Mapping:", getattr(full_dataset_train, 'class_to_idx', 'Explicit LabelMapper Active'))
    print("Val Class Mapping:  ", getattr(full_dataset_val, 'class_to_idx', 'Explicit LabelMapper Active'))
    
    _, train_sample_label = train_set[0]
    _, val_sample_label = val_set[0]
    print(f"Sanity Check -> Train Sample 0 Label: {train_sample_label} | Val Sample 0 Label: {val_sample_label}")

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    logger.info(
        "Dataset loaded: %d total frames. Stratified splitting by video.",
        len(full_dataset_train)
    )
    logger.info(
        "Train: %d frames (from %d videos). Val: %d frames (from %d videos).",
        len(train_indices), len(train_vids), len(val_indices), len(val_vids)
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
    total_epochs: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            assert outputs.shape == labels.shape, f"Shape mismatch: {outputs.shape} vs {labels.shape}"
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        predictions = (outputs >= 0.0).long()
        correct += (predictions == labels.long()).sum().item()
        total += images.size(0)
        num_batches += 1
        
        running_acc = (correct / max(total, 1)) * 100
        progress.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{running_acc:.1f}%'})

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            assert outputs.shape == labels.shape, f"Shape mismatch: {outputs.shape} vs {labels.shape}"
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predictions = (outputs >= 0.0).long()
        
        correct += (predictions == labels.long()).sum().item()
        total += images.size(0)
        
        if HAS_SKLEARN:
            all_labels.extend(labels.cpu().numpy())
            # Convert logits to probabilities for AUC
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            
        running_acc = (correct / max(total, 1)) * 100
        progress.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{running_acc:.1f}%'})

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    if len(all_labels) > 0 and epoch == 1:
        logger.debug(f"Sample True Targets: {all_labels[:5]}")
        logger.debug(f"Sample Predicted Probs: {[round(p, 4) for p in all_probs[:5]]}")
    
    auc = 0.0
    if HAS_SKLEARN and len(all_labels) > 0:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0  # Only one class present

    return avg_loss, accuracy, auc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SpatialForensicClassifier on preprocessed face crops.")
    parser.add_argument("--data-dir", type=str, default="data/processed/train", help="Path to face crops directory.")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Directory for saved models.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
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

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    if device.type == "cuda":
        summary = get_vram_summary()
        if summary:
            logger.info("GPU: %.0f MB total, %.0f MB allocated", summary["total_mb"], summary["allocated_mb"])

    if not HAS_SKLEARN:
        logger.warning("scikit-learn not found. AUC will not be calculated.")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return 1

    train_loader, val_loader = build_dataloaders(str(data_dir), batch_size=args.batch_size, num_workers=args.num_workers)

    # Stage 1: Freeze backbone, train only the head
    model = SpatialForensicClassifier(pretrained=True, freeze_backbone=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = None
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("models/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    history_file = log_dir / "training_history.json"
    history = []

    best_val_auc = 0.0
    start_time = time.time()

    logger.info("Starting Two-Stage training: %d total epochs, batch_size=%d", args.epochs, args.batch_size)
    logger.info("--- Stage 1 (Epoch 1-5): Training head only (lr=1e-4) ---")

    for epoch in range(1, args.epochs + 1):
        if epoch == 6:
            logger.info("--- Stage 2 (Epoch 6-10): Unfreezing entire backbone (lr=1e-5) ---")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device, epoch, args.epochs)
        
        if scheduler is not None:
            scheduler.step()

        # Clean Epoch Summary
        # Epoch 01/20 | Train Loss: 0.4512 | Train Acc: 81.2% | Val Loss: 0.3891 | Val Acc: 84.5% | Val AUC: 0.9123
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val AUC: {val_auc:.4f}")

        # JSON Logging
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc
        })
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            logger.info("[NEW BEST] Validation AUC improved to %.4f", val_auc)
            checkpoint_path = output_dir / "best_spatial_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_loss": val_loss,
                },
                str(checkpoint_path),
            )

        clear_gpu_cache()

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f min. Best val_auc=%.4f", elapsed / 60, best_val_auc)

    # Automatically call the reporting script
    logger.info("Generating final evaluation report...")
    report_script = Path("scripts/generate_report.py")
    if report_script.exists():
        subprocess.run([sys.executable, str(report_script)])

    return 0

if __name__ == "__main__":
    sys.exit(main())
