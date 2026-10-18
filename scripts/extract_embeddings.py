#!/usr/bin/env python3
"""Pre-compute spatial feature embeddings for video sequences.

Extracts 1280-dimensional feature representations from the trained EfficientNet-B0
spatial backbone (before classification head) for 30 uniformly sampled face crops per video.
Saves each video sequence as a PyTorch tensor of shape (30, 1280) to train the
TemporalAnomalyLSTM on authentic video dynamics.

Usage::

    python scripts/extract_embeddings.py \
        --data-dir data/raw/dfdc_videos \
        --output-dir data/processed/embeddings/train \
        --checkpoint models/checkpoints/best_spatial_model.pth \
        --frames-per-video 30 \
        --device cuda \
        --clean
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from deepfake_core.models.spatial_efficientnet import SpatialForensicClassifier
from deepfake_core.preprocessing.face_extractor import FaceExtractor
from deepfake_core.utils.vram_manager import clear_gpu_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_embeddings")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_eval_transforms() -> transforms.Compose:
    """Return standard ImageNet evaluation transforms matching training pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_spatial_backbone(checkpoint_path: Path, device: torch.device) -> SpatialForensicClassifier:
    """Initialize SpatialForensicClassifier, load weights, and set to eval mode."""
    model = SpatialForensicClassifier(pretrained=False, freeze_backbone=False)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info("Loading spatial weights from '%s'...", checkpoint_path)
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    logger.info("Spatial backbone loaded and set to eval mode on %s.", device)
    return model


def sample_video_frames(video_path: Path, max_frames: int = 30) -> list[np.ndarray]:
    """Sample frames uniformly from a video file and convert immediately from BGR to RGB.

    Args:
        video_path: Path to video file.
        max_frames: Total number of frames to sample.

    Returns:
        List of RGB frames as numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video file: %s", video_path)
        return []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback sequential read
            frames: list[np.ndarray] = []
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return frames

        # Uniform linear sampling across the duration
        sample_count = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return frames
    finally:
        cap.release()


def extract_faces_from_frames(
    frames: list[np.ndarray],
    face_extractor: FaceExtractor,
    target_count: int = 30,
) -> list[np.ndarray]:
    """Extract face crops from RGB frames and ensure exactly target_count crops via edge-padding.

    Args:
        frames: List of RGB video frames.
        face_extractor: MediaPipe FaceExtractor instance.
        target_count: Exact sequence length required (default: 30).

    Returns:
        List of exactly target_count 224x224 RGB face crops.
    """
    face_crops: list[np.ndarray] = []

    for frame in frames:
        crop = face_extractor.extract_face_crop(frame, is_bgr=False)
        if crop is not None:
            face_crops.append(crop)

    if not face_crops:
        return []

    # Sequence alignment to exactly target_count
    if len(face_crops) < target_count:
        # Edge-padding: duplicate last face crop to avoid synthetic zero vectors
        while len(face_crops) < target_count:
            face_crops.append(face_crops[-1])
    elif len(face_crops) > target_count:
        # Uniformly subsample target_count crops
        indices = np.linspace(0, len(face_crops) - 1, target_count, dtype=int)
        face_crops = [face_crops[i] for i in indices]

    return face_crops


def extract_video_embedding(
    video_path: Path,
    model: SpatialForensicClassifier,
    face_extractor: FaceExtractor,
    transform: transforms.Compose,
    device: torch.device,
    target_frames: int = 30,
) -> Optional[torch.Tensor]:
    """Extract (target_frames, 1280) spatial feature embedding tensor for a single video.

    Args:
        video_path: Path to .mp4 video file.
        model: SpatialForensicClassifier backbone.
        face_extractor: MediaPipe FaceExtractor.
        transform: ImageNet evaluation transforms.
        device: CUDA or CPU device.
        target_frames: Number of frames per video sequence (30).

    Returns:
        Tensor of shape (target_frames, 1280) or None if no faces found.
    """
    frames = sample_video_frames(video_path, max_frames=target_frames)
    if not frames:
        return None

    face_crops = extract_faces_from_frames(frames, face_extractor, target_count=target_frames)
    if not face_crops:
        return None

    # Prepare batch tensor: (30, 3, 224, 224)
    tensors = [transform(Image.fromarray(crop)) for crop in face_crops]
    batch = torch.stack(tensors).to(device)

    use_amp = (device.type == "cuda")
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = model.extract_features(batch)  # (30, 1280)
        else:
            embeddings = model.extract_features(batch)

    # Return as CPU float32 tensor
    return embeddings.cpu().float()


def process_video_class(
    class_label: str,
    input_dir: Path,
    output_dir: Path,
    model: SpatialForensicClassifier,
    face_extractor: FaceExtractor,
    transform: transforms.Compose,
    device: torch.device,
    target_frames: int = 30,
    overwrite: bool = False,
) -> int:
    """Process all video files in a class directory and save embedding tensors.

    Args:
        class_label: 'real' or 'fake'.
        input_dir: Directory containing input video files.
        output_dir: Directory where .pt files will be saved.
        model: Spatial model.
        face_extractor: FaceExtractor.
        transform: Evaluation transforms.
        device: Torch device.
        target_frames: Frames per sequence.
        overwrite: Whether to overwrite existing .pt files.

    Returns:
        Number of videos successfully converted to embeddings.
    """
    if not input_dir.exists():
        logger.warning("Input directory does not exist: %s", input_dir)
        return 0

    class_output_dir = output_dir / class_label
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video files
    video_files = [
        Path(p) for p in os.scandir(input_dir)
        if p.is_file() and os.path.splitext(p.name)[1].lower() in VIDEO_EXTENSIONS
    ]

    logger.info("Found %d video files for '%s' in %s", len(video_files), class_label.upper(), input_dir)
    if not video_files:
        return 0

    saved_count = 0
    skipped_count = 0
    no_faces_count = 0

    progress_bar = tqdm(video_files, desc=f"Extracting {class_label.upper()} embeddings", unit="video")
    for video_path in progress_bar:
        out_pt_path = class_output_dir / f"{video_path.stem}.pt"

        if out_pt_path.exists() and not overwrite:
            skipped_count += 1
            continue

        try:
            emb = extract_video_embedding(
                video_path=video_path,
                model=model,
                face_extractor=face_extractor,
                transform=transform,
                device=device,
                target_frames=target_frames,
            )

            if emb is not None:
                torch.save(emb, out_pt_path)
                saved_count += 1
            else:
                no_faces_count += 1

        except Exception as exc:
            logger.error("Failed to process video '%s': %s", video_path.name, exc)

        # Clear GPU cache periodically
        if saved_count % 20 == 0:
            clear_gpu_cache()

    logger.info(
        "Finished '%s': %d saved, %d skipped (existing), %d failed/no-faces",
        class_label.upper(),
        saved_count,
        skipped_count,
        no_faces_count,
    )
    return saved_count


def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract (30, 1280) temporal embeddings from video files using SpatialForensicClassifier.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/dfdc_videos",
        help="Root directory containing 'real' and 'fake' video subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/embeddings/train",
        help="Root directory where extracted .pt tensors will be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_spatial_model.pth",
        help="Path to trained spatial backbone checkpoint (.pth).",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=30,
        help="Exact number of uniformly sampled face crops per sequence (default: 30).",
    )
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe face detection confidence (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device (default: cuda).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt embedding files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing embeddings in output-dir before starting to prevent stale/static image pollution.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    # Resolve device with CUDA fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Clean old single-frame embeddings if requested
    if args.clean and output_dir.exists():
        logger.info("Cleaning old embeddings from %s...", output_dir)
        for sub in ["real", "fake"]:
            sub_path = output_dir / sub
            if sub_path.exists():
                shutil.rmtree(sub_path)
                sub_path.mkdir(parents=True, exist_ok=True)
        logger.info("Output directories cleaned.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained spatial model
    model = load_spatial_backbone(checkpoint_path, device=device)
    eval_transforms = get_eval_transforms()

    # 2. Initialize face extractor
    with FaceExtractor(min_confidence=args.face_confidence) as face_extractor:
        total_saved = 0
        for class_label in ["real", "fake"]:
            input_class_dir = data_dir / class_label
            saved = process_video_class(
                class_label=class_label,
                input_dir=input_class_dir,
                output_dir=output_dir,
                model=model,
                face_extractor=face_extractor,
                transform=eval_transforms,
                device=device,
                target_frames=args.frames_per_video,
                overwrite=args.overwrite or args.clean,
            )
            total_saved += saved

    logger.info("Embedding extraction complete. Total video sequence embeddings: %d", total_saved)


if __name__ == "__main__":
    main()
