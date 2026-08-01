#!/usr/bin/env python3
"""Batch face extraction from raw video datasets.

Reads MP4/AVI clips from a specified directory, extracts face crops
using the deepfake_core FaceExtractor, and saves 224×224 RGB images
as PNGs into data_cache/processed_faces/.

Usage::

    python scripts/preprocess_dataset.py \\
        --input-dir /path/to/raw/videos \\
        --output-dir data_cache/processed_faces \\
        --max-faces-per-video 30

The output directory structure mirrors the input::

    data_cache/processed_faces/
    ├── video_001/
    │   ├── frame_0000.png
    │   ├── frame_0005.png
    │   └── ...
    ├── video_002/
    │   └── ...
    └── ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Ensure deepfake_core is importable (works with `pip install -e .`)
from deepfake_core.preprocessing.face_extractor import FaceExtractor

logger = logging.getLogger(__name__)

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}


def discover_videos(input_dir: Path) -> list[Path]:
    """Recursively find all video files in a directory.

    Args:
        input_dir: Root directory to search.

    Returns:
        Sorted list of video file paths.
    """
    videos = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    videos.sort()
    logger.info("Discovered %d video files in '%s'", len(videos), input_dir)
    return videos


def sample_frames_from_video(
    video_path: Path,
    max_frames: int = 30,
) -> list[tuple[int, np.ndarray]]:
    """Sample frames uniformly from a video.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of (frame_index, bgr_frame) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = max_frames  # Fallback for streams

        sample_count = min(max_frames, total)
        indices = np.linspace(0, total - 1, sample_count, dtype=int)

        frames: list[tuple[int, np.ndarray]] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append((int(idx), frame))

        return frames

    finally:
        cap.release()


def process_single_video(
    video_path: Path,
    output_dir: Path,
    extractor: FaceExtractor,
    max_faces: int = 30,
) -> int:
    """Extract face crops from a single video and save to disk.

    Args:
        video_path: Path to the source video.
        output_dir: Base output directory for face crops.
        extractor: Initialized FaceExtractor instance.
        max_faces: Maximum face crops per video.

    Returns:
        Number of face crops saved.
    """
    # Create per-video subdirectory
    video_name = video_path.stem
    video_out = output_dir / video_name
    video_out.mkdir(parents=True, exist_ok=True)

    frames = sample_frames_from_video(video_path, max_frames=max_faces)
    saved_count = 0

    for frame_idx, frame in frames:
        crop = extractor.extract_face_crop(frame)
        if crop is not None:
            # Save as PNG (lossless)
            out_path = video_out / f"frame_{frame_idx:04d}.png"
            # Convert RGB → BGR for OpenCV saving
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), crop_bgr)
            saved_count += 1

    return saved_count


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the preprocessing script.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Extract face crops from video datasets for deepfake training.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_cache/processed_faces",
        help="Output directory for face crops (default: data_cache/processed_faces).",
    )
    parser.add_argument(
        "--max-faces-per-video",
        type=int,
        default=30,
        help="Max face crops to extract per video (default: 30).",
    )
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.5,
        help="Minimum face detection confidence (default: 0.5).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos
    videos = discover_videos(input_dir)
    if not videos:
        logger.error("No video files found in '%s'", input_dir)
        return 1

    # Process all videos
    total_faces = 0
    with FaceExtractor(min_confidence=args.face_confidence) as extractor:
        for video_path in tqdm(videos, desc="Processing videos", unit="video"):
            try:
                count = process_single_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    extractor=extractor,
                    max_faces=args.max_faces_per_video,
                )
                total_faces += count
                logger.debug("%s: %d faces extracted", video_path.name, count)
            except Exception as exc:
                logger.error("Failed to process '%s': %s", video_path.name, exc)
                continue

    logger.info(
        "Preprocessing complete. %d face crops extracted from %d videos → '%s'",
        total_faces,
        len(videos),
        output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
