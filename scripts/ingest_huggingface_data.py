#!/usr/bin/env python3
"""Ingest video datasets from Hugging Face and extract face crops.

Downloads balanced samples of REAL and FAKE videos from specified Hugging Face
datasets, saves them temporarily to disk, extracts 224x224 facial crops using
MediaPipe FaceExtractor, and saves them into the local SSD hot cache.

Usage::

    python scripts/ingest_huggingface_data.py --max-videos 15 --sample-rate 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import requests
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' is not installed. Please run `pip install -e .` again.")
    sys.exit(1)

# Ensure deepfake_core is importable
from deepfake_core.preprocessing.face_extractor import FaceExtractor

logger = logging.getLogger(__name__)


def save_temp_video(row: dict, temp_dir: Path, video_id: str) -> Optional[str]:
    """Inspect row fields and save video to a temporary file if necessary.

    Args:
        row: A single row/item from the Hugging Face dataset.
        temp_dir: Directory to store temporary video files.
        video_id: Unique identifier for the video.

    Returns:
        The path to the local video file, or None if extraction failed.
    """
    video_data = None

    # Check common keys for video data
    for key in ['video', 'file', 'url', 'path', 'video_url', 'download_url']:
        if key in row and row[key] is not None:
            video_data = row[key]
            break
            
    # Fallback: scan all values
    if not video_data:
        for val in row.values():
            if isinstance(val, dict) and ('bytes' in val or 'path' in val):
                video_data = val
                break
            elif isinstance(val, str) and (val.startswith('http') or val.endswith('.mp4') or val.endswith('.avi')):
                video_data = val
                break

    if not video_data:
        return None

    temp_path = temp_dir / f"{video_id}.mp4"

    try:
        if isinstance(video_data, dict):
            # Typical Hugging Face Video feature
            if 'path' in video_data and video_data['path'] and os.path.exists(video_data['path']):
                return video_data['path']
            if 'bytes' in video_data and video_data['bytes']:
                with open(temp_path, "wb") as f:
                    f.write(video_data['bytes'])
                return str(temp_path)
        elif isinstance(video_data, str):
            if video_data.startswith('http'):
                # Direct URL download
                r = requests.get(video_data, stream=True, timeout=15)
                r.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return str(temp_path)
            elif os.path.exists(video_data):
                # Local file path
                return video_data
    except Exception as exc:
        logger.warning("Failed to download or save video '%s': %s", video_id, exc)

    return None


def process_video(
    video_path: str,
    extractor: FaceExtractor,
    out_dir: Path,
    video_id: str,
    sample_rate: int
) -> int:
    """Process a single video, extracting and saving faces.

    Args:
        video_path: Path to the local video file.
        extractor: Initialized FaceExtractor instance.
        out_dir: Directory to save extracted face crops.
        video_id: Unique identifier for the video.
        sample_rate: Extract a face every N frames.

    Returns:
        Number of face crops successfully saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Failed to open video: %s", video_path)
        return 0

    saved_count = 0
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_rate == 0:
                crop = extractor.extract_face_crop(frame)
                if crop is not None:
                    # MediaPipe extractor returns RGB. cv2 expects BGR for saving.
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    out_path = out_dir / f"{video_id}_frame_{frame_idx:04d}.jpg"
                    # Save as high-quality JPEG
                    cv2.imwrite(str(out_path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1
                    
            frame_idx += 1
    finally:
        # Explicit release as requested for memory guardrails
        cap.release()
        
    return saved_count


def process_dataset(
    dataset_name: str,
    split: str,
    label_name: str,
    max_videos: int,
    sample_rate: int,
    temp_dir: Path,
    out_dir: Path,
    extractor: FaceExtractor
) -> tuple[int, int]:
    """Load dataset, download videos, and process faces.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to use.
        label_name: Class label ('real' or 'fake').
        max_videos: Maximum number of videos to process.
        sample_rate: Extract a face every N frames.
        temp_dir: Directory for temporary video storage.
        out_dir: Directory for saved face crops.
        extractor: FaceExtractor instance.

    Returns:
        A tuple of (videos_processed, total_faces_extracted).
    """
    try:
        logger.info("Loading dataset '%s' (split='%s') for %s class...", dataset_name, split, label_name.upper())
        # Use streaming to avoid downloading large datasets fully upfront
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as exc:
        logger.error("Failed to load dataset '%s': %s", dataset_name, exc)
        logger.error(
            "\n*** ACTION REQUIRED ***\n"
            "Make sure you have authenticated via `huggingface-cli login`.\n"
            "If this is a gated dataset, you must visit the dataset page on Hugging Face "
            "and accept the terms of use online first."
        )
        return 0, 0

    videos_processed = 0
    faces_extracted = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    progress = tqdm(total=max_videos, desc=f"Processing {label_name.upper()}", unit="vid")
    
    # We use enumerate on the dataset iterator
    try:
        dataset_iter = iter(dataset)
    except Exception as exc:
        logger.error("Failed to iterate dataset '%s': %s", dataset_name, exc)
        return 0, 0

    idx = 0
    while videos_processed < max_videos:
        try:
            row = next(dataset_iter)
        except StopIteration:
            logger.info("Reached end of dataset before max_videos.")
            break
        except Exception as exc:
            logger.warning("Error fetching row from dataset: %s", exc)
            continue
            
        video_id = f"{label_name}_{idx:05d}"
        idx += 1
        
        # Save temp video
        video_path = save_temp_video(row, temp_dir, video_id)
        if not video_path:
            # Not all rows might have videos, just skip
            continue
            
        # Process video
        count = process_video(video_path, extractor, out_dir, video_id, sample_rate)
        
        if count > 0:
            videos_processed += 1
            faces_extracted += count
            progress.update(1)
        else:
            logger.debug("Video %s skipped: no faces detected.", video_id)
            
        # Clean up temp file to keep disk usage low
        if video_path.startswith(str(temp_dir)) and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as e:
                logger.warning("Failed to delete temp video %s: %s", video_path, e)

    progress.close()
    return videos_processed, faces_extracted


def get_dir_size_mb(path: Path) -> float:
    """Calculate directory total file size in megabytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest datasets from Hugging Face for deepfake_core.")
    parser.add_argument("--max-videos", type=int, default=20, help="Maximum number of videos to process per class (default: 20).")
    parser.add_argument("--sample-rate", type=int, default=10, help="Extract 1 face every N frames (default: 10).")
    parser.add_argument("--fake-dataset", type=str, default="UniDataPro/deepfake-videos-dataset", help="HF Dataset for fake videos.")
    parser.add_argument("--real-dataset", type=str, default="UniDataPro/face-anti-spoofing", help="HF Dataset for real videos.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Paths setup according to project structure
    base_dir = Path("data_cache")
    
    temp_dir_real = base_dir / "sample_videos" / "real"
    temp_dir_fake = base_dir / "sample_videos" / "fake"
    
    out_dir_real = base_dir / "processed_faces" / "real"
    out_dir_fake = base_dir / "processed_faces" / "fake"

    logger.info("Initializing MediaPipe FaceExtractor (CPU mode)...")
    # Using context manager for FaceExtractor handles proper MediaPipe cleanup
    total_real_faces = 0
    total_fake_faces = 0
    
    with FaceExtractor(min_confidence=0.5) as extractor:
        # Process REAL videos
        logger.info("-" * 50)
        real_vids, real_faces = process_dataset(
            dataset_name=args.real_dataset,
            split=args.split,
            label_name="real",
            max_videos=args.max_videos,
            sample_rate=args.sample_rate,
            temp_dir=temp_dir_real,
            out_dir=out_dir_real,
            extractor=extractor
        )
        total_real_faces += real_faces
        
        # Process FAKE videos
        logger.info("-" * 50)
        fake_vids, fake_faces = process_dataset(
            dataset_name=args.fake_dataset,
            split=args.split,
            label_name="fake",
            max_videos=args.max_videos,
            sample_rate=args.sample_rate,
            temp_dir=temp_dir_fake,
            out_dir=out_dir_fake,
            extractor=extractor
        )
        total_fake_faces += fake_faces

    # Final Summary
    logger.info("=" * 50)
    logger.info("Data Ingestion Complete")
    logger.info("=" * 50)
    logger.info("Total REAL crops generated: %d", total_real_faces)
    logger.info("Total FAKE crops generated: %d", total_fake_faces)
    
    real_mb = get_dir_size_mb(out_dir_real)
    fake_mb = get_dir_size_mb(out_dir_fake)
    
    logger.info("Estimated footprint (REAL): %.2f MB", real_mb)
    logger.info("Estimated footprint (FAKE): %.2f MB", fake_mb)
    logger.info("Total SSD footprint: %.2f MB", real_mb + fake_mb)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
