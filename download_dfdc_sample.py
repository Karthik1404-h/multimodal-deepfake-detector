#!/usr/bin/env python3
"""Download and organize DFDC and multimodal video datasets.

Fetches the DFDC video subset from Kaggle, cross-references with video-level
metadata labels, and incorporates local video samples into:
    - data/raw/dfdc_videos/real/
    - data/raw/dfdc_videos/fake/

Usage::

    python download_dfdc_sample.py
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

import kaggle
import kagglehub
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_dfdc_sample")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def organize_existing_local_videos(target_real: Path, target_fake: Path) -> tuple[int, int]:
    """Incorporate all existing raw video samples into dfdc_videos directories.

    Args:
        target_real: Path to real directory.
        target_fake: Path to fake directory.

    Returns:
        Tuple of (real_copied, fake_copied).
    """
    logger.info("Scanning for existing local videos in data/raw...")
    real_copied = 0
    fake_copied = 0

    # 1. Existing deepfake_videos/video (authentic)
    real_sources = [
        Path("data/raw/deepfake_videos/video"),
        Path("data/raw/face_anti_spoofing"),
    ]
    for src_dir in real_sources:
        if src_dir.exists():
            for p in src_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    unique_name = f"local_real_{src_dir.name}_{p.parent.name}_{p.stem}{p.suffix.lower()}"
                    dest = target_real / unique_name
                    if not dest.exists():
                        shutil.copy2(p, dest)
                        real_copied += 1

    # Single real test video
    real_test = Path("data/raw/deepfake_videos/real_test_video.mp4")
    if real_test.exists():
        dest = target_real / "local_real_test_video.mp4"
        if not dest.exists():
            shutil.copy2(real_test, dest)
            real_copied += 1

    # 2. Existing deepfake_videos/deepfake (manipulated)
    fake_sources = [
        Path("data/raw/deepfake_videos/deepfake"),
    ]
    for src_dir in fake_sources:
        if src_dir.exists():
            for p in src_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    unique_name = f"local_fake_{src_dir.name}_{p.stem}{p.suffix.lower()}"
                    dest = target_fake / unique_name
                    if not dest.exists():
                        shutil.copy2(p, dest)
                        fake_copied += 1

    # Single fake test video
    fake_test = Path("data/raw/deepfake_videos/test_video.mp4")
    if fake_test.exists():
        dest = target_fake / "local_fake_test_video.mp4"
        if not dest.exists():
            shutil.copy2(fake_test, dest)
            fake_copied += 1

    logger.info("Local harvest complete: %d REAL videos, %d FAKE videos.", real_copied, fake_copied)
    return real_copied, fake_copied


def fetch_dfdc_video_dataset(
    target_dir: str | Path = "data/raw/dfdc_videos",
) -> tuple[int, int]:
    """Download DFDC sample videos from Kaggle and sort into real/ and fake/ directories.

    Args:
        target_dir: Target output directory.

    Returns:
        Tuple of (total_real, total_fake).
    """
    target_base = Path(target_dir)
    real_dir = target_base / "real"
    fake_dir = target_base / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # First, harvest all available local videos
    organize_existing_local_videos(real_dir, fake_dir)

    # Next, fetch DFDC video labels table
    logger.info("Fetching DFDC video metadata labels from Kaggle...")
    meta_path = kagglehub.dataset_download("basharallabadi/dfdc-video-audio-labels")
    csv_file = Path(meta_path) / "metadata.csv"

    label_map: dict[str, str] = {}
    if csv_file.exists():
        df_meta = pd.read_csv(csv_file)
        logger.info("Loaded metadata table with %d entries.", len(df_meta))
        for _, row in df_meta.iterrows():
            label_map[str(row["filename"]).strip()] = str(row["label"]).strip().upper()
    else:
        logger.warning("Metadata CSV not found in %s; using heuristic labeling.", meta_path)

    # Next, download DFDC video sample archive
    logger.info("Downloading DFDC video sample dataset from Kaggle...")
    temp_download_dir = Path("data/raw/dfdc_temp")
    temp_download_dir.mkdir(parents=True, exist_ok=True)

    try:
        kaggle.api.dataset_download_files(
            "swapnavasireddy/dfdc-sample-video",
            path=str(temp_download_dir),
            quiet=False,
            unzip=True,
        )
        logger.info("DFDC sample videos downloaded and extracted.")

        # Organize extracted .mp4 files
        real_count = 0
        fake_count = 0

        extracted_videos = list(temp_download_dir.rglob("*.mp4"))
        logger.info("Found %d DFDC sample videos to organize.", len(extracted_videos))

        for vfile in extracted_videos:
            name = vfile.name
            label = label_map.get(name)

            if label is None:
                # Heuristic fallback
                if "real" in name.lower():
                    label = "REAL"
                else:
                    label = "FAKE"

            dest_folder = real_dir if label == "REAL" else fake_dir
            dest_file = dest_folder / name

            if not dest_file.exists():
                shutil.copy2(vfile, dest_file)

            if label == "REAL":
                real_count += 1
            else:
                fake_count += 1

        logger.info("DFDC videos organized: %d REAL, %d FAKE.", real_count, fake_count)

    except Exception as exc:
        logger.error("Error during Kaggle DFDC download: %s", exc)
        logger.info("Continuing with currently available local videos.")

    finally:
        # Clean up temporary download directory
        if temp_download_dir.exists():
            shutil.rmtree(temp_download_dir, ignore_errors=True)

    total_real = len(list(real_dir.glob("*.mp4"))) + len(list(real_dir.glob("*.mov"))) + len(list(real_dir.glob("*.MOV")))
    total_fake = len(list(fake_dir.glob("*.mp4"))) + len(list(fake_dir.glob("*.mov"))) + len(list(fake_dir.glob("*.MOV")))

    logger.info("=== Dataset Summary in %s ===", target_base)
    logger.info("  --> Total REAL videos: %d in %s", total_real, real_dir)
    logger.info("  --> Total FAKE videos: %d in %s", total_fake, fake_dir)

    return total_real, total_fake


if __name__ == "__main__":
    fetch_dfdc_video_dataset()