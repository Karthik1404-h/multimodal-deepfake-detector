#!/usr/bin/env python3
"""Process local raw datasets into standardized face crops.

Reads .mp4 and .mov files directly from the local file system using OpenCV,
avoiding streaming or torchcodec issues. Extracts face crops using the modern
MediaPipe Tasks API (FaceExtractor) and saves them as 224x224 JPEG images.

Outputs are stored in data/processed/train/real and data/processed/train/fake.
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# Ensure deepfake_core is importable
from deepfake_core.preprocessing.face_extractor import FaceExtractor

logger = logging.getLogger(__name__)


def process_videos(
    input_dir: str,
    output_dir: str,
    label: str,
    sample_rate: int,
    extractor: FaceExtractor
) -> None:
    """Process a directory of videos, extract faces, and save crops."""
    
    # Locate all video files (.mp4, .mov, .avi) recursively
    extensions = ('**/*.mp4', '**/*.MOV', '**/*.mov', '**/*.avi')
    video_paths = []
    for ext in extensions:
        # Use glob.glob recursively as required
        video_paths.extend(glob.glob(os.path.join(input_dir, ext), recursive=True))
        
    video_paths = sorted(list(set(video_paths)))
    
    if not video_paths:
        logger.warning(f"No video files found in '{input_dir}'")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    total_faces = 0
    progress = tqdm(video_paths, desc=f"Processing {label.upper()} videos", unit="vid")
    
    for v_path in progress:
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            logger.warning(f"Failed to open video: {v_path}")
            continue
            
        v_name = Path(v_path).stem
        frame_idx = 0
        saved_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % sample_rate == 0:
                    # FaceExtractor expects a BGR frame (OpenCV default) and returns RGB
                    crop_rgb = extractor.extract_face_crop(frame)
                    
                    if crop_rgb is not None:
                        # Convert the extracted RGB crop back to BGR for cv2.imwrite
                        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                        
                        out_file = out_path / f"{label}_{v_name}_f{frame_idx:05d}.jpg"
                        cv2.imwrite(str(out_file), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        saved_count += 1
                        total_faces += 1
                
                frame_idx += 1
        finally:
            # Explicitly release the capture as per strict execution guardrails
            cap.release()
            
        if saved_count == 0:
            logger.debug(f"No faces detected in video: {v_name}")
            
    progress.close()
    logger.info(f"Finished {label.upper()}: Extracted {total_faces} face crops from {len(video_paths)} videos.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process local videos into face crops.")
    parser.add_argument("--real-dir", type=str, default="data/raw/face_anti_spoofing", help="Path to real videos")
    parser.add_argument("--fake-dir", type=str, default="data/raw/deepfake_videos", help="Path to fake videos")
    parser.add_argument("--out-real", type=str, default="data/processed/train/real", help="Output path for real crops")
    parser.add_argument("--out-fake", type=str, default="data/processed/train/fake", help="Output path for fake crops")
    parser.add_argument("--sample-rate", type=int, default=10, help="Process 1 frame every N frames")
    
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    logger.info("Starting local facial preprocessing pipeline...")

    # Initialize FaceExtractor with 20% padding around the face bounding box
    with FaceExtractor(min_confidence=0.5, crop_size=224, padding_ratio=0.2) as extractor:
        # Process REAL videos
        process_videos(
            input_dir=args.real_dir,
            output_dir=args.out_real,
            label="real",
            sample_rate=args.sample_rate,
            extractor=extractor
        )
        
        # Process FAKE videos
        process_videos(
            input_dir=args.fake_dir,
            output_dir=args.out_fake,
            label="fake",
            sample_rate=args.sample_rate,
            extractor=extractor
        )

    logger.info("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
