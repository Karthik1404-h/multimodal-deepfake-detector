"""Command-line interface for deepfake_core.

Provides a terminal-based entry point for running deepfake detection
on video files.

Usage::

    python -m deepfake_core.cli --video sample.mp4 --device cuda
    deepfake-detect --video sample.mp4 --device cuda --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the deepfake detection CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="deepfake-detect",
        description=(
            "deepfake_core CLI — Multimodal deepfake detection on video files. "
            "Analyzes spatial forensic features using EfficientNet-B0."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  deepfake-detect --video suspect.mp4 --device cuda\n"
            "  deepfake-detect --video clip.mp4 --device cpu --threshold 0.7\n"
            "  python -m deepfake_core.cli --video test.mp4 --max-frames 50\n"
        ),
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file (MP4, AVI, MKV, etc.).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device (default: cuda). Falls back to CPU if CUDA unavailable.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a trained model checkpoint (.pt). Uses untrained model if not provided.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5). Scores above this are FAKE.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum number of frames to sample from the video (default: 30).",
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
        help="Enable verbose (DEBUG) logging output.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments. Uses sys.argv if None.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Configure logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noisy third-party loggers
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger("deepfake_core.cli")

    # --- Validate input ---
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        return 1

    if not video_path.is_file():
        logger.error("Path is not a file: %s", video_path)
        return 1

    # --- Run detection ---
    try:
        from deepfake_core.pipeline import UnifiedDetector

        logger.info("Initializing detector on device='%s'...", args.device)

        detector = UnifiedDetector(
            device=args.device,
            model_path=args.model_path,
            threshold=args.threshold,
            max_frames=args.max_frames,
            face_confidence=args.face_confidence,
        )

        logger.info("Analyzing: %s", video_path.name)
        result = detector.predict_video_clip(str(video_path))

        # Enrich output with metadata
        output = {
            "video": str(video_path.name),
            **result,
            "device": args.device,
        }

        # Pretty-print JSON result
        print(json.dumps(output, indent=2))

        # Exit code based on detection result
        return 0

    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        return 1
    except ImportError as exc:
        logger.error(
            "Missing dependency: %s. Run 'pip install -e .' to install all requirements.",
            exc,
        )
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
