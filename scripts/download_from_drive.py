#!/usr/bin/env python3
"""Download datasets from Google Drive (Cold Vault) to local SSD.

Uses `gdown` for authenticated Google Drive downloads. Supports both
individual files and shared folder downloads.

Usage::

    # Download a single file by ID
    python scripts/download_from_drive.py \\
        --file-id 1ABcDeFgHiJkLmNoPqRsTuVwXyZ \\
        --output data_cache/sample_videos/dataset.zip

    # Download an entire shared folder
    python scripts/download_from_drive.py \\
        --folder-id 1ABcDeFgHiJkLmNoPqRsTuVwXyZ \\
        --output data_cache/sample_videos/

Prerequisites::

    pip install gdown
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def download_file(file_id: str, output_path: str, quiet: bool = False) -> Path:
    """Download a single file from Google Drive.

    Args:
        file_id: Google Drive file ID.
        output_path: Local destination path.
        quiet: Suppress download progress output.

    Returns:
        Path to the downloaded file.

    Raises:
        ImportError: If gdown is not installed.
        RuntimeError: If download fails.
    """
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required for Google Drive downloads. "
            "Install with: pip install gdown"
        ) from exc

    url = f"https://drive.google.com/uc?id={file_id}"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading file '%s' → '%s'", file_id, output)
    result = gdown.download(url, str(output), quiet=quiet)

    if result is None:
        raise RuntimeError(f"Failed to download file: {file_id}")

    logger.info("Download complete: %s (%.2f MB)", output.name, output.stat().st_size / 1e6)
    return output


def download_folder(folder_id: str, output_dir: str, quiet: bool = False) -> Path:
    """Download an entire shared folder from Google Drive.

    Args:
        folder_id: Google Drive folder ID.
        output_dir: Local destination directory.
        quiet: Suppress download progress output.

    Returns:
        Path to the output directory.

    Raises:
        ImportError: If gdown is not installed.
        RuntimeError: If download fails.
    """
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required for Google Drive downloads. "
            "Install with: pip install gdown"
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info("Downloading folder '%s' → '%s'", folder_id, output)

    gdown.download_folder(url, output=str(output), quiet=quiet)

    logger.info("Folder download complete → '%s'", output)
    return output


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Google Drive downloads.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = argparse.ArgumentParser(
        description="Download datasets from Google Drive (Cold Vault) to local SSD.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file-id",
        type=str,
        help="Google Drive file ID to download.",
    )
    group.add_argument(
        "--folder-id",
        type=str,
        help="Google Drive shared folder ID to download.",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Local output path (file or directory).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.file_id:
            download_file(args.file_id, args.output, quiet=args.quiet)
        else:
            download_folder(args.folder_id, args.output, quiet=args.quiet)
        return 0

    except ImportError as exc:
        logger.error("%s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Download failed: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
