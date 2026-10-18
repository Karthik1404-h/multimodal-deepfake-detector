"""Dataset loader for pre-computed temporal video embeddings.

Loads sequence tensors of shape (num_frames, 1280) and aligns them to a uniform
max_sequence_length via truncation or zero-padding.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TemporalEmbeddingDataset(Dataset):
    """PyTorch Dataset for video sequences of spatial feature embeddings.

    Scans 'real' and 'fake' subdirectories for .pt embedding tensors,
    assigns binary labels (0 for Real, 1 for Fake), and standardizes
    each video's sequence length via truncation or zero-padding.

    Args:
        root_dir: Root directory containing 'real' and 'fake' subfolders
            (e.g., 'data/processed/embeddings/train').
        max_sequence_length: Fixed number of time steps per sample (default: 30).
    """

    LABEL_REAL: int = 0
    LABEL_FAKE: int = 1

    def __init__(
        self,
        root_dir: str | Path,
        max_sequence_length: int = 30,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.max_sequence_length = max_sequence_length
        self.samples: list[tuple[Path, int]] = []

        self._load_samples()

    def _load_samples(self) -> None:
        """Scan real and fake subdirectories for .pt files."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Embeddings directory does not exist: {self.root_dir}")

        class_mapping = {
            "real": self.LABEL_REAL,
            "fake": self.LABEL_FAKE,
        }

        for class_name, label in class_mapping.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning("Class directory not found: %s", class_dir)
                continue

            # Scan efficiently using os.scandir
            count = 0
            with os.scandir(class_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".pt"):
                        self.samples.append((Path(entry.path), label))
                        count += 1

            logger.info("Loaded %d %s samples from %s", count, class_name.upper(), class_dir)

        if not self.samples:
            logger.warning("No .pt embedding files found in %s", self.root_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, align, and return a single video sequence embedding and its label.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
                - tensor: Float32 tensor of shape (max_sequence_length, 1280).
                - label: Float32 scalar tensor (0.0 for Real, 1.0 for Fake).
        """
        path, label = self.samples[idx]

        # Load embedding tensor (num_frames, 1280)
        embedding: torch.Tensor = torch.load(path, weights_only=True)

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        seq_len, feat_dim = embedding.shape

        # Sequence Alignment: Truncate or Zero-Pad
        if seq_len > self.max_sequence_length:
            # Truncate to maximum sequence length
            aligned_embedding = embedding[: self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            # Edge-pad by repeating the last valid frame (prevents synthetic zero-vector shock)
            pad_len = self.max_sequence_length - seq_len
            padding = embedding[-1:].repeat(pad_len, 1)
            aligned_embedding = torch.cat([embedding, padding], dim=0)
        else:
            aligned_embedding = embedding

        aligned_embedding = aligned_embedding.float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return aligned_embedding, label_tensor
