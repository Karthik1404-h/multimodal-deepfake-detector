"""Audio dataset loader for real vs. fake human voice deepfake audio.

Scans the dataset hierarchy ({USA|UK}/{female|male}/{speaker_id}/{category})
for audio files and converts them into standardized (1, 128, 128) Mel-spectrogram tensors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from deepfake_core.preprocessing.audio_extractor import AudioExtractor

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}


class DeepfakeAudioDataset(Dataset):
    """PyTorch Dataset for deepfake audio classification.

    Scans an audio dataset directory for real and synthetic speech samples,
    assigning labels:
        - 0.0 for REAL ('original')
        - 1.0 for FAKE ('synthetic_1', 'synthetic_2', 'synthetic_3', etc.)

    Converts each audio file on-the-fly into a normalized (1, 128, 128) Mel-spectrogram.

    Args:
        root_dir: Root path to the audio dataset (e.g., 'data/raw/audio_deepfake').
        extractor: Optional pre-initialized AudioExtractor instance.
    """

    LABEL_REAL: float = 0.0
    LABEL_FAKE: float = 1.0

    def __init__(
        self,
        root_dir: str | Path,
        extractor: Optional[AudioExtractor] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.extractor = extractor or AudioExtractor()
        self.samples: list[tuple[Path, float]] = []

        self._scan_dataset()

    def _scan_dataset(self) -> None:
        """Scan root_dir recursively for audio files and assign binary labels."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.root_dir}")

        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                parent_name = path.parent.name.lower()
                stem_name = path.stem.lower()

                # Determine label by parent directory or filename stem
                if parent_name == "original" or stem_name == "original":
                    self.samples.append((path, self.LABEL_REAL))
                elif parent_name.startswith("synthetic") or stem_name.startswith("synthetic"):
                    self.samples.append((path, self.LABEL_FAKE))
                else:
                    logger.debug("Skipping unrecognized audio file: %s", path)

        real_count = sum(1 for _, label in self.samples if label == self.LABEL_REAL)
        fake_count = sum(1 for _, label in self.samples if label == self.LABEL_FAKE)

        logger.info(
            "DeepfakeAudioDataset initialized from '%s': %d Real, %d Fake (%d total samples).",
            self.root_dir,
            real_count,
            fake_count,
            len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return Mel-spectrogram tensor and binary label.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
                - spectrogram: Float32 tensor of shape (1, 128, 128).
                - label: Float32 scalar tensor (0.0 for Real, 1.0 for Fake).
        """
        path, label = self.samples[idx]

        spectrogram = self.extractor.extract_mel_spectrogram(path)
        if spectrogram is None:
            # Fallback zero tensor in case of corrupted file
            logger.warning("Failed to extract spectrogram for '%s', using zero tensor.", path.name)
            spectrogram = torch.zeros((1, 128, 128), dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return spectrogram.float(), label_tensor
