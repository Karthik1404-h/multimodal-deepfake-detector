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
        self.cached_spectrograms: list[torch.Tensor] = []
        self.labels: list[float] = []

        self._scan_dataset()

    def _scan_dataset(self) -> None:
        """Scan root_dir recursively for audio files and extract 4-second Mel-spectrogram segments."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.root_dir}")

        audio_files: list[tuple[Path, float]] = []
        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                parent_name = path.parent.name.lower()
                stem_name = path.stem.lower()

                # Determine label by parent directory or filename stem
                if parent_name == "original" or stem_name == "original":
                    audio_files.append((path, self.LABEL_REAL))
                elif parent_name.startswith("synthetic") or stem_name.startswith("synthetic"):
                    audio_files.append((path, self.LABEL_FAKE))
                else:
                    logger.debug("Skipping unrecognized audio file: %s", path)

        chunk_len = 64000  # 4 seconds at 16kHz
        stride = 32000     # 2 seconds stride
        self.cached_spectrograms = []
        self.labels = []
        self.samples = []

        for path, label in audio_files:
            try:
                wf = self.extractor.extract_waveform(path)
                if wf is None or wf.numel() < 16000:
                    continue

                if wf.numel() <= chunk_len:
                    spec = self.extractor.waveform_to_mel_spectrogram(wf)
                    self.cached_spectrograms.append(spec)
                    self.labels.append(label)
                    self.samples.append((path, label))
                else:
                    for start in range(0, wf.numel() - chunk_len + 1, stride):
                        chunk = wf[start : start + chunk_len]
                        spec = self.extractor.waveform_to_mel_spectrogram(chunk)
                        self.cached_spectrograms.append(spec)
                        self.labels.append(label)
                        self.samples.append((path, label))
            except Exception as e:
                logger.warning("Error processing %s: %s", path, e)

        real_count = sum(1 for l in self.labels if l == self.LABEL_REAL)
        fake_count = sum(1 for l in self.labels if l == self.LABEL_FAKE)

        logger.info(
            "DeepfakeAudioDataset initialized from '%s': %d Real, %d Fake (%d total 4-second segments).",
            self.root_dir,
            real_count,
            fake_count,
            len(self.labels),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return Mel-spectrogram tensor and binary label.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
                - spectrogram: Float32 tensor of shape (1, 128, 128).
                - label: Float32 scalar tensor (0.0 for Real, 1.0 for Fake).
        """
        spectrogram = self.cached_spectrograms[idx]
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return spectrogram.float(), label_tensor
