"""Data loading and dataset primitives for deepfake_core."""

from deepfake_core.data.audio_dataset import DeepfakeAudioDataset
from deepfake_core.data.temporal_dataset import TemporalEmbeddingDataset

__all__ = ["TemporalEmbeddingDataset", "DeepfakeAudioDataset"]

