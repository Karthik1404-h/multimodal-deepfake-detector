"""Preprocessing sub-package for deepfake_core.

Provides face extraction and frame preprocessing utilities.
"""

from deepfake_core.preprocessing.audio_extractor import AudioExtractor
from deepfake_core.preprocessing.face_extractor import FaceExtractor

__all__ = ["FaceExtractor", "AudioExtractor"]

