"""deepfake_core — Modular multimodal deepfake detection library.

Provides spatial forensic analysis (EfficientNet-B0), face extraction
(MediaPipe), and a unified detection pipeline optimized for NVIDIA
RTX 4050 (6 GB VRAM) with FP16 mixed precision.

Quick start::

    from deepfake_core import UnifiedDetector

    detector = UnifiedDetector(device="cuda")
    result = detector.predict_video_clip("video.mp4")

Public API:
    - UnifiedDetector: Main detection pipeline
    - FaceExtractor: MediaPipe face crop extraction
    - SpatialForensicClassifier: EfficientNet-B0 binary classifier
"""

__version__ = "0.1.0"
__author__ = "Guru Karthikeya"

from deepfake_core.models.spatial_efficientnet import SpatialForensicClassifier
from deepfake_core.pipeline import UnifiedDetector
from deepfake_core.preprocessing.face_extractor import FaceExtractor

__all__ = [
    "FaceExtractor",
    "SpatialForensicClassifier",
    "UnifiedDetector",
    "__version__",
]
