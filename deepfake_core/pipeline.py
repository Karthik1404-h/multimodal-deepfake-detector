"""Unified deepfake detection pipeline.

Orchestrates face extraction, spatial forensic analysis, and (future)
lipsync/temporal analysis into a single inference call. Enforces FP16
mixed precision and explicit GPU cache management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from deepfake_core.models.spatial_efficientnet import SpatialForensicClassifier
from deepfake_core.preprocessing.face_extractor import FaceExtractor
from deepfake_core.utils.vram_manager import clear_gpu_cache, get_vram_usage

logger = logging.getLogger(__name__)


class UnifiedDetector:
    """Main entry point for multimodal deepfake detection.

    Combines spatial (EfficientNet-B0) analysis with a placeholder for
    future lipsync and temporal modules. Designed for low-VRAM deployment
    on NVIDIA RTX 4050 (6 GB).

    Usage::

        detector = UnifiedDetector(device="cuda")
        result = detector.predict_video_clip("suspect_video.mp4")
        print(result)
        # {"is_fake": True, "confidence": 0.92, "spatial_score": 0.92, "lipsync_score": 0.0}

    Args:
        device: Inference device ('cuda' or 'cpu'). Falls back to CPU
            if CUDA is unavailable.
        model_path: Path to a trained spatial model checkpoint (.pt).
            If None, uses an untrained model (for development/testing).
        threshold: Classification threshold. Scores above this are FAKE.
        max_frames: Maximum number of frames to sample from a video clip.
        face_confidence: Minimum confidence for face detection.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        max_frames: int = 30,
        face_confidence: float = 0.5,
    ) -> None:
        # Resolve device with CUDA fallback
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        self.threshold = threshold
        self.max_frames = max_frames
        self.face_confidence = face_confidence

        # --- Load spatial model ---
        self.spatial_model = SpatialForensicClassifier(pretrained=True, freeze_backbone=True)

        if model_path is not None:
            self._load_checkpoint(model_path)

        self.spatial_model.to(self.device)
        self.spatial_model.eval()

        # --- ImageNet normalization for EfficientNet ---
        self._transform = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 → CHW float [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(
            "UnifiedDetector initialized on %s (threshold=%.2f, max_frames=%d)",
            self.device,
            self.threshold,
            self.max_frames,
        )

    def predict_video_clip(self, video_path: str) -> dict:
        """Run full deepfake detection pipeline on a video clip.

        Pipeline steps:
            1. Sample frames uniformly from the video
            2. Extract face crops via MediaPipe
            3. Run spatial forensic classifier (FP16 autocast)
            4. Aggregate per-frame scores
            5. Clear GPU cache

        Args:
            video_path: Path to the video file (MP4, AVI, etc.).

        Returns:
            Dict with keys:
                - ``is_fake`` (bool): Final classification
                - ``confidence`` (float): Aggregated confidence score [0, 1]
                - ``spatial_score`` (float): Mean spatial classifier output
                - ``lipsync_score`` (float): Placeholder (always 0.0 for now)
                - ``frames_analyzed`` (int): Number of frames with detected faces

        Raises:
            FileNotFoundError: If video_path does not exist.
            RuntimeError: If video cannot be opened or no faces are detected.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            # Step 1: Sample frames
            frames = self._sample_frames(str(video_path))
            if not frames:
                raise RuntimeError(f"Could not read any frames from: {video_path}")

            logger.info("Sampled %d frames from '%s'", len(frames), video_path.name)

            # Step 2: Extract face crops
            face_crops = self._extract_faces(frames)
            if not face_crops:
                raise RuntimeError(
                    f"No faces detected in any of {len(frames)} sampled frames. "
                    f"Try lowering face_confidence (current: {self.face_confidence})"
                )

            logger.info("Extracted %d face crops", len(face_crops))

            # Step 3 & 4: Spatial inference with FP16
            spatial_score = self._run_spatial_inference(face_crops)

            # Lipsync placeholder (future module)
            lipsync_score = 0.0

            # Combined confidence (spatial only for now)
            confidence = spatial_score
            is_fake = confidence >= self.threshold

            result = {
                "is_fake": bool(is_fake),
                "confidence": round(float(confidence), 4),
                "spatial_score": round(float(spatial_score), 4),
                "lipsync_score": round(float(lipsync_score), 4),
                "frames_analyzed": len(face_crops),
            }

            logger.info("Prediction: %s (confidence=%.4f)", "FAKE" if is_fake else "REAL", confidence)
            return result

        finally:
            # Step 5: Always clear GPU cache after inference
            clear_gpu_cache()
            vram = get_vram_usage()
            if vram is not None:
                logger.debug("Post-inference VRAM: %.2f MB", vram)

    # --- Private Helpers ---

    def _load_checkpoint(self, model_path: str) -> None:
        """Load model weights from a checkpoint file.

        Args:
            model_path: Path to the .pt checkpoint file.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)

        # Support both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.spatial_model.load_state_dict(state_dict)
        logger.info("Loaded model checkpoint from '%s'", path.name)

    def _sample_frames(self, video_path: str) -> list[np.ndarray]:
        """Sample frames uniformly from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            List of BGR frames as numpy arrays.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # Fallback: read frames sequentially
                frames: list[np.ndarray] = []
                while len(frames) < self.max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                return frames

            # Uniform sampling
            sample_count = min(self.max_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            return frames

        finally:
            cap.release()

    def _extract_faces(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Extract face crops from a list of frames.

        Args:
            frames: List of BGR frames.

        Returns:
            List of 224×224 RGB face crops.
        """
        face_crops: list[np.ndarray] = []

        with FaceExtractor(min_confidence=self.face_confidence) as extractor:
            for frame in frames:
                crop = extractor.extract_face_crop(frame)
                if crop is not None:
                    face_crops.append(crop)

        return face_crops

    def _run_spatial_inference(self, face_crops: list[np.ndarray]) -> float:
        """Run spatial forensic classifier on face crops with FP16 autocast.

        Args:
            face_crops: List of 224×224 RGB numpy arrays.

        Returns:
            Mean probability score across all crops (0.0 = REAL, 1.0 = FAKE).
        """
        # Preprocess: stack into batch tensor
        tensors = [self._transform(crop) for crop in face_crops]
        batch = torch.stack(tensors).to(self.device)  # (B, 3, 224, 224)

        # Inference with mixed precision
        use_amp = self.device.type == "cuda"

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    probs = self.spatial_model(batch)  # (B, 1)
            else:
                probs = self.spatial_model(batch)

        # Aggregate: mean probability across all frames
        mean_score = probs.mean().item()
        return mean_score
