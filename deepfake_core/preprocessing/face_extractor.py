"""Face extraction pipeline using MediaPipe Face Detection.

Extracts and crops faces from video frames, resizing to 224×224 RGB
images suitable for input to the spatial forensic classifier.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# Modern MediaPipe Tasks API
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)


class FaceExtractor:
    """MediaPipe-based face detector and cropper.

    Designed as a context manager to ensure proper lifecycle management
    of MediaPipe resources. Uses the modern MediaPipe Tasks API.

    Usage::

        with FaceExtractor(min_confidence=0.5) as extractor:
            face = extractor.extract_face_crop(frame)
            if face is not None:
                # face is a 224×224 RGB numpy array
                process(face)

    Args:
        min_confidence: Minimum detection confidence threshold (0.0–1.0).
        crop_size: Output face crop dimensions (square).
        padding_ratio: Fractional padding around the detected bounding box
            to include forehead/chin context. 0.3 = 30% extra on each side.
    """

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

    def __init__(
        self,
        min_confidence: float = 0.5,
        crop_size: int = 224,
        padding_ratio: float = 0.3,
    ) -> None:
        self.min_confidence = min_confidence
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        self._detector: Optional[vision.FaceDetector] = None

    def _download_model_if_needed(self) -> str:
        """Ensure the TFLite model exists in the local models directory.
        
        Returns:
            Absolute path to the downloaded TFLite model file.
        """
        # Place it in a models/ directory at the project root
        project_root = Path.cwd()
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "blaze_face_short_range.tflite"
        
        if not model_path.exists():
            logger.info("Downloading MediaPipe face detection model to %s...", model_path)
            urllib.request.urlretrieve(self.MODEL_URL, str(model_path))
            logger.info("Download complete.")
            
        return str(model_path)

    # --- Context Manager Protocol ---

    def __enter__(self) -> "FaceExtractor":
        """Initialize the MediaPipe face detector."""
        model_path = self._download_model_if_needed()
        
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        
        logger.debug(
            "FaceExtractor initialized (confidence=%.2f, crop=%d, padding=%.2f)",
            self.min_confidence,
            self.crop_size,
            self.padding_ratio,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Release MediaPipe resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            logger.debug("FaceExtractor resources released.")

    # --- Public API ---

    def extract_face_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract and crop the primary face from a video frame.

        Args:
            frame: BGR image (OpenCV format) as a numpy array.
                   Shape: (H, W, 3), dtype: uint8.

        Returns:
            A 224×224 RGB numpy array of the cropped face, or None
            if no face is detected with sufficient confidence.

        Raises:
            RuntimeError: If called outside of a context manager block.
        """
        if self._detector is None:
            raise RuntimeError(
                "FaceExtractor must be used as a context manager. "
                "Use: `with FaceExtractor() as extractor:`"
            )

        # MediaPipe expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self._detector.detect(mp_image)

        if not results.detections:
            return None

        # Take the highest-confidence detection
        detection = max(results.detections, key=lambda d: d.categories[0].score if d.categories else 0)

        if not detection.categories or detection.categories[0].score < self.min_confidence:
            return None

        return self._crop_and_resize(frame_rgb, detection)

    def extract_all_faces(self, frame: np.ndarray) -> list[np.ndarray]:
        """Extract all detected faces from a frame.

        Args:
            frame: BGR image (OpenCV format).

        Returns:
            List of 224×224 RGB face crops. Empty list if no faces found.

        Raises:
            RuntimeError: If called outside of a context manager block.
        """
        if self._detector is None:
            raise RuntimeError(
                "FaceExtractor must be used as a context manager."
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self._detector.detect(mp_image)

        if not results.detections:
            return []

        crops: list[np.ndarray] = []
        for detection in results.detections:
            if detection.categories and detection.categories[0].score >= self.min_confidence:
                crop = self._crop_and_resize(frame_rgb, detection)
                if crop is not None:
                    crops.append(crop)

        return crops

    # --- Private Helpers ---

    def _crop_and_resize(
        self,
        frame_rgb: np.ndarray,
        detection,
    ) -> Optional[np.ndarray]:
        """Crop face region with padding and resize to target dimensions.

        Args:
            frame_rgb: RGB image array.
            detection: MediaPipe detection result object.

        Returns:
            Resized face crop or None if bounding box is invalid.
        """
        h, w, _ = frame_rgb.shape
        bbox = detection.bounding_box

        # Tasks API returns absolute pixels directly
        x_min = bbox.origin_x
        y_min = bbox.origin_y
        box_w = bbox.width
        box_h = bbox.height

        # Apply padding for contextual information (forehead, chin, ears)
        pad_x = int(box_w * self.padding_ratio)
        pad_y = int(box_h * self.padding_ratio)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_min + box_w + pad_x)
        y2 = min(h, y_min + box_h + pad_y)

        # Validate crop dimensions
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid face bounding box: (%d,%d)-(%d,%d)", x1, y1, x2, y2)
            return None

        face_crop = frame_rgb[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        # Resize to target dimensions (bilinear interpolation)
        face_resized = cv2.resize(
            face_crop,
            (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_LINEAR,
        )

        return face_resized
