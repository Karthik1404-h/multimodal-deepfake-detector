"""Face extraction pipeline using MediaPipe Face Detection.

Extracts and crops faces from video frames, resizing to 224×224 RGB
images suitable for input to the spatial forensic classifier.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class FaceExtractor:
    """MediaPipe-based face detector and cropper.

    Designed as a context manager to ensure proper lifecycle management
    of MediaPipe resources.

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

    def __init__(
        self,
        min_confidence: float = 0.5,
        crop_size: int = 224,
        padding_ratio: float = 0.3,
    ) -> None:
        self.min_confidence = min_confidence
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        self._detector: Optional[mp.solutions.face_detection.FaceDetection] = None

    # --- Context Manager Protocol ---

    def __enter__(self) -> "FaceExtractor":
        """Initialize the MediaPipe face detector."""
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range (< 2m), optimal for video crops
            min_detection_confidence=self.min_confidence,
        )
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
        results = self._detector.process(frame_rgb)

        if not results.detections:
            return None

        # Take the highest-confidence detection
        detection = max(results.detections, key=lambda d: d.score[0])

        if detection.score[0] < self.min_confidence:
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
        results = self._detector.process(frame_rgb)

        if not results.detections:
            return []

        crops: list[np.ndarray] = []
        for detection in results.detections:
            if detection.score[0] >= self.min_confidence:
                crop = self._crop_and_resize(frame_rgb, detection)
                if crop is not None:
                    crops.append(crop)

        return crops

    # --- Private Helpers ---

    def _crop_and_resize(
        self,
        frame_rgb: np.ndarray,
        detection: mp.solutions.face_detection.Detection,
    ) -> Optional[np.ndarray]:
        """Crop face region with padding and resize to target dimensions.

        Args:
            frame_rgb: RGB image array.
            detection: MediaPipe detection result.

        Returns:
            Resized face crop or None if bounding box is invalid.
        """
        h, w, _ = frame_rgb.shape
        bbox = detection.location_data.relative_bounding_box

        # Convert relative coords to absolute pixels
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

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

        # Resize to target dimensions (bilinear interpolation)
        face_resized = cv2.resize(
            face_crop,
            (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_LINEAR,
        )

        return face_resized
