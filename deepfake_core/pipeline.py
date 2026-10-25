"""Unified multimodal deepfake detection pipeline.

Orchestrates face extraction, spatial forensic analysis (EfficientNet-B0),
temporal sequence anomaly detection (Bidirectional LSTM with Attention),
and audio forensic analysis (AudioForensicCNN) into a unified inference pipeline.
Enforces FP16 mixed precision and explicit GPU cache management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from deepfake_core.models.audio_cnn import AudioForensicCNN
from deepfake_core.models.spatial_efficientnet import SpatialForensicClassifier
from deepfake_core.models.temporal_lstm import TemporalAnomalyLSTM
from deepfake_core.preprocessing.audio_extractor import AudioExtractor
from deepfake_core.preprocessing.face_extractor import FaceExtractor
from deepfake_core.utils.vram_manager import clear_gpu_cache, get_vram_usage

logger = logging.getLogger(__name__)


class UnifiedDetector:
    """Main entry point for 3-way multimodal deepfake detection.

    Fuses:
      1. Spatial forensic analysis (EfficientNet-B0 feature extractor & classification head)
      2. Temporal sequence anomaly analysis (Bidirectional LSTM with temporal attention)
      3. Audio spectrogram forensic analysis (AudioForensicCNN)

    Designed for low-VRAM deployment on NVIDIA RTX 4050 (6 GB).

    Usage::

        detector = UnifiedDetector(device="cuda")
        result = detector.analyze_video("suspect_video.mp4")
        print(result)

    Args:
        device: Inference device ('cuda' or 'cpu'). Falls back to CPU
            if CUDA is unavailable.
        model_path: Path to trained spatial model checkpoint (.pt or .pth).
            Defaults to 'models/checkpoints/best_spatial_model.pth'.
        temporal_model_path: Path to trained temporal LSTM checkpoint (.pt or .pth).
            Defaults to 'models/checkpoints/best_temporal_model.pth'.
        audio_model_path: Path to trained audio model checkpoint (.pt or .pth).
            Defaults to 'models/checkpoints/best_audio_model.pth'.
        threshold: Classification threshold for FAKE (default: 0.65).
        uncertain_threshold: Classification threshold for UNCERTAIN (default: 0.45).
        max_frames: Maximum number of frames to sample from a video clip.
        face_confidence: Minimum confidence for face detection.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = "models/checkpoints/best_spatial_model.pth",
        temporal_model_path: Optional[str] = "models/checkpoints/best_temporal_model.pth",
        audio_model_path: Optional[str] = "models/checkpoints/best_audio_model.pth",
        threshold: float = 0.65,
        uncertain_threshold: float = 0.45,
        max_frames: int = 30,
        face_confidence: float = 0.35,
        calibration_bias: float = 0.10,
        min_audio_rms: float = 0.002,
        enable_audio: bool = False,
    ) -> None:
        # Resolve device with CUDA fallback
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        self.threshold = threshold
        self.uncertain_threshold = uncertain_threshold
        self.max_frames = max_frames
        self.face_confidence = face_confidence
        self.calibration_bias = calibration_bias
        self.min_audio_rms = min_audio_rms
        self.enable_audio = enable_audio

        # --- 1. Load Spatial Forensic Classifier ---
        self.spatial_model = SpatialForensicClassifier(pretrained=True, freeze_backbone=True)
        if model_path is not None and Path(model_path).exists():
            self._load_spatial_checkpoint(model_path)
        elif model_path is not None and not Path(model_path).exists():
            logger.warning("Spatial model checkpoint not found: %s", model_path)
        self.spatial_model.to(self.device)
        self.spatial_model.eval()

        # --- 2. Load Temporal Anomaly LSTM ---
        self.temporal_model = TemporalAnomalyLSTM()
        if temporal_model_path is not None and Path(temporal_model_path).exists():
            self._load_temporal_checkpoint(temporal_model_path)
        elif temporal_model_path is not None and not Path(temporal_model_path).exists():
            logger.warning("Temporal model checkpoint not found: %s", temporal_model_path)
        self.temporal_model.to(self.device)
        self.temporal_model.eval()

        # --- 3. Load Audio Forensic Components ---
        self.audio_extractor = AudioExtractor()
        self.audio_model = AudioForensicCNN()
        if audio_model_path is not None and Path(audio_model_path).exists():
            self._load_audio_checkpoint(audio_model_path)
        elif audio_model_path is not None and not Path(audio_model_path).exists():
            logger.warning("Audio model checkpoint not found: %s", audio_model_path)
        self.audio_model.to(self.device)
        self.audio_model.eval()

        # --- ImageNet normalization matching scripts/extract_embeddings.py ---
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # PIL Image -> CHW float [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(
            "UnifiedDetector initialized on %s (fake_thresh=%.2f, uncertain_thresh=%.2f, max_frames=%d)",
            self.device,
            self.threshold,
            self.uncertain_threshold,
            self.max_frames,
        )

    def predict_video_clip(
        self,
        video_path: str,
        calibration_bias: Optional[float] = None,
    ) -> dict:
        """Run full 3-way multimodal deepfake detection pipeline on a video clip.

        Pipeline steps:
            1. Sample frames uniformly from the video.
            2. Extract face crops via MediaPipe FaceExtractor.
            3. Run SpatialForensicClassifier to extract (N, 1280) frame embeddings.
            4. Compute per-frame spatial probabilities and calculate weighted hybrid
               spatial score: (0.60 * mean_frame_prob) + (0.40 * top_k_mean_prob).
            5. Pass embeddings sequence (1, seq_len, 1280) through TemporalAnomalyLSTM
               with return_attention=True to compute temporal_score and attention weights.
            6. Combine video score: 0.30 * spatial_score + 0.70 * temporal_score.
            7. Extract audio waveform, check RMS voice activity energy (>= min_audio_rms),
               and evaluate Mel-spectrogram with AudioForensicCNN.
            8. Multimodal late fusion:
               - If active speech audio exists: final_score = max(video_score, audio_score)
               - If silent/ambient noise or no audio exists: final_score = video_score
            9. Determine tri-state verdict (FAKE >= 0.65, UNCERTAIN 0.45-0.65, REAL < 0.45).
            10. Clear GPU cache and return structured payload.

        Args:
            video_path: Path to the video file (MP4, AVI, MKV, etc.).
            calibration_bias: Optional calibration bias penalty applied to raw spatial
                probabilities to counteract compression domain shift. Defaults to self.calibration_bias.

        Returns:
            Dict containing:
                - verdict: "FAKE", "UNCERTAIN", or "REAL"
                - is_fake: True for FAKE, False for REAL, None for UNCERTAIN
                - final_score: Fused final probability score [0, 1]
                - confidence: Confidence score [0, 1]
                - calibration_bias: Effective calibration penalty applied
                - video_score: Combined spatial and temporal score [0, 1]
                - spatial_score: Weighted hybrid spatial frame score [0, 1]
                - temporal_score: Sequence-level temporal anomaly score [0, 1]
                - audio_score: Audio forensic score [0, 1] or None if no audio
                - has_audio: True if audio track was present and analyzed
                - temporal_attention_weights: List of frame attention weights [float]
                - mean_frame_prob: Mean spatial probability across all frames
                - top_k_mean_prob: Mean spatial probability across top 20% worst frames
                - frames_analyzed: Number of frames with detected faces

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

            # Step 2: Extract face crops with multi-scale distant subject support
            face_crops, is_fallback = self._extract_faces(frames)
            logger.info("Extracted %d face crops (contextual fallback: %s)", len(face_crops), is_fallback)

            # Step 3: Run spatial & temporal video pipeline with calibration bias
            effective_bias = self.calibration_bias if calibration_bias is None else calibration_bias
            raw_fake_probs, temporal_score, temporal_attention_weights = self._run_video_inference(
                face_crops, calibration_bias=effective_bias
            )

            # Step 4: Hybrid spatial scoring (Mean + Top-K 20%)
            mean_frame_prob = float(np.mean(raw_fake_probs))
            k = max(1, int(len(raw_fake_probs) * 0.20))
            top_k_mean_prob = float(np.mean(np.sort(raw_fake_probs)[-k:]))
            spatial_score = (0.60 * mean_frame_prob) + (0.40 * top_k_mean_prob)

            # Step 5: Combine video score (40% spatial + 60% temporal)
            # Balanced fusion between spatial artifact classification and temporal dynamics modeling
            video_score = (0.40 * spatial_score) + (0.60 * temporal_score)

            # Step 6: Audio forensic analysis with RMS voice activity gating
            audio_score, audio_rms = self._run_audio_inference(str(video_path))
            has_audio = (audio_score is not None)
            audio_gated = (audio_rms is not None and audio_score is None)

            # Step 7: Multimodal Late Fusion
            # When audio is active, only fuse if audio has high confidence (> threshold) or if enable_audio is True.
            # Neutral audio (~0.50-0.58) from underfit models is prevented from artificially inflating real video scores.
            if has_audio and audio_score is not None:
                if audio_score >= self.threshold:
                    # High audio deepfake confidence triggers override (voice clone deepfake detection)
                    final_score = max(float(video_score), float(audio_score))
                elif self.enable_audio:
                    if abs(audio_score - 0.50) < 0.12:
                        # Audio score is neutral (~0.50-0.54) with no discriminatory confidence: rely on visual biometric evidence
                        final_score = float(video_score)
                    else:
                        # Weighted combination when audio has moderate confidence
                        final_score = (0.75 * float(video_score)) + (0.25 * float(audio_score))
                else:
                    final_score = float(video_score)
            else:
                final_score = float(video_score)

            # Step 8: Tri-state classification
            # >= 0.65: FAKE
            # >= 0.45 and < 0.65: UNCERTAIN
            # < 0.45: REAL
            if final_score >= self.threshold:
                verdict = "FAKE"
                is_fake: bool | None = True
                confidence = final_score
            elif final_score >= self.uncertain_threshold:
                verdict = "UNCERTAIN"
                is_fake = None
                confidence = final_score
            else:
                verdict = "REAL"
                is_fake = False
                confidence = 1.0 - final_score

            result = {
                "verdict": verdict,
                "is_fake": is_fake,
                "final_score": round(float(final_score), 4),
                "confidence": round(float(confidence), 4),
                "calibration_bias": round(float(effective_bias), 4),
                "video_score": round(float(video_score), 4),
                "spatial_score": round(float(spatial_score), 4),
                "temporal_score": round(float(temporal_score), 4),
                "audio_score": round(float(audio_score), 4) if has_audio else None,
                "audio_rms": round(float(audio_rms), 4) if audio_rms is not None else None,
                "has_audio": has_audio,
                "audio_gated": audio_gated,
                "temporal_attention_weights": temporal_attention_weights,
                "mean_frame_prob": round(float(mean_frame_prob), 4),
                "top_k_mean_prob": round(float(top_k_mean_prob), 4),
                "frames_analyzed": len(face_crops),
                "face_fallback_used": is_fallback,
            }

            logger.info(
                "Prediction: %s (final=%.4f, video=%.4f [spatial=%.4f, temporal=%.4f], audio=%s, audio_rms=%s)",
                verdict,
                final_score,
                video_score,
                spatial_score,
                temporal_score,
                f"{audio_score:.4f}" if has_audio else ("GATED" if audio_gated else "N/A"),
                f"{audio_rms:.4f}" if audio_rms is not None else "N/A",
            )
            return result

        finally:
            # Step 9: Always clear GPU cache after inference
            clear_gpu_cache()
            vram = get_vram_usage()
            if vram is not None:
                logger.debug("Post-inference VRAM: %.2f MB", vram)

    # Alias for analyze_video
    analyze_video = predict_video_clip

    # --- Private Helpers ---

    def _load_spatial_checkpoint(self, model_path: str) -> None:
        """Load spatial model weights from a checkpoint file.

        Args:
            model_path: Path to the .pt or .pth checkpoint file.
        """
        path = Path(model_path)
        if not path.exists():
            logger.warning("Spatial model checkpoint not found: %s", path)
            return

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.spatial_model.load_state_dict(state_dict)
        logger.info("Loaded spatial model checkpoint from '%s'", path.name)

    # Backward compatibility alias
    _load_checkpoint = _load_spatial_checkpoint

    def _load_temporal_checkpoint(self, model_path: str) -> None:
        """Load temporal LSTM weights from a checkpoint file.

        Args:
            model_path: Path to the .pt or .pth checkpoint file.
        """
        path = Path(model_path)
        if not path.exists():
            logger.warning("Temporal model checkpoint not found: %s", path)
            return

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.temporal_model.load_state_dict(state_dict)
        logger.info("Loaded temporal model checkpoint from '%s'", path.name)

    def _load_audio_checkpoint(self, model_path: str) -> None:
        """Load audio CNN weights from a checkpoint file.

        Args:
            model_path: Path to the .pt or .pth checkpoint file.
        """
        path = Path(model_path)
        if not path.exists():
            logger.warning("Audio model checkpoint not found: %s", path)
            return

        checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.audio_model.load_state_dict(state_dict)
        logger.info("Loaded audio model checkpoint from '%s'", path.name)

    def _run_video_inference(
        self,
        face_crops: list[np.ndarray],
        calibration_bias: float = 0.10,
    ) -> tuple[np.ndarray, float, list[float]]:
        """Run spatial feature extraction, spatial frame scoring, and temporal LSTM inference.

        Args:
            face_crops: List of 224x224 RGB numpy arrays.
            calibration_bias: Calibration bias penalty (default: 0.10) subtracted from raw
                spatial probabilities to counteract H.264 compression domain shift.

        Returns:
            Tuple of:
                - raw_fake_probs: Calibrated numpy array of per-frame spatial probabilities (N,).
                - temporal_score: Probability score [0, 1] from TemporalAnomalyLSTM.
                - temporal_attention_weights: List of normalized attention weights per frame.
        """
        # Preprocess: convert RGB face crops to PIL Images and apply exact evaluation transforms
        tensors = [
            self._transform(Image.fromarray(crop) if isinstance(crop, np.ndarray) else crop)
            for crop in face_crops
        ]
        batch = torch.stack(tensors).to(self.device)

        use_amp = (self.device.type == "cuda")

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    # 1. Extract 1280-dim frame embeddings
                    embeddings = self.spatial_model.extract_features(batch)  # (N, 1280)
                    # 2. Compute per-frame spatial classification logits
                    spatial_logits = self.spatial_model.classifier(embeddings).squeeze(-1)  # (N,)
            else:
                embeddings = self.spatial_model.extract_features(batch)
                spatial_logits = self.spatial_model.classifier(embeddings).squeeze(-1)

            # Ensure embeddings are float32 for LSTM numerical stability
            embeddings = embeddings.float()

            # 3. Robust Sequence Alignment (uniform subsampling or edge-padding + masking)
            seq_len = embeddings.shape[0]
            max_seq_len = 30

            if seq_len > max_seq_len:
                # Uniformly sample max_seq_len frames across the entire clip
                indices = torch.linspace(0, seq_len - 1, max_seq_len, device=self.device).long()
                aligned_embeddings = embeddings[indices]
                mask = None
                valid_frames = max_seq_len
            elif seq_len < max_seq_len:
                # Edge-padding: duplicate last frame to prevent synthetic zero-vector shock
                pad_len = max_seq_len - seq_len
                padding = embeddings[-1:].repeat(pad_len, 1)
                aligned_embeddings = torch.cat([embeddings, padding], dim=0)
                # Boolean attention mask: True for authentic frames, False for padded
                mask = torch.zeros((1, max_seq_len), dtype=torch.bool, device=self.device)
                mask[0, :seq_len] = True
                valid_frames = seq_len
            else:
                aligned_embeddings = embeddings
                mask = None
                valid_frames = max_seq_len

            seq_embeddings = aligned_embeddings.unsqueeze(0)  # (1, 30, 1280)
            temp_logits, attn_weights = self.temporal_model(seq_embeddings, mask=mask, return_attention=True)

            raw_fake_probs = torch.sigmoid(spatial_logits).cpu().numpy().ravel()

            # Apply calibration bias to counter domain shift from video compression
            if calibration_bias > 0.0:
                raw_fake_probs = np.clip(raw_fake_probs - calibration_bias, 0.0, 1.0)

            temporal_score = float(torch.sigmoid(temp_logits).item())
            # Return attention distribution normalized across valid analyzed frames
            valid_attn = attn_weights.squeeze(0)[:valid_frames].cpu().numpy()
            sum_attn = float(np.sum(valid_attn))
            if sum_attn > 0:
                normalized_attn = (valid_attn / sum_attn).tolist()
            else:
                normalized_attn = valid_attn.tolist()

            temporal_attention_weights = [round(float(w), 4) for w in normalized_attn]

        return raw_fake_probs, temporal_score, temporal_attention_weights

    def _run_spatial_inference(self, face_crops: list[np.ndarray]) -> np.ndarray:
        """Run spatial forensic classifier on face crops with FP16 autocast.

        Args:
            face_crops: List of 224x224 RGB numpy arrays.

        Returns:
            Numpy array of raw fake probabilities for each crop (0.0 = REAL, 1.0 = FAKE).
        """
        raw_fake_probs, _, _ = self._run_video_inference(face_crops)
        return raw_fake_probs

    def _run_audio_inference(self, video_path: str) -> tuple[Optional[float], Optional[float]]:
        """Extract waveform, compute RMS energy for voice activity gating, and infer audio deepfake probability.

        Args:
            video_path: Path to the input video.

        Returns:
            Tuple of (audio_score, audio_rms).
            - If no audio stream exists: (None, None).
            - If audio RMS energy is below self.min_audio_rms (ambient noise/silence): (None, rms).
            - If active speech audio exists: (probability [0.0, 1.0], rms).
        """
        waveform = self.audio_extractor.extract_waveform(video_path)
        if waveform is None or waveform.numel() == 0:
            return None, None

        # Compute RMS energy of the audio waveform
        rms = float(torch.sqrt(torch.mean(waveform.float() ** 2)).item())

        # Gating: ignore audio if energy is below speech/voice activity threshold
        if rms < self.min_audio_rms:
            logger.info(
                "Audio track in '%s' is below voice activity threshold (RMS: %.4f < %.4f). Bypassing audio modality.",
                Path(video_path).name,
                rms,
                self.min_audio_rms,
            )
            return None, rms

        # Convert active speech waveform into standardized Mel-spectrogram
        spec = self.audio_extractor.waveform_to_mel_spectrogram(waveform)

        # spec shape: (1, 128, 128) -> add batch dim -> (1, 1, 128, 128)
        spec_batch = spec.unsqueeze(0).to(self.device)
        use_amp = (self.device.type == "cuda")

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logit = self.audio_model(spec_batch)
            else:
                logit = self.audio_model(spec_batch)

            prob = torch.sigmoid(logit).item()

        return float(prob), rms

    def _sample_frames(self, video_path: str) -> list[np.ndarray]:
        """Sample frames uniformly from a video file and convert immediately from BGR to RGB.

        Args:
            video_path: Path to the video file.

        Returns:
            List of RGB frames as numpy arrays.
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
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                return frames

            # Uniform sampling
            sample_count = min(self.max_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            return frames

        finally:
            cap.release()

    def _extract_faces(self, frames: list[np.ndarray]) -> tuple[list[np.ndarray], bool]:
        """Extract face crops from a list of RGB frames with multi-tier distant subject detection.

        Args:
            frames: List of RGB frames.

        Returns:
            Tuple of:
                - List of 224x224 RGB face crops.
                - Boolean flag indicating whether contextual fallback was used.
        """
        face_crops: list[Optional[np.ndarray]] = [None] * len(frames)
        last_valid_crop: Optional[np.ndarray] = None

        # Tier 1: Primary face extraction at user-configured confidence with full-range detector
        with FaceExtractor(min_confidence=self.face_confidence, model_type="full_range") as extractor:
            for idx, frame in enumerate(frames):
                crop = extractor.extract_face_crop(frame, is_bgr=False)
                if crop is not None:
                    face_crops[idx] = crop
                    last_valid_crop = crop

        valid_count = sum(1 for c in face_crops if c is not None)

        # Tier 2: Adaptive relaxed confidence (0.08) for distant / wide-angle subjects
        if valid_count < max(5, len(frames) // 4):
            relaxed_conf = min(0.12, max(0.06, self.face_confidence * 0.35))
            logger.info("Retrying face detection with high-sensitivity full-range (%.2f) for distant subjects...", relaxed_conf)
            with FaceExtractor(min_confidence=relaxed_conf, model_type="full_range") as extractor:
                for idx, frame in enumerate(frames):
                    if face_crops[idx] is None:
                        crop = extractor.extract_face_crop(frame, is_bgr=False)
                        if crop is not None:
                            face_crops[idx] = crop
                            if last_valid_crop is None:
                                last_valid_crop = crop

        valid_count = sum(1 for c in face_crops if c is not None)

        # Tier 3: Multi-Scale Zoom (zoom into upper-center 65% region to magnify distant faces)
        if valid_count < max(5, len(frames) // 4):
            logger.info("Attempting multi-scale zoom detection for distant/wide-shot video...")
            with FaceExtractor(min_confidence=0.08, model_type="full_range") as extractor:
                for idx, frame in enumerate(frames):
                    if face_crops[idx] is None:
                        h, w, _ = frame.shape
                        y1, y2 = int(0.02 * h), int(0.72 * h)
                        x1, x2 = int(0.15 * w), int(0.85 * w)
                        zoomed_roi = frame[y1:y2, x1:x2]
                        crop = extractor.extract_face_crop(zoomed_roi, is_bgr=False)
                        if crop is not None:
                            face_crops[idx] = crop
                            if last_valid_crop is None:
                                last_valid_crop = crop

        # Spatiotemporal sequence completion: forward-fill and back-fill across temporary occlusions
        first_valid_crop: Optional[np.ndarray] = next((c for c in face_crops if c is not None), None)
        if first_valid_crop is not None:
            current_crop = first_valid_crop
            resolved_crops: list[np.ndarray] = []
            for c in face_crops:
                if c is not None:
                    current_crop = c
                resolved_crops.append(current_crop)
            return resolved_crops, False

        # If no face detected across all tiers, raise descriptive error
        raise RuntimeError(
            f"No faces detected in any of {len(frames)} sampled frames. "
            "VerifAI requires a visible human subject. If the subject is distant or partially obscured, "
            "ensure they are facing the camera."
        )

