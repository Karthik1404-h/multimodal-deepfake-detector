"""Verification script for Multimodal Deepfake Detector.

Evaluates test video clips using the calibrated tri-modal pipeline:
- Spatial Forensics (EfficientNet-B0 hybrid scoring)
- Temporal Sequence Modeling (Bidirectional LSTM with attention)
- Audio Forensic Analysis (Mel-CNN with RMS energy voice activity gating)
- Multimodal Max-Confidence Late Fusion

Usage::

    python scripts/verify_fusion.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from deepfake_core.pipeline import UnifiedDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("verify_fusion")


def test_video(detector: UnifiedDetector, video_path: str, expected_type: str) -> dict | None:
    path = Path(video_path)
    if not path.exists():
        print(f"Skipping: {video_path} (file not found on disk)")
        return None

    print(f"\n=======================================================")
    print(f"Evaluating: {path.name} (Expected: {expected_type})")
    print(f"=======================================================")
    res = detector.predict_video_clip(str(path))

    audio_rms = res.get("audio_rms")
    audio_score = res.get("audio_score")
    if audio_score is not None:
        audio_str = f"{audio_score:.4f} (RMS: {audio_rms:.4f})"
    elif res.get("audio_gated"):
        audio_str = f"GATED (RMS: {audio_rms:.4f} < {detector.min_audio_rms:.3f})"
    else:
        audio_str = "None (No audio stream)"

    print(f"Frames Analyzed: {res.get('frames_analyzed')}")
    print(f"Audio Status:    {audio_str}")
    print(f"Spatial Score:   {res.get('spatial_score'):.4f} (Mean: {res.get('mean_frame_prob', 0.0):.4f}, Top-K: {res.get('top_k_mean_prob', 0.0):.4f})")
    print(f"Temporal Score:  {res.get('temporal_score'):.4f}")
    print(f"Video Score:     {res.get('video_score'):.4f} [30% Spatial + 70% Temporal]")
    print(f"-------------------------------------------------------")
    print(f"Final Score:     {res.get('final_score'):.4f}")
    print(f"Verdict:         {res.get('verdict')} (Confidence: {res.get('confidence'):.4f})")
    print(f"-------------------------------------------------------")

    final_score = res.get("final_score", 0.0)
    if expected_type.upper() == "FAKE":
        if final_score >= 0.65:
            print(f"--> RESULT: PASS (Final Score {final_score:.4f} >= 0.65 -> FAKE)")
        else:
            print(f"--> RESULT: CHECK (Final Score {final_score:.4f} vs 0.65 threshold -> {res.get('verdict')})")
    elif expected_type.upper() == "REAL":
        if final_score < 0.45:
            print(f"--> RESULT: PASS (Final Score {final_score:.4f} < 0.45 -> REAL)")
        else:
            print(f"--> RESULT: CHECK (Final Score {final_score:.4f} vs 0.45 threshold -> {res.get('verdict')})")

    return res


def main() -> None:
    print("Initializing UnifiedDetector on device='cuda' (with fallback to cpu)...")
    detector = UnifiedDetector(device="cuda")
    print(
        f"UnifiedDetector loaded (calibration_bias={detector.calibration_bias}, "
        f"min_audio_rms={detector.min_audio_rms})."
    )

    test_candidates = [
        ("data/raw/deepfake_videos/real_test_video.mp4", "REAL"),
        ("data/raw/dfdc_videos/fake/aagfhgtpmv.mp4", "FAKE"),
        ("data/raw/deepfake_videos/deepfake/2.mp4", "FAKE"),
        ("data/raw/deepfake_videos/deepfake/4.mp4", "FAKE"),
        ("data/raw/deepfake_videos/test_video.mp4", "FAKE"),
    ]

    for path, expected in test_candidates:
        test_video(detector, path, expected)


if __name__ == "__main__":
    main()
