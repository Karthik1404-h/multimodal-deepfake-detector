"""Audio extraction and Mel-spectrogram preprocessing for deepfake audio forensics.

Extracts the audio track from video files (.mp4, .mov, .avi, etc.), handles videos
without audio gracefully, and converts audio waveforms into standardized (1, 128, 128)
normalized Mel-spectrogram tensors ready for neural network batching.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def build_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """Construct a triangular Mel filterbank matrix in PyTorch.

    Args:
        sr: Audio sample rate in Hz.
        n_fft: FFT window size.
        n_mels: Number of Mel frequency bands.
        f_min: Lowest frequency in Hz.
        f_max: Highest frequency in Hz (defaults to sr / 2).

    Returns:
        Tensor of shape (n_mels, n_fft // 2 + 1).
    """
    if f_max is None:
        f_max = sr / 2.0

    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_min = hz_to_mel(f_min)
    m_max = hz_to_mel(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        if f_m > f_m_minus:
            filterbank[m - 1, f_m_minus:f_m] = (
                np.arange(f_m_minus, f_m) - f_m_minus
            ) / (f_m - f_m_minus)
        if f_m_plus > f_m:
            filterbank[m - 1, f_m:f_m_plus] = (
                f_m_plus - np.arange(f_m, f_m_plus)
            ) / (f_m_plus - f_m)

    return torch.from_numpy(filterbank)


class AudioExtractor:
    """Extracts and converts video audio into standardized Mel-spectrogram tensors.

    Supports extracting audio tracks via FFmpeg or native libraries, computing
    Mel-spectrograms, and standardizing tensors to (1, 128, 128) shape.

    Args:
        sample_rate: Target audio sampling rate in Hz (default: 16000).
        n_fft: FFT window size (default: 1024).
        hop_length: Number of audio samples between adjacent STFT columns (default: 512).
        n_mels: Number of Mel frequency bands (default: 128).
        target_size: Output spatial dimension (height, width) of spectrogram (default: (128, 128)).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        target_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_size = target_size

        # Precompute Mel filterbank matrix and Hann window
        self.mel_filterbank = build_mel_filterbank(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        self.window = torch.hann_window(n_fft)

        # Check for FFmpeg availability
        self.ffmpeg_path = shutil.which("ffmpeg")

    def has_audio_stream(self, video_path: str | Path) -> bool:
        """Check if a video file contains an audio stream.

        Args:
            video_path: Path to the input video.

        Returns:
            True if an audio stream is detected, False otherwise.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return False

        ffprobe_path = shutil.which("ffprobe")
        if not ffprobe_path:
            # Fall back to attempting extraction
            return True

        cmd = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            return bool(res.stdout.strip())
        except Exception as exc:
            logger.debug("ffprobe stream check error on %s: %s", video_path, exc)
            return True

    def extract_waveform(self, video_path: str | Path) -> Optional[torch.Tensor]:
        """Extract mono audio waveform normalized to [-1.0, 1.0] at 16000 Hz.

        Args:
            video_path: Path to the video file.

        Returns:
            1D float32 Tensor of audio samples, or None if no audio track exists.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning("Video file not found: %s", video_path)
            return None

        if not self.ffmpeg_path:
            logger.warning("FFmpeg executable not found on system PATH.")
            return None

        # Extract 16kHz mono 16-bit PCM WAV directly to pipe
        cmd = [
            self.ffmpeg_path,
            "-v", "error",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "wav",
            "pipe:1",
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res.returncode != 0 or len(res.stdout) <= 44:
                # No audio track or invalid stream
                logger.debug("No valid audio track extracted from: %s", video_path.name)
                return None

            with wave.open(io.BytesIO(res.stdout)) as wav:
                num_frames = wav.getnframes()
                if num_frames == 0:
                    return None
                raw_bytes = wav.readframes(num_frames)

            # Convert 16-bit PCM buffer to float32 tensor in [-1.0, 1.0]
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            waveform = torch.from_numpy(audio_np.copy())
            return waveform

        except subprocess.TimeoutExpired:
            logger.warning("Audio extraction timed out on: %s", video_path.name)
            return None
        except Exception as exc:
            logger.warning("Failed to extract audio from %s: %s", video_path.name, exc)
            return None

    def waveform_to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert a 1D audio waveform into a standardized (1, 128, 128) Mel-spectrogram.

        Args:
            waveform: 1D audio tensor of shape (num_samples,).

        Returns:
            Tensor of shape (1, 128, 128) containing normalized log Mel-spectrogram.
        """
        # Ensure minimum length for STFT
        if waveform.numel() < self.n_fft:
            pad_len = self.n_fft - waveform.numel()
            waveform = F.pad(waveform, (0, pad_len))

        # Compute STFT magnitude
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = stft.abs()  # (n_fft // 2 + 1, time_steps)

        # Map to Mel scale: (n_mels, time_steps)
        mel_spec = torch.matmul(self.mel_filterbank.to(magnitude.device), magnitude)

        # Log Mel compression
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))

        # Z-score Normalization
        norm_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        # Reshape to (1, 1, n_mels, time_steps) for spatial interpolation
        spec_4d = norm_mel.unsqueeze(0).unsqueeze(0)

        # Interpolate/pad to exact target size (e.g. 128, 128)
        spec_resized = F.interpolate(
            spec_4d,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Return (1, target_size[0], target_size[1]) -> (1, 128, 128)
        return spec_resized.squeeze(0)

    def extract_mel_spectrogram(self, video_path: str | Path) -> Optional[torch.Tensor]:
        """End-to-end extraction: video path -> standardized Mel-spectrogram.

        Args:
            video_path: Path to the input video file.

        Returns:
            Tensor of shape (1, 128, 128) if audio is present, or None if no audio track exists.
        """
        waveform = self.extract_waveform(video_path)
        if waveform is None or waveform.numel() == 0:
            return None

        return self.waveform_to_mel_spectrogram(waveform)
