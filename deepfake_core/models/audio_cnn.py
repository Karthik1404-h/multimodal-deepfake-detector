"""Audio forensic classifier using a 2D Convolutional Neural Network.

Processes (1, 128, 128) Mel-spectrogram representations of audio tracks to detect
synthetic speech, voice cloning, audio deepfakes, and vocoder artifacts.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AudioForensicCNN(nn.Module):
    """2D CNN classifier for Mel-spectrogram deepfake audio forensics.

    Architecture::

        Input Mel-Spectrogram (B, 1, 128, 128)
                    ↓
        Conv2d(1, 32) → BatchNorm → ReLU → MaxPool(2)   -> (B, 32, 64, 64)
                    ↓
        Conv2d(32, 64) → BatchNorm → ReLU → MaxPool(2)  -> (B, 64, 32, 32)
                    ↓
        Conv2d(64, 128) → BatchNorm → ReLU → MaxPool(2) -> (B, 128, 16, 16)
                    ↓
        Conv2d(128, 256) → BatchNorm → ReLU → MaxPool(2) -> (B, 256, 8, 8)
                    ↓
        AdaptiveAvgPool2d(1, 1) → Flatten               -> (B, 256)
                    ↓
        Dropout(0.3) → Linear(256, 64) → ReLU
                    ↓
        Dropout(0.2) → Linear(64, 1)
                    ↓
        Binary Logit (B,)

    Args:
        in_channels: Number of input spectrogram channels (default: 1).
        base_channels: Number of filters in first conv layer (default: 32).
        dropout_rate: Dropout rate in classifier head (default: 0.3).
    """

    LABEL_REAL: int = 0
    LABEL_FAKE: int = 1

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        c = base_channels

        # 4-stage convolutional feature extractor
        self.features = nn.Sequential(
            # Stage 1: (B, 1, 128, 128) -> (B, 32, 64, 64)
            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: (B, 32, 64, 64) -> (B, 64, 32, 32)
            nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: (B, 64, 32, 32) -> (B, 128, 16, 16)
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 4: (B, 128, 16, 16) -> (B, 256, 8, 8)
            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Binary classification head
        feature_dim = c * 8  # 256
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(64, 1),
        )

        self._init_weights()

        logger.info(
            "AudioForensicCNN initialized: params=%d, base_channels=%d",
            sum(p.numel() for p in self.parameters()),
            base_channels,
        )

    def _init_weights(self) -> None:
        """Initialize convolution and linear weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over input spectrograms.

        Args:
            x: Input tensor of shape (batch_size, 1, 128, 128) or (1, 128, 128).

        Returns:
            Logits tensor of shape (batch_size,). Values > 0 indicate FAKE.
        """
        if x.dim() == 3:
            # Add batch dimension if unbatched: (1, 128, 128) -> (1, 1, 128, 128)
            x = x.unsqueeze(0)

        feat = self.features(x)         # (B, 256, 8, 8)
        pooled = self.pool(feat)        # (B, 256, 1, 1)
        flattened = torch.flatten(pooled, 1)  # (B, 256)
        logits = self.classifier(flattened)   # (B, 1)

        return logits.squeeze(-1)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> dict[str, torch.Tensor]:
        """Inference helper returning raw probabilities and binary labels.

        Args:
            x: Input tensor of shape (B, 1, 128, 128).
            threshold: Classification threshold (default: 0.5).

        Returns:
            Dict containing 'logits', 'probabilities', and binary 'predictions'.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).long()

            return {
                "logits": logits,
                "probabilities": probabilities,
                "predictions": predictions,
            }
