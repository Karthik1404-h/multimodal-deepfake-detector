"""Spatial forensic classifier using EfficientNet-B0.

Binary classification head (Real=0, Fake=1) on top of a pretrained
EfficientNet-B0 backbone, optimized for the 6 GB VRAM budget.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SpatialForensicClassifier(nn.Module):
    """EfficientNet-B0 based binary deepfake classifier.

    Architecture::

        EfficientNet-B0 backbone (frozen or trainable)
            ↓
        AdaptiveAvgPool2d → Flatten
            ↓
        Dropout(0.3) → Linear(1280, 512) → ReLU → BatchNorm
            ↓
        Dropout(0.2) → Linear(512, 1) → Sigmoid

    Args:
        pretrained: Load ImageNet pretrained weights for the backbone.
        freeze_backbone: Freeze backbone parameters for transfer learning.
            Recommended for initial fine-tuning to prevent catastrophic forgetting.
        dropout_rate: Dropout probability for the classification head.
    """

    # Class labels for reference
    LABEL_REAL: int = 0
    LABEL_FAKE: int = 1

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.freeze_backbone = freeze_backbone

        # --- Backbone ---
        self.backbone = self._build_backbone(pretrained)

        if freeze_backbone:
            self._freeze_backbone_params()

        # --- Classification Head ---
        # EfficientNet-B0 outputs 1280-dim features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, 1),
        )

        # Initialize classifier weights
        self._init_classifier_weights()

        param_count = sum(p.numel() for p in self.parameters())
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SpatialForensicClassifier initialized: %d params (%d trainable), "
            "backbone_frozen=%s",
            param_count,
            trainable_count,
            freeze_backbone,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier.

        Args:
            x: Input tensor of shape (B, 3, 224, 224), normalized to
               ImageNet statistics.

        Returns:
            Tensor of shape (B, 1) with sigmoid probabilities.
            Values > 0.5 indicate FAKE, <= 0.5 indicate REAL.
        """
        features = self.backbone(x)       # (B, 1280)
        logits = self.classifier(features)  # (B, 1)
        return logits.squeeze(-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1280-dimensional global average pooled features before classification head.

        Args:
            x: Input tensor of shape (B, 3, 224, 224), normalized to ImageNet statistics.

        Returns:
            Tensor of shape (B, 1280) containing feature embeddings.
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> dict[str, torch.Tensor]:
        """Run inference and return structured predictions.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            threshold: Classification threshold (default 0.5).

        Returns:
            Dict with keys:
                - 'probabilities': Raw sigmoid outputs (B, 1)
                - 'predictions': Binary labels (B, 1), 1=FAKE
                - 'is_fake': Boolean tensor (B, 1)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            return {
                "probabilities": probs,
                "predictions": preds,
                "is_fake": preds.bool(),
            }

    # --- Private Helpers ---

    @staticmethod
    def _build_backbone(pretrained: bool) -> nn.Module:
        """Build EfficientNet-B0 feature extractor (no classification head).

        Args:
            pretrained: Whether to load ImageNet weights.

        Returns:
            nn.Module that maps (B, 3, 224, 224) → (B, 1280).
        """
        try:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for EfficientNet-B0. "
                "Install with: pip install torchvision"
            ) from exc

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)

        # Remove the original classifier — we use our own head
        # EfficientNet's features → avgpool → flatten gives us (B, 1280)
        backbone = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(1),
        )

        return backbone

    def _freeze_backbone_params(self) -> None:
        """Freeze all backbone parameters to prevent gradient updates."""
        frozen_count = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
            frozen_count += 1
        logger.debug("Froze %d backbone parameter tensors.", frozen_count)

    def _init_classifier_weights(self) -> None:
        """Initialize classifier head with Kaiming/Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == 1:
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def unfreeze_backbone(self, unfreeze_last_n: Optional[int] = None) -> None:
        """Selectively unfreeze backbone layers for fine-tuning.

        Args:
            unfreeze_last_n: Number of final backbone sub-modules to unfreeze.
                If None, unfreezes the entire backbone.
        """
        if unfreeze_last_n is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Unfroze entire backbone.")
        else:
            # backbone[0] is model.features which is a Sequential
            features = list(self.backbone[0].children())
            layers_to_unfreeze = features[-unfreeze_last_n:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info("Unfroze last %d backbone feature blocks.", unfreeze_last_n)
