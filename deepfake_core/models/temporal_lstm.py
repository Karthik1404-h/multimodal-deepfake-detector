"""Spatiotemporal LSTM architecture for sequence-level deepfake forensic analysis.

Processes sequences of frame-level feature embeddings (e.g. 1280-dim vectors from EfficientNet-B0)
using a bidirectional LSTM and a temporal attention pooling mechanism to detect temporal anomalies,
flickering, blinking irregularities, and boundary inconsistencies.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """Temporal attention pooling module for sequential embeddings.

    Computes normalized attention scores over sequence timesteps to produce
    a weighted context vector, prioritizing anomalous frames over stable ones.

    Args:
        feature_dim: Dimensionality of the input feature vector (e.g., hidden_dim * 2).
        attention_dim: Intermediate projection dimensionality.
    """

    def __init__(self, feature_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and weighted context vector.

        Args:
            x: Tensor of shape (batch_size, sequence_length, feature_dim).
            mask: Optional boolean tensor of shape (batch_size, sequence_length)
                where True indicates valid timesteps.

        Returns:
            Tuple of:
                - context: Weighted sum tensor of shape (batch_size, feature_dim).
                - weights: Attention distribution tensor of shape (batch_size, sequence_length).
        """
        # scores shape: (batch_size, sequence_length, 1)
        scores = self.projection(x)

        if mask is not None:
            # Mask padded positions with large negative value before softmax
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-1e9"))

        weights = F.softmax(scores, dim=1)  # (batch_size, sequence_length, 1)
        context = torch.sum(x * weights, dim=1)  # (batch_size, feature_dim)

        return context, weights.squeeze(-1)


class TemporalAnomalyLSTM(nn.Module):
    """Bidirectional LSTM with temporal attention for deepfake video detection.

    Architecture::

        Frame Feature Embeddings (B, T, 1280)
                    ↓
        Bidirectional LSTM (2 layers, hidden_dim=512)
                    ↓
        LSTM Outputs (B, T, 1024)
                    ↓
        Temporal Attention Pooling
                    ↓
        Context Vector (B, 1024)
                    ↓
        Dropout(0.3) → Linear(1024, 1)
                    ↓
        Sequence Output Logit (B,)

    Args:
        input_dim: Feature dimension per frame (default: 1280, matching EfficientNet-B0).
        hidden_dim: LSTM hidden state size (default: 512).
        num_layers: Number of stacked LSTM layers (default: 2).
        dropout: Dropout probability between LSTM layers and before the classifier head (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Bidirectional LSTM: output size = hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2

        # Temporal attention pooling over sequential outputs
        self.attention = TemporalAttention(
            feature_dim=lstm_output_dim,
            attention_dim=max(128, hidden_dim // 4),
        )

        # Classification head mapping pooled sequence context to binary logit
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_dim, 1),
        )

        self._init_weights()

        logger.info(
            "TemporalAnomalyLSTM initialized: input_dim=%d, hidden_dim=%d, "
            "num_layers=%d, bidirectional=True, params=%d",
            input_dim,
            hidden_dim,
            num_layers,
            sum(p.numel() for p in self.parameters()),
        )

    def _init_weights(self) -> None:
        """Initialize weights using Xavier / orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1.0 (LSTM standard best practice)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a sequence of frame embeddings.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).
            mask: Optional boolean mask of shape (batch_size, sequence_length)
                where True indicates valid frames and False indicates padding.
            return_attention: If True, also returns the attention weights
                distribution across the sequence.

        Returns:
            Logits tensor of shape (batch_size,).
            If return_attention=True, returns tuple (logits, attention_weights)
            where attention_weights has shape (batch_size, sequence_length).
        """
        if x.dim() != 3 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input tensor of shape (batch_size, sequence_length, {self.input_dim}), "
                f"but got tensor of shape {tuple(x.shape)}"
            )

        # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Attention pooling: context is (batch_size, hidden_dim * 2), weights is (batch_size, sequence_length)
        context, attn_weights = self.attention(lstm_out, mask=mask)

        # logits shape: (batch_size,)
        logits = self.classifier(context).squeeze(-1)

        if return_attention:
            return logits, attn_weights
        return logits

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Inference helper returning probabilities and binary decisions.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).
            threshold: Decision threshold for FAKE classification (default: 0.5).
            mask: Optional boolean mask of valid timesteps.

        Returns:
            Dict containing:
                - 'logits': Raw scalar logits of shape (batch_size,)
                - 'probabilities': Sigmoid probabilities in [0, 1] of shape (batch_size,)
                - 'predictions': Binary label (1=FAKE, 0=REAL) of shape (batch_size,)
                - 'attention_weights': Attention distribution of shape (batch_size, sequence_length)
        """
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward(x, mask=mask, return_attention=True)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).long()

            return {
                "logits": logits,
                "probabilities": probabilities,
                "predictions": predictions,
                "attention_weights": attn_weights,
            }
