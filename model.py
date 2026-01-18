#!/usr/bin/env python3
"""
model.py - Neural network architecture for hand shape classification
"""

import torch
import torch.nn as nn


class HandShapeClassifier(nn.Module):
    """
    MLP classifier for hand-formed shape classification.

    Input: 128 features (21*3*2 landmarks + 2 hand-present flags)
    Output: logits over num_classes
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        num_layers: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        layers = []

        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
