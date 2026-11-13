"""Small CNN used for MNIST training runs."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Two-conv-layer CNN with dropout before the classifier."""

    def __init__(self, dropout: float = 0.0, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        return self.classifier(self.features(inputs))


def build_model(dropout: float = 0.0, num_classes: int = 10) -> SmallCNN:
    """Construct the model with the run's dropout setting."""
    return SmallCNN(dropout=dropout, num_classes=num_classes)
