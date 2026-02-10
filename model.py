import torch
import torch.nn as nn
import constants as const


class SimpleCNN1D(nn.Module):
    """
1D Convolutional Neural Network for ICU alarm detection.

Input:
- x: Tensor of shape (batch_size, channels, time)
       e.g., multi-lead waveform signals.

Architecture:
1. Feature extractor:
   - 4 convolutional blocks: Conv1D -> BatchNorm -> ReLU -> MaxPool1D
   - Extra Conv1D + BatchNorm + ReLU
   - AdaptiveAvgPool1d collapses time dimension → fixed-length feature vector

2. Classifier (fully connected layers):
   - Flatten features
   - Dropout for regularization
   - Dense layer reduces feature dimension
   - ReLU + Dropout
   - Output: single logit for binary classification

Output:
- Tensor of shape (batch_size,) → logit for "True alarm" (1) vs "False alarm" (0)
- Apply sigmoid externally to convert logit to probability

Notes:
- BatchNorm stabilizes training across batches
- MaxPooling reduces temporal dimension while retaining key features
- Dropout prevents overfitting
"""
    def __init__(self, in_channels: int, base: int = const.MODEL_BASE, dropout: float = const.MODEL_DROPOUT):
        super().__init__()

        # Utility function to build a Conv1D block
        def block(cin, cout, kernel_size, padding):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(cout), #normalizamos para que la salida tenga media 0 y varianza 1
                nn.ReLU(inplace=True), # función de activación ReLU
                nn.MaxPool1d(2) # reduce temporal dimension by half
            )

        # Feature extractor
        self.features = nn.Sequential(
            block(in_channels, base, 7, 3),
            block(base, base * 2, 7, 3),
            block(base * 2, base * 4, 7, 3),
            block(base * 4, base * 4, 7, 3),
            nn.Conv1d(base * 4, base * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # Output shape: (batch_size, base*4, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),        # pasa de (batch_size, base*4, 1) a (batch_size, base*4), es decir quita la dimensión temporal
            nn.Dropout(dropout),
            nn.Linear(base * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),            # Output logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)  # Return shape: (batch_size,)


