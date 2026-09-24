"""Level 1 - CNN baseline.

log-Mel -> Conv2D x2 -> Dense -> sigmoid, per the methodology diagram.
Input is the raw waveform (1, 64000); the log-Mel front-end lives inside the
model so every model in this project takes the same input shape.
"""

import torch
import torch.nn as nn

from src.models.features import LogMelExtractor

EMBED_DIM = 64


class CNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.melspec = LogMelExtractor()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (32, 64, 251) -> (32, 32, 125)

            nn.Conv2d(32, EMBED_DIM, kernel_size=3, padding=1),
            nn.BatchNorm2d(EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 32, 125) -> (64, 16, 62)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (64, 1, 1), robust to any input size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(EMBED_DIM, 1)

    def forward_features(self, waveform):
        """waveform: (B, 1, 64000) -> (B, 64) pooled embedding."""
        x = self.melspec(waveform)  # (B, 1, 64, 251)
        x = self.conv(x)  # (B, 64, 16, 62)
        x = self.pool(x).flatten(1)  # (B, 64)
        return x

    def forward(self, waveform):
        """Returns raw logits (B,) - apply sigmoid / BCEWithLogitsLoss outside."""
        emb = self.dropout(self.forward_features(waveform))
        return self.classifier(emb).squeeze(-1)


if __name__ == "__main__":
    m = CNN()
    x = torch.randn(4, 1, 64_000)
    out = m(x)
    print("CNN output shape:", out.shape)  # expect (4,)
