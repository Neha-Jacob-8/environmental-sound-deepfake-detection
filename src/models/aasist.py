"""Level 2 - AASIST: raw waveform -> spectro-temporal graph attention.

Simplified, from-scratch reimplementation inspired by Jung et al., "AASIST:
Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention
Networks" (ICASSP 2022). Trained from scratch (no pretrained frontend),
matching the methodology diagram. See graph_attention.py for what is and
isn't a faithful reproduction of the original architecture.

Pipeline:
  raw waveform (1, 64000)
    -> learnable 1D conv frontend (waveform -> time-frequency map)
    -> residual 2D conv encoder -> fixed-size feature map (C, F, T)
    -> spectral graph: F nodes (one per frequency bin, pooled over time)
    -> temporal graph: T nodes (one per time frame, pooled over frequency)
    -> graph attention (self-attention) over each graph independently
    -> readout: concat(max+mean pool of spectral nodes, max+mean pool of
       temporal nodes) -> MLP classifier -> sigmoid
"""

import torch
import torch.nn as nn

from src.models.graph_attention import GraphAttentionLayer, graph_readout
from src.models.pooling import adaptive_avg_pool2d

CHANNELS = 64
F_OUT, T_OUT = 16, 32  # fixed spectral/temporal graph sizes (via adaptive pooling)
EMBED_DIM = 4 * CHANNELS  # readout dim: (max+mean) x (spectral+temporal)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SELU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return self.pool(out)


class AASIST(nn.Module):
    def __init__(self, n_heads=4, dropout=0.2):
        super().__init__()

        # Learnable "sinc-like" 1D frontend: turns the raw waveform into a
        # 2D time-frequency map, standing in for AASIST's SincConv frontend.
        self.frontend = nn.Conv1d(1, 70, kernel_size=128, stride=64, padding=0)
        self.frontend_bn = nn.BatchNorm1d(70)

        # Frontend gives F=70 (filters) x T=999 (hops). Only pool twice so the
        # encoder is downsampling into the adaptive-pool target below, not
        # upsampling from an over-shrunk map (70x999 -> 17x249 -> 16x32).
        self.encoder = nn.Sequential(
            ResidualBlock2D(1, 32, pool=True),
            ResidualBlock2D(32, 32, pool=True),
            ResidualBlock2D(32, CHANNELS, pool=False),
            ResidualBlock2D(CHANNELS, CHANNELS, pool=False),
        )

        self.spectral_gat = GraphAttentionLayer(CHANNELS, n_heads=n_heads, dropout=dropout)
        self.temporal_gat = GraphAttentionLayer(CHANNELS, n_heads=n_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward_features(self, waveform):
        """waveform: (B, 1, 64000) -> (B, EMBED_DIM) pooled graph embedding."""
        x = self.frontend(waveform)  # (B, 70, T1)
        x = self.frontend_bn(x.abs())
        x = x.unsqueeze(1)  # (B, 1, 70, T1) - treat the 70 filters as a frequency axis

        x = self.encoder(x)  # (B, C, F', T')
        # fixed (F_OUT, T_OUT) regardless of input length; see pooling.py for
        # why this is the functional form rather than nn.AdaptiveAvgPool2d
        x = adaptive_avg_pool2d(x, (F_OUT, T_OUT))

        spectral_nodes = x.mean(dim=3).transpose(1, 2)  # (B, F_OUT, C): pool over time
        temporal_nodes = x.mean(dim=2).transpose(1, 2)  # (B, T_OUT, C): pool over frequency

        spectral_nodes = self.spectral_gat(spectral_nodes)
        temporal_nodes = self.temporal_gat(temporal_nodes)

        spectral_readout = graph_readout(spectral_nodes)  # (B, 2C)
        temporal_readout = graph_readout(temporal_nodes)  # (B, 2C)
        return torch.cat([spectral_readout, temporal_readout], dim=-1)  # (B, 4C)

    def forward(self, waveform):
        """Returns raw logits (B,) - apply sigmoid / BCEWithLogitsLoss outside."""
        emb = self.dropout(self.forward_features(waveform))
        return self.classifier(emb).squeeze(-1)


if __name__ == "__main__":
    m = AASIST()
    x = torch.randn(4, 1, 64_000)
    out = m(x)
    print("AASIST output shape:", out.shape)  # expect (4,)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,}")
