"""Level 1 baseline: a small CNN over log-Mel spectrograms.

Deliberately modest. The training subset is 6,000 clips from 1,200 source
recordings, so a large network would memorise the source recordings rather than
learn generator artifacts - and memorising the source is exactly the failure
mode that makes a detector collapse on unseen generators.

Input  : (B, 1, 64, 251)   standardised log-Mel
Output : (B,)              a single logit, higher = more likely fake

Global average pooling over time rather than a flatten: a flatten would tie the
classifier to a fixed 251-frame input and let it key on *where* in the clip an
artifact sits. Artifacts are a property of the whole clip, so pooling is both
the smaller and the more honest choice.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, pool=2, dropout=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.body(x))


class LogMelCNN(nn.Module):
    """Four conv blocks -> global average pool -> one logit.

    Args:
        channels: width of each block
        dropout: 2-D dropout between blocks
        head_dropout: dropout before the classifier
        n_mels: only used for the shape self-check in `feature_dim`
    """

    def __init__(self, channels=(32, 64, 128, 128), dropout=0.1,
                 head_dropout=0.3, in_ch=1):
        super().__init__()
        blocks, c_prev = [], in_ch
        for c in channels:
            blocks.append(ConvBlock(c_prev, c, pool=2, dropout=dropout))
            c_prev = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(c_prev, 1),
        )
        self.out_channels = c_prev

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)          # (B, C)
        return self.head(x).squeeze(1)       # (B,)

    def embed(self, x):
        """Penultimate features - the fusion stage (Level 3) will need these."""
        return self.pool(self.features(x)).flatten(1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(name="cnn", **kw):
    if name != "cnn":
        raise ValueError(f"unknown model {name!r}")
    return LogMelCNN(**kw)
