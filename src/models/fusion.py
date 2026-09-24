"""Proposed - Feature Fusion: CNN branch (+) BEATs+AASIST branch -> joint classifier.

Both branches consume the same raw waveform input and each contributes its
pooled embedding (see forward_features() in cnn.py / beats_aasist.py); the
embeddings are concatenated and passed through a small joint classifier
head. Branches can optionally be initialized from already-trained Level 1 /
Level 3 checkpoints (recommended - train those two first) and optionally
frozen so only the fusion head is trained.
"""

import torch
import torch.nn as nn

from src.models.beats_aasist import EMBED_DIM as BEATS_EMBED_DIM
from src.models.beats_aasist import BeatsAASIST
from src.models.cnn_waveform import EMBED_DIM as CNN_EMBED_DIM
from src.models.cnn_waveform import CNN

JOINT_DIM = CNN_EMBED_DIM + BEATS_EMBED_DIM


class Fusion(nn.Module):
    def __init__(self, dropout=0.3, freeze_branches=False,
                 cnn_checkpoint=None, beats_aasist_checkpoint=None):
        super().__init__()
        self.cnn_branch = CNN(dropout=dropout)
        self.beats_aasist_branch = BeatsAASIST(dropout=dropout)

        if cnn_checkpoint is not None:
            state = torch.load(cnn_checkpoint, map_location="cpu")
            self.cnn_branch.load_state_dict(state["state_dict"])
        if beats_aasist_checkpoint is not None:
            state = torch.load(beats_aasist_checkpoint, map_location="cpu")
            self.beats_aasist_branch.load_state_dict(state["state_dict"])

        if freeze_branches:
            for p in self.cnn_branch.parameters():
                p.requires_grad = False
            for p in self.beats_aasist_branch.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(JOINT_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, waveform):
        """Returns raw logits (B,) - apply sigmoid / BCEWithLogitsLoss outside."""
        cnn_emb = self.cnn_branch.forward_features(waveform)  # (B, CNN_EMBED_DIM)
        beats_emb = self.beats_aasist_branch.forward_features(waveform)  # (B, BEATS_EMBED_DIM)
        joint = self.dropout(torch.cat([cnn_emb, beats_emb], dim=-1))
        return self.classifier(joint).squeeze(-1)


if __name__ == "__main__":
    m = Fusion()
    x = torch.randn(2, 1, 64_000)
    out = m(x)
    print("Fusion output shape:", out.shape)  # expect (2,)
