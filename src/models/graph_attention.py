"""Graph self-attention layer shared by the AASIST and BEATs+AASIST backends.

This is a compact, from-scratch layer in the spirit of the heterogeneous
stacking graph-attention layer (HS-GAL) from Jung et al., "AASIST: Audio
Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks"
(ICASSP 2022) - each node attends to every other node in its graph (a fully
connected graph over frequency bins, or over time frames) and the graph is
summarized by concatenating max- and mean-pooled node embeddings, matching
AASIST's "max graph operation" + readout. This is a simplified,
course-project-scope reimplementation, not a line-by-line port of the
original repo - node/edge counts and the exact heterogeneous-graph merge
step are reduced so it trains on a CPU in reasonable time.
"""

import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """Multi-head self-attention over a set of graph nodes, with a residual
    connection and layer norm (a standard GAT/Transformer-encoder block)."""

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes):
        """nodes: (B, N, dim) -> (B, N, dim), attention-updated node embeddings."""
        b, n, d = nodes.shape
        q = self.q_proj(nodes).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(nodes).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(nodes).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)  # (B,H,N,N)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)  # (B,N,dim)
        nodes = self.norm1(nodes + self.dropout(self.out_proj(out)))
        nodes = self.norm2(nodes + self.dropout(self.ffn(nodes)))
        return nodes


def graph_readout(nodes):
    """(B, N, dim) -> (B, 2*dim): concat of max-pool and mean-pool over nodes.

    This is the "graph pooling" step that turns a variable-size set of node
    embeddings into one fixed-size graph-level vector (AASIST's max graph
    operation, extended with mean-pooling for a slightly richer summary).
    """
    max_pool = nodes.max(dim=1).values
    mean_pool = nodes.mean(dim=1)
    return torch.cat([max_pool, mean_pool], dim=-1)
