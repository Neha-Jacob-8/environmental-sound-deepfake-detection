"""Backend-safe adaptive pooling.

torch's adaptive_avg_pool is not implemented on Apple's MPS backend when the
input size is not an exact multiple of the output size (pytorch#96056). That is
the normal case in this project - AASIST pools 17x249 down to 16x32, and the
Level 3 frontend pools ~199 frames down to 40 nodes - so the models would run
on CPU and CUDA but die on a Mac GPU.

Falling back to CPU for just this one op keeps the result bit-identical on every
backend. The tensors are small by the time they reach it (the conv stack above
has already reduced them), so the round trip costs far less than moving the
whole model to CPU would.
"""

import torch.nn.functional as F


def _needs_cpu(x, sizes):
    return x.device.type == "mps" and any(
        dim % size for dim, size in zip(x.shape[-len(sizes):], sizes)
    )


def adaptive_avg_pool2d(x, size):
    """(B, C, H, W) -> (B, C, size[0], size[1])."""
    if _needs_cpu(x, size):
        return F.adaptive_avg_pool2d(x.cpu(), size).to(x.device)
    return F.adaptive_avg_pool2d(x, size)


def adaptive_avg_pool1d(x, size):
    """(B, C, L) -> (B, C, size)."""
    if _needs_cpu(x, (size,)):
        return F.adaptive_avg_pool1d(x.cpu(), size).to(x.device)
    return F.adaptive_avg_pool1d(x, size)
