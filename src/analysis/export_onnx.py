"""Export the log-Mel CNN to ONNX, front-end included, for in-browser inference.

The browser must get the *same* answer as PyTorch, which means the log-Mel
front-end has to travel with the model rather than being reimplemented in
JavaScript - mel filterbanks and dB conversion are exactly where a hand port
silently diverges.

torchaudio's MelSpectrogram cannot be exported: it uses a complex STFT and ONNX
has no complex type. So the STFT is rebuilt here as a real-valued conv1d against
a precomputed DFT basis, which is mathematically identical and exports cleanly:

    X[k] = sum_n x[n] w[n] exp(-2*pi*i*k*n/N)
         = sum_n x[n] w[n] cos(...)  -  i * sum_n x[n] w[n] sin(...)
    power = real^2 + imag^2

The mel filterbank is lifted straight off the torchaudio transform rather than
recomputed, so the two cannot drift.

    python -m src.analysis.export_onnx --verify
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import (  # noqa: E402
    CLIP_LEN, HOP, N_FFT, N_MELS, SR, logmel_stats,
)
from src.models import load_checkpoint  # noqa: E402

TOP_DB = 80.0
AMIN = 1e-10


class ExportableDetector(nn.Module):
    """Raw waveform (B, 64000) -> (p_fake, axis_pos, embedding).

    axis_pos places the clip on the same real -> seen-fake axis the analysis
    uses: 0.0 is the real centroid, 1.0 the seen-fake centroid. It is a linear
    projection of the 128-d embedding, so unlike the t-SNE map it can be
    computed for a clip the model has never seen - which is what lets a user
    drop their own audio onto the same picture.
    """

    def __init__(self, cnn, mean, std, real_c=None, axis=None, scale=1.0):
        super().__init__()
        ref = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)

        # Real-valued DFT basis, windowed, as conv1d filters.
        win = torch.hann_window(N_FFT, periodic=True)
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64).unsqueeze(1)
        ang = 2.0 * math.pi * k * n / N_FFT
        w64 = win.to(torch.float64)
        basis = torch.cat([torch.cos(ang) * w64, -torch.sin(ang) * w64], dim=0)
        self.register_buffer("dft", basis.to(torch.float32).unsqueeze(1))

        # Lifted from torchaudio so the filterbank cannot drift from the
        # reference implementation.
        self.register_buffer("fb", ref.mel_scale.fb.clone())          # (513, 64)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, 1, -1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, 1, -1, 1))
        self.cnn = cnn
        self.n_freq = N_FFT // 2 + 1

        dim = cnn.out_channels
        self.register_buffer("real_c", torch.zeros(dim) if real_c is None
                             else torch.as_tensor(real_c, dtype=torch.float32))
        self.register_buffer("axis", torch.zeros(dim) if axis is None
                             else torch.as_tensor(axis, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, wav):
        # Peak-normalise, matching EnvSDDDataset(normalize=True).
        peak = wav.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        x = (wav / peak).unsqueeze(1)                       # (B, 1, T)

        # center=True in torch.stft means reflect-padding by n_fft // 2.
        x = torch.nn.functional.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
        spec = torch.nn.functional.conv1d(x, self.dft, stride=HOP)   # (B, 2F, T')
        power = spec[:, :self.n_freq] ** 2 + spec[:, self.n_freq:] ** 2

        mel = torch.matmul(power.transpose(1, 2), self.fb).transpose(1, 2)
        db = 10.0 * torch.log10(mel.clamp_min(AMIN))
        # AmplitudeToDB's top_db floor, per clip.
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - TOP_DB)

        feat = (db.unsqueeze(1) - self.mean) / self.std

        emb = self.cnn.embed(feat)                          # (B, 128)
        logit = self.cnn.head(emb).squeeze(1)               # (B,)
        axis_pos = ((emb - self.real_c) @ self.axis) / self.scale
        return torch.sigmoid(logit), axis_pos, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/models/logmel_cnn_best.pt")
    ap.add_argument("--out", default="docs/model/logmel_cnn.onnx")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    cnn, ck = load_checkpoint(a.checkpoint, "cpu")
    mean, std = logmel_stats()

    # Axis geometry, from the same embeddings the app plots.
    d = np.load("results/tables/embeddings.npz", allow_pickle=True)
    emb, groups = d["emb"].astype(np.float64), d["group"]
    real_c = emb[groups == "real"].mean(0)
    seen_c = emb[groups == "seen"].mean(0)
    axis = seen_c - real_c
    axis /= np.linalg.norm(axis)
    scale = float((seen_c - real_c) @ axis)
    print(f"axis scale (real centroid -> seen-fake centroid): {scale:.4f}")

    model = ExportableDetector(cnn, mean, std, real_c, axis, scale).eval()

    if a.verify:
        from src.datasets.envsdd_dataset import EnvSDDDataset
        raw = EnvSDDDataset(split="test", mode="waveform", normalize=False)
        ref_ds = EnvSDDDataset(split="test", mode="logmel")
        idx = list(range(0, len(raw), max(1, len(raw) // 64)))[:64]
        wav = torch.stack([raw[i][0][0] for i in idx])
        with torch.no_grad():
            got = model(wav)[0]
            want = torch.sigmoid(cnn(torch.stack([ref_ds[i][0] for i in idx])))
        d = (got - want).abs()
        print(f"vs the PyTorch pipeline over {len(idx)} real clips: "
              f"max |diff| {d.max():.3e}, mean {d.mean():.3e}")
        if d.max() > 1e-4:
            sys.exit("front-end does not match - refusing to export")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, CLIP_LEN)
    torch.onnx.export(
        model, (dummy,), a.out,
        input_names=["waveform"],
        output_names=["p_fake", "axis_pos", "embedding"],
        dynamic_axes={"waveform": {0: "batch"}, "p_fake": {0: "batch"},
                      "axis_pos": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=a.opset,
    )
    size = Path(a.out).stat().st_size
    print(f"exported -> {a.out}  ({size/1e6:.2f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    probe = torch.randn(3, CLIP_LEN)
    with torch.no_grad():
        want = [t.numpy() for t in model(probe)]
    got = sess.run(None, {"waveform": probe.numpy()})
    for name, g, w in zip(["p_fake", "axis_pos", "embedding"], got, want):
        print(f"  onnxruntime vs torch [{name:9s}] max |diff| {np.abs(g - w).max():.3e}")


if __name__ == "__main__":
    main()
