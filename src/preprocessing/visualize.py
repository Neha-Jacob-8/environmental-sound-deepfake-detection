"""Waveform + log-Mel spectrogram plots for one REAL and one FAKE clip.

Purely illustrative: this demonstrates the preprocessing pipeline for a
project review, it does not train anything.

Picks one REAL clip and one FAKE clip from the manifest (same source
recording when possible, so the pair is directly comparable) and saves:

    results/plots/real_waveform.png
    results/plots/fake_waveform.png
    results/plots/real_mel_spectrogram.png
    results/plots/fake_mel_spectrogram.png
    results/plots/preprocessing_comparison.png

Usage:
    python -m src.preprocessing.visualize
    python -m src.preprocessing.visualize --split test --generator G03
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: safe for scripts, servers, and Colab alike
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.generators import GENERATOR_NAMES  # noqa: E402
from src.preprocessing.verify_subset import TARGET_SR  # noqa: E402

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


def load_manifest(root="data"):
    path = Path(root) / "metadata" / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_subset.py and verify_subset.py "
            "first to build the manifest before visualizing."
        )
    return pd.read_csv(path)


def select_pair(manifest, split=None, generator=None):
    """Pick one REAL clip and one FAKE clip to visualize.

    Prefers a REAL clip and a fake clip that share the same (split,
    source_id), so the pair is the same underlying recording - real vs. a
    generated version of it - which makes for a clearer side-by-side figure.
    Falls back to any REAL/FAKE clip if no matched pair is available.
    """
    df = manifest.copy()
    if split is not None:
        df = df[df.split == split]
    if df.empty:
        raise ValueError(f"No rows in manifest for split={split!r}")

    fakes = df[df.generator != "REAL"]
    if generator is not None:
        fakes = fakes[fakes.generator == generator]
    if fakes.empty:
        raise ValueError("No FAKE clips found matching the requested filters")

    reals = df[df.generator == "REAL"]
    if reals.empty:
        raise ValueError("No REAL clips found matching the requested filters")

    # Try to find a matched (split, source_id) pair first.
    if "source_id" in df.columns:
        for _, fake_row in fakes.iterrows():
            match = reals[
                (reals.split == fake_row.split)
                & (reals.source_id == fake_row.source_id)
            ]
            if len(match):
                return match.iloc[0], fake_row

    return reals.iloc[0], fakes.iloc[0]


def load_audio(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:  # defensive: processed clips should already be mono
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), sr


def clip_title(row):
    kind = "REAL" if row.generator == "REAL" else "FAKE"
    label = f"{kind}"
    if row.generator != "REAL":
        gen_name = GENERATOR_NAMES.get(row.generator, row.generator)
        label += f" ({row.generator} - {gen_name})"
    if "source_dataset" in row and pd.notna(row.source_dataset):
        label += f", source: {row.source_dataset}"
    return label


def plot_waveform(wav, sr, title, out_path):
    t = np.arange(len(wav)) / sr
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, wav, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, t[-1] if len(t) else 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_log_mel(wav, sr):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)


def plot_mel_spectrogram(wav, sr, title, out_path):
    log_mel = compute_log_mel(wav, sr)
    duration = len(wav) / sr
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        log_mel, origin="lower", aspect="auto", cmap="magma",
        extent=[0, duration, 0, N_MELS],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel frequency bin")
    fig.colorbar(im, ax=ax, format="%+2.0f dB", label="Log power (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return log_mel


def plot_combined(real_wav, fake_wav, sr, real_title, fake_title, out_path):
    real_mel = compute_log_mel(real_wav, sr)
    fake_mel = compute_log_mel(fake_wav, sr)
    duration = len(real_wav) / sr

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    t = np.arange(len(real_wav)) / sr
    axes[0, 0].plot(t, real_wav, linewidth=0.5, color="tab:blue")
    axes[0, 0].set_title(f"REAL waveform\n{real_title}")
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude")

    t = np.arange(len(fake_wav)) / sr
    axes[0, 1].plot(t, fake_wav, linewidth=0.5, color="tab:red")
    axes[0, 1].set_title(f"FAKE waveform\n{fake_title}")
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("Amplitude")

    im0 = axes[1, 0].imshow(
        real_mel, origin="lower", aspect="auto", cmap="magma",
        extent=[0, duration, 0, N_MELS],
    )
    axes[1, 0].set_title("REAL log-Mel spectrogram")
    axes[1, 0].set_xlabel("Time (seconds)")
    axes[1, 0].set_ylabel("Mel frequency bin")
    fig.colorbar(im0, ax=axes[1, 0], format="%+2.0f dB")

    im1 = axes[1, 1].imshow(
        fake_mel, origin="lower", aspect="auto", cmap="magma",
        extent=[0, duration, 0, N_MELS],
    )
    axes[1, 1].set_title("FAKE log-Mel spectrogram")
    axes[1, 1].set_xlabel("Time (seconds)")
    axes[1, 1].set_ylabel("Mel frequency bin")
    fig.colorbar(im1, ax=axes[1, 1], format="%+2.0f dB")

    fig.suptitle("Preprocessing comparison: REAL vs. FAKE", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--split", default=None,
                     help="restrict clip selection to one split "
                          "(default: any split in the manifest)")
    ap.add_argument("--generator", default=None,
                     help="restrict the FAKE clip to one generator, e.g. G03")
    a = ap.parse_args()

    manifest = load_manifest(a.root)
    real_row, fake_row = select_pair(manifest, split=a.split, generator=a.generator)

    # Reconstruct from split/filename rather than trusting the manifest's
    # "path" column: older manifests written on Windows store backslash
    # paths there, which silently fail to resolve on Linux/Colab.
    real_path = Path(a.root) / "processed" / real_row.split / real_row.filename
    fake_path = Path(a.root) / "processed" / fake_row.split / fake_row.filename
    for p in (real_path, fake_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"Selected clip is missing on disk: {p}")

    print(f"REAL clip: {real_path}")
    print(f"FAKE clip: {fake_path}")

    real_wav, real_sr = load_audio(real_path)
    fake_wav, fake_sr = load_audio(fake_path)
    if real_sr != TARGET_SR or fake_sr != TARGET_SR:
        raise ValueError(
            f"Expected {TARGET_SR} Hz audio, got real={real_sr}, fake={fake_sr}. "
            "Run verify_subset.py first to confirm standardization."
        )

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_title = clip_title(real_row)
    fake_title = clip_title(fake_row)

    plot_waveform(real_wav, real_sr, real_title, out_dir / "real_waveform.png")
    print(f"saved -> {out_dir / 'real_waveform.png'}")
    plot_waveform(fake_wav, fake_sr, fake_title, out_dir / "fake_waveform.png")
    print(f"saved -> {out_dir / 'fake_waveform.png'}")

    plot_mel_spectrogram(real_wav, real_sr, real_title, out_dir / "real_mel_spectrogram.png")
    print(f"saved -> {out_dir / 'real_mel_spectrogram.png'}")
    plot_mel_spectrogram(fake_wav, fake_sr, fake_title, out_dir / "fake_mel_spectrogram.png")
    print(f"saved -> {out_dir / 'fake_mel_spectrogram.png'}")

    plot_combined(
        real_wav, fake_wav, TARGET_SR, real_title, fake_title,
        out_dir / "preprocessing_comparison.png",
    )
    print(f"saved -> {out_dir / 'preprocessing_comparison.png'}")


if __name__ == "__main__":
    main()
