"""PyTorch Dataset over a verified EnvSDD subset.

Reads data/metadata/manifest.csv (written by verify_subset.py) and returns either
raw waveforms (for AASIST) or log-Mel spectrograms (for the CNN baseline).

Amplitude normalisation happens here rather than on disk, so the stored clips
stay untouched and the augmentation experiments can work from the raw audio.

Usage:
    from src.datasets.envsdd_dataset import EnvSDDDataset, make_loader

    train = EnvSDDDataset(split="train", mode="logmel")
    seen  = EnvSDDDataset(split="test", mode="logmel", generators=["G01","G02","G03","G04"])
    unseen= EnvSDDDataset(split="test", mode="logmel", generators=["G05","G06","G07"])
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.generators import (  # noqa: E402
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)

SR = 16_000
CLIP_LEN = 64_000          # 4.000 s
N_FFT = 1024
HOP = 256
N_MELS = 64                # -> (64 mels, 251 frames)
STATS_PATH = "data/metadata/logmel_stats.npz"


class EnvSDDDataset(Dataset):
    """One clip per item.

    Args:
        split: "train" | "validation" | "test"
        mode: "waveform" -> (1, 64000); "logmel" -> (1, N_MELS, frames)
        generators: keep only these generator ids. REAL clips are always kept,
            since a fake-only set has nothing to discriminate against.
        manifest: path to manifest.csv
        normalize: peak-normalise each clip to unit amplitude
        standardize: subtract the train-split per-mel-bin mean and divide by its
            std (logmel mode only). Stats always come from the *train* split,
            whatever split this Dataset holds, so val/test never leak into them.
        augment: optional callable(np.ndarray) -> np.ndarray applied to the raw
            waveform before normalisation (Experiment 5).
    """

    def __init__(self, split, mode="logmel", generators=None,
                 manifest="data/metadata/manifest.csv", normalize=True,
                 augment=None, standardize=True):
        if mode not in ("waveform", "logmel"):
            raise ValueError(f"mode must be 'waveform' or 'logmel', got {mode!r}")

        mpath = Path(manifest)
        if not mpath.exists():
            raise FileNotFoundError(
                f"{mpath} not found. Run fetch_subset.py then verify_subset.py."
            )

        df = pd.read_csv(mpath)
        df = df[df.split == split]
        if generators is not None:
            keep = set(generators) | {"REAL"}
            df = df[df.generator.isin(keep)]
        if df.empty:
            raise ValueError(
                f"No clips for split={split!r} generators={generators!r}. "
                f"Manifest has splits {sorted(pd.read_csv(mpath).split.unique())}."
            )

        self.df = df.reset_index(drop=True)
        self.split, self.mode = split, mode
        self.normalize, self.augment = normalize, augment

        self.standardize = bool(standardize) and mode == "logmel"

        if mode == "logmel":
            import torchaudio
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            )
            self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        if self.standardize:
            mean, std = logmel_stats(manifest=manifest)
            self._mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1)
            self._std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, sr = sf.read(row.path, dtype="float32", always_2d=False)

        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            raise ValueError(f"{row.path}: expected {SR} Hz, got {sr}")

        # Length is guaranteed by verify_subset.py, but guard anyway so a
        # hand-added clip cannot produce a ragged batch.
        if len(wav) < CLIP_LEN:
            wav = np.pad(wav, (0, CLIP_LEN - len(wav)))
        elif len(wav) > CLIP_LEN:
            wav = wav[:CLIP_LEN]

        if self.augment is not None:
            wav = self.augment(wav)

        if self.normalize:
            peak = np.abs(wav).max()
            if peak > 1e-8:
                wav = wav / peak

        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))

        if self.mode == "logmel":
            x = self.to_db(self.melspec(x)).unsqueeze(0)   # (1, mels, frames)
            if self.standardize:
                x = (x - self._mean) / self._std
        else:
            x = x.unsqueeze(0)                              # (1, samples)

        return x, torch.tensor(int(row.label), dtype=torch.float32)

    def class_weights(self):
        """pos_weight for BCEWithLogitsLoss.

        Fakes outnumber reals 4:1 in train and 7:1 in test, so unweighted BCE
        drifts toward predicting 'fake'.
        """
        n_fake = int((self.df.label == 1).sum())
        n_real = int((self.df.label == 0).sum())
        if n_fake == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_real / n_fake, dtype=torch.float32)

    def describe(self):
        g = self.df.generator.value_counts().sort_index().to_dict()
        return (f"{self.split}: {len(self.df)} clips, "
                f"{self.df.source_id.nunique()} source groups, "
                f"real={int((self.df.label == 0).sum())} "
                f"fake={int((self.df.label == 1).sum())}, {g}")


def logmel_stats(manifest="data/metadata/manifest.csv", cache=STATS_PATH,
                 split="train", force=False):
    """Per-mel-bin mean and std of the log-Mel features, over `split`.

    Unstandardised log-Mel sits around mean 8 / std 14 with a ~35 dB tilt across
    mel bins, which makes the first conv layer spend its early training just
    rescaling its input and makes the learning rate fussy to pick.

    Computed once and cached to disk. Always taken from the train split so that
    validation and test statistics never leak into training.
    """
    cpath = Path(cache)
    if cpath.exists() and not force:
        d = np.load(cpath)
        return d["mean"], d["std"]

    ds = EnvSDDDataset(split=split, mode="logmel", manifest=manifest,
                       standardize=False)

    # Streaming sums rather than stacking every clip: the full train split at
    # scale would not fit in memory as one array.
    n = 0
    total = np.zeros(N_MELS, dtype=np.float64)
    total_sq = np.zeros(N_MELS, dtype=np.float64)
    for i in range(len(ds)):
        x = ds[i][0][0].numpy().astype(np.float64)   # (mels, frames)
        n += x.shape[1]
        total += x.sum(axis=1)
        total_sq += (x ** 2).sum(axis=1)

    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 1e-12))

    cpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cpath, mean=mean, std=std, n_frames=n, split=split)
    print(f"logmel stats over {len(ds)} {split} clips -> {cpath}")
    return mean, std


def make_loader(split, mode="logmel", generators=None, batch_size=32,
                shuffle=None, num_workers=4, **kw):
    ds = EnvSDDDataset(split=split, mode=mode, generators=generators, **kw)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=False,          # no effect on MPS
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def seen_unseen_test_loaders(mode="logmel", batch_size=32, **kw):
    """The core generalisation experiment: G01-G04 vs G05-G07.

    Both draw from the test split so they share the same real clips and the same
    source-domain mix, leaving the generator as the only difference.
    """
    return (
        make_loader("test", mode, SEEN_GENERATORS, batch_size, shuffle=False, **kw),
        make_loader("test", mode, UNSEEN_GENERATORS, batch_size, shuffle=False, **kw),
    )
