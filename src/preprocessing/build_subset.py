"""Stream EnvSDD from HuggingFace and materialise a balanced local subset.

Streaming avoids pulling the full 60 GB repo, but a streaming IterableDataset
re-fetches over HTTP every epoch and cannot shuffle or random-access. So we
stream *once*, keep a per-generator quota, and write the result to local disk as
plain WAV + a metadata CSV. Training then reads from disk at full speed.

The HF train/validation/test splits are used as-is. We deliberately do not build
our own splits: a real clip and its derived fakes share no recoverable source ID
in this dataset, so any custom split risks scattering them across train and val.

Usage:
    # metadata only, no audio written - fast sanity check
    python -m src.preprocessing.build_subset --split test --per-generator 20 --dry-run

    # real subset
    python -m src.preprocessing.build_subset --split train      --per-generator 1500
    python -m src.preprocessing.build_subset --split validation --per-generator 400
    python -m src.preprocessing.build_subset --split test       --per-generator 400
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.generators import (  # noqa: E402
    ALL_GENERATORS,
    SEEN_GENERATORS,
    resolve_generator,
)

REPO = "EnvSDD/EnvSDD"
TARGET_SR = 16_000
CLIP_SECONDS = 4
TARGET_LEN = TARGET_SR * CLIP_SECONDS  # 64,000 samples


def decode_audio(audio_field):
    """Return (waveform_1d_float32, sample_rate).

    datasets 5.x returns a torchcodec AudioDecoder for Audio features; older
    versions return a dict with 'array'/'sampling_rate'. Handle both.
    """
    if isinstance(audio_field, dict):
        return np.asarray(audio_field["array"], dtype=np.float32), int(
            audio_field["sampling_rate"]
        )

    samples = audio_field.get_all_samples()
    wav = samples.data.numpy().astype(np.float32)
    return wav, int(samples.sample_rate)


def standardise(wav, sr):
    """Mono, 16 kHz, exactly 4 s, peak-normalised."""
    wav = np.squeeze(wav)
    if wav.ndim > 1:  # (channels, samples) -> mono
        wav = wav.mean(axis=0)

    if sr != TARGET_SR:
        import torch
        import torchaudio

        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, TARGET_SR
        ).numpy()

    # Pad short clips with silence, centre-truncate long ones.
    if wav.shape[0] < TARGET_LEN:
        wav = np.pad(wav, (0, TARGET_LEN - wav.shape[0]))
    elif wav.shape[0] > TARGET_LEN:
        start = (wav.shape[0] - TARGET_LEN) // 2
        wav = wav[start : start + TARGET_LEN]

    peak = np.abs(wav).max()
    if peak > 1e-8:
        wav = wav / peak
    return wav.astype(np.float32)


def build(split, per_generator, out_root, dry_run, max_rows):
    from datasets import load_dataset

    audio_dir = Path(out_root) / "processed" / split
    meta_path = Path(out_root) / "metadata" / f"{split}_subset.csv"
    if not dry_run:
        audio_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"opening stream: {REPO} [{split}] ...", flush=True)
    t0 = time.time()
    ds = load_dataset(REPO, split=split, streaming=True)
    print(f"stream open in {time.time() - t0:.1f}s", flush=True)

    kept = Counter()
    sources = Counter()
    seen_rows = 0
    rows_out = []
    # REAL needs a quota too, and reals are outnumbered by fakes in every split.
    quotas = {"REAL": per_generator}
    # train/validation contain only the four seen generators; test has all seven.
    expected = ["REAL"] + (ALL_GENERATORS if split == "test" else SEEN_GENERATORS)

    t0 = time.time()
    for row in ds:
        seen_rows += 1
        if max_rows and seen_rows > max_rows:
            print(f"hit --max-rows={max_rows}, stopping scan", flush=True)
            break

        gen = resolve_generator(row["attack_type"], row["generative_model"])
        if kept[gen] >= quotas.get(gen, per_generator):
            continue

        idx = kept[gen]
        stem = f"{gen}_{idx:06d}"

        if not dry_run:
            wav, sr = decode_audio(row["audio"])
            wav = standardise(wav, sr)
            sf.write(audio_dir / f"{stem}.wav", wav, TARGET_SR, subtype="PCM_16")

        rows_out.append(
            {
                "filename": f"{stem}.wav",
                "label": 0 if row["label"] == "real" else 1,
                "generator": gen,
                "attack_type": row["attack_type"],
                "generative_model": row["generative_model"],
                "source_dataset": row["dataset"],
                "orig_filename": row["filename"],
                "split": split,
            }
        )
        kept[gen] += 1
        sources[row["dataset"]] += 1

        total = sum(kept.values())
        if total % 100 == 0:
            print(
                f"  kept {total} / scanned {seen_rows} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

        # Stop as soon as every generator expected in this split is full.
        # Keyed on the expected set, not on what we have seen so far, so we
        # never stop early just because a later generator has not appeared yet.
        if all(kept[g] >= quotas.get(g, per_generator) for g in expected):
            print("all quotas filled, stopping scan", flush=True)
            break

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n--- {split} ---", flush=True)
    print(f"scanned {seen_rows} rows, kept {len(rows_out)}", flush=True)
    print("per generator:", dict(sorted(kept.items())), flush=True)
    print("per source dataset:", dict(sources.most_common()), flush=True)
    print(f"metadata -> {meta_path}", flush=True)
    if not dry_run:
        print(f"audio    -> {audio_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True,
                   choices=["train", "validation", "test", "remain"])
    p.add_argument("--per-generator", type=int, default=500,
                   help="clips to keep per generator (and for REAL)")
    p.add_argument("--out", default="data")
    p.add_argument("--dry-run", action="store_true",
                   help="collect metadata only, write no audio")
    p.add_argument("--max-rows", type=int, default=0,
                   help="stop after scanning N rows (0 = no limit)")
    a = p.parse_args()
    build(a.split, a.per_generator, a.out, a.dry_run, a.max_rows)


if __name__ == "__main__":
    main()
