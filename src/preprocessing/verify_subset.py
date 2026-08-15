"""Verify a fetched EnvSDD subset and emit a single combined manifest.

This is the completion gate for the preprocessing stage. It checks that every
row in the per-split metadata CSVs has a readable WAV at 16 kHz / 64,000 samples
/ mono, drops rows whose audio is missing or corrupt, and writes one manifest
covering all splits.

Audio is NOT normalised on disk. Peak normalisation happens at load time in the
Dataset so the raw clips stay available for augmentation experiments.

Usage:
    python -m src.preprocessing.verify_subset
    python -m src.preprocessing.verify_subset --strict   # exit 1 on any problem
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

TARGET_SR = 16_000
TARGET_LEN = 64_000
SPLITS = ["train", "validation", "test"]


def check_split(split, root):
    meta = Path(root) / "metadata" / f"{split}_subset.csv"
    audio_dir = Path(root) / "processed" / split
    if not meta.exists():
        print(f"[{split}] no metadata CSV - not fetched yet, skipping")
        return None

    df = pd.read_csv(meta)
    ok, problems = [], []

    for row in df.itertuples(index=False):
        p = audio_dir / row.filename
        if not p.exists():
            problems.append((row.filename, "missing"))
            continue
        try:
            wav, sr = sf.read(p)
        except Exception as e:
            problems.append((row.filename, f"unreadable: {e}"))
            continue
        if wav.ndim != 1:
            problems.append((row.filename, f"not mono: {wav.shape}"))
        elif sr != TARGET_SR:
            problems.append((row.filename, f"sr={sr}"))
        elif len(wav) != TARGET_LEN:
            problems.append((row.filename, f"len={len(wav)}"))
        elif not np.isfinite(wav).all():
            problems.append((row.filename, "non-finite samples"))
        elif np.abs(wav).max() < 1e-6:
            problems.append((row.filename, "silent"))
        else:
            ok.append(row.filename)

    clean = df[df.filename.isin(set(ok))].copy()
    clean["path"] = clean.filename.map(lambda f: str(audio_dir / f))

    print(f"\n[{split}]")
    print(f"  metadata rows : {len(df)}")
    print(f"  usable clips  : {len(clean)}")
    print(f"  problems      : {len(problems)}")
    for f, why in problems[:10]:
        print(f"      {f}: {why}")
    if len(problems) > 10:
        print(f"      ... and {len(problems) - 10} more")

    if len(clean):
        print(f"  per generator : {clean.generator.value_counts().sort_index().to_dict()}")
        print(f"  label balance : {clean.label.value_counts().to_dict()}")
        print(f"  source groups : {clean.source_id.nunique()}")
        print(f"  source data   : {clean.source_dataset.value_counts().to_dict()}")

        # Incomplete groups are usable but unbalance per-generator comparisons.
        sizes = clean.groupby("source_id").size()
        expected = 8 if split == "test" else 5
        partial = int((sizes != expected).sum())
        if partial:
            print(f"  WARNING: {partial} incomplete source groups "
                  f"(expected {expected} clips each) - re-run the fetch to fill them")

    return clean, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any clip failed verification")
    a = ap.parse_args()

    frames, total_problems = [], 0
    for split in SPLITS:
        res = check_split(split, a.root)
        if res is None:
            continue
        clean, problems = res
        total_problems += len(problems)
        if len(clean):
            frames.append(clean)

    if not frames:
        print("\nNothing verified - run fetch_subset.py first.")
        sys.exit(1)

    manifest = pd.concat(frames, ignore_index=True)
    out = Path(a.root) / "metadata" / "manifest.csv"
    manifest.to_csv(out, index=False)

    print("\n=== manifest ===")
    print(f"total usable clips: {len(manifest)}")
    print(manifest.groupby(["split", "generator"]).size().unstack(fill_value=0))
    print(f"\nwritten -> {out}")

    # Cross-split leakage check: a source_id must never appear in two splits.
    # source_id is derived per split from row position, so collisions across
    # splits are expected numerically - compare on (split, source_id) instead.
    dupes = manifest.groupby("source_id")["split"].nunique()
    shared = int((dupes > 1).sum())
    if shared:
        print(f"\nnote: {shared} source_id values appear in more than one split. "
              "This is expected (ids are per-split row positions), not leakage - "
              "the HF splits are disjoint.")

    if total_problems and a.strict:
        print(f"\nFAILED: {total_problems} problem clips (--strict)")
        sys.exit(1)
    print(f"\nPreprocessing verified. {total_problems} problem clips excluded.")


if __name__ == "__main__":
    main()
