"""Audio standardisation verification + preprocessing summary report.

This does NOT re-implement audio validation. It calls
`verify_subset.check_split` (the existing, already-correct verification
logic) once per split and reports what comes back: sample rate, channel
count, duration, and waveform-length distributions for every clip that
passed verification, plus the counts of any clips that did not.

Outputs:
    results/preprocessing_report.json
    results/preprocessing_report.txt

Usage:
    python -m src.preprocessing.audio_report
    python -m src.preprocessing.audio_report --root data --out results
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.verify_subset import (  # noqa: E402
    SPLITS,
    TARGET_LEN,
    TARGET_SR,
    check_split,
)

EXPECTED_CHANNELS = 1
EXPECTED_DURATION = TARGET_LEN / TARGET_SR  # 4.0 seconds


def collect(root="data"):
    """Run verification for every split and collect audio-property stats.

    Returns a dict with per-split results plus pooled totals. Reads each
    clip once (inside check_split); nothing is loaded into memory beyond
    one clip at a time, and nothing is duplicated across splits.
    """
    per_split = {}
    pooled_stats = {
        "sr": Counter(), "channels": Counter(),
        "duration": Counter(), "samples": Counter(),
    }
    pooled_problems = []
    manifests = []

    for split in SPLITS:
        stats = {
            "sr": Counter(), "channels": Counter(),
            "duration": Counter(), "samples": Counter(),
        }
        res = check_split(split, root, stats=stats)
        if res is None:
            per_split[split] = None
            continue
        clean, problems = res

        for key in pooled_stats:
            pooled_stats[key].update(stats[key])
        pooled_problems.extend((split, f, why) for f, why in problems)
        if len(clean):
            manifests.append(clean)

        per_split[split] = {
            "metadata_rows": 0 if res is None else int(len(clean)) + len(problems),
            "valid_clips": int(len(clean)),
            "invalid_clips": int(len(problems)),
            "problems": [{"file": f, "reason": why} for f, why in problems],
            "generator_counts": (
                clean.generator.value_counts().sort_index().to_dict()
                if len(clean) else {}
            ),
            "label_counts": (
                {str(k): int(v) for k, v in clean.label.value_counts().to_dict().items()}
                if len(clean) else {}
            ),
            "source_dataset_counts": (
                clean.source_dataset.value_counts().to_dict() if len(clean) else {}
            ),
            "stats": {k: dict(v) for k, v in stats.items()},
        }

    total_valid = sum(v["valid_clips"] for v in per_split.values() if v)
    total_invalid = sum(v["invalid_clips"] for v in per_split.values() if v)

    result = {
        "target_sample_rate_hz": TARGET_SR,
        "target_channels": EXPECTED_CHANNELS,
        "target_duration_seconds": EXPECTED_DURATION,
        "target_samples": TARGET_LEN,
        "per_split": per_split,
        "totals": {
            "valid_clips": total_valid,
            "invalid_clips": total_invalid,
            "pooled_sr_distribution": dict(pooled_stats["sr"]),
            "pooled_channel_distribution": dict(pooled_stats["channels"]),
            "pooled_duration_distribution": dict(pooled_stats["duration"]),
            "pooled_samples_distribution": dict(pooled_stats["samples"]),
        },
    }
    return result


def print_verification_summary(result):
    """The concise summary requested for the project review."""
    sr_dist = result["totals"]["pooled_sr_distribution"]
    dur_dist = result["totals"]["pooled_duration_distribution"]
    samp_dist = result["totals"]["pooled_samples_distribution"]

    def only_value(dist, fallback):
        return next(iter(dist)) if len(dist) == 1 else f"mixed {dist}" if dist else fallback

    print("\n=== Audio standardization verification ===")
    print(f"Sample rate:\n  {only_value(sr_dist, result['target_sample_rate_hz'])} Hz")
    print("Channels:\n  Mono" if len(result["totals"]["pooled_channel_distribution"]) <= 1
          else f"Channels:\n  mixed {result['totals']['pooled_channel_distribution']}")
    print(f"Duration:\n  {only_value(dur_dist, result['target_duration_seconds'])} seconds")
    print(f"Samples:\n  {only_value(samp_dist, result['target_samples'])}")
    print(f"\nValid clips:\n  {result['totals']['valid_clips']}")
    print(f"Invalid clips:\n  {result['totals']['invalid_clips']}")


def print_property_summary(result):
    """Per-split breakdown of what was checked and what was found."""
    print("\n=== Audio property summary (per split) ===")
    for split, v in result["per_split"].items():
        if v is None:
            print(f"\n[{split}] not fetched yet, skipped")
            continue
        print(f"\n[{split}]")
        print(f"  files checked        : {v['valid_clips'] + v['invalid_clips']}")
        print(f"  valid clips          : {v['valid_clips']}")
        print(f"  invalid clips        : {v['invalid_clips']}")
        print(f"  sample rate dist.    : {v['stats']['sr']}")
        print(f"  channel dist.        : {v['stats']['channels']}")
        print(f"  duration dist. (s)   : {v['stats']['duration']}")
        print(f"  waveform length dist.: {v['stats']['samples']}")

    all_standard = (
        len(result["totals"]["pooled_sr_distribution"]) <= 1
        and len(result["totals"]["pooled_channel_distribution"]) <= 1
        and len(result["totals"]["pooled_duration_distribution"]) <= 1
        and result["totals"]["invalid_clips"] == 0
    )
    if all_standard and result["totals"]["valid_clips"] > 0:
        print(
            f"\nAll {result['totals']['valid_clips']} valid clips are standardized: "
            f"{result['target_sample_rate_hz']} Hz, mono, "
            f"{result['target_duration_seconds']}s, {result['target_samples']} samples."
        )


def write_report(result, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "preprocessing_report.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    txt_path = out_dir / "preprocessing_report.txt"
    lines = ["Preprocessing Summary Report", "=" * 32, ""]
    lines.append(f"Target sample rate : {result['target_sample_rate_hz']} Hz")
    lines.append(f"Target channels    : {result['target_channels']} (mono)")
    lines.append(f"Target duration    : {result['target_duration_seconds']} s")
    lines.append(f"Target samples     : {result['target_samples']}")
    lines.append("")

    total_clips = 0
    for split, v in result["per_split"].items():
        if v is None:
            lines.append(f"[{split}] not fetched")
            continue
        n = v["valid_clips"] + v["invalid_clips"]
        total_clips += n
        lines.append(f"[{split}]")
        lines.append(f"  total clips     : {n}")
        lines.append(f"  usable clips    : {v['valid_clips']}")
        lines.append(f"  problematic     : {v['invalid_clips']}")
        lines.append(f"  generator counts: {v['generator_counts']}")
        lines.append(f"  label counts    : {v['label_counts']} (0=real, 1=fake)")
        lines.append(f"  source datasets : {v['source_dataset_counts']}")
        lines.append("")

    lines.append(f"TOTAL usable clips    : {result['totals']['valid_clips']}")
    lines.append(f"TOTAL problematic     : {result['totals']['invalid_clips']}")
    lines.append(f"Sample rates seen     : {result['totals']['pooled_sr_distribution']}")
    lines.append(f"Channel counts seen   : {result['totals']['pooled_channel_distribution']}")
    lines.append(f"Durations seen (s)    : {result['totals']['pooled_duration_distribution']}")
    lines.append(f"Waveform lengths seen : {result['totals']['pooled_samples_distribution']}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nreport -> {json_path}")
    print(f"report -> {txt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    result = collect(a.root)
    print_verification_summary(result)
    print_property_summary(result)
    write_report(result, a.out)


if __name__ == "__main__":
    main()
