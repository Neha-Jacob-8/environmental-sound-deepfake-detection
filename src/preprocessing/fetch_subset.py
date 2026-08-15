"""Fetch an EnvSDD subset clip-by-clip via the HF datasets-server rows endpoint.

Why not `load_dataset(streaming=True)`: the repo is stored as 398 MB parquet
shards, and parquet reads whole row groups, so streaming pulls ~80 MB before it
yields a single row. On a slow link that stalls or times out outright.

The rows endpoint instead serves each clip as its own ~128 KB WAV, already at
16 kHz / 64,000 samples / 4.000 s. That turns one impossible 398 MB transfer
into many small resumable ones.

Source grouping
---------------
Rows are laid out in fixed blocks, one block per source recording:
    train/validation: 5 rows  (1 real + G01-G04)   -> source_id = row_idx // 5
    test:             8 rows  (1 real + G01-G07)   -> source_id = row_idx // 8
Verified by arithmetic: 139055/5=27811, 39710/5=7942, 39768/8=4971.

We therefore sample whole *source groups*, never individual rows. This keeps a
real clip and all its derived fakes together and yields a perfectly balanced
subset for free (N groups -> N reals and N clips per generator).

Usage:
    python -m src.preprocessing.fetch_subset --split test  --groups 40 --dry-run
    python -m src.preprocessing.fetch_subset --split train --groups 1200
    python -m src.preprocessing.fetch_subset --split validation --groups 300
    python -m src.preprocessing.fetch_subset --split test  --groups 300

Re-running resumes: clips already on disk are skipped.
"""

import argparse
import csv
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.generators import resolve_generator  # noqa: E402

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "EnvSDD/EnvSDD"
BLOCK = {"train": 5, "validation": 5, "test": 8}
MAX_ROWS_PER_REQUEST = 100


# Verified against the HF size endpoint and confirmed by exact block arithmetic
# (139055/5=27811, 39710/5=7942, 39768/8=4971). Used as a fallback so a DNS blip
# at startup cannot abort a run that would otherwise succeed.
KNOWN_ROWS = {"train": 139055, "validation": 39710, "test": 39768}


def get_split_size(split, retries=3):
    """Row count for a split, falling back to the verified constant."""
    for attempt in range(retries):
        try:
            info = requests.get(
                "https://datasets-server.huggingface.co/size",
                params={"dataset": DATASET, "config": "default"}, timeout=60,
            ).json()
            n = next(s["num_rows"] for s in info["size"]["splits"]
                     if s["split"] == split)
            if n != KNOWN_ROWS.get(split):
                print(f"note: {split} now has {n} rows "
                      f"(expected {KNOWN_ROWS.get(split)}) - dataset updated?",
                      flush=True)
            return n
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"  size lookup retry {attempt + 1}/{retries} in {wait}s: "
                      f"{type(e).__name__}", flush=True)
                time.sleep(wait)
    print(f"size endpoint unreachable, using known row count for {split}",
          flush=True)
    return KNOWN_ROWS[split]


def get_rows(split, offset, length, retries=6):
    """One page of rows. Signed audio URLs expire, so download promptly."""
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": split,
        "offset": offset,
        "length": length,
    }
    for attempt in range(retries):
        try:
            r = requests.get(ROWS_URL, params=params, timeout=90)
            if r.status_code == 429:  # rate limited
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["rows"]
        except Exception as e:
            if attempt == retries - 1:
                raise
            # Exponential backoff: a dropped link or DNS outage needs far longer
            # than a linear 3/6/9s ramp to recover. 5/10/20/40/80s ~= 2.5 min.
            wait = 5 * (2 ** attempt)
            print(f"    retry {attempt + 1}/{retries} at offset {offset} "
                  f"in {wait}s: {type(e).__name__}", flush=True)
            time.sleep(wait)
    return []


def download_clip(url, dest, retries=4):
    if dest.exists() and dest.stat().st_size > 1000:
        return "skip"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)  # atomic, so a kill mid-write cannot leave a bad wav
            return "ok"
        except Exception:
            if attempt == retries - 1:
                return "fail"
            time.sleep(2 * (attempt + 1))
    return "fail"


def plan_offsets(split, n_groups, n_chunks, total_rows, seed):
    """Pick contiguous runs of source groups spread across the whole split.

    Source datasets sit in contiguous regions of the index (train starts with
    UrbanSound8K and reaches TUTASC2019Dev by row 70k), so evenly spaced chunks
    give coverage of every source dataset while keeping requests contiguous and
    cheap. Fully random group picks would be one HTTP request per group.
    """
    block = BLOCK[split]
    total_groups = total_rows // block
    n_groups = min(n_groups, total_groups)
    n_chunks = max(1, min(n_chunks, n_groups))

    # Each chunk's run must fit inside its own region, otherwise neighbouring
    # runs collide. Shrink the chunk count until it does; requesting the whole
    # split collapses this to a single contiguous sweep, which is correct.
    while n_chunks > 1 and total_groups // n_chunks < -(-n_groups // n_chunks):
        n_chunks -= 1

    # Distribute the remainder so the plan delivers exactly n_groups rather than
    # silently truncating (25 groups over 10 chunks must be 25, not 20).
    base, extra = divmod(n_groups, n_chunks)
    sizes = [base + (1 if i < extra else 0) for i in range(n_chunks)]

    # Give each chunk its own non-overlapping region of the index and place its
    # run at a seeded random offset inside that region. Drawing starts
    # independently would let two runs overlap and collapse into fewer unique
    # source groups than requested.
    region = max(1, total_groups // n_chunks)
    groups_per_request = max(1, MAX_ROWS_PER_REQUEST // block)
    rng = random.Random(seed)

    requests_plan = []
    for i, size in enumerate(sizes):
        lo = i * region
        hi = max(lo, min(lo + region, total_groups) - size)
        g = rng.randint(lo, hi)
        remaining = size
        while remaining > 0:
            take = min(groups_per_request, remaining, total_groups - g)
            if take <= 0:
                break
            requests_plan.append((g * block, take * block))
            g += take
            remaining -= take
    return requests_plan


def build(split, n_groups, n_chunks, out_root, dry_run, seed, workers):
    block = BLOCK[split]
    audio_dir = Path(out_root) / "processed" / split
    meta_path = Path(out_root) / "metadata" / f"{split}_subset.csv"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = get_split_size(split)
    print(f"{split}: {total_rows} rows, block={block}, "
          f"{total_rows // block} source groups", flush=True)

    plan = plan_offsets(split, n_groups, n_chunks, total_rows, seed)
    print(f"plan: {len(plan)} requests covering "
          f"{sum(l for _, l in plan) // block} groups", flush=True)

    rows_out, stats = [], {"ok": 0, "skip": 0, "fail": 0}
    failed_pages = []
    t0 = time.time()

    for i, (offset, length) in enumerate(plan):
        try:
            page = get_rows(split, offset, length)
        except Exception as e:
            print(f"  page {offset} failed permanently: {type(e).__name__}",
                  flush=True)
            failed_pages.append(offset)
            continue

        jobs = []
        for r in page:
            row, idx = r["row"], r["row_idx"]
            gen = resolve_generator(row["attack_type"], row["generative_model"])
            source_id = idx // block
            stem = f"{split}_{source_id:06d}_{gen}"
            dest = audio_dir / f"{stem}.wav"
            meta_row = {
                "filename": f"{stem}.wav",
                "label": 0 if row["label"] == "real" else 1,
                "generator": gen,
                "source_id": source_id,
                "row_idx": idx,
                "attack_type": row["attack_type"],
                "generative_model": row["generative_model"],
                "source_dataset": row["dataset"],
                "orig_filename": row["filename"],
                "split": split,
            }
            if dry_run:
                rows_out.append(meta_row)
            else:
                jobs.append((row["audio"][0]["src"], dest, meta_row))

        if jobs:
            # Only record a clip in the metadata once its audio is actually on
            # disk. Writing rows for failed downloads makes the CSV claim files
            # that do not exist.
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(lambda j: download_clip(j[0], j[1]), jobs))
            for (_, _, meta_row), res in zip(jobs, results):
                stats[res] += 1
                if res in ("ok", "skip"):
                    rows_out.append(meta_row)

        done = sum(l for _, l in plan[: i + 1])
        el = time.time() - t0
        print(f"  [{i + 1}/{len(plan)}] rows={done} "
              f"ok={stats['ok']} skip={stats['skip']} fail={stats['fail']} "
              f"{el:.0f}s", flush=True)

    # Sanity check: every complete block must contain exactly one real clip.
    from collections import Counter
    per_group = Counter()
    for r in rows_out:
        if r["label"] == 0:
            per_group[r["source_id"]] += 1
    bad = [g for g, c in per_group.items() if c != 1]
    if bad:
        print(f"WARNING: {len(bad)} source groups without exactly one real clip "
              f"- block assumption may be wrong for this split", flush=True)

    if rows_out:
        with open(meta_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    gen_counts = Counter(r["generator"] for r in rows_out)
    src_counts = Counter(r["source_dataset"] for r in rows_out)
    print(f"\n--- {split} ---", flush=True)
    print(f"rows: {len(rows_out)}  groups: {len(set(r['source_id'] for r in rows_out))}",
          flush=True)
    print(f"per generator: {dict(sorted(gen_counts.items()))}", flush=True)
    print(f"per source dataset: {dict(src_counts.most_common())}", flush=True)
    print(f"downloads: {stats}", flush=True)
    print(f"metadata -> {meta_path}", flush=True)

    requested = sum(l for _, l in plan) // block
    got = len(set(r["source_id"] for r in rows_out))
    if failed_pages or stats["fail"] or got < requested:
        print(f"\nINCOMPLETE: {got}/{requested} source groups, "
              f"{len(failed_pages)} metadata pages and {stats['fail']} clips "
              f"failed. Re-run the same command to fill the gaps "
              f"(the plan is seeded, so it retries exactly these).", flush=True)
    else:
        print(f"\nCOMPLETE: {got}/{requested} source groups.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True,
                   choices=["train", "validation", "test"])
    p.add_argument("--groups", type=int, default=200,
                   help="number of source recordings to fetch")
    p.add_argument("--chunks", type=int, default=12,
                   help="spread groups over this many regions of the split")
    p.add_argument("--out", default="data")
    p.add_argument("--dry-run", action="store_true",
                   help="metadata only, download no audio")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--workers", type=int, default=8,
                   help="parallel clip downloads")
    a = p.parse_args()
    build(a.split, a.groups, a.chunks, a.out, a.dry_run, a.seed, a.workers)


if __name__ == "__main__":
    main()
