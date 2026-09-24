"""Score every trained checkpoint on the test split, side by side.

    python -m src.evaluation.compare
    python -m src.evaluation.compare --checkpoints "results/models/*_best.pt"

The comparison that matters is not which model has the lowest EER overall - it
is which one loses the least when the generator changes. A model that wins on
seen generators and collapses on unseen ones is worse, for this project's
question, than one that is mediocre on both. So the table is sorted by the
generalisation gap, and the pooled seen/unseen columns sit next to it.
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader              # noqa: E402
from src.evaluation.metrics import (                             # noqa: E402
    per_generator_eer,
    seen_unseen_summary,
)
from src.models import LEVEL, input_mode, load_checkpoint        # noqa: E402
from src.preprocessing.generators import (                       # noqa: E402
    ALL_GENERATORS,
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)
from src.training.train import pick_device, score_loader         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", default="results/models/*_best.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/tables/model_comparison.csv")
    a = p.parse_args()

    paths = sorted(glob.glob(a.checkpoints))
    if not paths:
        print(f"no checkpoints matched {a.checkpoints!r}. Train something first:")
        print("    python -m src.training.train --model logmel_cnn")
        sys.exit(1)

    device = pick_device(a.device)
    rows, per_gen_rows = [], []

    for path in paths:
        try:
            model, ck = load_checkpoint(path, device)
        except Exception as e:
            print(f"skipping {path}: {type(e).__name__}: {e}")
            continue

        name = ck["model"]
        mode = ck.get("input_mode") or input_mode(name)
        loader = make_loader(a.split, mode=mode, batch_size=a.batch_size,
                             shuffle=False, num_workers=a.num_workers,
                             normalize=ck["normalize"])
        scores, labels, _ = score_loader(model, loader, device)
        gens = loader.dataset.df.generator.to_numpy()

        summary = seen_unseen_summary(labels, scores, gens,
                                      SEEN_GENERATORS, UNSEEN_GENERATORS)
        per_gen = per_generator_eer(labels, scores, gens)

        row = {
            "model": name,
            "level": LEVEL.get(name, "?"),
            "input": mode,
            "normalized": ck["normalize"],
            "params": sum(q.numel() for q in model.parameters() if q.requires_grad),
            "val_eer": ck.get("val_eer", float("nan")),
            "seen_eer": summary["seen"]["eer"],
            "unseen_eer": summary["unseen"]["eer"],
            "gap": summary["gap"],
            "test_auc": summary["unseen"]["auc"],
            "checkpoint": path,
        }
        row.update({f"eer_{g}": per_gen[g]["eer"] for g in per_gen})
        rows.append(row)
        per_gen_rows.extend(
            {"model": name, "generator": g,
             "seen": g in SEEN_GENERATORS, **m}
            for g, m in sorted(per_gen.items())
        )
        print(f"scored {name:14s} seen {row['seen_eer']:.4f}  "
              f"unseen {row['unseen_eer']:.4f}  gap {row['gap']:+.4f}")

    df = pd.DataFrame(rows).sort_values("gap").reset_index(drop=True)

    cols = ["model", "level", "input", "normalized", "params", "val_eer",
            "seen_eer", "unseen_eer", "gap"]
    print(f"\n=== test split, sorted by generalisation gap (smaller is better) ===")
    print(df[cols].to_string(
        index=False,
        formatters={c: "{:.4f}".format
                    for c in ("val_eer", "seen_eer", "unseen_eer", "gap")}
                   | {"params": "{:,}".format}))

    gen_cols = [f"eer_{g}" for g in ALL_GENERATORS if f"eer_{g}" in df.columns]
    if gen_cols:
        print("\n=== per-generator EER ===")
        print(df.set_index("model")[gen_cols].to_string(
            formatters={c: "{:.4f}".format for c in gen_cols}))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    long_out = Path(a.out).with_name("model_comparison_per_generator.csv")
    pd.DataFrame(per_gen_rows).to_csv(long_out, index=False)
    print(f"\nwritten -> {a.out}")
    print(f"written -> {long_out}")


if __name__ == "__main__":
    main()
