"""Train a model over several seeds, with and without augmentation.

Two questions at once, because they are the same experiment.

**Is a single run trustworthy?** AASIST's validation EER swings 0.098 between
adjacent epochs, which is most of its entire seen-unseen gap. Any number from
one run is suspect at that noise level, so every result here is mean +/- std
over seeds rather than a single figure.

**Does augmentation narrow the gap?** That is the project's remaining novelty.
Augmentation is applied to the training split only, so the comparison is like
for like, and the number that matters is the gap - not the seen EER, which
augmentation may well cost something.

    python -m src.analysis.seed_sweep --seeds 1 2 3
    python -m src.analysis.seed_sweep --model cnn --epochs 20
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader              # noqa: E402
from src.evaluation.metrics import per_generator_eer             # noqa: E402
from src.evaluation.metrics import seen_unseen_summary           # noqa: E402
from src.models import input_mode, load_checkpoint               # noqa: E402
from src.preprocessing.generators import (                       # noqa: E402
    ALL_GENERATORS,
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)
from src.training.train import pick_device, score_loader         # noqa: E402


def score(ckpt, device):
    model, ck = load_checkpoint(ckpt, device)
    mode = ck.get("input_mode") or input_mode(ck["model"])
    loader = make_loader("test", mode=mode, batch_size=64, shuffle=False,
                         num_workers=0, normalize=ck["normalize"])
    s, y, _ = score_loader(model, loader, device)
    gens = loader.dataset.df.generator.to_numpy()
    summ = seen_unseen_summary(y, s, gens, SEEN_GENERATORS, UNSEEN_GENERATORS)
    per = per_generator_eer(y, s, gens)
    return summ, per, ck.get("val_eer", float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logmel_cnn")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--augment-p", type=float, default=0.5)
    ap.add_argument("--out", default="results/tables/seed_sweep.csv")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    device = pick_device()
    Path("results/models").mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    for condition in ("baseline", "augmented"):
        for seed in a.seeds:
            tag = f"sweep_{a.model}_{condition}_s{seed}"
            ckpt = Path(f"results/models/{tag}_best.pt")
            if not (a.skip_existing and ckpt.exists()):
                cmd = [sys.executable, "-u", "-m", "src.training.train",
                       "--model", a.model, "--epochs", str(a.epochs),
                       "--seed", str(seed), "--num-workers", "4",
                       "--out", str(ckpt),
                       "--history", f"results/tables/{tag}_history.csv"]
                if condition == "augmented":
                    cmd += ["--augment", "--augment-p", str(a.augment_p)]
                print(f"\n=== {condition} seed {seed} "
                      f"({time.time() - t0:.0f}s elapsed) ===", flush=True)
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout[-1500:], r.stderr[-1500:])
                    sys.exit(f"training failed: {condition} seed {seed}")
                for line in r.stdout.splitlines():
                    if "best val EER" in line:
                        print("   ", line.strip())

            summ, per, val = score(ckpt, device)
            row = {"model": a.model, "condition": condition, "seed": seed,
                   "val_eer": val, "seen_eer": summ["seen"]["eer"],
                   "unseen_eer": summ["unseen"]["eer"], "gap": summ["gap"]}
            row.update({f"eer_{g}": per[g]["eer"] for g in ALL_GENERATORS if g in per})
            rows.append(row)
            print(f"    seen {row['seen_eer']:.4f}  unseen {row['unseen_eer']:.4f}  "
                  f"gap {row['gap']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"\n=== {a.model}: mean +/- std over {len(a.seeds)} seeds ===\n")
    print(f"{'condition':11s} {'seen EER':>16s} {'unseen EER':>16s} {'gap':>16s}")
    agg = {}
    for c in ("baseline", "augmented"):
        d = df[df.condition == c]
        agg[c] = d
        cells = "".join(
            f"{d[k].mean():9.4f} +/-{d[k].std():.4f}" for k in
            ("seen_eer", "unseen_eer", "gap"))
        print(f"{c:11s} {cells}")

    b, g = agg["baseline"], agg["augmented"]
    d_gap = g.gap.mean() - b.gap.mean()
    pooled = np.sqrt((b.gap.std() ** 2 + g.gap.std() ** 2) / 2) or 1e-9
    print(f"\n  augmentation moved the gap by {d_gap:+.4f} "
          f"({abs(d_gap) / pooled:.1f}x the seed-to-seed spread)")
    print(f"  seen EER cost: {g.seen_eer.mean() - b.seen_eer.mean():+.4f}")
    verdict = ("narrowed the gap" if d_gap < 0 else "widened the gap")
    strong = abs(d_gap) > pooled
    print(f"  -> augmentation {verdict}"
          f"{'' if strong else ', but by less than the seed noise'}")
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
