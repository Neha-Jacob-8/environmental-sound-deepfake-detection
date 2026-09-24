"""Bootstrap confidence intervals for the seen/unseen EER gap.

Resamples whole **source groups**, not individual clips. Clips inside a group
come from one recording, so their scores are correlated; resampling clips would
treat them as independent and report intervals that are too narrow.

    python -m src.evaluation.bootstrap_ci --n-boot 2000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader          # noqa: E402
from src.evaluation.metrics import eer                       # noqa: E402
from src.models import input_mode, load_checkpoint           # noqa: E402
from src.preprocessing.generators import (                   # noqa: E402
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)
from src.training.train import pick_device, score_loader     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/models/logmel_cnn_best.pt")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/tables/bootstrap_ci.csv",
                   help="shared table; this model's row is replaced in place")
    a = p.parse_args()

    dev = pick_device()
    model, ck = load_checkpoint(a.checkpoint, dev)
    mode = ck.get("input_mode") or input_mode(ck["model"])
    print(f"model={ck['model']}, input mode={mode}")

    loader = make_loader("test", mode=mode, batch_size=64, shuffle=False,
                         num_workers=a.num_workers, normalize=ck["normalize"])
    scores, labels, _ = score_loader(model, loader, dev)
    df = loader.dataset.df
    gens, sid = df.generator.to_numpy(), df.source_id.to_numpy()

    seen_m = (labels == 0) | np.isin(gens, SEEN_GENERATORS)
    unseen_m = (labels == 0) | np.isin(gens, UNSEEN_GENERATORS)

    groups = np.unique(sid)
    by_group = {g: np.flatnonzero(sid == g) for g in groups}
    rng = np.random.default_rng(a.seed)

    def pooled(mask, idx):
        sel = idx[mask[idx]]
        return eer(labels[sel], scores[sel])[0]

    boot = np.empty((a.n_boot, 2))
    for b in range(a.n_boot):
        pick = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        boot[b] = pooled(seen_m, idx), pooled(unseen_m, idx)

    gap = boot[:, 1] - boot[:, 0]
    point_seen = pooled(seen_m, np.arange(len(labels)))
    point_unseen = pooled(unseen_m, np.arange(len(labels)))

    print(f"\ngroup-level bootstrap: {a.n_boot} resamples of "
          f"{len(groups)} test source groups\n")
    for name, point, col in (("seen   (G01-G04)", point_seen, boot[:, 0]),
                             ("unseen (G05-G07)", point_unseen, boot[:, 1])):
        lo, hi = np.percentile(col, [2.5, 97.5])
        print(f"{name:18s} EER {point:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    lo, hi = np.percentile(gap, [2.5, 97.5])
    print(f"{'GAP':18s}     {point_unseen - point_seen:+.4f}   "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"\nP(gap > 0) = {(gap > 0).mean():.4f}")

    # One shared table the API reads, with this model's row replaced rather
    # than appended, so re-running never leaves two rows for one model.
    s_lo, s_hi = np.percentile(boot[:, 0], [2.5, 97.5])
    u_lo, u_hi = np.percentile(boot[:, 1], [2.5, 97.5])
    row = {
        "model": ck["model"], "n_resamples": a.n_boot,
        "seen_eer": point_seen, "seen_lo": s_lo, "seen_hi": s_hi,
        "unseen_eer": point_unseen, "unseen_lo": u_lo, "unseen_hi": u_hi,
        "gap": point_unseen - point_seen, "gap_lo": lo, "gap_hi": hi,
        "p_gap_positive": float((gap > 0).mean()),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(df):
        df = df[df.model != ck["model"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
