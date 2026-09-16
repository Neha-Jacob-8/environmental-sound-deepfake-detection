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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader          # noqa: E402
from src.evaluation.metrics import eer                       # noqa: E402
from src.models.cnn import LogMelCNN                         # noqa: E402
from src.preprocessing.generators import (                   # noqa: E402
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)
from src.training.train import pick_device, score_loader     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/models/cnn_best.pt")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = pick_device()
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model = LogMelCNN(**ck.get("model_kwargs", {}))
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    loader = make_loader("test", mode="logmel", batch_size=64, shuffle=False,
                         num_workers=a.num_workers)
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


if __name__ == "__main__":
    main()
