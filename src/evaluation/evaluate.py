"""Evaluate a trained detector on the test split - the core result.

    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --checkpoint results/models/cnn_best.pt

Reports EER per generator, then pooled over seen (G01-G04) and unseen (G05-G07)
generators, and the gap between them. The gap is the headline number.

Every generator is scored against the *same* pool of real clips from the test
split, so the only thing that changes between rows is the fake side. Both the
seen and unseen pools also draw from the same test split, so they share a source
-domain mix - without that, an apparent generalisation failure could just be the
test split's unseen source domains (Clotho, DCASE2023Task7) showing up.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader              # noqa: E402
from src.evaluation.metrics import (                             # noqa: E402
    per_generator_eer,
    seen_unseen_summary,
)
from src.models import input_mode, load_checkpoint               # noqa: E402
from src.preprocessing.generators import (                       # noqa: E402
    GENERATOR_NAMES,
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)
from src.training.train import pick_device, score_loader         # noqa: E402


def load_model(checkpoint, device):
    model, ck = load_checkpoint(checkpoint, device)
    print(f"loaded {checkpoint}  model={ck['model']}  epoch {ck.get('epoch')}  "
          f"val EER {ck.get('val_eer', float('nan')):.4f}")
    return model, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/models/logmel_cnn_best.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None,
                   help="default results/tables/<model>_test_per_generator.csv")
    a = p.parse_args()

    device = pick_device(a.device)
    model, ck = load_model(a.checkpoint, device)
    a.out = a.out or f"results/tables/{ck['model']}_test_per_generator.csv"

    mode = ck.get("input_mode") or input_mode(ck["model"])
    loader = make_loader(a.split, mode=mode, batch_size=a.batch_size,
                         shuffle=False, num_workers=a.num_workers)
    print(loader.dataset.describe())

    scores, labels, _ = score_loader(model, loader, device)
    gens = loader.dataset.df.generator.to_numpy()
    assert len(gens) == len(scores), "loader order does not match the manifest"

    per_gen = per_generator_eer(labels, scores, gens)
    summary = seen_unseen_summary(labels, scores, gens,
                                  SEEN_GENERATORS, UNSEEN_GENERATORS)

    rows = []
    print(f"\n{'gen':5s} {'generator':22s} {'seen?':7s} "
          f"{'EER':>8s} {'AUC':>8s} {'F1':>8s} {'n_fake':>7s}")
    for g in sorted(per_gen):
        m = per_gen[g]
        seen = "seen" if g in SEEN_GENERATORS else "UNSEEN"
        print(f"{g:5s} {GENERATOR_NAMES.get(g, ''):22s} {seen:7s} "
              f"{m['eer']:8.4f} {m['auc']:8.4f} {m['f1']:8.4f} {m['n_fake']:7d}")
        rows.append({"generator": g, "name": GENERATOR_NAMES.get(g, ""),
                     "seen": g in SEEN_GENERATORS, **m})

    print(f"\n{'pooled':28s} {'EER':>8s} {'AUC':>8s}")
    for k in ("seen", "unseen"):
        m = summary[k]
        print(f"{k + ' (' + ('G01-G04' if k == 'seen' else 'G05-G07') + ')':28s} "
              f"{m['eer']:8.4f} {m['auc']:8.4f}")
        rows.append({"generator": f"POOLED_{k.upper()}", "name": k,
                     "seen": k == "seen", **m})
    print(f"{'generalisation gap':28s} {summary['gap']:+8.4f}"
          "   <- unseen EER minus seen EER")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
