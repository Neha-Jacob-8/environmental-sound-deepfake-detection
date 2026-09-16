"""Full metrics report for a trained detector: train, validation and test.

    python -m src.evaluation.report

Threshold policy
----------------
Accuracy, precision, recall and F1 all need a decision threshold; EER and AUC do
not. The threshold used here is the EER point **of the validation split**, then
applied unchanged to train and test. Picking it on test instead would tune the
operating point on the data being reported, which inflates every thresholded
number.

Read accuracy with care: fakes outnumber reals 4:1 in train/validation and 7:1
in test, so a model that answers "fake" every time already scores 80% / 87.5%.
Balanced accuracy and EER are the numbers that survive that imbalance.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader          # noqa: E402
from src.evaluation.metrics import eer                       # noqa: E402
from src.models.cnn import LogMelCNN                         # noqa: E402
from src.preprocessing.generators import (                   # noqa: E402
    GENERATOR_NAMES,
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)
from src.training.train import pick_device, score_loader     # noqa: E402


def row(y, s, thr):
    pred = (s >= thr).astype(int)
    e, _ = eer(y, s)
    return {
        "n": len(y), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "accuracy": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan"),
        "eer": e,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/models/cnn_best.pt")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default="results/tables/level1_cnn_report.csv")
    a = p.parse_args()

    dev = pick_device()
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model = LogMelCNN(**ck.get("model_kwargs", {}))
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    print(f"checkpoint: epoch {ck['epoch']}, val EER {ck['val_eer']:.4f}\n")

    data = {}
    for split in ("train", "validation", "test"):
        ld = make_loader(split, mode="logmel", batch_size=64, shuffle=False,
                         num_workers=a.num_workers)
        s, y, _ = score_loader(model, ld, dev)
        data[split] = (y, s, ld.dataset.df.generator.to_numpy())

    # Threshold fixed on validation, then reused everywhere.
    thr = eer(*data["validation"][:2])[1]
    print(f"decision threshold (validation EER point): {thr:.4f}\n")

    rows = {sp: row(y, s, thr) for sp, (y, s, _) in data.items()}
    tab = pd.DataFrame(rows).T
    cols = ["n", "n_real", "n_fake", "accuracy", "bal_acc", "precision",
            "recall", "f1", "auc", "eer"]
    print("=== overall ===")
    print(tab[cols].to_string(
        formatters={c: "{:.4f}".format for c in cols[3:]}))

    print("\n=== confusion matrices (rows = true real/fake, cols = pred) ===")
    for sp, (y, s, _) in data.items():
        cm = confusion_matrix(y, (s >= thr).astype(int))
        print(f"\n{sp}:\n{cm}")

    print("\n=== test, per generator (each vs the same 300 real clips) ===")
    y, s, g = data["test"]
    real = y == 0
    per = []
    for gen in sorted(set(g[y == 1])):
        sel = real | (g == gen)
        r = row(y[sel], s[sel], thr)
        r.update(generator=gen, name=GENERATOR_NAMES.get(gen, ""),
                 seen="seen" if gen in SEEN_GENERATORS else "UNSEEN")
        per.append(r)
    pt = pd.DataFrame(per).set_index("generator")
    print(pt[["name", "seen", "accuracy", "bal_acc", "recall", "f1", "auc", "eer"]]
          .to_string(formatters={c: "{:.4f}".format
                                 for c in ["accuracy", "bal_acc", "recall", "f1", "auc", "eer"]}))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([tab.assign(scope="split"), pt.assign(scope="generator")])
    out.to_csv(a.out)
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
