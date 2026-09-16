"""Train the Level 1 log-Mel CNN.

    python -m src.training.train                      # defaults
    python -m src.training.train --epochs 60 --lr 3e-4

Model selection is on **validation EER**, not validation loss. Loss is dominated
by the easy majority of clips, while EER is the metric the result is reported in,
and the two do not peak at the same epoch.

A caveat that shapes how the result must be read: the validation split contains
only G01-G04, the same generators as training. It therefore cannot be used to
select for generalisation - doing so would tune on the very thing the test split
is meant to measure. Validation here answers "has this model finished learning
the seen generators", nothing more.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader          # noqa: E402
from src.evaluation.metrics import all_metrics               # noqa: E402
from src.models.cnn import LogMelCNN                         # noqa: E402


def pick_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def score_loader(model, loader, device, criterion=None):
    """Fake-ness scores and labels for every clip in `loader`."""
    model.eval()
    scores, labels, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            losses.append(criterion(logits, y).item() * len(y))
        scores.append(logits.detach().float().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    loss = sum(losses) / len(labels) if losses else float("nan")
    return scores, labels, loss


def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8,
                   help="stop after this many epochs with no val-EER improvement")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--out", default="results/models/cnn_best.pt")
    p.add_argument("--history", default="results/tables/cnn_history.csv")
    a = p.parse_args()

    set_seed(a.seed)
    device = pick_device(a.device)

    train_loader = make_loader("train", mode="logmel", batch_size=a.batch_size,
                               num_workers=a.num_workers)
    val_loader = make_loader("validation", mode="logmel", batch_size=a.batch_size,
                             shuffle=False, num_workers=a.num_workers)

    print(train_loader.dataset.describe())
    print(val_loader.dataset.describe())

    model = LogMelCNN(dropout=a.dropout, head_dropout=a.head_dropout).to(device)
    pos_weight = train_loader.dataset.class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr,
                           weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3)

    print(f"\ndevice={device}  params={model.n_params():,}  "
          f"pos_weight={pos_weight.item():.3f}  lr={a.lr}")
    print(f"{'ep':>3} {'train_loss':>11} {'val_loss':>9} {'val_eer':>8} "
          f"{'val_auc':>8} {'lr':>8} {'sec':>6}")

    best_eer, best_epoch, history = float("inf"), -1, []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.history).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, criterion, device)
        scores, labels, val_loss = score_loader(model, val_loader, device, criterion)
        m = all_metrics(labels, scores)
        sched.step(val_loss)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"{ep:3d} {tr_loss:11.4f} {val_loss:9.4f} {m['eer']:8.4f} "
              f"{m['auc']:8.4f} {lr_now:8.2e} {dt:6.1f}"
              + ("  *" if m["eer"] < best_eer else ""))

        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss,
                        "val_eer": m["eer"], "val_auc": m["auc"],
                        "val_f1": m["f1"], "lr": lr_now, "seconds": dt})

        if m["eer"] < best_eer:
            best_eer, best_epoch = m["eer"], ep
            torch.save({
                "state_dict": model.state_dict(),
                "model": "LogMelCNN",
                "model_kwargs": {"dropout": a.dropout,
                                 "head_dropout": a.head_dropout},
                "epoch": ep, "val_eer": best_eer, "args": vars(a),
            }, a.out)
        elif ep - best_epoch >= a.patience:
            print(f"\nearly stop: no val-EER improvement in {a.patience} epochs")
            break

    pd.DataFrame(history).to_csv(a.history, index=False)
    print(f"\nbest val EER {best_eer:.4f} at epoch {best_epoch}")
    print(f"checkpoint -> {a.out}")
    print(f"history    -> {a.history}")
    print("\nNow evaluate on the test split (the only split with G05-G07):")
    print("    python -m src.evaluation.evaluate")


if __name__ == "__main__":
    main()
