"""Train any of the project's detectors.

    python -m src.training.train                             # logmel_cnn
    python -m src.training.train --model aasist --epochs 20
    python -m src.training.train --model fusion --epochs 15 \
        --cnn_checkpoint results/models/cnn_best.pt \
        --beats_aasist_checkpoint results/models/beats_aasist_best.pt

Every model returns raw logits (B,), so the loop below is identical for all of
them; only `--model` changes. Each one declares the Dataset mode it needs (raw
waveform or log-Mel) in the registry, so the right input is built automatically
- see src/models/__init__.py.

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
from src.models import (                                     # noqa: E402
    LEVEL,
    MODEL_NAMES,
    build_model,
    input_mode,
)


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
    p.add_argument("--model", default="logmel_cnn", choices=MODEL_NAMES)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8,
                   help="stop after this many epochs with no val-EER improvement")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dropout", type=float, default=None,
                   help="default depends on the model")
    p.add_argument("--head-dropout", type=float, default=0.3,
                   help="logmel_cnn only")
    # Fusion initialises its two branches from already-trained checkpoints.
    # Both spellings accepted so run_training.sh works either way.
    p.add_argument("--cnn-checkpoint", "--cnn_checkpoint", dest="cnn_checkpoint",
                   default=None, help="fusion only: Level 1 branch weights")
    p.add_argument("--beats-aasist-checkpoint", "--beats_aasist_checkpoint",
                   dest="beats_aasist_checkpoint", default=None,
                   help="fusion only: Level 3 branch weights")
    p.add_argument("--freeze-branches", action="store_true",
                   help="fusion only: train just the joint head")
    p.add_argument("--out", default=None,
                   help="default results/models/<model>_best.pt")
    p.add_argument("--history", default=None,
                   help="default results/tables/<model>_history.csv")
    a = p.parse_args()

    a.out = a.out or f"results/models/{a.model}_best.pt"
    a.history = a.history or f"results/tables/{a.model}_history.csv"

    set_seed(a.seed)
    device = pick_device(a.device)

    # The registry says whether this model eats waveforms or spectrograms.
    mode = input_mode(a.model)
    train_loader = make_loader("train", mode=mode, batch_size=a.batch_size,
                               num_workers=a.num_workers)
    val_loader = make_loader("validation", mode=mode, batch_size=a.batch_size,
                             shuffle=False, num_workers=a.num_workers)

    print(f"model={a.model} (level {LEVEL[a.model]}), input mode={mode}")
    print(train_loader.dataset.describe())
    print(val_loader.dataset.describe())

    model_kwargs = {}
    if a.dropout is not None:
        model_kwargs["dropout"] = a.dropout
    if a.model == "logmel_cnn":
        model_kwargs.setdefault("dropout", 0.1)
        model_kwargs["head_dropout"] = a.head_dropout
    if a.model == "fusion":
        model_kwargs.update(cnn_checkpoint=a.cnn_checkpoint,
                            beats_aasist_checkpoint=a.beats_aasist_checkpoint,
                            freeze_branches=a.freeze_branches)

    model = build_model(a.model, **model_kwargs).to(device)
    pos_weight = train_loader.dataset.class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Only optimise what is actually trainable: the Level 3 frontend is frozen,
    # and fusion can freeze both branches.
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad],
                           lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3)

    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"\ndevice={device}  trainable params={n_params:,}  "
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
                "model": a.model,
                "model_kwargs": model_kwargs,
                "input_mode": mode,
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
    print(f"    python -m src.evaluation.evaluate --checkpoint {a.out}")


if __name__ == "__main__":
    main()
