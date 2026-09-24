"""Fast sanity check: one training step and one eval step for every model.

    python -m src.training.smoke_test
    python -m src.training.smoke_test --models aasist fusion

Catches shape and wiring bugs in seconds, before committing to a real run. It
uses whatever is in data/processed/ and reports nothing about accuracy - two
batches cannot say anything about that.

`beats_aasist` and `fusion` download torchaudio's pretrained wav2vec2
checkpoint (~360 MB) the first time they run; after that it is cached.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader            # noqa: E402
from src.evaluation.metrics import all_metrics                 # noqa: E402
from src.models import LEVEL, MODEL_NAMES, build_model, input_mode  # noqa: E402
from src.training.train import pick_device, score_loader       # noqa: E402


def run_one(name, device, batch_size):
    print(f"\n--- {name}  (level {LEVEL[name]}) ---", flush=True)
    t0 = time.time()

    mode = input_mode(name)
    train_loader = make_loader("train", mode=mode, batch_size=batch_size,
                               num_workers=0)
    val_loader = make_loader("validation", mode=mode, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    model = build_model(name).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"input mode      : {mode}")
    print(f"trainable params: {trainable:,}" +
          (f"   frozen: {frozen:,}" if frozen else ""))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # one training step
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    logits = model(x)
    assert logits.shape == y.shape, \
        f"{name}: logits {tuple(logits.shape)} != labels {tuple(y.shape)}"
    loss = criterion(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    print(f"input {tuple(x.shape)} -> logits {tuple(logits.shape)}  "
          f"loss {loss.item():.4f}")

    # one eval pass over a couple of batches
    scores, labels, _ = score_loader(model, val_loader, device)
    m = all_metrics(labels, scores)
    print(f"eval on {len(labels)} val clips: EER {m['eer']:.4f} "
          f"(meaningless this early - shape check only)")
    print(f"ok in {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=MODEL_NAMES,
                   choices=MODEL_NAMES)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = pick_device(a.device)
    print(f"device: {device}")

    failed = []
    for name in a.models:
        try:
            run_one(name, device, a.batch_size)
        except Exception:
            failed.append(name)
            print(f"FAILED: {name}")
            traceback.print_exc()

    print("\n" + "=" * 52)
    if failed:
        print(f"{len(failed)} of {len(a.models)} models FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"all {len(a.models)} models passed")


if __name__ == "__main__":
    main()
