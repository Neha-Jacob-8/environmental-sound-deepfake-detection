"""Drop the frozen pretrained frontend from checkpoints that embed it.

Level 3 and fusion checkpoints are ~363 MB each, and essentially all of that is
torchaudio's WAV2VEC2_BASE weights. Those weights are frozen, identical in every
checkpoint, deterministic, and already cached by torchaudio - `load_frontend()`
fetches them when the model is constructed, so storing them again in each
checkpoint buys nothing.

Stripping them takes four checkpoints from ~1.45 GB to a few MB. The trainable
parameters - the only thing a training run actually produced - are untouched.

    python -m src.analysis.slim_checkpoints --dry-run
    python -m src.analysis.slim_checkpoints
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Every parameter reachable through a frozen pretrained frontend.
# Note the ".model." - AASIST's own `frontend` is a LEARNABLE Conv1d whose keys
# are frontend.weight / frontend.bias. Only the wav2vec2 wrapper nests a .model
# submodule, so this prefix matches the frozen weights and nothing else.
FROZEN_PREFIXES = ("frontend.model.", "beats_aasist_branch.frontend.model.")


def is_frozen(key):
    return key.startswith(FROZEN_PREFIXES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/models")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    total_before = total_after = 0
    for p in sorted(Path(a.dir).glob("*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        key = "state_dict" if "state_dict" in ck else "model_state"
        sd = ck.get(key)
        if sd is None:
            continue
        frozen = [k for k in sd if is_frozen(k)]
        before = p.stat().st_size
        total_before += before
        if not frozen:
            total_after += before
            print(f"  {p.name:30s} {before/1e6:7.1f} MB   nothing frozen to drop")
            continue

        n = sum(sd[k].numel() for k in frozen)
        if a.dry_run:
            print(f"  {p.name:30s} {before/1e6:7.1f} MB   would drop "
                  f"{len(frozen)} tensors / {n/1e6:.1f}M params")
            total_after += before - n * 4
            continue

        for k in frozen:
            del sd[k]
        ck[key] = sd
        ck["frontend_stripped"] = True
        torch.save(ck, p)
        after = p.stat().st_size
        total_after += after
        print(f"  {p.name:30s} {before/1e6:7.1f} MB -> {after/1e6:5.1f} MB   "
              f"dropped {n/1e6:.1f}M frozen params")

    print(f"\n  {'would be' if a.dry_run else 'now'}: "
          f"{total_before/1e9:.2f} GB -> {total_after/1e9:.2f} GB "
          f"(saves {(total_before-total_after)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
