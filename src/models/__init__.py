"""Model registry.

Five detectors across the four project levels. Every model takes ONE tensor
and returns raw logits of shape (B,) - BCEWithLogitsLoss is applied outside -
so `train.py` and the evaluation scripts are identical for all of them and
only `--model` changes.

What differs is the input each one wants, which is why every entry declares an
INPUT_MODE that `EnvSDDDataset(mode=...)` is built with:

| name           | level    | input             | source                        |
|----------------|----------|-------------------|-------------------------------|
| logmel_cnn     | 1        | (1, 64, 251)      | cnn.py                        |
| cnn            | 1        | (1, 64000)        | cnn_waveform.py               |
| aasist         | 2        | (1, 64000)        | aasist.py                     |
| beats_aasist   | 3        | (1, 64000)        | beats_aasist.py               |
| fusion         | proposed | (1, 64000)        | fusion.py                     |

Two Level 1 CNNs, deliberately:

  `logmel_cnn` takes the spectrogram from the Dataset, which standardises it
  per mel bin using statistics computed once over the train split. That is the
  baseline the README results were measured with.

  `cnn` takes the raw waveform and computes its own log-Mel internally,
  standardising each clip against its own mean and std. The fusion model needs
  this variant, because its CNN branch has to consume the same waveform the
  BEATs branch does.

The difference is not cosmetic: per-clip standardisation discards absolute
level, which may itself carry a generator cue, while per-bin standardisation
over the train split keeps it and flattens the ~35 dB tilt across mel bins
instead. Both are defensible; keeping them side by side makes the choice
measurable rather than assumed.
"""

import torch

from src.models.aasist import AASIST
from src.models.beats_aasist import BeatsAASIST
from src.models.cnn import LogMelCNN
from src.models.cnn_waveform import CNN
from src.models.fusion import Fusion

# Checkpoints written before the registry existed stored the class name.
_LEGACY_NAMES = {"LogMelCNN": "logmel_cnn", "CNN": "cnn"}

# Which EnvSDDDataset mode each model must be fed.
INPUT_MODE = {
    "logmel_cnn": "logmel",
    "cnn": "waveform",
    "aasist": "waveform",
    "beats_aasist": "waveform",
    "fusion": "waveform",
}

LEVEL = {
    "logmel_cnn": "1 (baseline)",
    "cnn": "1",
    "aasist": "2",
    "beats_aasist": "3",
    "fusion": "proposed",
}

_BUILDERS = {
    "logmel_cnn": LogMelCNN,
    "cnn": CNN,
    "aasist": AASIST,
    "beats_aasist": BeatsAASIST,
    "fusion": Fusion,
}

MODEL_NAMES = list(_BUILDERS)


def build_model(name, **kwargs):
    """Instantiate a model by registry name.

    Unknown names raise rather than falling back to a default, so a typo in
    --model surfaces immediately instead of silently training the wrong thing.
    """
    if name not in _BUILDERS:
        raise ValueError(
            f"unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}"
        )
    return _BUILDERS[name](**kwargs)


def load_checkpoint(path, device="cpu"):
    """Rebuild the model a checkpoint was written from, and return (model, ck).

    The checkpoint records which model wrote it, so every evaluation script
    works on any of the five without being told which one it is looking at.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    name = ck.get("model", "logmel_cnn")
    name = _LEGACY_NAMES.get(name, name)

    kwargs = dict(ck.get("model_kwargs", {}))
    # A fusion checkpoint records the branch checkpoints it was *seeded* from.
    # Those weights are already inside state_dict, so re-reading the files here
    # would be wasted work - and would fail outright if they have since moved.
    for k in ("cnn_checkpoint", "beats_aasist_checkpoint", "freeze_branches"):
        kwargs.pop(k, None)

    model = build_model(name, **kwargs)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    ck["model"] = name
    return model, ck


def input_mode(name):
    name = _LEGACY_NAMES.get(name, name)
    if name not in INPUT_MODE:
        raise ValueError(f"unknown model {name!r}")
    return INPUT_MODE[name]


__all__ = ["AASIST", "BeatsAASIST", "CNN", "Fusion", "LogMelCNN",
           "build_model", "input_mode", "load_checkpoint",
           "INPUT_MODE", "LEVEL", "MODEL_NAMES"]
