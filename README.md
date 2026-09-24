# Robust Environmental Sound Deepfake Detection Against Unseen Audio Generators

Detecting AI-generated environmental sound, with the central question being whether
a detector generalises to **audio generators it never saw during training**.

Built on the [EnvSDD](https://huggingface.co/datasets/EnvSDD/EnvSDD) dataset
(Interspeech 2025, CC BY 4.0), following the ESDD 2026 challenge protocol.

## Research question

Can combining complementary acoustic representations improve robustness of
environmental sound deepfake detection against unseen audio generators?

Models are trained on generators **G01–G04** and evaluated on **G05–G07**.

## Generators

The dataset has no `G01`–`G07` column. Generator IDs are reconstructed from the
composite key `(attack_type, generative_model)` — see
[`src/preprocessing/generators.py`](src/preprocessing/generators.py).

| ID | Type | Generator | Split |
|----|------|-----------|-------|
| G01 | TTA | AudioLDM | train + test |
| G02 | TTA | AudioLDM 2 | train + test |
| G03 | TTA | AudioGen | train + test |
| G04 | ATA | AudioLDM | train + test |
| G05 | TTA | AudioLCM | **test only** |
| G06 | TTA | TangoFlux | **test only** |
| G07 | ATA | AudioLDM 2 | **test only** |

Note that G07 is AudioLDM 2 in audio-to-audio mode while G02 is the same model in
text-to-audio mode. Only **G05 and G06 are architecturally unseen**; G07 tests an
unseen conditioning mode for a seen architecture, and should be expected to be
easier. Reporting per-generator EER rather than a single pooled number makes this
visible.

## Dataset structure

Rows are laid out in fixed blocks, one block per source recording:

| Split | Rows | Block | Source groups |
|-------|------|-------|---------------|
| train | 139,055 | 5 (1 real + G01–G04) | 27,811 |
| validation | 39,710 | 5 | 7,942 |
| test | 39,768 | 8 (1 real + G01–G07) | 4,971 |

`source_id = row_idx // block_size`. Sampling operates on whole source groups, so a
real clip and all of its derived fakes always stay together — this is what prevents
data leakage, and it yields a perfectly balanced subset for free.

Source recordings sit in contiguous regions of the index, so sampling is spread
across the split via `--chunks`:

| Split | Layout |
|-------|--------|
| train | UrbanSound8K → TUTSED2016Dev → TUTASC2019Dev |
| validation | UrbanSound8K → TUTSED2016Dev → TUTSED2017Dev → TUTASC2019Dev |
| test | UrbanSound8K → TUTSED2017Dev → TUTASC2019Dev → DCASE2023Task7 → Clotho |

**Clotho and DCASE2023Task7 appear only in test.** The test set therefore carries
unseen *source domains* in addition to unseen *generators*. To attribute a result
to the generator alone, compute seen-EER on the test split's G01–G04 rows and
unseen-EER on its G05–G07 rows, so both sit on the same domain mix.

## Setup

```bash
pip install -r requirements.txt
```

## Preprocessing

Downloads clips individually (~128 KB each) via the HuggingFace datasets-server,
rather than pulling 500 MB parquet shards. Resumable — re-run any command to fill
gaps; clips already on disk are skipped.

```bash
# pilot, a few minutes
python3 -u -m src.preprocessing.fetch_subset --split train      --groups 40 --chunks 20
python3 -u -m src.preprocessing.fetch_subset --split validation --groups 20 --chunks 10
python3 -u -m src.preprocessing.fetch_subset --split test       --groups 20 --chunks 10

# full working subset (~9,900 clips, ~1.3 GB)
python3 -u -m src.preprocessing.fetch_subset --split train      --groups 1200 --chunks 20
python3 -u -m src.preprocessing.fetch_subset --split validation --groups 300  --chunks 10
python3 -u -m src.preprocessing.fetch_subset --split test       --groups 300  --chunks 10

# verify and build the manifest
python3 -m src.preprocessing.verify_subset
```

Or all of it at once:

```bash
./run_preprocessing.sh
TRAIN=40 VAL=20 TEST=20 ./run_preprocessing.sh    # smaller pilot
```

Or as a notebook — [`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb)
runs the same pipeline cell by cell, with a manifest summary, spectrograms of one
source group and a Dataset smoke test at the end. It detects Colab and, when it
finds it, clones the repo, installs the dependencies, and zips the result up at the
end so the subset survives the VM being wiped.

`--groups` counts **source recordings**, not clips (train/val groups are 5 clips,
test groups are 8).

### Output

```
data/
├── processed/<split>/          16 kHz mono WAV, 64,000 samples (4.000 s)
│                               named <split>_<source_id>_<generator>.wav
└── metadata/
    ├── <split>_subset.csv      intermediate, written during fetching
    └── manifest.csv            verified, merged - this is what downstream reads
```

Audio is **not** amplitude-normalised on disk. Normalisation happens at load time
in the Dataset so raw clips stay available for augmentation experiments.

`manifest.csv` is regenerated (not appended) — re-run `verify_subset.py` after
every fetch.

### Standardisation report and plots

```bash
python3 -m src.preprocessing.audio_report               # results/preprocessing_report.{json,txt}
python3 -m src.preprocessing.visualize                  # waveform + log-Mel PNGs
python3 -m src.preprocessing.visualize --split test --generator G06   # an unseen generator
```

`audio_report.py` reuses `verify_subset`'s per-clip checks to print the sample
rate / channel / duration distributions and confirm every clip is standardised.
`visualize.py` picks a REAL clip and a FAKE clip from the same source recording
where it can, and writes waveform, log-Mel and side-by-side comparison plots to
`results/plots/`.

## Notes

- `src/preprocessing/build_subset.py` uses `datasets.load_dataset(streaming=True)`.
  It is superseded by `fetch_subset.py` and kept only for reference: parquet reads
  whole row groups, so streaming pulls ~80 MB before yielding a single row, which
  is impractical on a slow connection.
- The full dataset is 60 GB (218,533 clips across dev + test, ~28 GB as WAV).
  The subset here is ~4.5% of it. For full-scale runs, use a CUDA GPU environment
  where `load_dataset("EnvSDD/EnvSDD")` downloads at datacenter speed.

## Planned work

| Stage | Model | `--model` | Status |
|-------|-------|-----------|--------|
| Preprocessing | — | — | done |
| Level 1 | CNN on log-Mel spectrograms | `logmel_cnn` | trained + evaluated |
| Level 1 | CNN on raw waveform | `cnn` | trained + evaluated |
| Level 2 | AASIST | `aasist` | trained + evaluated |
| Level 3 | BEATs + AASIST | `beats_aasist` | trained + evaluated |
| Novelty | CNN + BEATs feature fusion | `fusion` | trained + evaluated |
| Novelty | Augmentation | — | not started |

Primary metric is **EER**, reported per generator, alongside F1 and AUC.

## Detection models

Five detectors. Every one returns raw logits `(B,)`, so the training loop and
the evaluation scripts are identical across them and only `--model` changes.
What differs is the input each wants, which the registry in
[`src/models/__init__.py`](src/models/__init__.py) declares and `train.py` reads,
so the right `EnvSDDDataset` mode is built automatically.

| `--model` | Level | Input | Front-end | Back-end |
|---|---|---|---|---|
| `logmel_cnn` | 1 | `(1, 64, 251)` | log-Mel from the Dataset, standardised per mel bin over the train split | 4 conv blocks → global avg pool |
| `cnn` | 1 | `(1, 64000)` | log-Mel computed in-model, standardised per clip | Conv2D ×2 → dense |
| `aasist` | 2 | `(1, 64000)` | learnable 1-D conv | spectro-temporal graph attention |
| `beats_aasist` | 3 | `(1, 64000)` | frozen pretrained SSL model | graph attention |
| `fusion` | novelty | `(1, 64000)` | `cnn` branch ⊕ `beats_aasist` branch | joint classifier |

**Two Level 1 CNNs, deliberately.** `logmel_cnn` standardises per mel bin using
statistics computed once over the train split; `cnn` standardises each clip
against its own mean and std. Per-clip standardisation discards absolute level,
which may itself carry a generator cue; per-bin standardisation keeps it and
flattens the ~35 dB tilt across mel bins instead. Both are defensible, so both
are kept — the choice is measurable rather than assumed. `fusion` needs the
waveform variant, because its CNN branch must consume the same input its BEATs
branch does.

**Level 2 and the graph-attention back-end** are a compact from-scratch
reimplementation in the spirit of Jung et al., *AASIST* (ICASSP 2022) — not a
line-by-line port. Sizes are reduced so it trains on CPU in reasonable time; see
the docstring in [`graph_attention.py`](src/models/graph_attention.py).

**Level 3's front-end** is torchaudio's frozen `WAV2VEC2_BASE`, not Microsoft's
BEATs checkpoint, which is not pip-installable. `load_frontend()` in
[`beats_aasist.py`](src/models/beats_aasist.py) is the drop-in extension point;
everything downstream is front-end agnostic.

### Running

```bash
python3 -m src.training.smoke_test        # one train + eval step per model, seconds
./run_training.sh                         # train and evaluate all five
MODELS="logmel_cnn aasist" ./run_training.sh
```

or individually:

```bash
python3 -m src.training.train --model aasist --epochs 20
python3 -m src.evaluation.evaluate --checkpoint results/models/aasist_best.pt
python3 -m src.evaluation.compare         # all checkpoints side by side
```

`fusion` initialises its branches from trained checkpoints, so train `cnn` and
`beats_aasist` first:

```bash
python3 -m src.training.train --model fusion --epochs 15 \
    --cnn_checkpoint results/models/cnn_best.pt \
    --beats_aasist_checkpoint results/models/beats_aasist_best.pt
```

`compare` sorts by **generalisation gap**, not by overall EER: a model that wins
on seen generators and collapses on unseen ones is worse, for this project's
question, than one that is mediocre on both.

### Note for Apple Silicon

`adaptive_avg_pool` is unimplemented on MPS for non-divisible sizes
([pytorch#96056](https://github.com/pytorch/pytorch/issues/96056)), which AASIST
and Level 3 both hit. [`src/models/pooling.py`](src/models/pooling.py) falls back
to CPU for that one op, keeping results identical on every backend.

## Results

All five models scored on the same 2,400-clip test split, through the same
`src/evaluation/compare.py`, so every number below is computed identically:

```bash
./run_training.sh                    # train everything
python3 -m src.evaluation.compare    # the table below
```

| `--model` | Level | Input | Params | seen EER | unseen EER | gap |
|---|---|---|---|---|---|---|
| `cnn` | 1 | waveform | 19,073 | 0.2133 | 0.2700 | +0.0567 |
| **`logmel_cnn`** | 1 | log-Mel | 240,737 | **0.0242** | **0.0833** | +0.0592 |
| `fusion` | novelty | waveform | 324,227 | 0.2033 | 0.2967 | +0.0933 |
| `beats_aasist` | 3 | waveform | 263,937 | 0.2400 | 0.3433 | +0.1033 |
| `aasist` | 2 | waveform | 269,427 | 0.1133 | 0.2333 | +0.1200 |

Per generator:

| | G01 | G02 | G03 | G04 | G05 | G06 | G07 |
|---|---|---|---|---|---|---|---|
| `logmel_cnn` | 0.0100 | 0.0133 | 0.0367 | 0.0300 | 0.0967 | 0.0667 | 0.0833 |
| `aasist` | 0.1000 | 0.1033 | 0.1733 | 0.1033 | 0.2867 | 0.2367 | 0.1133 |
| `cnn` | 0.1767 | 0.1900 | 0.1800 | 0.2933 | 0.1333 | 0.1633 | **0.5100** |
| `fusion` | 0.1767 | 0.2233 | 0.2100 | 0.1933 | 0.1767 | 0.2033 | **0.5133** |
| `beats_aasist` | 0.2200 | 0.2767 | 0.2467 | 0.2267 | 0.3100 | 0.2700 | **0.5100** |

Levels 2, 3 and fusion were trained on a Colab T4; see
[`notebooks/colab_full_run.ipynb`](notebooks/colab_full_run.ipynb) for that run.

### Four things these numbers say

**The simplest model wins, by a wide margin.** `logmel_cnn` reaches 0.0242 seen
and 0.0833 unseen, roughly 5× better than AASIST and an order of magnitude
better than the Level 3 and fusion models. The likely reason is data: 1,200
training source recordings is far too little for a graph-attention network or a
frozen SSL front-end to pay off, while a small log-Mel CNN with per-mel-bin
standardisation fits that budget comfortably. Scale the training split before
concluding anything about the architectures themselves.

**Fusion did not beat its own branches.** It sits between `cnn` (0.2133/0.2700)
and `beats_aasist` (0.2400/0.3433) on seen generators and is worse than `cnn` on
unseen ones. Combining two weak branches did not produce a strong one — which is
a result worth reporting, not a bug to hide.

**Three models are at chance on G07.** `cnn`, `fusion` and `beats_aasist` all
score ~0.51 on G07 (AudioLDM 2 in audio-to-audio mode) — literally coin-flip.
`aasist` manages 0.1133 and `logmel_cnn` 0.0833 on the same clips, so the
information is there and those three simply fail to use it. G07 preserves the
source recording's structure, so it is the unseen generator that most resembles
a real clip.

**The gap is real but not the whole story.** For `aasist`, bootstrapping over the
300 test source groups gives a gap of +0.1200, 95% CI [+0.0900, +0.1467],
positive in 2000/2000 resamples. Note that a small gap alone is not good news:
`fusion` has a *negative* gap in one scoring only because it is weak everywhere.
Read the gap alongside the seen EER, never on its own.

### Waveform normalisation is part of the checkpoint

The Level 2/3/fusion models were trained on **raw** waveforms; `logmel_cnn` was
trained on peak-normalised ones. This is not a free choice at evaluation time. A
model trained on raw audio and scored on peak-normalised audio sees a different
input distribution and collapses — AASIST scores 0.5067 on G01 that way, against
0.1000 scored correctly, i.e. chance instead of its real number.

Checkpoints therefore record the setting, and every evaluation script reads it
back, so a checkpoint is always scored the way it was trained. `cnn` is the one
model immune to the difference, because per-clip log-Mel standardisation cancels
a scalar gain exactly.

## Credits

Preprocessing was built jointly. The detection models — AASIST, the
graph-attention back-end, BEATs+AASIST, the waveform CNN and the fusion model —
together with `audio_report.py`, `visualize.py` and the Colab pipeline, are the
work of a project partner, merged in here and adapted to this repository's
Dataset and training interfaces. The Level 2, 3 and fusion results were produced
by her Colab GPU run, recorded in `notebooks/colab_full_run.ipynb`; they are
rescored here through this repository's metrics so all five models are directly
comparable.

## References

- ESDD 2026: Environmental Sound Deepfake Detection Challenge Evaluation Plan
- ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge Evaluation Plan
- EnvSDD dataset — [arXiv:2505.19203](https://arxiv.org/abs/2505.19203)
