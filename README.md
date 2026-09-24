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
| Level 1 | CNN on raw waveform | `cnn` | implemented, not yet trained |
| Level 2 | AASIST | `aasist` | implemented, not yet trained |
| Level 3 | BEATs + AASIST | `beats_aasist` | implemented, not yet trained |
| Novelty | CNN + BEATs feature fusion | `fusion` | implemented, not yet trained |
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

### Level 1 — log-Mel CNN (240,737 params)

```bash
python3 -m src.training.train           # ~3 min on MPS, early-stops around epoch 31
python3 -m src.evaluation.evaluate      # per-generator EER on the test split
python3 -m src.evaluation.report        # accuracy/precision/recall/F1 for all splits
python3 -m src.evaluation.bootstrap_ci  # group-level 95% CIs
```

Or as a notebook: [`notebooks/training.ipynb`](notebooks/training.ipynb) runs the
same code with training curves, per-generator bars, score distributions, ROC,
confusion matrix and the bootstrap, all inline.

Trained on G01–G04 over 1,200 source groups.

| Split | Accuracy | Bal. acc | Precision | Recall | F1 | AUC | EER |
|---|---|---|---|---|---|---|---|
| train | 0.9990 | 0.9991 | 0.9998 | 0.9990 | 0.9994 | 1.0000 | 0.0008 |
| validation | 0.9987 | 0.9992 | 1.0000 | 0.9983 | 0.9992 | 1.0000 | 0.0017 |
| test | 0.9300 | 0.9286 | 0.9889 | 0.9305 | 0.9588 | 0.9805 | 0.0729 |

Accuracy is not a useful number here: fakes outnumber reals 7:1 on test, so
answering "fake" every time already scores 87.5%. EER is the reported metric.

Pooled over the test split, with 95% CIs bootstrapped over the 300 test source
groups (whole groups resampled — clips within a group are correlated):

| | EER | 95% CI |
|---|---|---|
| seen (G01–G04) | 0.0242 | [0.0167, 0.0333] |
| unseen (G05–G07) | 0.0833 | [0.0633, 0.1033] |
| **generalisation gap** | **+0.0592** | [+0.0411, +0.0767] |

The gap is positive in 2000/2000 bootstrap resamples.

**Run-to-run variation.** Two runs of this identical configuration gave gaps of
+0.0608 and +0.0592 — stable. Individual generators are far noisier at 300 test
groups (G06 moved 0.0933 → 0.0667 between the two), so per-generator EERs should
not be compared at this scale without CIs. Enlarging the test split is the cheap
fix; see the note on scaling below.

**What the failure is not.** Train and validation accuracy sit within 0.03pp of
each other, so this is not classic overfitting and regularisation has no gap to
close. The model breaks along a different axis — across *generators* — which the
validation split cannot see, because it contains only G01–G04. Precision holds on
test while recall falls on the unseen generators: roughly one unseen fake in seven
is passed as real.

## Credits

Preprocessing was built jointly. The detection models — AASIST, the
graph-attention back-end, BEATs+AASIST, the waveform CNN and the fusion model —
together with `audio_report.py`, `visualize.py` and the Colab pipeline, are the
work of a project partner, merged in here and adapted to this repository's
Dataset and training interfaces.

## References

- ESDD 2026: Environmental Sound Deepfake Detection Challenge Evaluation Plan
- ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge Evaluation Plan
- EnvSDD dataset — [arXiv:2505.19203](https://arxiv.org/abs/2505.19203)
