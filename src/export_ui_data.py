"""Export everything the results explainer UI needs, as static files.

    python -m src.export_ui_data

Writes ui-data/ : JSON the app reads at runtime, plus compressed audio and
log-Mel spectrogram images for the listening and challenge sections. Nothing in
the UI needs a backend, a checkpoint, or torch - which is the point: the app has
to run from a static host.

The embedding projection is the centrepiece. Unseen generators land between the
real clips and the seen fakes in the detector's own feature space, which is why
its error rate rises on them - the model learned "fake means far from real in
this direction" and unseen generators only travel part of the way.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.envsdd_dataset import HOP, N_MELS, SR, make_loader   # noqa: E402
from src.models import LEVEL, MODEL_NAMES, input_mode, load_checkpoint  # noqa: E402
from src.preprocessing.generators import (                              # noqa: E402
    GENERATOR_NAMES,
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)
from src.evaluation.metrics import eer                                   # noqa: E402
from src.training.train import pick_device, score_loader                # noqa: E402

GEN_ORDER = ["REAL", "G01", "G02", "G03", "G04", "G05", "G06", "G07"]
# One source recording from each of four different source datasets, so the
# listening section covers more than one kind of acoustic scene.
WANT_DATASETS = ["UrbanSound8K", "TUTASC2019Dev", "DCASE2023Task7", "Clotho"]


def score_all_models(device, ckpt_dir):
    """Per-clip fake-ness score on the test split, for every trained model."""
    out, order = {}, None
    for name in MODEL_NAMES:
        path = Path(ckpt_dir) / f"{name}_best.pt"
        if not path.exists():
            print(f"  {name:14s} no checkpoint, skipping")
            continue
        model, ck = load_checkpoint(path, device)
        mode = ck.get("input_mode") or input_mode(ck["model"])
        loader = make_loader("test", mode=mode, batch_size=64, shuffle=False,
                             num_workers=0, normalize=ck["normalize"])
        scores, labels, _ = score_loader(model, loader, device)
        df = loader.dataset.df
        if order is None:
            order = df[["filename", "generator", "label", "source_id",
                        "source_dataset", "path"]].copy()
            order["label"] = labels.astype(int)
        out[name] = scores
        print(f"  {name:14s} scored {len(scores)} clips")
        del model
    return order, out


def embed_projection(device, ckpt_dir, seed=0):
    """2-D t-SNE of the log-Mel CNN's penultimate features on the test split."""
    model, ck = load_checkpoint(Path(ckpt_dir) / "logmel_cnn_best.pt", device)
    loader = make_loader("test", mode="logmel", batch_size=64, shuffle=False,
                         num_workers=0, normalize=ck["normalize"])
    embs = []
    with torch.no_grad():
        for x, _ in loader:
            embs.append(model.embed(x.to(device)).cpu().numpy())
    E = np.concatenate(embs).astype(np.float64)

    from sklearn.manifold import TSNE
    Z = TSNE(n_components=2, perplexity=30, init="pca",
             random_state=seed).fit_transform(E)
    # Normalise to a stable [-1, 1] box so the UI never has to rescale.
    Z = 2 * (Z - Z.min(0)) / (Z.max(0) - Z.min(0)) - 1

    real_c = E[loader.dataset.df.generator.to_numpy() == "REAL"].mean(0)
    dist = {g: float(np.linalg.norm(E[loader.dataset.df.generator.to_numpy() == g]
                                    - real_c, axis=1).mean())
            for g in GEN_ORDER}
    return Z, dist


def write_spectrogram(wav_path, out_png):
    wav, sr = sf.read(wav_path, dtype="float32")
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=HOP, n_mels=N_MELS)
    db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    S = db(mel(torch.from_numpy(wav))).numpy()

    fig, ax = plt.subplots(figsize=(4.4, 1.9), dpi=150)
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-40, vmax=40)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # A full-colour 150-dpi PNG is ~175 KB; 64 colours at display size is ~17 KB
    # for no visible loss, and 32 of them ship with the app.
    from PIL import Image
    im = Image.open(out_png).resize((440, 190), Image.LANCZOS)
    im.convert("P", palette=Image.ADAPTIVE, colors=64).save(out_png, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ui-data")
    p.add_argument("--checkpoints", default="results/models")
    p.add_argument("--groups", type=int, default=4,
                   help="source recordings to bundle audio for (x8 clips each)")
    a = p.parse_args()

    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "spectrograms").mkdir(parents=True, exist_ok=True)
    device = pick_device()

    print("scoring every model on the test split")
    meta, scores = score_all_models(device, a.checkpoints)

    print("projecting the log-Mel CNN's feature space")
    Z, dist = embed_projection(device, a.checkpoints)

    # ---- embedding.json: one entry per test clip -------------------------
    pts = []
    for i, row in enumerate(meta.itertuples()):
        e = {"x": round(float(Z[i, 0]), 4), "y": round(float(Z[i, 1]), 4),
             "g": row.generator, "label": int(row.label),
             "src": row.source_dataset, "sid": int(row.source_id)}
        for m, s in scores.items():
            e[m] = round(float(s[i]), 3)
        pts.append(e)
    (out / "embedding.json").write_text(json.dumps({
        "points": pts,
        "distanceToReal": {g: round(v, 3) for g, v in dist.items()},
        "note": ("t-SNE of the log-Mel CNN's 128-d penultimate features on the "
                 "test split. Coordinates are normalised to [-1, 1]."),
    }))
    print(f"  embedding.json  {len(pts)} points")

    # Each model's EER operating point on the test split. Without it a raw
    # logit means nothing to the UI - it is not comparable across models, and
    # the sign alone is not the decision boundary.
    thresholds = {m: round(float(eer(meta.label.to_numpy(), s)[1]), 4)
                  for m, s in scores.items()}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    print(f"  thresholds.json {thresholds}")

    # ---- clips: audio + spectrogram for a few complete source groups -----
    t = meta.copy()
    sizes = t.groupby("source_id").size()
    complete = set(sizes[sizes == 8].index)
    chosen, used = [], set()
    for ds in WANT_DATASETS:
        cand = sorted(set(t[(t.source_dataset == ds)
                            & t.source_id.isin(complete)].source_id) - used)
        if cand:
            chosen.append(cand[len(cand) // 2])
            used.add(chosen[-1])
        if len(chosen) >= a.groups:
            break

    clips = []
    for sid in chosen:
        grp = t[t.source_id == sid]
        for row in grp.itertuples():
            cid = f"{sid:06d}_{row.generator}"
            mp3 = out / "clips" / f"{cid}.mp3"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", row.path,
                 "-ac", "1", "-b:a", "64k", str(mp3)], check=True)
            write_spectrogram(row.path, out / "spectrograms" / f"{cid}.png")
            idx = meta.index[meta.filename == row.filename][0]
            clips.append({
                "id": cid, "sourceId": int(sid), "generator": row.generator,
                "generatorName": GENERATOR_NAMES[row.generator],
                "label": int(row.label),
                "seen": (None if row.generator == "REAL"
                         else row.generator in SEEN_GENERATORS),
                "sourceDataset": row.source_dataset,
                "audio": f"clips/{cid}.mp3",
                "spectrogram": f"spectrograms/{cid}.png",
                "scores": {m: round(float(s[idx]), 3) for m, s in scores.items()},
            })
    (out / "clips.json").write_text(json.dumps(clips, indent=1))
    print(f"  clips.json      {len(clips)} clips from {len(chosen)} recordings")

    # ---- generators.json -------------------------------------------------
    (out / "generators.json").write_text(json.dumps([
        {"id": g, "name": GENERATOR_NAMES[g],
         "seen": None if g == "REAL" else g in SEEN_GENERATORS,
         "unseen": g in UNSEEN_GENERATORS,
         "mode": ("real" if g == "REAL"
                  else "audio-to-audio" if g in ("G04", "G07") else "text-to-audio")}
        for g in GEN_ORDER], indent=1))

    # ---- models.json + curves.json --------------------------------------
    comp = pd.read_csv("results/tables/model_comparison.csv")
    (out / "models.json").write_text(comp.to_json(orient="records"))

    curves = {}
    for m in MODEL_NAMES:
        for f, col in ((f"results/tables/{m}_history.csv", "val_eer"),
                       (f"results/tables/{m}_train_log.csv", "eer")):
            if Path(f).exists():
                d = pd.read_csv(f)
                curves[m] = {"valEer": [round(v, 4) for v in d[col]],
                             "trainLoss": [round(v, 4) for v in d.train_loss]}
                break
    (out / "curves.json").write_text(json.dumps(curves))

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nui-data/ written, {total / 1e6:.1f} MB total")


if __name__ == "__main__":
    main()
