"""Extract the detector's penultimate embedding for every test clip.

The generalisation gap is usually reported as two EER numbers. This looks at
where it comes from: if the detector has learned "fake means far from real in
this direction", then generators it never trained on should sit *partway* along
that direction rather than at the far end - close enough to real to be missed.

Writes results/tables/embeddings.npz (embeddings + labels) and, with --project,
a 2-D t-SNE plus centroid geometry for the UI.

    python -m src.analysis.embeddings --project
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.envsdd_dataset import make_loader          # noqa: E402
from src.models import input_mode, load_checkpoint           # noqa: E402
from src.preprocessing.generators import (                   # noqa: E402
    GENERATOR_NAMES,
    SEEN_GENERATORS,
    UNSEEN_GENERATORS,
)
from src.training.train import pick_device                   # noqa: E402


@torch.no_grad()
def extract(model, loader, device):
    embs, logits = [], []
    for x, _ in loader:
        x = x.to(device)
        e = model.embed(x)
        embs.append(e.float().cpu().numpy())
        logits.append(model.head(e).squeeze(1).float().cpu().numpy())
    return np.concatenate(embs), np.concatenate(logits)


def group_of(gen):
    if gen == "REAL":
        return "real"
    return "seen" if gen in SEEN_GENERATORS else "unseen"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/models/logmel_cnn_best.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--project", action="store_true", help="also run t-SNE")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    a = p.parse_args()

    device = pick_device()
    model, ck = load_checkpoint(a.checkpoint, device)
    if not hasattr(model, "embed"):
        sys.exit(f"{ck['model']} has no embed(); use logmel_cnn.")

    loader = make_loader(a.split, mode=ck.get("input_mode") or input_mode(ck["model"]),
                         batch_size=64, shuffle=False, num_workers=a.num_workers,
                         normalize=ck["normalize"])
    df = loader.dataset.df
    emb, logit = extract(model, loader, device)
    gens = df.generator.to_numpy()
    groups = np.array([group_of(g) for g in gens])
    print(f"{ck['model']}: {emb.shape[0]} clips, {emb.shape[1]}-d embeddings")

    # --- centroid geometry -------------------------------------------------
    real_c = emb[groups == "real"].mean(0)
    rows = []
    for g in ["REAL"] + SEEN_GENERATORS + UNSEEN_GENERATORS:
        sel = gens == g
        d = np.linalg.norm(emb[sel] - real_c, axis=1)
        rows.append({"generator": g, "name": GENERATOR_NAMES[g],
                     "group": group_of(g), "n": int(sel.sum()),
                     "dist_to_real_centroid": float(d.mean()),
                     "dist_std": float(d.std()),
                     "mean_logit": float(logit[sel].mean())})
    geo = pd.DataFrame(rows)
    print("\n=== mean distance to the REAL centroid (128-d) ===")
    print(geo[["generator", "name", "group", "dist_to_real_centroid",
               "mean_logit"]].to_string(
        index=False, formatters={"dist_to_real_centroid": "{:.3f}".format,
                                 "mean_logit": "{:+.3f}".format}))

    by_group = {g: float(np.linalg.norm(emb[groups == g] - real_c, axis=1).mean())
                for g in ("real", "seen", "unseen")}
    print("\nby group:", {k: round(v, 3) for k, v in by_group.items()})

    out = {"model": ck["model"], "group_distance": by_group,
           "per_generator": geo.to_dict("records")}

    if a.project:
        from sklearn.manifold import TSNE
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        xy = TSNE(n_components=2, perplexity=a.perplexity, init="pca",
                  random_state=a.seed).fit_transform(StandardScaler().fit_transform(emb))
        print(f"\nt-SNE done: {xy.shape}")

        cent = {g: xy[groups == g].mean(0).tolist() for g in ("real", "seen", "unseen")}
        print("2-D centroids:", {k: [round(v, 1) for v in c] for k, c in cent.items()})

        # Is the 3-way structure real, or is t-SNE inventing it?
        acc2 = cross_val_score(KNeighborsClassifier(15), xy, groups, cv=5).mean()
        acc128 = cross_val_score(KNeighborsClassifier(15), emb, groups, cv=5).mean()
        print(f"5-fold kNN 3-way accuracy: 2-D {acc2:.3f}   128-d {acc128:.3f}   "
              f"(chance {1/3:.3f})")

        out.update(tsne_centroids=cent, knn_acc_2d=float(acc2),
                   knn_acc_128d=float(acc128))
        np.savez_compressed("results/tables/embeddings.npz", emb=emb, xy=xy,
                            logit=logit, generator=gens, group=groups,
                            filename=df.filename.to_numpy(),
                            source_id=df.source_id.to_numpy())
        print("written -> results/tables/embeddings.npz")

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    geo.to_csv("results/tables/embedding_geometry.csv", index=False)
    Path("results/tables/embedding_summary.json").write_text(json.dumps(out, indent=2))
    print("written -> results/tables/embedding_geometry.csv, embedding_summary.json")


if __name__ == "__main__":
    main()
