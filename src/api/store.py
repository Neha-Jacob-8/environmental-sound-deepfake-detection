"""Reads results/tables/ and turns it into the shapes the dashboard wants.

Everything here derives from files the training and evaluation scripts already
write. Nothing is duplicated: retrain a model, re-run evaluate/compare, and the
API serves the new numbers on the next request without a rebuild or a redeploy.

Tables are cached per file mtime, so a re-run is picked up automatically but a
busy endpoint does not re-parse CSVs on every call.
"""

import json
from pathlib import Path

import pandas as pd

from src.preprocessing.generators import (
    ALL_GENERATORS,
    GENERATOR_NAMES,
    SEEN_GENERATORS,
)

LABELS = {"logmel_cnn": "Log-Mel CNN", "cnn": "Waveform CNN", "aasist": "AASIST",
          "beats_aasist": "BEATs + AASIST", "fusion": "Feature Fusion"}
# Fixed categorical order: a model keeps its colour however the table is sorted.
COLORS = {"logmel_cnn": "#2a78d6", "cnn": "#eb6834", "aasist": "#1baf7a",
          "beats_aasist": "#eda100", "fusion": "#e87ba4"}
ORDER = ["logmel_cnn", "cnn", "aasist", "beats_aasist", "fusion"]

CONDITIONING = {"G01": "text-to-audio", "G02": "text-to-audio",
                "G03": "text-to-audio", "G04": "audio-to-audio",
                "G05": "text-to-audio", "G06": "text-to-audio",
                "G07": "audio-to-audio"}
GEN_NOTES = {
    "G04": "Audio-to-audio: starts from the real recording and preserves its "
           "structure, so it resembles real audio more closely than the "
           "text-to-audio generators.",
    "G07": "The same architecture as G02 in a different conditioning mode, so "
           "it tests an unseen *mode* rather than an unseen architecture. Only "
           "G05 and G06 are architecturally new.",
}


class Store:
    def __init__(self, root="."):
        self.root = Path(root).resolve()
        self.tables = self.root / "results" / "tables"
        self._cache: dict[str, tuple[float, object]] = {}

    # -- plumbing ---------------------------------------------------------
    def _csv(self, name):
        p = self.tables / name
        if not p.exists():
            return None
        key, mtime = f"csv:{name}", p.stat().st_mtime
        hit = self._cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
        df = pd.read_csv(p)
        self._cache[key] = (mtime, df)
        return df

    def available(self):
        c = self._csv("model_comparison.csv")
        return [] if c is None else [m for m in ORDER if m in set(c.model)]

    # -- endpoints --------------------------------------------------------
    def models(self):
        c = self._csv("model_comparison.csv")
        if c is None:
            return []
        c = c.set_index("model")
        out = []
        for m in ORDER:
            if m not in c.index:
                continue
            r = c.loc[m]
            out.append({
                "id": m, "label": LABELS[m], "level": str(r.level),
                "input": str(r["input"]), "params": int(r.params),
                "normalized": bool(r.normalized) if "normalized" in r else None,
                "valEer": round(float(r.val_eer), 4),
                "seenEer": round(float(r.seen_eer), 4),
                "unseenEer": round(float(r.unseen_eer), 4),
                "gap": round(float(r.gap), 4),
                "color": COLORS[m],
            })
        return out

    def per_generator(self):
        c = self._csv("model_comparison.csv")
        if c is None:
            return {}
        c = c.set_index("model")
        return {
            m: {g: round(float(c.loc[m, f"eer_{g}"]), 4)
                for g in ALL_GENERATORS if f"eer_{g}" in c.columns}
            for m in ORDER if m in c.index
        }

    def curves(self):
        out = {}
        for m in ORDER:
            for fn, col in ((f"{m}_history.csv", "val_eer"), (f"{m}_train_log.csv", "eer")):
                d = self._csv(fn)
                if d is None:
                    continue
                out[m] = {
                    "valEer": [round(float(v), 4) for v in d[col]],
                    "trainLoss": [round(float(v), 4) for v in d["train_loss"]],
                }
                break
        return out

    def confidence(self):
        d = self._csv("bootstrap_ci.csv")
        if d is None:
            return {}
        return {
            r.model: {
                "seen": [round(float(r.seen_lo), 4), round(float(r.seen_hi), 4)],
                "unseen": [round(float(r.unseen_lo), 4), round(float(r.unseen_hi), 4)],
                "gap": [round(float(r.gap_lo), 4), round(float(r.gap_hi), 4)],
                "nResamples": int(r.n_resamples),
                "pGapPositive": round(float(r.p_gap_positive), 4),
            }
            for r in d.itertuples()
        }

    def generators(self):
        return [
            {"id": g, "name": GENERATOR_NAMES[g].split(" (")[0],
             "mode": CONDITIONING[g], "seen": g in SEEN_GENERATORS,
             "notes": GEN_NOTES.get(g)}
            for g in ALL_GENERATORS
        ]

    def dataset(self):
        man = self.root / "data" / "metadata" / "manifest.csv"
        if not man.exists():
            return None
        key, mtime = "manifest", man.stat().st_mtime
        hit = self._cache.get(key)
        if not hit or hit[0] != mtime:
            df = pd.read_csv(man, usecols=["split", "source_id"])
            self._cache[key] = (mtime, df)
        df = self._cache[key][1]
        counts = df.split.value_counts()
        return {
            "name": "EnvSDD", "protocol": "ESDD 2026",
            "totalClips": int(len(df)),
            "trainClips": int(counts.get("train", 0)),
            "valClips": int(counts.get("validation", 0)),
            "testClips": int(counts.get("test", 0)),
            "testSourceGroups": int(df[df.split == "test"].source_id.nunique()),
            "sampleRate": "16 kHz mono", "duration": "4.000 s",
        }

    def embeddings(self):
        p = self.root / "docs" / "ui-data.json"
        if not p.exists():
            return None
        key, mtime = "emb", p.stat().st_mtime
        hit = self._cache.get(key)
        if not hit or hit[0] != mtime:
            self._cache[key] = (mtime, json.loads(p.read_text()))
        return self._cache[key][1]

    def summary(self):
        models = self.models()
        if not models:
            return {"ready": False, "reason": "no scored checkpoints yet"}
        best = min(models, key=lambda m: m["unseenEer"])
        return {
            "ready": True,
            "title": "Robust Environmental Sound Deepfake Detection "
                     "Against Unseen Audio Generators",
            "researchQuestion": "Can combining complementary acoustic "
                                "representations improve robustness of "
                                "environmental sound deepfake detection against "
                                "unseen audio generators?",
            "bestModel": best["id"],
            "bestSeenEer": best["seenEer"],
            "bestUnseenEer": best["unseenEer"],
            "bestGap": best["gap"],
            "nModels": len(models),
            "dataset": self.dataset(),
        }
