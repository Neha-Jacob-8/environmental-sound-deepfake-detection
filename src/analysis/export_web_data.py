"""Regenerate the dashboard's data module from the real result tables.

    python -m src.analysis.export_web_data

web/src/data/researchData.ts started life as numbers pasted into a prompt, which
means it is a snapshot: retrain anything and the dashboard keeps showing the old
figures with no indication they are stale. This rewrites the measured parts of
that file straight from results/tables/, so the site cannot silently disagree
with the repository.

Only the measured constants are generated - MODELS, PER_GENERATOR, CURVES and
CONFIDENCE. Everything else in the module (the interfaces, the generator
descriptions, the palette, dataset stats) is hand-written prose and is left
exactly as it is.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TARGET = Path("web/src/data/researchData.ts")
LABELS = {"logmel_cnn": "Log-Mel CNN", "cnn": "Waveform CNN", "aasist": "AASIST",
          "beats_aasist": "BEATs + AASIST", "fusion": "Feature Fusion"}
# Fixed categorical order - a model keeps its colour however the table is sorted.
COLORS = {"logmel_cnn": "#2a78d6", "cnn": "#eb6834", "aasist": "#1baf7a",
          "beats_aasist": "#eda100", "fusion": "#e87ba4"}
GENS = ["G01", "G02", "G03", "G04", "G05", "G06", "G07"]
ORDER = ["logmel_cnn", "cnn", "aasist", "beats_aasist", "fusion"]


def block(name, body):
    return f"{name}{body}"


def build(root: Path):
    comp = pd.read_csv(root / "results/tables/model_comparison.csv").set_index("model")

    models = []
    for m in ORDER:
        if m not in comp.index:
            print(f"  note: {m} has no checkpoint scored yet, skipping")
            continue
        r = comp.loc[m]
        models.append(
            f'  {{ id: "{m}", label: "{LABELS[m]}", level: "{r.level}", '
            f'input: "{r.input}", params: {int(r.params)}, '
            f"valEer: {r.val_eer:.4f}, seenEer: {r.seen_eer:.4f}, "
            f"unseenEer: {r.unseen_eer:.4f}, gap: {r.gap:.4f}, "
            f'color: "{COLORS[m]}" }},'
        )

    per_gen = []
    for m in ORDER:
        if m not in comp.index:
            continue
        vals = ", ".join(f"{g}:{comp.loc[m, f'eer_{g}']:.4f}" for g in GENS)
        per_gen.append(f"  {m}: {{ {vals} }},")

    curves = []
    for m in ORDER:
        for fn, col in ((f"{m}_history.csv", "val_eer"), (f"{m}_train_log.csv", "eer")):
            p = root / "results/tables" / fn
            if not p.exists():
                continue
            d = pd.read_csv(p)
            eer = ",".join(f"{v:.4f}" for v in d[col])
            loss = ",".join(f"{v:.4f}" for v in d["train_loss"])
            curves.append(f"  {m}: {{ valEer:[{eer}], trainLoss:[{loss}] }},")
            break

    return ("\n".join(models), "\n".join(per_gen), "\n".join(curves))


def replace_array(src, marker, body, close):
    """Swap one generated block, leaving the surrounding file untouched."""
    start = src.index(marker) + len(marker)
    end = src.index(close, start)
    return src[:start] + "\n" + body + "\n" + src[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file is out of date instead of rewriting it")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    target = root / TARGET

    models, per_gen, curves = build(root)
    src = original = target.read_text()
    src = replace_array(src, "export const MODELS: ModelInfo[] = [", models, "\n];")
    src = replace_array(
        src, "export const PER_GENERATOR: Record<string, Record<string, number>> = {",
        per_gen, "\n};")
    src = replace_array(
        src,
        "export const CURVES: Record<string, { valEer: number[]; trainLoss: number[] }> = {",
        curves, "\n};")

    if a.check:
        if src != original:
            print(f"OUT OF DATE: {TARGET} does not match results/tables/")
            print("Run: python -m src.analysis.export_web_data")
            sys.exit(1)
        print(f"up to date: {TARGET}")
        return

    if src == original:
        print(f"unchanged: {TARGET} already matches results/tables/")
    else:
        target.write_text(src)
        print(f"rewritten: {TARGET}")


if __name__ == "__main__":
    main()
