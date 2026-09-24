"""Check that the API serves exactly what is in results/tables/.

    python -m src.api.verify                       # against a running server
    python -m src.api.verify --url http://host:8000

Reads the source files directly, reads the API, and compares value by value. A
dashboard is only trustworthy if its numbers can be traced back to the files the
training runs wrote, so this makes that a command rather than an assumption.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)

    try:
        api = requests.get(f"{a.url}/api/bundle", timeout=30).json()
    except Exception as e:
        sys.exit(f"cannot reach {a.url}: {type(e).__name__}: {e}")

    checks, failures = 0, []

    def cmp(label, got, want, tol=5e-5):
        nonlocal checks
        checks += 1
        if isinstance(want, float) and abs(float(got) - want) > tol:
            failures.append(f"{label}: API {got} != file {want}")
        elif not isinstance(want, float) and got != want:
            failures.append(f"{label}: API {got!r} != file {want!r}")

    # -- model_comparison.csv --------------------------------------------
    comp = pd.read_csv(root / "results/tables/model_comparison.csv").set_index("model")
    api_models = {m["id"]: m for m in api["models"]}
    cmp("model count", len(api_models), len(comp))
    for mid, row in comp.iterrows():
        m = api_models.get(mid)
        if m is None:
            failures.append(f"{mid}: in CSV but missing from the API")
            continue
        cmp(f"{mid}.seenEer", m["seenEer"], round(float(row.seen_eer), 4))
        cmp(f"{mid}.unseenEer", m["unseenEer"], round(float(row.unseen_eer), 4))
        cmp(f"{mid}.gap", m["gap"], round(float(row.gap), 4))
        cmp(f"{mid}.valEer", m["valEer"], round(float(row.val_eer), 4))
        cmp(f"{mid}.params", m["params"], int(row.params))
        for g in ["G01", "G02", "G03", "G04", "G05", "G06", "G07"]:
            cmp(f"{mid}.{g}", api["perGenerator"][mid][g],
                round(float(row[f"eer_{g}"]), 4))

    # -- bootstrap_ci.csv -------------------------------------------------
    ci_path = root / "results/tables/bootstrap_ci.csv"
    if ci_path.exists():
        ci = pd.read_csv(ci_path).set_index("model")
        cmp("confidence rows", len(api["confidence"]), len(ci))
        for mid, row in ci.iterrows():
            c = api["confidence"].get(mid)
            if c is None:
                failures.append(f"{mid}: CI in CSV but missing from the API")
                continue
            cmp(f"{mid}.ci.gap_lo", c["gap"][0], round(float(row.gap_lo), 4))
            cmp(f"{mid}.ci.gap_hi", c["gap"][1], round(float(row.gap_hi), 4))

    # -- training logs ----------------------------------------------------
    for mid, curve in api["curves"].items():
        for fn, col in ((f"{mid}_history.csv", "val_eer"), (f"{mid}_train_log.csv", "eer")):
            p = root / "results/tables" / fn
            if not p.exists():
                continue
            d = pd.read_csv(p)
            cmp(f"{mid}.curve length", len(curve["valEer"]), len(d))
            cmp(f"{mid}.curve[0]", curve["valEer"][0], round(float(d[col].iloc[0]), 4))
            cmp(f"{mid}.curve[-1]", curve["valEer"][-1], round(float(d[col].iloc[-1]), 4))
            break

    # -- manifest ---------------------------------------------------------
    man = root / "data/metadata/manifest.csv"
    if man.exists():
        df = pd.read_csv(man, usecols=["split", "source_id"])
        ds = api["dataset"]
        cmp("dataset.totalClips", ds["totalClips"], int(len(df)))
        for split, key in (("train", "trainClips"), ("validation", "valClips"),
                           ("test", "testClips")):
            cmp(f"dataset.{key}", ds[key], int((df.split == split).sum()))

    print(f"compared {checks} values against results/tables/ and the manifest")
    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all values match their source files exactly")


if __name__ == "__main__":
    main()
