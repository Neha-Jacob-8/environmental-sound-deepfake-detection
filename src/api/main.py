"""HTTP API for the results dashboard.

    python -m src.api.main                  # http://127.0.0.1:8000
    python -m src.api.main --reload

Everything is read from results/tables/ per request (cached on file mtime), so
retraining a model and re-running `evaluate`/`compare` changes what the site
shows without rebuilding or redeploying the frontend.

/api/predict runs the real detector on uploaded audio. It is the one thing a
static bundle cannot do, and it is why this is a server rather than a JSON file.
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.api.store import Store  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
store = Store(ROOT)

app = FastAPI(title="EnvSDD detection results", version="1.0")
app.add_middleware(
    CORSMiddleware,
    # Vite dev server; a same-origin deployment needs none of this.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000", "http://localhost:4173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_model = None       # loaded on first /api/predict, not at import
_lda = None


def _need(value, what):
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        raise HTTPException(503, f"{what} is not available yet - see README")
    return value


@app.get("/api/health")
def health():
    return {"ok": True, "models": store.available()}


@app.get("/api/summary")
def summary():
    return store.summary()


@app.get("/api/models")
def models():
    return _need(store.models(), "results/tables/model_comparison.csv")


@app.get("/api/per-generator")
def per_generator():
    return _need(store.per_generator(), "per-generator EER")


@app.get("/api/curves")
def curves():
    return _need(store.curves(), "training logs")


@app.get("/api/confidence")
def confidence():
    return store.confidence()


@app.get("/api/generators")
def generators():
    return store.generators()


@app.get("/api/dataset")
def dataset():
    return _need(store.dataset(), "data/metadata/manifest.csv")


@app.get("/api/embeddings")
def embeddings():
    return _need(store.embeddings(), "docs/ui-data.json")


@app.get("/api/bundle")
def bundle():
    """Everything the dashboard needs, in one round trip."""
    return {
        "summary": store.summary(),
        "models": store.models(),
        "perGenerator": store.per_generator(),
        "curves": store.curves(),
        "confidence": store.confidence(),
        "generators": store.generators(),
        "dataset": store.dataset(),
    }


@app.get("/api/audio/{filename}")
def audio(filename: str):
    # Resolve and confine to the audio directory: a bare join would let
    # "../../etc/passwd" escape it.
    base = (ROOT / "docs" / "audio").resolve()
    path = (base / filename).resolve()
    if not str(path).startswith(str(base)) or not path.is_file():
        raise HTTPException(404, "clip not found")
    return FileResponse(path, media_type="audio/wav")


def _load_model():
    global _model, _lda
    if _model is not None:
        return _model, _lda
    import torch

    from src.analysis.export_onnx import ExportableDetector
    from src.datasets.envsdd_dataset import logmel_stats
    from src.models import load_checkpoint

    ckpt = ROOT / "results" / "models" / "logmel_cnn_best.pt"
    if not ckpt.exists():
        raise HTTPException(503, "no trained detector - run src.training.train")
    cnn, _ = load_checkpoint(ckpt, "cpu")
    mean, std = logmel_stats()

    d = np.load(ROOT / "results" / "tables" / "embeddings.npz", allow_pickle=True)
    emb, groups = d["emb"].astype(np.float64), d["group"]
    real_c = emb[groups == "real"].mean(0)
    axis = emb[groups == "seen"].mean(0) - real_c
    axis /= np.linalg.norm(axis)
    scale = float((emb[groups == "seen"].mean(0) - real_c) @ axis)

    _model = ExportableDetector(cnn, mean, std, real_c, axis, scale).eval()
    _lda = (d["lda_mean"], d["lda_scalings"]) if "lda_mean" in d.files else None
    return _model, _lda


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Run the detector on an uploaded clip.

    The model expects 4.000 s of 16 kHz mono. Anything else is resampled,
    downmixed and padded or trimmed here, and the response says what was done
    so the caller is never guessing what was actually scored.
    """
    import librosa
    import torch

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        wav, sr = librosa.load(io.BytesIO(raw), sr=16_000, mono=True)
    except Exception as e:
        raise HTTPException(415, f"could not decode audio: {type(e).__name__}")

    original_seconds = round(len(wav) / 16_000, 3)
    if len(wav) < 64_000:
        wav = np.pad(wav, (0, 64_000 - len(wav)))
        action = "padded with silence to 4.000 s"
    elif len(wav) > 64_000:
        wav = wav[:64_000]
        action = "trimmed to the first 4.000 s"
    else:
        action = "used as-is"

    model, lda = _load_model()
    with torch.no_grad():
        p_fake, axis_pos, emb = model(torch.from_numpy(wav[None, :]).float())

    emb_np = emb[0].numpy().astype(np.float64)
    point = None
    if lda is not None:
        xy = (emb_np - lda[0]) @ lda[1]
        point = [round(float(xy[0]), 3), round(float(xy[1]), 3)]

    return {
        "pFake": round(float(p_fake[0]), 4),
        "verdict": "likely AI-generated" if float(p_fake[0]) >= 0.5 else "likely real",
        "axisPos": round(float(axis_pos[0]), 3),
        "ldaPoint": point,
        "input": {"originalSeconds": original_seconds, "action": action,
                  "sampleRate": 16_000},
        "caveats": [
            "Trained on 4-second environmental field recordings. Speech, music "
            "or silence is out of distribution and the result is not meaningful.",
            "A course research model: 0.0242 EER on generators it trained on, "
            "0.0833 on ones it did not. Not a production tool.",
        ],
    }


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    uvicorn.run("src.api.main:app", host=a.host, port=a.port, reload=a.reload)


if __name__ == "__main__":
    main()
