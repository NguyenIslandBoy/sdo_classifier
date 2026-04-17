"""
main.py — FastAPI inference endpoint for SDO solar flare classification.

Endpoints:
    GET  /health       — liveness check
    GET  /model/info   — model metadata
    POST /predict      — upload AIA 171Å image → flare class + confidence
"""

import io
import json
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pt"
META_PATH  = BASE_DIR / "run_metadata.json"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SDO Flare Classifier",
    description = "Binary solar flare risk classification from AIA 171Å imagery.",
    version     = "0.1.0",
)

# ── Model loading (once at startup) ───────────────────────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = None
meta       = {}
CLASS_NAMES = ["quiet", "active"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@app.on_event("startup")
async def load_model():
    global model, meta

    # ── Load model ─────────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))
    from model import SDOFlareClassifier

    model = SDOFlareClassifier(num_classes=2, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # ── Load metadata ──────────────────────────────────────────────────────────
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status" : "ok",
        "device" : str(device),
        "model"  : "loaded" if model is not None else "not loaded",
    }


@app.get("/model/info")
def model_info():
    return meta


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Validate file type ─────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Send JPEG or PNG."
        )

    # ── Read and preprocess image ──────────────────────────────────────────────
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    tensor = val_transforms(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # ── Inference ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                        # [1, 2]
        probs  = torch.softmax(logits, dim=1)[0]      # [2]
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    pred_class = int(probs.argmax().item())
    confidence = round(float(probs[pred_class].item()), 4)

    return JSONResponse({
        "prediction"    : CLASS_NAMES[pred_class],
        "confidence"    : confidence,
        "probabilities" : {
            "quiet"  : round(float(probs[0].item()), 4),
            "active" : round(float(probs[1].item()), 4),
        },
        "latency_ms"    : latency_ms,
        "model"         : "EfficientNetB3",
        "note"          : "Trained on SDOBenchmark example subset (416 samples).",
    })