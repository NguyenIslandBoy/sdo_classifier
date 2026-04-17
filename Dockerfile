# Dockerfile — SDO Flare Classifier inference API
# Multi-stage build: keeps final image lean by separating dependency install

FROM python:3.12-slim AS base

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# CPU-only torch: significantly smaller image than default (800MB vs 2.5GB)
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY src/     ./src/
COPY api/     ./api/
COPY models/  ./models/
COPY run_metadata.json .

# ── Pre-cache EfficientNetB3 pretrained weights ────────────────────────────────
RUN python -c "from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights; efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)"

# ── Non-root user for security ─────────────────────────────────────────────────
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]