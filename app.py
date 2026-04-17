"""
app.py — Gradio demo for SDO Solar Flare Classifier.
Deployed on Hugging Face Spaces (CPU).
"""

import io
import json
import os
import urllib.request
from pathlib import Path

import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_model_full_binary.pt"

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["quiet", "active"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Load model ─────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(BASE_DIR / "src"))
from model import SDOFlareClassifier

device = torch.device("cpu")
net    = None

def get_model():
    global net
    if net is None:
        net = SDOFlareClassifier(num_classes=2, dropout=0.3)
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        net.to(device)
        net.eval()
        print("Model loaded.")
    return net


# ── Inference function ─────────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return "No image provided.", {}

    tensor = val_transforms(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = get_model()(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_class = int(probs.argmax().item())
    confidence = float(probs[pred_class].item())

    label  = CLASS_NAMES[pred_class]
    result = {
        "quiet" : float(probs[0].item()),
        "active": float(probs[1].item()),
    }

    summary = (
        f"**Prediction: {label.upper()}**\n\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"{'Active region detected — elevated flare risk.' if label == 'active' else 'Quiet sun — low flare activity expected.'}\n\n"
        f"*Model: EfficientNetB3 trained on SDOBenchmark (416 samples)*"
    )
    return summary, result


# ── Example images ─────────────────────────────────────────────────────────────
EXAMPLES = []
examples_dir = BASE_DIR / "examples"
if examples_dir.exists():
    EXAMPLES = [[str(p)] for p in sorted(examples_dir.glob("*.jpg"))[:4]]


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="SDO Solar Flare Classifier") as demo:
    gr.Markdown("""
    # 🌞 SDO Solar Flare Classifier
    **Binary solar flare risk classification from NASA Solar Dynamics Observatory AIA 171Å imagery.**

    Upload an AIA 171Å solar image to predict whether the active region shows elevated flare risk.

    > **Note:** Model trained on SDOBenchmark example subset (416 samples).
    > Performance will improve significantly with the full 8,336-sample dataset.
    > Best validation F1: 0.743 | Test F1: 0.69
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="AIA 171Å Solar Image")
            submit_btn  = gr.Button("Classify", variant="primary")
        with gr.Column():
            text_output = gr.Markdown(label="Prediction")
            conf_output = gr.Label(label="Class Probabilities", num_top_classes=2)

    submit_btn.click(
        fn      = predict,
        inputs  = [image_input],
        outputs = [text_output, conf_output],
    )

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=image_input)

    gr.Markdown("""
    ---
    **Data:** [SDOBenchmark](https://i4ds.github.io/SDOBenchmark/) |
    **Model:** EfficientNetB3 (ImageNet pretrained, fine-tuned) |
    **Code:** [GitHub](https://github.com/YOUR_USERNAME/sdo-flare-detection)
    """)

demo.launch()