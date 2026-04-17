"""
trainer.py — Training loop for SDOFlareClassifier.

Design decisions:
- Weighted cross-entropy loss: corrects for class imbalance (quiet >> active)
- AdamW optimiser: better weight decay than Adam, standard for fine-tuning
- ReduceLROnPlateau scheduler: halves LR if val loss stalls for 3 epochs
- Two-phase training:
    Phase 1 (epochs 1-5):  head only, backbone frozen, higher LR
    Phase 2 (epochs 6-15): top 3 backbone blocks unfrozen, lower LR
- Best model checkpoint saved by val F1 (not val loss) — F1 is more
  meaningful than loss for imbalanced binary classification
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, classification_report
import numpy as np


def compute_class_weights(dataset, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted cross-entropy.
    Downweights the majority class (quiet) automatically.
    """
    from collections import Counter
    if hasattr(dataset, "dataset"):
        # Subset — access underlying samples via indices
        labels = [dataset.dataset.samples[i][1] for i in dataset.indices]
    else:
        labels = [s[1] for s in dataset.samples]

    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls, cnt in counts.items():
        weights[cls] = total / (num_classes * cnt)
    return weights


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimiser.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(imgs)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(imgs)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, all_preds, all_labels
