"""
transforms.py — Image preprocessing and augmentation for SDOBenchmark.

Design decisions documented here for reproducibility:
- Resize to 224x224: matches EfficientNet pretrained input size
- Normalise with ImageNet stats: pretrained backbone expects this range
- Augmentation only on training set, NOT validation/test
- Vertical flip only (not horizontal): per SDOBenchmark FAQ, horizontal
  flip changes solar rotation direction which alters underlying physics.
  Vertical flip is physically valid — no known difference between upper
  and lower solar hemispheres.
- No random crop: risk of cutting out the active region entirely
- Colour jitter on brightness/contrast only: simulates detector degradation
  over time (known issue in SDO data — detectors darken with age)
"""

import torchvision.transforms as T

# ── ImageNet normalisation stats ───────────────────────────────────────────────
# Used because EfficientNet backbone was pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INPUT_SIZE = 224


def get_train_transforms() -> T.Compose:
    """
    Augmented transforms for training set.
    Each augmentation is physically justified for solar imagery.
    """
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.2,   # simulates detector sensitivity drift
            contrast=0.2,     # simulates atmospheric/instrument variation
            saturation=0.0,   # grayscale solar images — no saturation meaning
            hue=0.0,          # same — hue shift has no physical basis here
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> T.Compose:
    """
    Deterministic transforms for validation and test sets.
    No augmentation — only resize and normalise.
    """
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalise(tensor):
    """
    Reverse ImageNet normalisation for visualisation purposes.
    tensor: [C, H, W] float32
    Returns: [C, H, W] float32 in [0, 1]
    """
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)
