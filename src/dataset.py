"""
dataset.py — SDOBenchmark dataset loader with flare severity classification.

Classification scheme (GOES scale):
    0: quiet    — peak_flux < 1e-6  (no flare / B-class)
    1: moderate — peak_flux in [1e-6, 1e-4)  (C/M-class)
    2: strong   — peak_flux >= 1e-4  (X-class)

Actual folder structure:
    <split>/
        <AR_number>/
            <sample_id>/          ← matches `id` column in meta_data.csv
                <ts>__<wl>.jpg    ← e.g. 2012-05-14T041857__171.jpg
                ...               ← 4 timestamps x 10 wavelengths = up to 40 images
"""

import os
import re
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

# ── Flare classification thresholds (GOES scale) ──────────────────────────────
FLUX_THRESHOLDS = [1e-6, 1e-4]
CLASS_NAMES     = ["quiet", "moderate", "strong"]

# ── Wavelength used for single-channel training ────────────────────────────────
PRIMARY_WAVELENGTH = "171"


def flux_to_class(peak_flux: float) -> int:
    """Map continuous peak_flux to 3-class label."""
    if peak_flux < FLUX_THRESHOLDS[0]:
        return 0
    elif peak_flux < FLUX_THRESHOLDS[1]:
        return 1
    else:
        return 2


def load_metadata(data_dir: str) -> pd.DataFrame:
    """
    Load meta_data.csv and add classification labels.

    Args:
        data_dir: path to training/ or test/ directory

    Returns:
        DataFrame with columns: id, start, end, peak_flux, label, class_name
    """
    csv_path = os.path.join(data_dir, "meta_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"meta_data.csv not found in {data_dir}")

    df = pd.read_csv(csv_path)
    df["label"]      = df["peak_flux"].apply(flux_to_class)
    df["class_name"] = df["label"].apply(lambda x: CLASS_NAMES[x])
    return df


def find_image(sample_dir: str, wavelength: str, timestep_idx: int = -1) -> str | None:
    """
    Find a JPEG for a given wavelength inside a flat sample folder.

    Files are named: <timestamp>__<wavelength>.jpg
    e.g. 2012-05-14T041857__171.jpg

    Args:
        sample_dir:   full path to the sample folder (the `id` folder)
        wavelength:   AIA channel string, e.g. "171"
        timestep_idx: which timestamp to use after sorting ascending.
                      -1 = most recent (closest to prediction window, default)
                       0 = earliest

    Returns:
        full image path, or None if not found
    """
    if not os.path.isdir(sample_dir):
        return None

    pattern = re.compile(r"__(" + re.escape(wavelength) + r")\.jpg$")
    matches = sorted([
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if pattern.search(f)
    ])

    if not matches:
        return None

    return matches[timestep_idx]


def find_sample_dir(data_dir: str, sample_id: str) -> str | None:
    """
    Locate the sample folder.

    CSV id format : 11389_2012_01_01_19_06_00_0
    Actual path   : training/11389/2012_01_01_19_06_00_0/
    Fix: strip the AR number prefix from the id to get the subfolder name.
    """
    ar_number      = sample_id.split("_")[0]
    subfolder_name = sample_id[len(ar_number) + 1:]  # strip "11389_"
    candidate      = os.path.join(data_dir, ar_number, subfolder_name)
    if os.path.isdir(candidate):
        return candidate

    # Fallback: search all AR subdirs
    for ar_dir in os.listdir(data_dir):
        candidate = os.path.join(data_dir, ar_dir, subfolder_name)
        if os.path.isdir(candidate):
            return candidate

    return None


class SDOFlareDataset(Dataset):
    """
    PyTorch Dataset for SDOBenchmark solar flare classification.

    Args:
        data_dir:     path to training/ or test/ directory
        wavelength:   AIA channel to use (default: "171")
        timestep_idx: which of the 4 sorted timestamps to use (default: -1, most recent)
        transform:    optional torchvision transforms

    Returns per item:
        image (Tensor [C, H, W]): float32, normalised to [0, 1]
        label (int):              0=quiet, 1=moderate, 2=strong
        sample_id (str):          for traceability
    """

    def __init__(
        self,
        data_dir:     str,
        wavelength:   str = PRIMARY_WAVELENGTH,
        timestep_idx: int = -1,
        binary:       bool = False,
        transform           = None,
    ):
        """
        binary=True: merges moderate+strong → 1 (active), quiet → 0.
        Use for example subset where strong class has <5 samples.
        """
        self.data_dir     = data_dir
        self.wavelength   = wavelength
        self.timestep_idx = timestep_idx
        self.binary       = binary
        self.class_names  = ["quiet", "active"] if binary else CLASS_NAMES
        self.transform    = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        self.metadata = load_metadata(data_dir)
        self._build_index()

    def _build_index(self):
        """Build list of (image_path, label, sample_id) — skip missing images."""
        self.samples = []
        missing = 0

        for _, row in self.metadata.iterrows():
            sample_dir = find_sample_dir(self.data_dir, row["id"])

            if sample_dir is None:
                missing += 1
                continue

            img_path = find_image(sample_dir, self.wavelength, self.timestep_idx)

            if img_path is None:
                missing += 1
                continue

            label = int(row["label"])
            if self.binary:
                label = 0 if label == 0 else 1
            self.samples.append((img_path, label, row["id"]))

        print(f"[SDOFlareDataset] Loaded {len(self.samples)} samples "
              f"({missing} skipped — missing images) from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, sample_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, sample_id

    def class_distribution(self) -> pd.Series:
        """Return class counts — check imbalance before training."""
        labels = [s[1] for s in self.samples]
        return pd.Series(labels).map(lambda x: self.class_names[x]).value_counts()
