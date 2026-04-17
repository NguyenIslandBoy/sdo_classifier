"""
model.py — EfficientNetB3 fine-tuned for solar flare binary classification.

Architecture decisions:
- EfficientNetB3: good accuracy/parameter tradeoff; same family as CUB-200
  work so transfer learning intuitions carry over directly
- Pretrained on ImageNet: meaningful even for solar imagery — low-level
  edge and texture detectors transfer well across domains
- Frozen backbone initially: prevents destroying pretrained weights before
  the new head is stable (standard fine-tuning practice)
- Custom head: Dropout(0.3) before final linear — reduces overfitting on
  the small SDOBenchmark training set
"""

import torch
import torch.nn as nn
from torchvision import models


class SDOFlareClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # ── Backbone: EfficientNetB3 pretrained ───────────────────────────────
        weights        = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        self.backbone  = models.efficientnet_b3(weights=weights)

        # ── Freeze all backbone layers initially ──────────────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ── Replace classifier head ───────────────────────────────────────────
        # EfficientNetB3 classifier: [Dropout, Linear(1536 -> 1000)]
        # We replace with our own head for num_classes output
        in_features = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_top_blocks(self, n_blocks: int = 3):
        """
        Unfreeze the last n blocks of the EfficientNet backbone for fine-tuning.
        Call after head is stable (e.g. after epoch 3-5).
        EfficientNetB3 has 7 MBConv blocks (features[1] through features[7]).
        """
        blocks = list(self.backbone.features.children())
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"Unfrozen top {n_blocks} blocks — "
              f"trainable params: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")

    def count_trainable(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "pct": 100*trainable/total}
