# MLCellReader/cell_classifier.py
"""
Small CNN that reads one breaker body cell (trip amps or poles) as a
classification over a closed vocabulary, plus the runtime wrapper the parser
will use as a confidence-gated second opinion next to EasyOCR.

The vocabulary is whatever classes had enough human labels at training time
(e.g. "15", "20", ..., "blank"). Values outside it are never predicted, so the
caller must keep EasyOCR as the fallback whenever the classifier is unsure.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from MLCellReader.cell_data import MODEL_INPUT_H, MODEL_INPUT_W, normalize_cell_image

MODEL_FILENAME = "model.pt"
POOLED_WIDTH = 8  # keeps left-to-right order so "20" and "02" stay distinguishable


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CellNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32), nn.MaxPool2d(2),
            _conv_block(32, 64), nn.MaxPool2d(2),
            _conv_block(64, 128), nn.MaxPool2d(2),
            _conv_block(128, 128),
            nn.AdaptiveAvgPool2d((1, POOLED_WIDTH)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * POOLED_WIDTH, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def save_checkpoint(path: Path, model: CellNet, classes: Sequence[str], target: str, meta: dict) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(classes),
            "target": target,
            "input_hw": [MODEL_INPUT_H, MODEL_INPUT_W],
            "meta": meta,
        },
        path,
    )


class CellClassifier:
    """Loads a trained checkpoint and classifies raw grayscale cell crops."""

    def __init__(self, model: CellNet, classes: List[str], target: str, device: torch.device):
        self.model = model.eval()
        self.classes = classes
        self.target = target
        self.device = device

    @classmethod
    def load(cls, model_dir_or_file: str | Path, device: str | None = None) -> "CellClassifier":
        path = Path(model_dir_or_file).expanduser()
        if path.is_dir():
            path = path / MODEL_FILENAME
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
        if checkpoint.get("input_hw") != [MODEL_INPUT_H, MODEL_INPUT_W]:
            raise ValueError(
                f"{path} was trained for input {checkpoint.get('input_hw')}, "
                f"but this code normalizes to {[MODEL_INPUT_H, MODEL_INPUT_W]}"
            )
        model = CellNet(len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(torch_device)
        return cls(model, list(checkpoint["classes"]), checkpoint["target"], torch_device)

    @torch.no_grad()
    def predict_batch(self, gray_crops: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        """Return (class_name, softmax_confidence) per crop."""
        if not gray_crops:
            return []
        batch = np.stack([normalize_cell_image(crop) for crop in gray_crops])[:, None]
        logits = self.model(torch.from_numpy(batch).to(self.device))
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        return [(self.classes[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]

    def predict(self, gray_crop: np.ndarray) -> Tuple[str, float]:
        return self.predict_batch([gray_crop])[0]
