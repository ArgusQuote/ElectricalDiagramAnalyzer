# MLCellReader/cell_fallback.py
"""
Runtime bridge between the trained cell classifiers and the breaker parser.

The parser only asks for a reading after both of its EasyOCR passes (de-gridded
and raw) failed to yield a usable value for a cell. The classifier's answer is
returned as the text EasyOCR would have produced for that cell ("20", "2",
"20/2", "20A", "2P"), so the parser's own value checks still decide whether the
row becomes a breaker. Cross-validation on the labeled corpus showed this
"fill only what OCR missed" use cuts cell errors sharply, while letting the
model override successful OCR reads introduced silent wrong values (20 -> 15).

Model lookup: ``$CELL_CLASSIFIER_MODELS_DIR`` if set, else ``<data_root>/models``;
in either, ``amps_latest.txt`` / ``poles_latest.txt`` name the run directory.
When the models, torch, or the files are unavailable the fallback reports itself
disabled and the parser behaves exactly like its OCR-only predecessor.
"""
from __future__ import annotations

import os
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from MLCellReader.cell_data import (
    BLANK_CLASS,
    ROLE_COMBO,
    ROLE_POLES,
    ROLE_TRIP,
    TARGET_AMPS,
    TARGET_POLES,
    data_root,
    models_dir,
)

MODELS_DIR_ENV_VAR = "CELL_CLASSIFIER_MODELS_DIR"
DISABLE_ENV_VAR = "CELL_CLASSIFIER_DISABLE"

# Below this softmax confidence the classifier is ignored and the cell stays
# unread, exactly as before. Not yet tuned on full-pipeline batch runs.
MIN_CONFIDENCE = 0.5

_TARGETS_BY_ROLE = {
    ROLE_TRIP: (TARGET_AMPS,),
    ROLE_POLES: (TARGET_POLES,),
    ROLE_COMBO: (TARGET_AMPS, TARGET_POLES),
}


def _resolve_models_dir() -> Path:
    override = os.environ.get(MODELS_DIR_ENV_VAR)
    return Path(override).expanduser() if override else models_dir(data_root())


def _latest_model_dir(root: Path, target: str) -> Optional[Path]:
    pointer = root / f"{target}_latest.txt"
    if not pointer.is_file():
        return None
    run_dir = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
    return run_dir if run_dir.is_dir() else None


class CellFallbackReader:
    """Lazily loads the amps / poles classifiers once per process and reads cells."""

    _shared: Optional["CellFallbackReader"] = None
    _shared_lock = threading.Lock()

    def __init__(self):
        self.classifiers: Dict[str, object] = {}
        self.disabled_reason: Optional[str] = None
        self.stats: Counter = Counter()
        self._load()

    @classmethod
    def shared(cls) -> "CellFallbackReader":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
                if not cls._shared.enabled:
                    print(f"[cell_fallback] classifier fallback OFF ({cls._shared.disabled_reason}); "
                          "parser stays OCR-only")
            return cls._shared

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    def _load(self) -> None:
        if os.environ.get(DISABLE_ENV_VAR, "").strip() not in ("", "0"):
            self.disabled_reason = f"disabled by ${DISABLE_ENV_VAR}"
            return
        root = _resolve_models_dir()
        try:
            from MLCellReader.cell_classifier import CellClassifier
        except ImportError as exc:
            self.disabled_reason = f"classifier runtime unavailable: {exc}"
            return

        for target in (TARGET_AMPS, TARGET_POLES):
            run_dir = _latest_model_dir(root, target)
            if run_dir is None:
                self.disabled_reason = f"no {target} model under {root}"
                self.classifiers.clear()
                return
            try:
                self.classifiers[target] = CellClassifier.load(run_dir)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                self.disabled_reason = f"failed to load {run_dir}: {type(exc).__name__}: {exc}"
                self.classifiers.clear()
                return
        print(f"[cell_fallback] loaded classifiers from {root}: "
              f"{ {t: c.classes for t, c in self.classifiers.items()} }")

    def read_cell(self, role: str, gray_crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Read one raw (not de-gridded) body-cell crop for a trip / poles / combo column.

        Returns (ocr_like_text, confidence) or None when the model is disabled,
        unsure, or sees an empty cell.
        """
        targets = _TARGETS_BY_ROLE.get(role)
        if not self.enabled or targets is None or gray_crop is None or gray_crop.size == 0:
            return None

        readings = {}
        confs = []
        for target in targets:
            label, conf = self.classifiers[target].predict(gray_crop)
            if conf < MIN_CONFIDENCE:
                self.stats[f"{role}_unsure"] += 1
                return None
            readings[target] = label
            confs.append(conf)

        amps = readings.get(TARGET_AMPS)
        poles = readings.get(TARGET_POLES)
        amps = None if amps == BLANK_CLASS else amps
        poles = None if poles == BLANK_CLASS else poles

        if role == ROLE_TRIP:
            text = amps
        elif role == ROLE_POLES:
            text = poles
        elif amps and poles:
            text = f"{amps}/{poles}"
        elif amps:
            text = f"{amps}A"
        elif poles:
            text = f"{poles}P"
        else:
            text = None

        if not text:
            self.stats[f"{role}_blank"] += 1
            return None
        self.stats[f"{role}_read"] += 1
        return text, min(confs)
