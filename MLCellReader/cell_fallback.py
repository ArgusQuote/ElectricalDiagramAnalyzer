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

Which models load is pinned in ``production_models.json`` next to this file: a
run-directory name (resolved under the models directory, so the same pin works
on every host), the model file's sha256, and the per-target confidence floor and
allowed classes. A missing or mismatched model disables the fallback instead of
loading something unverified; the parser then behaves exactly like its OCR-only
predecessor.

Environment:
  CELL_CLASSIFIER_DISABLE=1       turn the fallback off
  CELL_CLASSIFIER_MODELS_DIR=...  models directory (default <data_root>/models)
  CELL_CLASSIFIER_DEVICE=cuda     run on GPU (default cpu, so the fallback never
                                  competes with EasyOCR workers for VRAM)
  CELL_CLASSIFIER_USE_LATEST=1    ignore the pin and load ``<target>_latest.txt``
                                  (experiments only; skips the sha256 check)
"""
from __future__ import annotations

import hashlib
import json
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
DEVICE_ENV_VAR = "CELL_CLASSIFIER_DEVICE"
USE_LATEST_ENV_VAR = "CELL_CLASSIFIER_USE_LATEST"

DEFAULT_DEVICE = "cpu"
PIN_FILE = Path(__file__).with_name("production_models.json")
MODEL_FILENAME = "model.pt"

# Used only when a target's pin (or the unpinned latest model) gives no floor.
DEFAULT_MIN_CONFIDENCE = 0.5

_TARGETS_BY_ROLE = {
    ROLE_TRIP: (TARGET_AMPS,),
    ROLE_POLES: (TARGET_POLES,),
    ROLE_COMBO: (TARGET_AMPS, TARGET_POLES),
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() not in ("", "0")


def _resolve_models_dir() -> Path:
    override = os.environ.get(MODELS_DIR_ENV_VAR)
    return Path(override).expanduser() if override else models_dir(data_root())


def _latest_model_dir(root: Path, target: str) -> Optional[Path]:
    pointer = root / f"{target}_latest.txt"
    if not pointer.is_file():
        return None
    run_dir = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
    return run_dir if run_dir.is_dir() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pins(pin_file: Path = PIN_FILE) -> Dict[str, dict]:
    """Read the per-target production pins ({target: {run, sha256, ...}})."""
    with pin_file.open(encoding="utf-8") as f:
        return json.load(f)["models"]


class CellFallbackReader:
    """Lazily loads the amps / poles classifiers once per process and reads cells."""

    _shared: Optional["CellFallbackReader"] = None
    _shared_lock = threading.Lock()

    def __init__(self):
        self.classifiers: Dict[str, object] = {}
        self.model_info: Dict[str, dict] = {}
        self.min_confidence: Dict[str, float] = {}
        self.allowed_classes: Dict[str, Optional[frozenset]] = {}
        self.device: str = os.environ.get(DEVICE_ENV_VAR, "").strip() or DEFAULT_DEVICE
        self.pinned: bool = not _env_flag(USE_LATEST_ENV_VAR)
        self.disabled_reason: Optional[str] = None
        self.stats: Counter = Counter()
        self._load()

    @classmethod
    def shared(cls) -> "CellFallbackReader":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
                if cls._shared.enabled:
                    print(f"[cell_fallback] classifier fallback ON: {cls._shared.describe()}")
                else:
                    print(f"[cell_fallback] classifier fallback OFF ({cls._shared.disabled_reason}); "
                          "parser stays OCR-only")
            return cls._shared

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    def describe(self) -> dict:
        """JSON-safe summary of what is loaded, for per-job result logging."""
        return {
            "enabled": self.enabled,
            "disabledReason": self.disabled_reason,
            "pinned": self.pinned,
            "device": self.device,
            "models": self.model_info,
            "minConfidence": self.min_confidence,
        }

    def _disable(self, reason: str) -> None:
        self.disabled_reason = reason
        self.classifiers.clear()

    def _run_dirs(self, root: Path) -> Optional[Dict[str, Tuple[Path, Optional[dict]]]]:
        """Map each target to (run_dir, pin or None); None after disabling."""
        targets = (TARGET_AMPS, TARGET_POLES)
        if not self.pinned:
            dirs = {}
            for target in targets:
                run_dir = _latest_model_dir(root, target)
                if run_dir is None:
                    self._disable(f"no {target}_latest.txt model under {root}")
                    return None
                dirs[target] = (run_dir, None)
            return dirs

        try:
            pins = load_pins()
        except (OSError, ValueError, KeyError) as exc:
            self._disable(f"cannot read {PIN_FILE.name}: {type(exc).__name__}: {exc}")
            return None

        dirs = {}
        for target in targets:
            pin = pins.get(target)
            if not pin or not pin.get("run") or not pin.get("sha256"):
                self._disable(f"{PIN_FILE.name} has no complete pin for {target}")
                return None
            dirs[target] = (root / pin["run"], pin)
        return dirs

    def _load(self) -> None:
        if _env_flag(DISABLE_ENV_VAR):
            self.disabled_reason = f"disabled by ${DISABLE_ENV_VAR}"
            return
        root = _resolve_models_dir()
        try:
            from MLCellReader.cell_classifier import CellClassifier
        except ImportError as exc:
            self.disabled_reason = f"classifier runtime unavailable: {exc}"
            return

        run_dirs = self._run_dirs(root)
        if run_dirs is None:
            return

        for target, (run_dir, pin) in run_dirs.items():
            model_file = run_dir / MODEL_FILENAME
            if not model_file.is_file():
                self._disable(f"missing {model_file}")
                return
            actual_sha = _sha256(model_file)
            if pin is not None and actual_sha != pin["sha256"]:
                self._disable(
                    f"{target} model sha256 mismatch for {model_file}: "
                    f"expected {pin['sha256'][:12]}..., found {actual_sha[:12]}..."
                )
                return
            try:
                classifier = CellClassifier.load(run_dir, device=self.device)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                self._disable(f"failed to load {run_dir}: {type(exc).__name__}: {exc}")
                return

            pin = pin or {}
            allowed = pin.get("allowed_classes")
            self.classifiers[target] = classifier
            self.min_confidence[target] = float(pin.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
            self.allowed_classes[target] = frozenset(allowed) if allowed else None
            self.model_info[target] = {
                "run": run_dir.name,
                "sha256": actual_sha,
                "classes": classifier.classes,
                "allowedClasses": sorted(allowed, key=str) if allowed else None,
            }

    def read_cell(self, role: str, gray_crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Read one raw (not de-gridded) body-cell crop for a trip / poles / combo column.

        Returns (ocr_like_text, confidence) or None when the model is disabled,
        unsure, predicts a class it is not trusted on, or sees an empty cell.
        """
        targets = _TARGETS_BY_ROLE.get(role)
        if not self.enabled or targets is None or gray_crop is None or gray_crop.size == 0:
            return None

        readings = {}
        confs = []
        for target in targets:
            label, conf = self.classifiers[target].predict(gray_crop)
            if conf < self.min_confidence[target]:
                self.stats[f"{role}_unsure"] += 1
                return None
            allowed = self.allowed_classes[target]
            if allowed is not None and label != BLANK_CLASS and label not in allowed:
                self.stats[f"{role}_untrusted_class"] += 1
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
