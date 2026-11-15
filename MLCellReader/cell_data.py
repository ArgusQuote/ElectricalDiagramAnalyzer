# MLCellReader/cell_data.py
"""
Shared storage layout, label schema, and image normalization for the
breaker-cell (trip amps / poles) classifier.

Everything that touches the annotation data goes through this module so the
harvester, the labeling server, the trainer, and the runtime classifier all
agree on where files live and what a crop looks like when it reaches the model.

Data root layout (default ``~/Documents/CellAnnotations``, kept outside the
repo because the crops come from customer drawings):

    crops/<role>/<cell_id>.png   untouched grayscale cell crop
    manifest.jsonl               one record per harvested crop (append-only)
    labels.jsonl                 one record per human label (append-only,
                                 the last record for a cell_id wins)
    models/<target>_<stamp>/     trained classifiers
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import cv2
import numpy as np

DEFAULT_DATA_ROOT = Path("~/Documents/CellAnnotations").expanduser()
DATA_ROOT_ENV_VAR = "CELL_ANNOTATIONS_ROOT"

MANIFEST_FILENAME = "manifest.jsonl"
LABELS_FILENAME = "labels.jsonl"
CROPS_DIRNAME = "crops"
MODELS_DIRNAME = "models"

# Cell roles as the parser names its body columns.
ROLE_TRIP = "trip"
ROLE_POLES = "poles"
ROLE_COMBO = "combo"
CELL_ROLES = (ROLE_TRIP, ROLE_POLES, ROLE_COMBO)

# What a classifier predicts, and which crop roles carry that information.
TARGET_AMPS = "amps"
TARGET_POLES = "poles"
TARGET_ROLES = {
    TARGET_AMPS: (ROLE_TRIP, ROLE_COMBO),
    TARGET_POLES: (ROLE_POLES, ROLE_COMBO),
}

# Label states written by the labeling server.
STATE_LABELED = "labeled"
STATE_BLANK = "blank"            # the cell is genuinely empty / SPACE / SPARE / '--'
STATE_UNREADABLE = "unreadable"  # a human cannot read it either -> excluded from training
LABEL_STATES = (STATE_LABELED, STATE_BLANK, STATE_UNREADABLE)

# Class name used for empty cells in every classifier.
BLANK_CLASS = "blank"

VALID_POLES = (1, 2, 3)

# Crops are letterboxed onto a fixed canvas. Body cells are wide and short, and
# combo cells ("20 A / 1") are wider still, so the canvas keeps a 4:1 aspect.
MODEL_INPUT_H = 48
MODEL_INPUT_W = 192


def data_root(override: Optional[str | Path] = None) -> Path:
    """Resolve the annotation data root: explicit arg, then env var, then default."""
    if override:
        return Path(override).expanduser()
    env_value = os.environ.get(DATA_ROOT_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_DATA_ROOT


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_FILENAME


def labels_path(root: Path) -> Path:
    return root / LABELS_FILENAME


def crops_dir(root: Path, role: str) -> Path:
    return root / CROPS_DIRNAME / role


def models_dir(root: Path) -> Path:
    return root / MODELS_DIRNAME


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file, skipping blank lines; missing file yields nothing."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[cell_data] Skipping corrupt line {line_number} in {path}: {exc}")


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Append records to a JSONL file and return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_manifest(root: Path) -> Dict[str, dict]:
    """Return manifest records keyed by cell id (first record for an id wins)."""
    records: Dict[str, dict] = {}
    for record in iter_jsonl(manifest_path(root)):
        cell_id = record.get("id")
        if cell_id and cell_id not in records:
            records[cell_id] = record
    return records


def load_labels(root: Path) -> Dict[str, dict]:
    """Return the latest human label per cell id."""
    labels: Dict[str, dict] = {}
    for record in iter_jsonl(labels_path(root)):
        cell_id = record.get("id")
        if cell_id:
            labels[cell_id] = record
    return labels


def make_label_record(
    cell_id: str,
    state: str,
    amps: Optional[int] = None,
    poles: Optional[int] = None,
) -> dict:
    """Build a validated label record; raises ValueError on an impossible label."""
    if state not in LABEL_STATES:
        raise ValueError(f"Unknown label state {state!r}")
    if state != STATE_LABELED:
        amps = None
        poles = None
    # 0 is allowed because some drawings literally print "0 A" (e.g. shunt-trip rows).
    if amps is not None and (not isinstance(amps, int) or amps < 0):
        raise ValueError(f"amps must be a non-negative integer, got {amps!r}")
    if poles is not None and poles not in VALID_POLES:
        raise ValueError(f"poles must be one of {VALID_POLES}, got {poles!r}")
    if state == STATE_LABELED and amps is None and poles is None:
        raise ValueError("A 'labeled' cell needs amps and/or poles; use 'blank' for empty cells")
    return {
        "id": cell_id,
        "state": state,
        "amps": amps,
        "poles": poles,
        "ts": int(time.time()),
    }


def class_for_target(label: dict, target: str) -> Optional[str]:
    """
    Map a human label to a classifier class name for one target.

    Returns None when the label carries no usable information for the target
    (unreadable cell, or a combo cell where the human only filled the other
    half), which excludes the sample from that target's training set.
    """
    state = label.get("state")
    if state == STATE_BLANK:
        return BLANK_CLASS
    if state != STATE_LABELED:
        return None
    value = label.get(target)
    return None if value is None else str(int(value))


def normalize_cell_image(gray: np.ndarray) -> np.ndarray:
    """
    Letterbox a grayscale cell crop onto the model canvas.

    Aspect ratio is preserved (digit shapes must not be squashed) and the
    padding uses the crop's own background level so the pad edge is not
    mistaken for a grid line. Returns float32 in [0, 1] with ink = 1.
    """
    if gray is None or gray.size == 0:
        raise ValueError("Empty cell crop")
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    scale = min(MODEL_INPUT_H / h, MODEL_INPUT_W / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)

    background = int(np.percentile(gray, 90))
    canvas = np.full((MODEL_INPUT_H, MODEL_INPUT_W), background, dtype=np.uint8)
    y0 = (MODEL_INPUT_H - new_h) // 2
    x0 = (MODEL_INPUT_W - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return 1.0 - canvas.astype(np.float32) / 255.0


def list_labeled_samples(root: Path, target: str) -> List[dict]:
    """
    Join manifest + labels into training samples for one target.

    Each sample is ``{"id", "image", "class", "source_pdf", "role", "weak_class"}``
    where ``weak_class`` is what the current OCR parser produced for the same
    crop, kept so the trainer can report an EasyOCR baseline on the same rows.
    """
    if target not in TARGET_ROLES:
        raise ValueError(f"Unknown target {target!r}; choose from {sorted(TARGET_ROLES)}")

    manifest = load_manifest(root)
    labels = load_labels(root)
    roles = TARGET_ROLES[target]
    weak_key = "weak_amps" if target == TARGET_AMPS else "weak_poles"

    samples = []
    for cell_id, label in labels.items():
        record = manifest.get(cell_id)
        if record is None or record.get("role") not in roles:
            continue
        cls = class_for_target(label, target)
        if cls is None:
            continue
        weak_value = record.get(weak_key)
        samples.append(
            {
                "id": cell_id,
                "image": str(root / record["image"]),
                "class": cls,
                "source_pdf": record.get("source_pdf_md5") or record.get("source_pdf", ""),
                "role": record["role"],
                "weak_class": BLANK_CLASS if weak_value is None else str(int(weak_value)),
            }
        )
    return samples
