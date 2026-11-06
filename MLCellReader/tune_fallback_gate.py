#!/usr/bin/env python3
# MLCellReader/tune_fallback_gate.py
"""
Choose the cell-classifier confidence floor and trusted classes from k-fold runs.

    python MLCellReader/tune_fallback_gate.py                 # newest 5-fold set per target
    python MLCellReader/tune_fallback_gate.py --folds 5 --out /tmp/gate.json

Each fold model is scored only on the PDFs it was held out from, in the mode the
parser uses: OCR's reading stands whenever it parsed, and the model may only
fill a cell OCR returned nothing for. For every candidate floor the report gives
how many cells the model fills, how many of those fills are wrong (a value
where the cell is blank, or the wrong value), and the total cell errors; per
predicted class it gives how often a fill of that class is right, which is what
``allowed_classes`` in ``production_models.json`` should be based on.

Combo cells are scored per target (amps and poles separately); in the parser a
combo fill needs both targets above their floors, so combo results are at least
as conservative as reported here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from MLCellReader.cell_classifier import CellClassifier  # noqa: E402
from MLCellReader.cell_fallback import DEFAULT_MIN_CONFIDENCE, PIN_FILE, load_pins  # noqa: E402
from MLCellReader.cell_data import (  # noqa: E402
    BLANK_CLASS,
    TARGET_AMPS,
    TARGET_POLES,
    data_root,
    list_labeled_samples,
    models_dir,
)

THRESHOLDS = (0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9)
BATCH_SIZE = 256
FOLD_RUN_RE = re.compile(r"^(?P<target>amps|poles)_(?P<stamp>\d{8}_\d{6})_fold(?P<fold>\d+)of(?P<folds>\d+)$")


def newest_fold_set(root: Path, target: str, folds: int) -> List[Path]:
    """Newest run directory for each fold index of one target."""
    newest: Dict[int, Path] = {}
    for run_dir in sorted(root.iterdir()):
        m = FOLD_RUN_RE.match(run_dir.name)
        if not m or m["target"] != target or int(m["folds"]) != folds:
            continue
        if not (run_dir / "model.pt").is_file():
            continue
        newest[int(m["fold"])] = run_dir  # sorted by stamp, so the last one wins
    missing = [f for f in range(folds) if f not in newest]
    if missing:
        raise SystemExit(f"[tune] {target}: no completed run for fold(s) {missing} under {root}")
    return [newest[f] for f in range(folds)]


def held_out_predictions(run_dirs: List[Path], samples: List[dict]) -> List[dict]:
    """Predict every sample with the one fold model that never trained on its PDF."""
    rows = []
    for run_dir in run_dirs:
        meta = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["meta"]
        val_pdfs = set(meta["val_pdfs"])
        held_out = [s for s in samples if s["source_pdf"] in val_pdfs]
        classifier = CellClassifier.load(run_dir)
        for start in range(0, len(held_out), BATCH_SIZE):
            chunk = held_out[start:start + BATCH_SIZE]
            crops = [cv2.imread(s["image"], cv2.IMREAD_GRAYSCALE) for s in chunk]
            kept = [(s, c) for s, c in zip(chunk, crops) if c is not None and c.size]
            preds = classifier.predict_batch([c for _, c in kept])
            for (sample, _), (label, conf) in zip(kept, preds):
                rows.append({**sample, "pred": label, "conf": conf, "fold_run": run_dir.name})
        print(f"[tune] {run_dir.name}: {len(held_out)} held-out cells")
    return rows


def sweep(rows: List[dict]) -> List[dict]:
    """Fill-only-when-OCR-missed outcome at each candidate floor."""
    ocr_errors = sum(r["weak_class"] != r["class"] for r in rows)
    results = []
    for threshold in THRESHOLDS:
        filled = right = phantom = wrong_value = errors = 0
        for r in rows:
            final = r["weak_class"]
            if r["weak_class"] == BLANK_CLASS and r["pred"] != BLANK_CLASS and r["conf"] >= threshold:
                final = r["pred"]
                filled += 1
                if r["pred"] == r["class"]:
                    right += 1
                elif r["class"] == BLANK_CLASS:
                    phantom += 1
                else:
                    wrong_value += 1
            errors += final != r["class"]
        results.append({
            "threshold": threshold,
            "filled": filled,
            "fill_right": right,
            "fill_phantom": phantom,
            "fill_wrong_value": wrong_value,
            "total_errors": errors,
            "ocr_only_errors": ocr_errors,
        })
    return results


def per_class_fills(rows: List[dict], threshold: float) -> Dict[str, dict]:
    """How trustworthy a fill of each predicted class is at one floor."""
    stats: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["weak_class"] == BLANK_CLASS and r["pred"] != BLANK_CLASS and r["conf"] >= threshold:
            stats[r["pred"]]["filled"] += 1
            stats[r["pred"]]["right"] += r["pred"] == r["class"]
    return {
        cls: {"filled": c["filled"], "right": c["right"], "precision": round(c["right"] / c["filled"], 3)}
        for cls, c in sorted(stats.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
    }


def evaluate_pin(rows: List[dict], pin: dict) -> dict:
    """Fill-only outcome under one target's production_models.json gate."""
    floor = float(pin.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
    allowed = set(pin["allowed_classes"]) if pin.get("allowed_classes") else None
    filled = right = phantom = 0
    for r in rows:
        if r["weak_class"] != BLANK_CLASS or r["pred"] == BLANK_CLASS or r["conf"] < floor:
            continue
        if allowed is not None and r["pred"] not in allowed:
            continue
        filled += 1
        right += r["pred"] == r["class"]
        phantom += r["class"] == BLANK_CLASS
    missed = sum(r["weak_class"] == BLANK_CLASS and r["class"] != BLANK_CLASS for r in rows)
    return {
        "min_confidence": floor,
        "allowed_classes": sorted(allowed, key=int) if allowed else None,
        "ocr_missed_cells": missed,
        "filled": filled,
        "fill_right": right,
        "fill_phantom": phantom,
        "fill_wrong_value": filled - right - phantom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="Write the full report as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = data_root(args.data_root)
    pins = load_pins()
    report = {}
    for target in (TARGET_POLES, TARGET_AMPS):
        run_dirs = newest_fold_set(models_dir(root), target, args.folds)
        rows = held_out_predictions(run_dirs, list_labeled_samples(root, target))
        results = sweep(rows)
        report[target] = {
            "fold_runs": [d.name for d in run_dirs],
            "cells": len(rows),
            "sweep": results,
            "per_class_fills": {str(t): per_class_fills(rows, t) for t in THRESHOLDS},
            "pinned_gate": evaluate_pin(rows, pins.get(target, {})),
        }
        print(f"\n[tune] ===== {target}: {len(rows)} held-out cells =====")
        print(f"[tune] {'floor':>5} {'filled':>7} {'right':>6} {'phantom':>8} {'wrongval':>9} "
              f"{'errors':>7} (OCR only: {results[0]['ocr_only_errors']})")
        for r in results:
            print(f"[tune] {r['threshold']:>5.2f} {r['filled']:>7} {r['fill_right']:>6} "
                  f"{r['fill_phantom']:>8} {r['fill_wrong_value']:>9} {r['total_errors']:>7}")
        print(f"[tune] pinned gate ({PIN_FILE.name}): {report[target]['pinned_gate']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[tune] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
