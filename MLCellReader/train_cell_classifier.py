#!/usr/bin/env python3
# MLCellReader/train_cell_classifier.py
"""
Train the breaker-cell classifier for one target from human labels.

    python MLCellReader/train_cell_classifier.py --target poles
    python MLCellReader/train_cell_classifier.py --target amps

Validation PDFs are held out as whole documents (never row-by-row), because
rows from the same drawing are near-duplicates and would inflate the score.
The report compares the model with what EasyOCR + the parser produced on the
*same* held-out crops, which is the number that says whether ML is worth wiring
into the parser.

Output: <data_root>/models/<target>_<stamp>/{model.pt, metrics.json, val_errors.jsonl}
and <data_root>/models/<target>_latest.txt pointing at the newest run.

``--smoke-weak-labels`` trains on the OCR suggestions instead of human labels.
It only proves the plumbing runs; its scores mean nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from MLCellReader.cell_classifier import MODEL_FILENAME, CellNet, save_checkpoint  # noqa: E402
from MLCellReader.cell_data import (  # noqa: E402
    BLANK_CLASS,
    TARGET_AMPS,
    TARGET_ROLES,
    data_root,
    list_labeled_samples,
    load_manifest,
    models_dir,
    normalize_cell_image,
)

CONFIDENCE_THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.99)


# ---------------------------------------------------------------- data

def weak_label_samples(root: Path, target: str) -> List[dict]:
    """Plumbing-test samples labeled by the current OCR (see module docstring)."""
    weak_key = "weak_amps" if target == TARGET_AMPS else "weak_poles"
    samples = []
    for record in load_manifest(root).values():
        if record.get("role") not in TARGET_ROLES[target]:
            continue
        value = record.get(weak_key)
        cls = BLANK_CLASS if value is None else str(int(value))
        samples.append(
            {
                "id": record["id"],
                "image": str(root / record["image"]),
                "class": cls,
                "source_pdf": record.get("source_pdf_md5") or "",
                "role": record["role"],
                "weak_class": cls,
            }
        )
    return samples


def split_by_pdf(samples: List[dict], val_fraction: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Hold out whole PDFs until at least val_fraction of the samples are in validation."""
    by_pdf: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        by_pdf[sample["source_pdf"]].append(sample)

    pdfs = sorted(by_pdf)
    if len(pdfs) < 2:
        print("[train] WARNING: only one source PDF; falling back to a random row split "
              "(validation score will be optimistic)")
        shuffled = samples[:]
        random.Random(seed).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * val_fraction))
        return shuffled[cut:], shuffled[:cut]

    random.Random(seed).shuffle(pdfs)
    target_val = val_fraction * len(samples)
    val_pdfs, val_count = set(), 0
    for pdf in pdfs:
        if val_count >= target_val or len(val_pdfs) == len(pdfs) - 1:
            break
        val_pdfs.add(pdf)
        val_count += len(by_pdf[pdf])

    train = [s for s in samples if s["source_pdf"] not in val_pdfs]
    val = [s for s in samples if s["source_pdf"] in val_pdfs]
    return train, val


def split_by_pdf_fold(samples: List[dict], folds: int, fold: int) -> Tuple[List[dict], List[dict]]:
    """
    K-fold split over whole PDFs, so every PDF is held out exactly once across the K runs.

    PDFs are assigned largest-first to the currently smallest fold, which keeps the
    folds close in size even though one drawing can hold a large share of the cells.
    """
    by_pdf: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        by_pdf[sample["source_pdf"]].append(sample)
    if len(by_pdf) < folds:
        raise ValueError(f"Only {len(by_pdf)} source PDFs; cannot make {folds} folds")

    fold_sizes = [0] * folds
    fold_of: Dict[str, int] = {}
    for pdf in sorted(by_pdf, key=lambda p: (-len(by_pdf[p]), p)):
        smallest = min(range(folds), key=lambda i: (fold_sizes[i], i))
        fold_of[pdf] = smallest
        fold_sizes[smallest] += len(by_pdf[pdf])

    train = [s for s in samples if fold_of[s["source_pdf"]] != fold]
    val = [s for s in samples if fold_of[s["source_pdf"]] == fold]
    return train, val


def augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Perturb a raw crop the way different drawings / scans / row bands differ."""
    h, w = gray.shape[:2]
    background = int(np.percentile(gray, 90))

    # Row-band jitter: each edge moves in (crop) or out (pad) by up to 8%.
    top, bottom = (int(rng.uniform(-0.08, 0.08) * h) for _ in range(2))
    left, right = (int(rng.uniform(-0.08, 0.08) * w) for _ in range(2))
    gray = cv2.copyMakeBorder(
        gray, max(0, -top), max(0, -bottom), max(0, -left), max(0, -right),
        cv2.BORDER_CONSTANT, value=background,
    )
    h2, w2 = gray.shape[:2]
    gray = gray[max(0, top): h2 - max(0, bottom), max(0, left): w2 - max(0, right)]
    if gray.size == 0 or min(gray.shape[:2]) < 4:
        return augment_fallback(h, w, background)

    if rng.random() < 0.3:
        angle = rng.uniform(-2.0, 2.0)
        matrix = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    if rng.random() < 0.4:  # lower source resolution
        factor = rng.uniform(0.35, 0.9)
        small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        if min(small.shape[:2]) >= 2:
            gray = cv2.resize(small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:  # stroke weight
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.erode(gray, kernel) if rng.random() < 0.5 else cv2.dilate(gray, kernel)
    if rng.random() < 0.2:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    alpha, beta = rng.uniform(0.7, 1.3), rng.uniform(-25, 25)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    if rng.random() < 0.2:
        noise = rng.normal(0, 8, gray.shape)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return gray


def augment_fallback(h: int, w: int, background: int) -> np.ndarray:
    return np.full((max(h, 4), max(w, 4)), background, dtype=np.uint8)


class CellDataset(Dataset):
    def __init__(self, samples: List[dict], class_to_idx: Dict[str, int], train: bool, seed: int):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.images = []
        for sample in samples:
            image = cv2.imread(sample["image"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Missing crop {sample['image']}")
            self.images.append(image)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.train:
            image = augment(image, self.rng)
        tensor = torch.from_numpy(normalize_cell_image(image))[None]
        return tensor, self.class_to_idx[self.samples[index]["class"]]


# ---------------------------------------------------------------- training

def class_weights(train: List[dict], classes: Sequence[str]) -> torch.Tensor:
    """Inverse-sqrt frequency, so the dominant '20' / '1' classes do not drown out '3'."""
    counts = Counter(s["class"] for s in train)
    weights = [1.0 / np.sqrt(max(1, counts[c])) for c in classes]
    scale = len(weights) / sum(weights)
    return torch.tensor([w * scale for w in weights], dtype=torch.float32)


@torch.no_grad()
def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, confs, truths, loss_sum = [], [], [], 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss_sum += float(criterion(logits, targets))
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        preds.extend(idx.tolist())
        confs.extend(conf.tolist())
        truths.extend(targets.tolist())
    return np.array(preds), np.array(confs), np.array(truths), loss_sum / max(1, len(truths))


def balanced_accuracy(preds: np.ndarray, truths: np.ndarray) -> float:
    recalls = [float((preds[truths == c] == c).mean()) for c in np.unique(truths)]
    return float(np.mean(recalls)) if recalls else 0.0


def train_model(train_ds, val_ds, classes, weights, args, device) -> Tuple[CellNet, dict]:
    model = CellNet(len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    best_state, best_score, best_epoch, history = None, -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += float(loss) * len(targets)
        scheduler.step()

        preds, _, truths, val_loss = predict_dataset(model, val_loader, device)
        score = balanced_accuracy(preds, truths)
        accuracy = float((preds == truths).mean()) if len(truths) else 0.0
        history.append({"epoch": epoch, "train_loss": running / len(train_ds),
                        "val_loss": val_loss, "val_acc": accuracy, "val_bal_acc": score})
        print(f"[train] epoch {epoch:3d}  train_loss={running / len(train_ds):.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={accuracy:.4f}  val_bal_acc={score:.4f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_bal_acc": best_score, "history": history}


# ---------------------------------------------------------------- reporting

def build_report(model, val_ds, val, classes, device, batch_size) -> Tuple[dict, List[dict]]:
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, confs, truths, _ = predict_dataset(model, loader, device)

    true_names = [classes[t] for t in truths]
    pred_names = [classes[p] for p in preds]
    ocr_names = [s["weak_class"] for s in val]

    confusion = {t: Counter() for t in classes}
    for t, p in zip(true_names, pred_names):
        confusion[t][p] += 1

    per_class = {}
    for cls in classes:
        rows = [i for i, t in enumerate(true_names) if t == cls]
        if rows:
            per_class[cls] = {
                "n": len(rows),
                "model_acc": sum(pred_names[i] == cls for i in rows) / len(rows),
                "ocr_acc": sum(ocr_names[i] == cls for i in rows) / len(rows),
            }

    gated = []
    for threshold in CONFIDENCE_THRESHOLDS:
        kept = confs >= threshold
        gated.append({
            "threshold": threshold,
            "coverage": float(kept.mean()) if len(kept) else 0.0,
            "accuracy_when_confident": float((preds[kept] == truths[kept]).mean()) if kept.any() else None,
            # The deployment plan: trust the model when confident, otherwise keep OCR.
            "hybrid_acc": float(np.mean([
                (pred_names[i] if kept[i] else ocr_names[i]) == true_names[i]
                for i in range(len(true_names))
            ])) if len(true_names) else None,
        })

    # The two numbers that decide whether the model helps: cells OCR got wrong that the
    # model gets right, and cells OCR got right that the model would break.
    ocr_wrong = [i for i in range(len(true_names)) if ocr_names[i] != true_names[i]]
    ocr_right = [i for i in range(len(true_names)) if ocr_names[i] == true_names[i]]

    n = len(true_names)
    report = {
        "val_samples": n,
        "model_acc": sum(p == t for p, t in zip(pred_names, true_names)) / n if n else None,
        "ocr_acc": sum(o == t for o, t in zip(ocr_names, true_names)) / n if n else None,
        "ocr_wrong": len(ocr_wrong),
        "model_fixes_ocr_wrong": sum(pred_names[i] == true_names[i] for i in ocr_wrong),
        "ocr_right": len(ocr_right),
        "model_breaks_ocr_right": sum(pred_names[i] != true_names[i] for i in ocr_right),
        "per_class": per_class,
        "confidence_gating": gated,
        "confusion": {t: dict(c) for t, c in confusion.items() if c},
    }
    errors = [
        {"id": s["id"], "true": t, "model": p, "model_conf": round(float(c), 4), "ocr": o}
        for s, t, p, c, o in zip(val, true_names, pred_names, confs, ocr_names)
        if p != t or o != t
    ]
    return report, errors


def print_report(report: dict) -> None:
    print("\n[report] ===== held-out PDFs =====")
    print(f"[report] samples: {report['val_samples']}")
    print(f"[report] EasyOCR+parser accuracy: {report['ocr_acc']:.4f}")
    print(f"[report] model accuracy:          {report['model_acc']:.4f}")
    print(f"[report] OCR wrong on {report['ocr_wrong']} cells; model fixes {report['model_fixes_ocr_wrong']}")
    print(f"[report] OCR right on {report['ocr_right']} cells; model breaks {report['model_breaks_ocr_right']}")
    print("[report] per class (n, model, ocr):")
    for cls, row in report["per_class"].items():
        print(f"[report]   {cls:>6}  n={row['n']:5d}  model={row['model_acc']:.3f}  ocr={row['ocr_acc']:.3f}")
    print("[report] confidence gating (use model only when conf >= threshold, else OCR):")
    for row in report["confidence_gating"]:
        acc = row["accuracy_when_confident"]
        print(f"[report]   >= {row['threshold']:.2f}  coverage={row['coverage']:.3f}  "
              f"model_acc_there={'n/a' if acc is None else f'{acc:.4f}'}  hybrid_acc={row['hybrid_acc']:.4f}")


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the breaker-cell classifier.")
    parser.add_argument("--target", choices=sorted(TARGET_ROLES), required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-class-count", type=int, default=5,
                        help="Classes with fewer labels are left to EasyOCR")
    parser.add_argument("--folds", type=int, default=None,
                        help="K-fold over whole PDFs; use with --fold (overrides --val-fraction)")
    parser.add_argument("--fold", type=int, default=None, help="Which fold to hold out (0-based)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: cuda when available)")
    parser.add_argument("--smoke-weak-labels", action="store_true",
                        help="Train on OCR suggestions (plumbing test only)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = data_root(args.data_root)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = weak_label_samples(root, args.target) if args.smoke_weak_labels \
        else list_labeled_samples(root, args.target)
    counts = Counter(s["class"] for s in samples)
    classes = sorted((c for c, n in counts.items() if n >= args.min_class_count),
                     key=lambda c: (c == BLANK_CLASS, int(c) if c.isdigit() else 0))
    dropped = {c: n for c, n in counts.items() if n < args.min_class_count}
    samples = [s for s in samples if s["class"] in classes]

    print(f"[train] target={args.target} data_root={root}")
    print(f"[train] classes ({len(classes)}): {[(c, counts[c]) for c in classes]}")
    if dropped:
        print(f"[train] too rare, left to EasyOCR: {dropped}")
    if len(classes) < 2 or len(samples) < 20:
        print("[train] Not enough labeled data yet (need >= 2 classes and >= 20 samples). "
              "Label more crops with MLCellReader/label_server.py.")
        return 1

    if (args.folds is None) != (args.fold is None):
        print("[train] --folds and --fold must be given together")
        return 2
    if args.folds is not None:
        if not 0 <= args.fold < args.folds:
            print(f"[train] --fold must be in 0..{args.folds - 1}")
            return 2
        train, val = split_by_pdf_fold(samples, args.folds, args.fold)
    else:
        train, val = split_by_pdf(samples, args.val_fraction, args.seed)
    print(f"[train] train={len(train)}  val={len(val)}  "
          f"val PDFs={sorted({s['source_pdf'][:10] for s in val})}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds = CellDataset(train, class_to_idx, train=True, seed=args.seed)
    val_ds = CellDataset(val, class_to_idx, train=False, seed=args.seed)

    start = time.perf_counter()
    model, training = train_model(train_ds, val_ds, classes, class_weights(train, classes), args, device)
    report, errors = build_report(model, val_ds, val, classes, device, args.batch_size)
    print_report(report)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_SMOKE" if args.smoke_weak_labels else ""
    if args.folds is not None:
        suffix += f"_fold{args.fold}of{args.folds}"
    out_dir = models_dir(root) / f"{args.target}_{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "trained_at": stamp,
        "smoke_weak_labels": args.smoke_weak_labels,
        "class_counts": {c: counts[c] for c in classes},
        "dropped_rare_classes": dropped,
        "train_samples": len(train),
        "val_samples": len(val),
        "val_pdfs": sorted({s["source_pdf"] for s in val}),
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_seconds": round(time.perf_counter() - start, 1),
    }
    save_checkpoint(out_dir / MODEL_FILENAME, model.cpu(), classes, args.target, meta)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "training": training, "report": report}, handle, indent=2)
    with open(out_dir / "val_errors.jsonl", "w", encoding="utf-8") as handle:
        for row in errors:
            handle.write(json.dumps(row) + "\n")
    # Fold runs are evaluations, not the model to ship, so they never become "latest".
    if not args.smoke_weak_labels and args.folds is None:
        (models_dir(root) / f"{args.target}_latest.txt").write_text(str(out_dir) + "\n")

    print(f"\n[train] saved {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
