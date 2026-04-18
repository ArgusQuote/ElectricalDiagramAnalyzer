#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dev harness for the ML-based PageFilter (V4).

Demonstrates how to call PageFilterV4, which uses a fine-tuned
MobileNetV2 model instead of OCR/regex/footprint heuristics.

Usage:
    # Single file (default paths)
    python DevEnv/PageFilterMLDevEnv.py

    # Specific PDF
    python DevEnv/PageFilterMLDevEnv.py --pdf ~/Documents/new\ panels/derek2.pdf

    # Batch all PDFs in a folder
    python DevEnv/PageFilterMLDevEnv.py --pdf_dir ~/Documents/new\ panels

    # Custom model or confidence threshold
    python DevEnv/PageFilterMLDevEnv.py --model ~/path/to/model.pt --threshold 0.7

    # Enable debug JSON logs
    python DevEnv/PageFilterMLDevEnv.py --debug
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from PageFilter.PageFilterV4 import PageFilter

# ---------- DEFAULTS ----------
DEFAULT_PDF = Path("~/Documents/new panels/derekfirst.pdf").expanduser()
DEFAULT_OUT_DIR = Path("~/Documents/Diagrams/PdfOutput_ML").expanduser()
DEFAULT_MODEL = Path(
    "~/Documents/TableAnnotations/models_classifier/best/model.pt"
).expanduser()


def run_filter(pdf_path: Path, out_dir: Path, model_path: Path,
               threshold: float, debug: bool) -> None:
    """Run the ML page filter on a single PDF and print results."""
    print(f"\n{'=' * 60}")
    print(f"  PDF:        {pdf_path.name}")
    print(f"  Model:      {model_path}")
    print(f"  Threshold:  {threshold}")
    print(f"  Output:     {out_dir}")
    print(f"{'=' * 60}")

    pf = PageFilter(
        output_dir=str(out_dir),
        model_path=str(model_path),
        confidence_threshold=threshold,
        render_dpi=150,
        verbose=True,
        debug=debug,
        use_ghostscript_letter=True,
        letter_orientation="landscape",
        gs_use_cropbox=True,
        gs_compat="1.7",
    )

    t0 = time.perf_counter()
    kept, dropped, filtered_pdf, log_path = pf.readPdf(str(pdf_path))
    elapsed = time.perf_counter() - t0

    print(f"\n--- Results ({elapsed:.2f}s) ---")
    print(f"  Kept pages:    {kept}")
    print(f"  Dropped pages: {dropped}")
    if filtered_pdf:
        size_kb = os.path.getsize(filtered_pdf) / 1024
        print(f"  Output PDF:    {filtered_pdf}  ({size_kb:.0f} KB)")
    else:
        print("  Output PDF:    (none — no pages kept)")
    if log_path:
        print(f"  Debug log:     {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dev harness for ML-based PageFilter (V4)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pdf", type=str, default=None,
        help="Path to a single PDF to filter")
    group.add_argument(
        "--pdf_dir", type=str, default=None,
        help="Directory of PDFs to batch-filter")
    parser.add_argument(
        "--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL),
        help=f"Path to model.pt checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for keeping a page (default: 0.5)")
    parser.add_argument(
        "--debug", action="store_true",
        help="Write per-page JSON decision log")

    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser()
    model_path = Path(args.model).expanduser()

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Train one with:  python MLTableDetection/train_page_classifier.py")
        return 1

    pdfs: list[Path] = []
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir).expanduser()
        pdfs = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
        if not pdfs:
            print(f"[ERROR] No PDFs found in {pdf_dir}")
            return 1
    elif args.pdf:
        p = Path(args.pdf).expanduser()
        if not p.is_file():
            print(f"[ERROR] PDF not found: {p}")
            return 1
        pdfs = [p]
    else:
        if not DEFAULT_PDF.is_file():
            print(f"[ERROR] Default PDF not found: {DEFAULT_PDF}")
            print("Pass --pdf or --pdf_dir to specify input.")
            return 1
        pdfs = [DEFAULT_PDF]

    print(f"Processing {len(pdfs)} PDF(s) ...")
    for pdf in pdfs:
        run_filter(pdf, out_dir, model_path, args.threshold, args.debug)

    print(f"\nAll outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
