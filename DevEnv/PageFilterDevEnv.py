#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PageFilterDevEnv.py
A minimal dev harness for PageFilter (labels + E-sheet + footprints), modeled after your reference.

Usage examples:
  # Single file (default paths)
  python PageFilterDevEnv.py

  # Specific file
  python PageFilterDevEnv.py --pdf ~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/generic2.pdf

  # Batch all PDFs in a folder (non-recursive)
  python PageFilterDevEnv.py --pdf_dir ~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf

  # Enable debug JSON logs and bump DPI
  python PageFilterDevEnv.py --debug --dpi 400
"""

import sys
import os
from pathlib import Path
import time
import argparse

# ----- Path setup (same pattern as your reference) -----
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports from your project
from PageFilter.PageFilter import PageFilter

# ----- Defaults (same locations you used) -----
DEFAULT_INPUT_PDF = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/good_tester_combo.pdf").expanduser()
DEFAULT_IN_DIR   = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf").expanduser()
DEFAULT_OUT_DIR  = Path("~/ElectricalDiagramAnalyzer/DevEnv/PdfOuput").expanduser()


def run_one(pdf_path: Path, out_dir: Path, args) -> None:
    """Run PageFilter on a single PDF and print a concise summary."""
    t0 = time.time()

    FILTER_OUT_DIR = out_dir
    FILTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Construct PageFilter with dev-friendly knobs (mirrors your reference)
    FILTER = PageFilter(
        output_dir=str(FILTER_OUT_DIR),
        dpi=args.dpi,                     # raster DPI used only for undecided pages
        longest_cap_px=args.longest_cap_px,
        proc_scale=args.proc_scale,
        use_ocr=not args.no_ocr,
        ocr_gpu=args.ocr_gpu,
        verbose=not args.quiet,
        debug=args.debug,                 # when True, writes JSON to <out>/filter_debug/<base>_filter_log.json
        rect_w_fr_range=(args.rect_w_min, args.rect_w_max),
        rect_h_fr_range=(args.rect_h_min, args.rect_h_max),
        min_rectangularity=args.min_rectangularity,
        min_rect_count=args.min_rect_count,
        label_zoom=args.label_zoom,
        hard_hit_score=args.hard_hit_score,
        min_hit_score=args.min_hit_score,
    )

    kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(pdf_path))
    dt = time.time() - t0

    # Summary
    base = pdf_path.name
    kept_n = len(kept_pages)
    dropped_n = len(dropped_pages)
    print(f"[PageFilter] {base}: kept={kept_n} dropped={dropped_n} time={dt:.2f}s")
    if filtered_pdf:
        print(f"  → filtered_pdf: {filtered_pdf}")
    if args.debug and log_json:
        print(f"  → debug_log:    {log_json}")


def main():
    ap = argparse.ArgumentParser(description="Dev harness for PageFilter")
    src = ap.add_mutually_exclusive_group()

    src.add_argument("--pdf", type=str, default=None,
                     help="Path to a single PDF to process")
    src.add_argument("--pdf_dir", type=str, default=None,
                     help="Directory of PDFs to process (non-recursive)")

    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
                    help="Output directory for filtered PDFs and debug logs")

    # Performance / rendering
    ap.add_argument("--dpi", type=int, default=350, help="Raster DPI used only on undecided pages")
    ap.add_argument("--longest_cap_px", type=int, default=9000, help="Cap longest rasterized side in pixels (or set to -1 to disable)")
    ap.add_argument("--proc_scale", type=float, default=0.5, help="Downscale factor for footprint processing (0.5 = half-size)")

    # OCR controls
    ap.add_argument("--no_ocr", action="store_true", help="Disable OCR entirely")
    ap.add_argument("--ocr_gpu", action="store_true", help="Enable GPU for EasyOCR (if available)")
    ap.add_argument("--debug", action="store_true", help="Write debug JSON logs")
    ap.add_argument("--quiet", action="store_true", help="Reduce stdout")

    # Footprint shape thresholds
    ap.add_argument("--rect_w_min", type=float, default=0.10, help="Min footprint width fraction of page")
    ap.add_argument("--rect_w_max", type=float, default=0.55, help="Max footprint width fraction of page")
    ap.add_argument("--rect_h_min", type=float, default=0.10, help="Min footprint height fraction of page")
    ap.add_argument("--rect_h_max", type=float, default=0.60, help="Max footprint height fraction of page")
    ap.add_argument("--min_rectangularity", type=float, default=0.70, help="Min rectangularity (area_cnt/area_box)")
    ap.add_argument("--min_rect_count", type=int, default=2, help="Min number of qualifying footprints to KEEP")

    # Label (Stage 0) ROI and scoring (matches your requested behavior)
    ap.add_argument("--label_expand_up", type=float, default=0.20,
                help="Expand upward (as a fraction of page height) from the corner crop top to form the label band")
    ap.add_argument("--label_zoom", type=float, default=2.4, help="Zoom for label OCR")
    ap.add_argument("--hard_hit_score", type=float, default=6.5, help="Score threshold for GOLD pages (return only those)")
    ap.add_argument("--min_hit_score", type=float, default=3.0, help="Score threshold for soft hits")

    args = ap.parse_args()

    # Normalize caps
    if args.longest_cap_px is not None and args.longest_cap_px < 0:
        args.longest_cap_px = None

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve inputs
    targets = []
    if args.pdf:
        targets = [Path(args.pdf).expanduser()]
    elif args.pdf_dir:
        pdir = Path(args.pdf_dir).expanduser()
        targets = sorted([p for p in pdir.iterdir() if p.suffix.lower() == ".pdf"])
    else:
        # Default single file (your reference one)
        targets = [DEFAULT_INPUT_PDF]

    if not targets:
        print("[WARN] No input PDFs found.")
        return

    print(f"[DevEnv] Processing {len(targets)} PDF(s)")
    for pdf_path in targets:
        if not pdf_path.exists():
            print(f"[SKIP] Missing file: {pdf_path}")
            continue
        run_one(pdf_path, out_dir, args)


if __name__ == "__main__":
    main()
