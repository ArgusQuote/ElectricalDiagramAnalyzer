#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run one PDF through both DevTestEnv (V7 header / APIv8) and DevTestEnvV2 (V8 header / APIv9),
then print a side-by-side comparison of header parser outputs.
"""
import os
import sys
import json
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from PageFilter.PageFilterV3 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch

# One PDF to compare
INPUT_PDF = Path("~/Documents/pdfToScan/good_tester_combo.pdf").expanduser()
OUT_V7 = Path("~/Documents/DevEnv_CompareV7").expanduser()
OUT_V8 = Path("~/Documents/DevEnv_CompareV8").expanduser()

def run_pipeline(input_pdf: Path, out_base: Path, use_v8_header: bool):
    """Run filter + finder + parser; use_v8_header=True => APIv9 (V8), False => APIv8 (V7). Returns list of header attrs per crop."""
    if use_v8_header:
        from OcrLibrary.BreakerTableParserAPIv9 import BreakerTablePipeline
        label = "V8"
    else:
        from OcrLibrary.BreakerTableParserAPIv8 import BreakerTablePipeline
        label = "V7"

    out_base.mkdir(parents=True, exist_ok=True)
    filter_dir = out_base / "PdfOutput"
    finder_dir = out_base / "PanelSearchOutput"
    filter_dir.mkdir(parents=True, exist_ok=True)
    finder_dir.mkdir(parents=True, exist_ok=True)

    # PageFilter
    pf = PageFilter(output_dir=str(filter_dir), dpi=400, longest_cap_px=9000, proc_scale=0.5,
                    use_ocr=True, ocr_gpu=False, verbose=False, debug=False,
                    rect_w_fr_range=(0.10, 0.55), rect_h_fr_range=(0.10, 0.60),
                    min_rectangularity=0.70, min_rect_count=2)
    kept, dropped, filtered_pdf, _ = pf.readPdf(str(input_pdf))
    pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept) > 0) else input_pdf

    # PanelBoardSearch
    finder = PanelBoardSearch(output_dir=str(finder_dir), dpi=400, render_dpi=1400, aa_level=8,
                              render_colorspace="gray", verbose=False,
                              min_void_area_fr=0.004, min_void_w_px=90, min_void_h_px=90,
                              max_void_area_fr=0.30, void_w_fr_range=(0.20, 0.60), void_h_fr_range=(0.15, 0.55),
                              min_whitespace_area_fr=0.01, margin_shave_px=6, pad=6)
    crops = finder.readPdf(str(pdf_for_finder))
    if not crops:
        return label, []

    pipe = BreakerTablePipeline(debug=False)
    headers = []
    for img_path in crops:
        result = pipe.run(img_path)
        hdr = (result.get("results") or {}).get("header") or {}
        attrs = (hdr.get("attrs") or {}).copy()
        headers.append({"name": hdr.get("name"), "attrs": attrs, "image": str(img_path)})
    return label, headers


def main():
    if not INPUT_PDF.is_file():
        print(f"PDF not found: {INPUT_PDF}")
        return

    print("Running pipeline with V7 header (APIv8)...")
    v7_label, v7_headers = run_pipeline(INPUT_PDF, OUT_V7, use_v8_header=False)
    print("Running pipeline with V8 header (APIv9)...")
    v8_label, v8_headers = run_pipeline(INPUT_PDF, OUT_V8, use_v8_header=True)

    n = max(len(v7_headers), len(v8_headers))
    print("\n" + "=" * 80)
    print("PanelHeaderParser V7 vs V8 — header comparison")
    print("=" * 80)

    for i in range(n):
        h7 = v7_headers[i] if i < len(v7_headers) else {}
        h8 = v8_headers[i] if i < len(v8_headers) else {}
        a7 = h7.get("attrs") or {}
        a8 = h8.get("attrs") or {}

        print(f"\n--- Crop {i + 1} ---")
        print(f"  {'Field':<20} {'V7 (APIv8)':<24} {'V8 (APIv9)':<24} {'Match':<8}")
        print("  " + "-" * 76)

        for key in ["name", "voltage", "amperage", "mainBreakerAmperage", "intRating"]:
            val7 = h7.get("name") if key == "name" else a7.get(key)
            val8 = h8.get("name") if key == "name" else a8.get(key)
            if key == "mainBreakerAmperage" and "mainBreakerAmperage" not in a7 and "mainBreakerAmperage" not in a8:
                continue
            v7_str = str(val7) if val7 is not None else "None"
            v8_str = str(val8) if val8 is not None else "None"
            match = "OK" if val7 == val8 else "DIFF"
            print(f"  {key:<20} {v7_str:<24} {v8_str:<24} {match:<8}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary: V7 and V8 should match on name, voltage, bus (amperage), main, AIC after the voltage fix.")
    print("=" * 80)


if __name__ == "__main__":
    main()
