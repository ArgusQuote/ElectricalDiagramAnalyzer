#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import sys
import json
from pathlib import Path

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- IMPORTS ----------
from PageFilter.PageFilterV3 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch
# Use the SAME Parser API you used in your analyzer tests:
from OcrLibrary.BreakerTableParserAPIv9 import BreakerTablePipeline, API_VERSION

# ---------- IO PATHS ----------
# Input: directory containing PDFs to analyze (all *.pdf in this folder)
INPUT_DIR       = Path("~/Documents/pdfToScan").expanduser()
# Output base: each PDF gets a subdir and a unique dump file
OUT_BASE        = Path("~/Documents/DevEnv2_FullAnalysis").expanduser()

def process_one_pdf(input_pdf: Path, filter_out_dir: Path, finder_out_dir: Path, pipe_out_dir: Path, pipe: BreakerTablePipeline) -> Path | None:
    """Run full pipeline for one PDF; write JSON dump to pipe_out_dir. Returns dump path or None."""
    # ---- 1) PageFilter ----
    print("\n[PageFilter] starting…")
    FILTER = PageFilter(
        output_dir=str(filter_out_dir),
        dpi=400,
        longest_cap_px=9000,
        proc_scale=0.5,
        use_ocr=True,
        ocr_gpu=False,
        verbose=True,
        debug=False,
        rect_w_fr_range=(0.10, 0.55),
        rect_h_fr_range=(0.10, 0.60),
        min_rectangularity=0.70,
        min_rect_count=2,
    )
    kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(input_pdf))
    print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")

    pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept_pages) > 0) else input_pdf
    print(f"[PanelFinder] using PDF: {pdf_for_finder}")

    # ---- 2) PanelBoardSearch ----
    print("\n[PanelFinder] starting…")
    FINDER = PanelBoardSearch(
        output_dir=str(finder_out_dir),
        dpi=400,
        render_dpi=1400,
        aa_level=8,
        render_colorspace="gray",
        min_void_area_fr=0.004,
        min_void_w_px=90,
        min_void_h_px=90,
        max_void_area_fr=0.30,
        void_w_fr_range=(0.20, 0.60),
        void_h_fr_range=(0.15, 0.55),
        min_whitespace_area_fr=0.01,
        margin_shave_px=6,
        pad=6,
        verbose=True,
    )
    crops = FINDER.readPdf(str(pdf_for_finder))
    print(f"[PanelFinder] wrote {len(crops)} crop(s) to {finder_out_dir}")

    if not crops:
        print("[Pipeline] No panel crops found — skipping.")
        return None

    all_results = []
    total_hdr_breakers = 0
    total_tbl_breakers = 0

    for img_path in crops:
        print(f"\n\n=========================")
        print(f"Analyzing crop: {img_path}")
        print(f"=========================")

        result = pipe.run(img_path)
        stages  = result.get("results") or {}
        ana_res = stages.get("analyzer") or {}
        hdr_res = stages.get("header") or {}
        tbl_res = stages.get("parser") or {}

        # ---- Analyzer summary ----
        print("\n=== ANALYZER ===")
        print("header_y        :", ana_res.get("header_y"))
        print("footer_y        :", ana_res.get("footer_y"))

        # ---- Header summary ----
        print("\n=== HEADER PARSER ===")
        print("name    :", (hdr_res or {}).get("name"))
        print("attrs   :", (hdr_res or {}).get("attrs"))
        hdr_breakers = ((hdr_res or {}).get("attrs") or {}).get("detected_breakers") or []

        # ---- Table summary (tally only) ----
        print("\n=== TABLE PARSER (summary) ===")
        if tbl_res:
            spaces            = tbl_res.get("spaces")
            detected_breakers = tbl_res.get("detected_breakers") or []
            breaker_counts    = tbl_res.get("breakerCounts") or {}
            gfi_counts        = tbl_res.get("gfiBreakerCounts") or {}

            print("spaces               :", spaces)
            print("detected breakers    :", len(detected_breakers))
            print("Breakers (tally):")

            if not breaker_counts:
                print("  (none detected)")
            else:
                # sort by poles, then amps numerically: 1P_20A, 1P_30A, 3P_100A, ...
                def _sort_key(item):
                    key, _count = item
                    m = re.match(r"(\d+)P_(\d+)A", key)
                    if not m:
                        return (9999, 9999, key)
                    p = int(m.group(1))
                    a = int(m.group(2))
                    return (p, a, key)

                for key, count in sorted(breaker_counts.items(), key=_sort_key):
                    m = re.match(r"(\d+)P_(\d+)A", key)
                    if m:
                        poles = int(m.group(1))
                        amps  = int(m.group(2))
                        print(f"  {poles} P, {amps} A, count - {count}")
                    else:
                        print(f"  {key}, count - {count}")
            # --- GFI summary ---
            if gfi_counts:
                print("GFI Breakers (tally):")
                for key, count in sorted(gfi_counts.items(), key=_sort_key):
                    m = re.match(r"(\d+)P_(\d+)A", key)
                    if m:
                        poles = int(m.group(1))
                        amps  = int(m.group(2))
                        print(f"  {poles} P, {amps} A, GFI count - {count}")
                    else:
                        print(f"  {key}, GFI count - {count}")

        else:
            print("parser  : None")
            detected_breakers = []

        total_hdr_breakers += len(hdr_breakers)
        total_tbl_breakers += len(detected_breakers)

        all_results.append({
            "image": img_path,
            "results": result,
            "header_breakers": hdr_breakers,
            "table_breakers": detected_breakers,
        })

    # ---- Write JSON dump (unique file per PDF) ----
    dump_path = pipe_out_dir / "pipeline_dump.json"
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[WROTE] {dump_path}")
    except Exception as e:
        print(f"[WARN] Could not write dump: {e}")
        return None

    print(f"\n[Pipeline] Done. crops={len(crops)} | header_breakers={total_hdr_breakers} | table_breakers={total_tbl_breakers}")
    return dump_path


def main():
    """Discover all PDFs in INPUT_DIR, run pipeline for each, write one output file per PDF."""
    if not INPUT_DIR.is_dir():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDFs found in {INPUT_DIR}")
        return

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    pipe = BreakerTablePipeline(debug=True)
    print("[ParserAPI] API_VERSION:", API_VERSION)

    written = []
    for i, input_pdf in enumerate(pdfs, 1):
        stem = input_pdf.stem
        print(f"\n{'='*60}")
        print(f"PDF {i}/{len(pdfs)}: {input_pdf.name}")
        print(f"{'='*60}")

        filter_out_dir = OUT_BASE / stem / "PdfOutput"
        finder_out_dir = OUT_BASE / stem / "PanelSearchOutput"
        pipe_out_dir = OUT_BASE / stem / "ParserOutput"
        for d in (filter_out_dir, finder_out_dir, pipe_out_dir):
            d.mkdir(parents=True, exist_ok=True)

        dump_path = process_one_pdf(input_pdf, filter_out_dir, finder_out_dir, pipe_out_dir, pipe)
        if dump_path:
            written.append((input_pdf.name, str(dump_path)))

    print(f"\n[SUMMARY] Processed {len(pdfs)} PDF(s), wrote {len(written)} output file(s).")
    for name, path in written:
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
