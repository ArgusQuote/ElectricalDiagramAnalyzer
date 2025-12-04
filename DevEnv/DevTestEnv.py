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
from PageFilter.PageFilter import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV15 import PanelBoardSearch
# Use the SAME Parser API you used in your analyzer tests:
from OcrLibrary.BreakerTableParserAPIv5 import BreakerTablePipeline, API_VERSION

# ---------- IO PATHS (fixed typos: PdfOutput / PanelSearchOutput) ----------
INPUT_PDF       = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/chucksmall.pdf").expanduser()
FILTER_OUT_DIR  = Path("~/ElectricalDiagramAnalyzer/DevEnv/PdfOutput").expanduser()
FINDER_OUT_DIR  = Path("~/ElectricalDiagramAnalyzer/DevEnv/PanelSearchOutput").expanduser()
PIPE_OUT_DIR    = Path("~/ElectricalDiagramAnalyzer/DevEnv/ParserOutput").expanduser()

for d in (FILTER_OUT_DIR, FINDER_OUT_DIR, PIPE_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

def main():
    # ---- 1) PageFilter ----
    print("\n[PageFilter] starting…")
    FILTER = PageFilter(
        output_dir=str(FILTER_OUT_DIR),   # filtered PDF will be written here
        dpi=400,                          # raster DPI used only for undecided pages
        longest_cap_px=9000,
        proc_scale=0.5,
        use_ocr=True,
        ocr_gpu=False,
        verbose=True,
        debug=False,                      # True -> JSON log at output_dir/filter_debug/
        rect_w_fr_range=(0.10, 0.55),
        rect_h_fr_range=(0.10, 0.60),
        min_rectangularity=0.70,
        min_rect_count=2,
    )
    kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(INPUT_PDF))
    print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")

    # Choose PDF for finder: filtered if we kept something, else original
    pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept_pages) > 0) else INPUT_PDF
    print(f"[PanelFinder] using PDF: {pdf_for_finder}")

    # ---- 2) PanelBoardSearch ----
    print("\n[PanelFinder] starting…")
    FINDER = PanelBoardSearch(
        output_dir=str(FINDER_OUT_DIR),
        dpi=400,
        render_dpi=1400,
        aa_level=8,
        render_colorspace="gray",
        min_void_area_fr=0.004,
        min_void_w_px=90,
        min_void_h_px=90,
        max_void_area_fr=0.30,
        void_w_fr_range=(0.20, 0.60),
        void_h_fr_range=(0.20, 0.55),
        min_whitespace_area_fr=0.01,
        margin_shave_px=6,
        pad=6,
        verbose=True,
        # one-box settings (defaults are fine, but you can loosen slightly if needed):
        # onebox_min_rel_area=0.02, onebox_max_rel_area=0.75,
        # onebox_aspect_range=(0.4, 3.0), onebox_min_side_px=80,
    )
    crops = FINDER.readPdf(str(pdf_for_finder))
    print(f"[PanelFinder] wrote {len(crops)} crop(s) to {FINDER_OUT_DIR}")

    if not crops:
        print("[Pipeline] No panel crops found — exiting.")
        return

    # ---- 3) BreakerTableParser API (same as your analyzer test env) ----
    print("\n[ParserAPI] starting…")
    pipe = BreakerTablePipeline(debug=True)
    print("[ParserAPI] API_VERSION:", API_VERSION)

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
        print("header_y_orig   :", ana_res.get("header_y_original"))
        print("header_token_y  :", ana_res.get("header_token_y"))
        print("header_rule_src :", ana_res.get("header_rule_source"))
        print("footer_y        :", ana_res.get("footer_y"))
        print("spaces          :", ana_res.get("spaces_detected"))
        print("gridless        :", ana_res.get("gridless_path"))
        print("overlay         :", ana_res.get("page_overlay_path"))

        # ---- Header summary ----
        print("\n=== HEADER PARSER ===")
        print("name    :", (hdr_res or {}).get("name"))
        print("attrs   :", (hdr_res or {}).get("attrs"))
        hdr_breakers = ((hdr_res or {}).get("attrs") or {}).get("detected_breakers") or []
        print(f"\n--- Breakers found in HEADER attrs: {len(hdr_breakers)} ---")

        # ---- Table summary (tally only) ----
        print("\n=== TABLE PARSER (summary) ===")
        if tbl_res:
            spaces            = tbl_res.get("spaces")
            detected_breakers = tbl_res.get("detected_breakers") or []
            breaker_counts    = tbl_res.get("breakerCounts") or {}

            print("spaces        :", spaces)
            print("rows parsed   :", len(detected_breakers))
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

    # ---- Write JSON dump ----
    dump_path = PIPE_OUT_DIR / "full_pipeline_breaker_dump.json"
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[WROTE] {dump_path}")
    except Exception as e:
        print(f"[WARN] Could not write dump: {e}")

    print(f"\n[Pipeline] Done. crops={len(crops)} | header_breakers={total_hdr_breakers} | table_breakers={total_tbl_breakers}")

if __name__ == "__main__":
    main()
