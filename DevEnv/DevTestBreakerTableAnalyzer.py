#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
from pathlib import Path

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- IMPORTS ----------
from VisualDetectionToolLibrary.PanelSearchToolV11 import PanelBoardSearch
from OcrLibrary.BreakerTableParserAPIv2 import BreakerTablePipeline, API_VERSION

# ---------- INPUTS ----------
PDF_PATH = '~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/good_tester_combo.pdf'
OUTPUT_DIR = '~/ElectricalDiagramAnalyzer/DevEnv/PanelSearchOutput'
# ================================================================

def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default

def _one_line_breaker(b: dict) -> str:
    num  = _first(b, ["ckt","cct","circuit","circuit_number","number","idx"], default="?")
    desc = _first(b, ["desc","description","label","load","cktdesc","ckt_desc"], default="")
    trip = _first(b, ["trip","amps","amperage","size","rating"], default="")
    poles= _first(b, ["poles","pole","ph","phase"], default="")
    side = _first(b, ["side","left_right","L/R","lr"], default="")
    qty  = _first(b, ["count","qty","quantity"], default="")
    extras = []
    if side not in ("", None): extras.append(f"side={side}")
    if poles not in ("", None): extras.append(f"poles={poles}")
    if qty not in ("", None): extras.append(f"qty={qty}")
    if trip not in ("", None): extras.append(f"trip={trip}")
    meta = ("  [" + ", ".join(extras) + "]") if extras else ""
    return f"  - CKT {num}: {desc}{meta}"

def main():
    pdf_path = str(Path(PDF_PATH).expanduser())
    out_dir = Path(OUTPUT_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Running PanelSearchToolV11...")
    finder = PanelBoardSearch(
        output_dir=str(out_dir),
        dpi=400,
        min_whitespace_area_fr=0.01,
        margin_shave_px=6,
        min_void_area_fr=0.004,
        min_void_w_px=90,
        min_void_h_px=90,
        max_void_area_fr=0.50,
        pad=6,
        debug=False,
        verbose=True,
        save_masked_shape_crop=False,
        replace_multibox=True,
    )
    crops = finder.readPdf(pdf_path)
    print(f">>> Found {len(crops)} panel image(s)")

    if not crops:
        print(">>> No panels detected — exiting.")
        return

    # ---------- Run OCR API ----------
    pipe = BreakerTablePipeline(debug=True)
    print("\n>>> API Version:", API_VERSION)

    all_results = []

    for img_path in crops:
        print(f"\n\n=========================\nAnalyzing: {img_path}\n=========================")
        result = pipe.run(img_path)
        stages  = result.get("results") or {}
        ana_res = stages.get("analyzer") or {}
        hdr_res = stages.get("header") or {}
        tbl_res = stages.get("parser") or {}

        # ---- Summaries ----
        print("\n=== ANALYZER ===")
        print("header_y:", ana_res.get("header_y"))
        print("footer_y:", ana_res.get("footer_y"))
        print("spaces  :", ana_res.get("spaces_detected"), "→", ana_res.get("spaces_corrected"))
        print("gridless:", ana_res.get("gridless_path"))
        print("overlay :", ana_res.get("page_overlay_path"))

        print("\n=== HEADER PARSER ===")
        print("name    :", (hdr_res or {}).get("name"))
        print("attrs   :", (hdr_res or {}).get("attrs"))

        hdr_breakers = ((hdr_res or {}).get("attrs") or {}).get("detected_breakers") or []
        print(f"\n--- Breakers found in HEADER attrs: {len(hdr_breakers)} ---")
        for b in hdr_breakers:
            print(_one_line_breaker(b))

        print("\n=== TABLE PARSER ===")
        if tbl_res:
            tbl_breakers = tbl_res.get("detected_breakers") or []
            print("spaces  :", tbl_res.get("spaces"))
            print(f"breakers: {len(tbl_breakers)}")
            print("\n--- Breakers from TABLE parser ---")
            for b in tbl_breakers:
                print(_one_line_breaker(b))
        else:
            print("parser  : None")
            tbl_breakers = []

        all_results.append({
            "image": img_path,
            "results": result,
            "header_breakers": hdr_breakers,
            "table_breakers": tbl_breakers,
        })

    # ---------- Write JSON dump ----------
    dump_path = out_dir / "breaker_dump_all.json"
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[WROTE] {dump_path}")
    except Exception as e:
        print(f"[WARN] Could not write dump: {e}")

if __name__ == "__main__":
    main()
