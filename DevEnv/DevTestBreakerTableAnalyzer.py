#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
from pathlib import Path

# ----- Path setup (same as your snippet) -----
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ----- Imports -----
from OcrLibrary.BreakerTableAnalyzer3 import BreakerTableAnalyzer
from OcrLibrary.PanelHeaderParserV4 import PanelParser as PanelHeaderParser
from OcrLibrary.BreakerTableParser4 import BreakerTableParser

# ====== SET YOUR TEST IMAGE HERE ==========================================
IMG_PATH = '~/ElectricalDiagramAnalyzer/DevEnv/PanelSearchOuput/generic2_electrical_filtered_page001_table01_rect.png'
# ==========================================================================

def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default

def _one_line_breaker(b: dict) -> str:
    # Try common field names across variants
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
    img = str(Path(IMG_PATH).expanduser())

    # 1) Analyzer
    analyzer = BreakerTableAnalyzer(debug=True)
    ana_res = analyzer.analyze(img)

    # 2) Header parser (share OCR)
    header_parser = PanelHeaderParser(debug=True)
    if hasattr(analyzer, "reader"):
        header_parser.reader = analyzer.reader

    hy   = ana_res.get("header_y")
    gray = ana_res.get("gray")
    header_kwargs = {}
    if isinstance(hy, (int, float)) and hasattr(gray, "shape"):
        H = float(gray.shape[0])
        header_ratio = max(0.0, min(1.0, float(hy) / H))
        header_kwargs["header_y_ratio"] = header_ratio

    hdr_res = header_parser.parse_panel(img, **header_kwargs)

    # 3) Table parser (unless header is 'false positive')
    hdr_name = (hdr_res or {}).get("name", "")
    should_run_parser = not (isinstance(hdr_name, str) and hdr_name.strip().lower() == "false positive")
    if should_run_parser:
        table_parser = BreakerTableParser(debug=True, reader=getattr(analyzer, "reader", None))
        tbl_res = table_parser.parse_from_analyzer(ana_res or {})
    else:
        tbl_res = None
        print("[INFO] Skipping breaker table parser due to header false positive")

    # ---- Summaries
    print("\n=== ANALYZER ===")
    print("header_y:", ana_res.get("header_y"))
    print("footer_y:", ana_res.get("footer_y"))
    print("spaces  :", ana_res.get("spaces_detected"), "→", ana_res.get("spaces_corrected"))
    print("gridless:", ana_res.get("gridless_path"))
    print("overlay :", ana_res.get("page_overlay_path"))

    print("\n=== HEADER PARSER ===")
    print("name    :", (hdr_res or {}).get("name"))
    print("attrs   :", (hdr_res or {}).get("attrs"))

    # Breakers from HEADER attrs (if any)
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

    # Optional JSON dump with full breaker objects for diffing/inspection
    out_dir = os.path.dirname(img)
    dump_path = os.path.join(out_dir, "breaker_dump.json")
    try:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": img,
                    "analyzer": {k: v for k, v in ana_res.items() if k in ("header_y","footer_y","spaces_detected","spaces_corrected","gridless_path","page_overlay_path")},
                    "header": hdr_res,
                    "table": tbl_res,
                    "header_breakers": hdr_breakers,
                    "table_breakers": tbl_breakers,
                },
                f, indent=2, ensure_ascii=False, default=str
            )
        print(f"\n[WROTE] {dump_path}")
    except Exception as e:
        print(f"[WARN] Could not write {dump_path}: {e}")

if __name__ == "__main__":
    main()
