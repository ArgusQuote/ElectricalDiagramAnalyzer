#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import shutil
from pathlib import Path

import numpy as np
import cv2

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- IMPORTS ----------
from OcrLibrary.BreakerTableParserAPIv2 import BreakerTablePipeline, API_VERSION

# ---------- INPUTS ----------
PNG_PATH   = '~/Documents/Diagrams/CaseStudy4/ELECTRICAL SET (Mark Up)_electrical_altered2_page002_table01_rect.png'
OUTPUT_DIR = '~/Documents/Diagrams/Case5Panel'   # destination where we WANT artifacts

# ================================================================
def _copy_if_exists(src_path: str | None, dest_dir: Path):
    if not src_path:
        return None
    p = Path(os.path.expanduser(src_path))
    if p.is_file():
        out = dest_dir / p.name
        try:
            shutil.copy2(str(p), str(out))
            return str(out)
        except Exception:
            return None
    return None

def _save_gray_and_header(ana_res: dict, dest_dir: Path):
    """If analyzer returned a gray image and header line, save them to OUTPUT_DIR."""
    gray = ana_res.get("gray", None)
    if gray is not None and isinstance(gray, np.ndarray):
        # save analyzer_gray.png
        gray_path = dest_dir / "analyzer_gray.png"
        g = gray
        if g.ndim == 3 and g.shape[2] == 3:
            cv2.imwrite(str(gray_path), g)
        else:
            cv2.imwrite(str(gray_path), g if g.dtype == np.uint8 else g.astype(np.uint8))

        # header line overlay if header_y present
        if "header_y" in ana_res:
            y = int(ana_res["header_y"] or 0)
            if g.ndim == 2:
                vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            else:
                vis = g.copy()
            y = max(0, min(vis.shape[0]-1, y))
            cv2.line(vis, (0, y), (vis.shape[1]-1, y), (0, 0, 255), 3)
            cv2.imwrite(str(dest_dir / "header_line_overlay.png"), vis)
        return True
    return False

def main():
    png_path = str(Path(PNG_PATH).expanduser())
    out_dir = Path(OUTPUT_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Run OCR API ----------
    pipe = BreakerTablePipeline(debug=True)   # pipeline has no idea about OUTPUT_DIR
    print("\n>>> API Version:", API_VERSION)

    print(f"\n\n=========================\nAnalyzing: {png_path}\n=========================")
    result = pipe.run(png_path)
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

    # ---- Collect artifacts into OUTPUT_DIR ----
    saved_any = False

    # 1) Copy any file paths the analyzer produced
    for key in ("gridless_path", "page_overlay_path", "header_overlay_path"):
        copied = _copy_if_exists(ana_res.get(key), out_dir)
        if copied:
            print(f"[OK] Copied {key} → {copied}")
            saved_any = True

    # 2) If analyzer returned an in-memory gray image, save it (and a header line overlay)
    if _save_gray_and_header(ana_res, out_dir):
        print(f"[OK] Saved analyzer_gray.png (and header_line_overlay.png if header_y present) to {out_dir}")
        saved_any = True

    if not saved_any:
        print(f"[INFO] No analyzer artifacts to save; OUTPUT_DIR remains but may be empty: {out_dir}")
        print("       (The pipeline itself doesn’t write to OUTPUT_DIR unless its submodules do.)")

if __name__ == "__main__":
    main()
