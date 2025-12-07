#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DevEnv/RunBreakerTableAnalyzerHF_simple.py

Hard-coded test for BreakerTableAnalyzer9 + HF overlay.

Edit:
  SRC_IMAGE  -> path to a single panel image
  DEBUG_ROOT -> where you want all debug / overlay images to be written
"""

import os
import sys
from pathlib import Path

# ========= EDIT THESE TWO LINES =========
SRC_IMAGE  = Path("~/Documents/Diagrams/CaseStudy_VectorCrop_Run3/generic3_page001_panel01.png").expanduser()
DEBUG_ROOT = Path("~/Documents/Diagrams/HF_Debug").expanduser()
# =======================================

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from OcrLibrary.BreakerTableAnalyzer12 import BreakerTableAnalyzer, ANALYZER_VERSION


def main():
    src_path = os.path.abspath(os.path.expanduser(SRC_IMAGE))
    if not os.path.exists(src_path):
        print(f"ERROR: SRC_IMAGE does not exist: {src_path}")
        return

    debug_root = os.path.abspath(os.path.expanduser(DEBUG_ROOT))
    os.makedirs(debug_root, exist_ok=True)

    print(f"[RunBreakerTableAnalyzerHF_simple] ANALYZER_VERSION = {ANALYZER_VERSION}")
    print(f"[RunBreakerTableAnalyzerHF_simple] SRC_IMAGE       = {src_path}")
    print(f"[RunBreakerTableAnalyzerHF_simple] DEBUG_ROOT      = {debug_root}")

    analyzer = BreakerTableAnalyzer(debug=True)
    analyzer.debug_root_dir = debug_root

    res = analyzer.analyze(src_path)

    print("\n=== SIMPLE HEADER / FOOTER RESULT ===")
    print(f"src_path        : {res.get('src_path')}")
    print(f"header_y        : {res.get('header_y')}")
    print(f"header_bottom_y : {res.get('header_bottom_y')}")
    print(f"footer_y        : {res.get('footer_y')}")
    print(f"footer_token_val: {res.get('footer_token_val')}")
    print(f"panel_size      : {res.get('panel_size')}")
    print(f"debug_dir       : {res.get('debug_dir')}")

    # Expected overlay path from BreakerTableAnalyzer's debug overlay
    dbg_dir = res.get("debug_dir") or debug_root
    base = os.path.splitext(os.path.basename(res["src_path"]))[0]
    overlay_path = os.path.join(dbg_dir, f"{base}_hf_overlay.png")

    print("\nIf everything is wired correctly, the HF overlay should be at:")
    print(f"  {overlay_path}")


if __name__ == "__main__":
    main()
