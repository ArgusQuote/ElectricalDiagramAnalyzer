#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

# ---------- HARD-CODED PATHS ----------
SOURCE_PATH = "/home/marco/Documents/Diagrams/CaseStudy_VectorCrop_Run15"
DEBUG_OUTPUT_DIR = "/home/marco/Documents/Diagrams/BreakerDebug3"

DEBUG_MODE = True  # set False if you want no debug.png outputs

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from OcrLibrary.BreakerTableAnalyzer11 import BreakerTableAnalyzer, ANALYZER_VERSION


def _collect_images(path: str) -> List[Path]:
    """Return list of images from file, folder, or glob."""
    p = Path(path).expanduser()

    if p.is_file():
        return [p]

    if p.is_dir():
        return sorted(
            [
                f
                for f in p.iterdir()
                if f.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
                }
            ]
        )

    # treat as glob
    return sorted(
        [
            f
            for f in Path(".").glob(path)
            if f.suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
            }
        ]
    )


def run_single(analyzer: BreakerTableAnalyzer, img_path: Path):
    print("\n" + "=" * 90)
    print(f"[PROCESSING] {img_path}")
    print("=" * 90)

    try:
        result = analyzer.analyze(str(img_path))
    except Exception as e:
        print(f"ERROR while analyzing {img_path}: {e}")
        return

    header_y = result.get("header_y")
    footer_y = result.get("footer_y")
    centers = result.get("centers", [])
    snap_note = result.get("snap_note")
    gridless_path = result.get("gridless_path")
    overlay_page = result.get("page_overlay_path")
    overlay_cols = result.get("column_overlay_path")

    print(f" Header Y ........... {header_y}")
    print(f" Footer Y ........... {footer_y}")
    print(f" Rows found.......... {len(centers)}")
    print(f" Centers............  {centers}")
    print(f" Gridless image ..... {gridless_path}")
    print(f" Overlay (page) ..... {overlay_page}")
    print(f" Overlay (columns) .. {overlay_cols}")
    if snap_note:
        print(f" Snap note: {snap_note}")


def main():
    print(f"\n[DevEnv] Running BreakerTableAnalyzer v{ANALYZER_VERSION}")
    print(f"[DevEnv] SOURCE_PATH = {SOURCE_PATH}")
    print(f"[DevEnv] Debug Mode  = {DEBUG_MODE}")
    print(f"[DevEnv] Output dir  = {DEBUG_OUTPUT_DIR}\n")

    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    analyzer = BreakerTableAnalyzer(
        debug=DEBUG_MODE,
        bottom_trim_frac=0.15,
        top_trim_frac=0.50,
        upscale_factor=1.0,
    )

    # Tell analyzer to put ALL overlays here
    analyzer.debug_root_dir = DEBUG_OUTPUT_DIR

    imgs = _collect_images(SOURCE_PATH)
    if not imgs:
        print("❌  No images found. Fix SOURCE_PATH.")
        return

    print(f"Found {len(imgs)} image(s). Running...\n")

    for p in imgs:
        run_single(analyzer, p)

    print("\n✔ Done.")


if __name__ == "__main__":
    main()
