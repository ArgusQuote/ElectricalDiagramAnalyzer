#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import easyocr

# ---------------- PATH SETUP ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from AnchoringClasses.BreakerHeaderFinder import BreakerHeaderFinder, HeaderResult


def prep_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR -> enhanced gray, upscale if too small."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    H, W = g.shape
    if H < 1600:
        s = 1600.0 / H
        g = cv2.resize(g, (int(W * s), int(H * s)), interpolation=cv2.INTER_CUBIC)
    return g


def main():
    # ---- EDIT THESE PATHS ----
    SOURCE_IMAGE = (
        "/home/marco/Documents/Diagrams/CaseStudy_VectorCrop_Run15/"
        "ELECTRICAL SET (Mark Up)_electrical_filtered_page002_panel02.png"
    )
    OUT_DIR = "/home/marco/Documents/Diagrams/HeaderFinderDebug3"
    # --------------------------

    os.makedirs(OUT_DIR, exist_ok=True)

    img = cv2.imread(SOURCE_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read file: {SOURCE_IMAGE}")

    gray = prep_gray(img)

    # EasyOCR reader
    reader = easyocr.Reader(["en"], gpu=False)

    # NEW header finder class
    header_finder = BreakerHeaderFinder(
        reader,
        debug=True,           # overlay + band saved only when True
        debug_dir=OUT_DIR,    # where to dump header band PNG
    )

    # Run analysis
    header_res: HeaderResult = header_finder.analyze_rows(gray)

    print("Header Y:", header_res.header_y)

    # Build overlay on FULL-SIZE image (same size as source)
    ocr_overlay = header_finder.draw_ocr_overlay(img)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(SOURCE_IMAGE).stem
    overlay_path = os.path.join(OUT_DIR, f"{base}_header_overlay_{ts}.png")
    cv2.imwrite(overlay_path, ocr_overlay)
    print("Overlay saved to:", overlay_path)


if __name__ == "__main__":
    main()
