#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import easyocr

# ---------------- PATH SETUP (optional) ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from AnchoringClasses.BreakerHeaderFinder import BreakerHeaderFinder, HeaderResult


def prep_gray(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    H, W = g.shape
    if H < 1600:
        s = 1600.0 / H
        g = cv2.resize(g, (int(W * s), int(H * s)), interpolation=cv2.INTER_CUBIC)
    return g


def save_header_overlay(
    gray: np.ndarray,
    header_res: HeaderResult,
    out_path: str,
):
    """
    Header-only overlay:
      - horizontal row borders
      - HEADER_TEXT line
      - HEADER line
      - FIRST_BREAKER_LINE
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape

    borders = header_res.dbg.lines
    centers = header_res.centers
    header_y = header_res.header_y

    # ----------------------------------------------------------------------
    # draw row borders lightly for context
    # ----------------------------------------------------------------------
    for y in borders:
        cv2.line(vis, (0, int(y)), (W - 1, int(y)), (220, 220, 220), 1)

    # ----------------------------------------------------------------------
    # HEADER + FIRST BREAKER LINE
    # ----------------------------------------------------------------------
    first_breaker_line_y = None

    if header_y is not None:
        hy = int(header_y)

        # "header text" baseline (slightly above header rule)
        header_text_y = max(0, hy - 8)
        cv2.line(vis, (0, header_text_y), (W - 1, header_text_y), (0, 255, 255), 1)
        cv2.putText(
            vis,
            "HEADER_TEXT",
            (12, max(16, header_text_y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # actual header rule
        cv2.line(vis, (0, hy), (W - 1, hy), (255, 255, 0), 2)
        cv2.putText(
            vis,
            "HEADER",
            (12, max(16, hy + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # FIRST_BREAKER_LINE = first border below header
        for b in sorted(borders):
            if b > hy + 2:
                first_breaker_line_y = int(b)
                break

    if first_breaker_line_y is not None:
        y = first_breaker_line_y
        cv2.line(vis, (0, y), (W - 1, y), (0, 165, 255), 2)
        cv2.putText(
            vis,
            "FIRST_BREAKER_LINE",
            (12, max(16, y + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    # ----------------------------------------------------------------------
    # optional: small text block with stats at bottom
    # ----------------------------------------------------------------------
    stats = []
    stats.append(f"rows={len(centers)}")
    if header_y is not None:
        stats.append(f"header_y={int(header_y)}")
    if first_breaker_line_y is not None:
        stats.append(f"first_breaker_y={int(first_breaker_line_y)}")

    if stats:
        cv2.putText(
            vis,
            " | ".join(stats),
            (12, H - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def main():
    # ------------------------------------------------------------------
    # HARD-CODED PATHS: edit these for whatever header test you want
    # ------------------------------------------------------------------
    SOURCE_IMAGE = (
        "/home/marco/Documents/Diagrams/testfolder/"
        "generic3_page001_panel05.png"
    )
    OUT_DIR = "/home/marco/Documents/Diagrams/AnchorDebug_HeaderOnly2"

    os.makedirs(OUT_DIR, exist_ok=True)

    img = cv2.imread(SOURCE_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read file: {SOURCE_IMAGE}")

    gray = prep_gray(img)

    # EasyOCR: use CPU to avoid CUDA OOM headaches
    reader = easyocr.Reader(["en"], gpu=False)

    header_finder = BreakerHeaderFinder(reader, debug=True)
    # ensure header crop (analysis band) is written here when debug=True
    header_finder.debug_dir = OUT_DIR

    # 1) HEADER + ROW STRUCTURE
    header_res: HeaderResult = header_finder.analyze_rows(gray)

    print("Header Y:", header_res.header_y)
    print("Row count:", len(header_res.centers))
    print("Row centers:", header_res.centers)
    print("Structural footer (from header finder):", header_res.footer_struct_y)
    print("Spaces detected:", header_res.spaces_detected)
    print("Spaces corrected:", header_res.spaces_corrected)
    print("Footer snapped?:", header_res.footer_snapped)
    print("Snap note:", header_res.snap_note)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(SOURCE_IMAGE).stem
    overlay_path = os.path.join(OUT_DIR, f"{base}_header_only_{ts}.png")
    save_header_overlay(gray, header_res, overlay_path)
    print("Overlay:", overlay_path)
    print("Header band (if saved):", os.path.join(OUT_DIR, "header_band.png"))


if __name__ == "__main__":
    main()
