#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract horizontal + vertical line masks from big_whitespace_mask.png.

Inputs:
  - SRC_MASK: black shapes + lines on white background

Outputs:
  - big_whitespace_hlines.png  (horizontal lines only)
  - big_whitespace_vlines.png  (vertical lines only)
  - big_whitespace_hvlines.png (all lines combined)
"""

import os
from pathlib import Path
import cv2
import numpy as np

# ========= EDIT THESE =========
SRC_MASK = Path("~/Documents/Diagrams/Whitespace_finding3/big_whitespace_mask.png").expanduser()
OUT_H = Path("~/Documents/Diagrams/Whitespace_finding4/big_whitespace_hlines.png").expanduser()
OUT_V = Path("~/Documents/Diagrams/Whitespace_finding4/big_whitespace_vlines.png").expanduser()
OUT_HV = Path("~/Documents/Diagrams/Whitespace_finding4/big_whitespace_hvlines.png").expanduser()
# ==============================

# Fraction of image size used for structuring element length
# (controls minimum line length we pick up)
H_LINE_LEN_FRAC = 1 / 40.0   # width / 40
V_LINE_LEN_FRAC = 1 / 40.0   # height / 40


def main():
    src_path = os.path.abspath(str(SRC_MASK))
    out_h = os.path.abspath(str(OUT_H))
    out_v = os.path.abspath(str(OUT_V))
    out_hv = os.path.abspath(str(OUT_HV))

    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read mask image: {src_path}")

    h, w = img.shape[:2]
    print(f"[INFO] Loaded {src_path}, size={w}x{h}")

    # Binary invert: lines/rects become white (255) on black background
    _, bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # --- Horizontal lines ---
    h_len = max(10, int(w * H_LINE_LEN_FRAC))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # --- Vertical lines ---
    v_len = max(10, int(h * V_LINE_LEN_FRAC))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Combine
    hv_lines = cv2.bitwise_or(h_lines, v_lines)

    # Invert back so lines are black on white (for easier viewing)
    h_out = cv2.bitwise_not(h_lines)
    v_out = cv2.bitwise_not(v_lines)
    hv_out = cv2.bitwise_not(hv_lines)

    os.makedirs(os.path.dirname(out_h), exist_ok=True)
    cv2.imwrite(out_h, h_out)
    cv2.imwrite(out_v, v_out)
    cv2.imwrite(out_hv, hv_out)

    print(f"[INFO] Wrote horizontal line mask to: {out_h}")
    print(f"[INFO] Wrote vertical line mask to:   {out_v}")
    print(f"[INFO] Wrote combined line mask to:   {out_hv}")


if __name__ == "__main__":
    main()
