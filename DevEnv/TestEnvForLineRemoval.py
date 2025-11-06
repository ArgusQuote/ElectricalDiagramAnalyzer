#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find lines (H/V) and remove them from the image.
Outputs per input:
  * *_lines_mask.png     : white lines on black (binary mask)
  * *_lines_removed.png  : original with detected lines inpainted (removed)

Adjust HSCALE_DEN / VSCALE_DEN (smaller => longer kernels) or MIN_LEN_PX as needed.
Increase LINES_ERASE_DILATE or INPAINT_RADIUS if thin line remnants remain.
"""

import os, glob
from pathlib import Path
import cv2
import numpy as np

# ---------------- PATHS AS VARIABLES (edit these) ----------------
SRC_PATH = os.path.expanduser("~/Documents/Diagrams/PanelSearchOutput")
OUT_DIR  = os.path.expanduser("~/Documents/Diagrams/PngOutput")
GLOB_PATTERN = "*.png"
# -----------------------------------------------------------------

# Morphology params (tune for your images)
HSCALE_DEN = 40          # horiz kernel length = W // HSCALE_DEN  (smaller => longer kernel)
VSCALE_DEN = 40          # vert  kernel length = H // VSCALE_DEN
MIN_LEN_PX = 15          # minimum kernel length (pixels)
DILATE_EDGES_FOR_MASK = False  # slightly thicken detected lines in mask for robustness

# Line removal params
LINES_ERASE_DILATE = 1   # extra dilation before inpaint (0..3). Raise if faint line residue remains.
INPAINT_RADIUS     = 2   # 1..4 typical. Larger removes thicker lines but may blur nearby features.


def _ensure_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _otsu_ink_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Return bw_inv where ink (text+lines)=255, paper=0."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw_inv


def _extract_lines(bw_inv: np.ndarray,
                   h_den: int = HSCALE_DEN,
                   v_den: int = VSCALE_DEN,
                   min_len_px: int = MIN_LEN_PX,
                   dilate_edges: bool = DILATE_EDGES_FOR_MASK) -> np.ndarray:
    """
    Keep only long H/V lines from the ink mask.
    Returns a single-channel mask with white lines on black.
    """
    H, W = bw_inv.shape[:2]
    h_len = max(min_len_px, W // max(5, h_den))
    v_len = max(min_len_px, H // max(5, v_den))

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    hlines = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, hk, iterations=1)
    vlines = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, vk, iterations=1)

    if dilate_edges:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        hlines = cv2.dilate(hlines, k, iterations=1)
        vlines = cv2.dilate(vlines, k, iterations=1)

    lines = cv2.bitwise_or(hlines, vlines)
    return lines


def _remove_lines(img_bgr: np.ndarray, lines_mask: np.ndarray,
                  extra_dilate: int = LINES_ERASE_DILATE,
                  radius: int = INPAINT_RADIUS) -> np.ndarray:
    """
    Remove lines by inpainting the regions indicated by lines_mask.
    Optionally dilates the mask a bit so line edges are fully removed.
    """
    mask = lines_mask.copy()
    if extra_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + 2*extra_dilate, 1 + 2*extra_dilate))
        mask = cv2.dilate(mask, k, iterations=1)
    # Inpaint where mask>0
    return cv2.inpaint(img_bgr, mask, max(1, int(radius)), cv2.INPAINT_TELEA)


def _process_one(path: str, out_dir: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] could not read {path}")
        return
    img = _ensure_bgr(img)

    bw_inv = _otsu_ink_mask(img)
    lines_mask = _extract_lines(bw_inv)

    # Save pure mask (white lines on black)
    base = Path(path).stem
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir_p / f"{base}_lines_mask.png"
    cv2.imwrite(str(mask_path), lines_mask)

    # Remove lines
    lines_removed = _remove_lines(img, lines_mask,
                                  extra_dilate=LINES_ERASE_DILATE,
                                  radius=INPAINT_RADIUS)
    removed_path = out_dir_p / f"{base}_lines_removed.png"
    cv2.imwrite(str(removed_path), lines_removed)

    print(f"[OK] {base}: wrote\n      {mask_path}\n      {removed_path}")

def main():
    src = SRC_PATH
    out = OUT_DIR
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, GLOB_PATTERN)))
        if not files:
            print(f"[INFO] no files matching {GLOB_PATTERN} in {src}")
            return
        for f in files:
            _process_one(f, out)
    else:
        if os.path.isfile(src):
            _process_one(src, out)
        else:
            raise FileNotFoundError(f"SRC_PATH not found or not a valid image: {src}")


if __name__ == "__main__":
    main()
