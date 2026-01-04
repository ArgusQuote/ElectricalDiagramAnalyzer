#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1) Load a PDF page or raster image.
2) Find large white-space regions and tint them with ONE uniform color.
3) On the tinted image, detect white-ish "table" regions and draw magenta boxes.
4) Save:
     - white_mask.png           : basic white pixels mask
     - big_whitespace_mask.png  : only big whitespace blobs
     - tinted_whitespace.png    : original + uniform tint over big whitespace
     - tables_whitemask.png     : white-ish regions on the tinted image (tables pop out)
     - tables_whitemask_closed.png : after morphology (clean blobs)
     - tables_overlay.png       : original image with magenta boxes over tables
"""

import os
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

# ========= EDIT THESE LINES =========
SRC_PATH = Path("~/Documents/Diagrams/NoAmps.pdf").expanduser()
OUT_DIR  = Path("~/Documents/Diagrams/Whitespace_finding3").expanduser()
# ====================================

# For detecting basic "white" pixels from the original
WHITE_THRESH = 240        # gray >= this is considered white-ish
MIN_WHITESPACE_FRACTION = 0.15   # big whitespace blobs: >= 15% of image pixels

# Tint settings for whitespace
TINT_COLOR_BGR = (200, 255, 200)  # pale greenish tint (B,G,R)
TINT_ALPHA     = 0.50             # how strong tint is in [0..1]

# For detecting tables on the tinted image (HSV)
TABLE_WHITE_V_MIN   = 220     # bright
TABLE_WHITE_S_MAX   = 40      # low saturation (white/gray)
TABLE_MIN_FRACTION  = 0.01    # tables: >= 1% of image area
TABLE_MIN_WIDTH_PX  = 120
TABLE_MIN_HEIGHT_PX = 100

# Morphology to connect table interiors
TABLE_CLOSE_KERNEL  = (15, 15)
TABLE_CLOSE_ITER    = 2


def load_as_bgr(path: Path) -> np.ndarray:
    """
    Load a PDF (first page) or a raster image into a BGR np.ndarray.
    """
    path_str = os.path.abspath(str(path))
    if not os.path.exists(path_str):
        raise FileNotFoundError(path_str)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(path_str)
        if len(doc) == 0:
            raise RuntimeError(f"PDF has no pages: {path_str}")
        page = doc[0]

        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 1:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif pix.n == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise RuntimeError(f"Unexpected channel count from pixmap: {pix.n}")
        return img_bgr
    else:
        img = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path_str}")
        return img


def main():
    src_path = SRC_PATH
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_as_bgr(src_path)
    h, w = img.shape[:2]
    total_pixels = h * w

    print(f"[INFO] Source: {src_path}")
    print(f"[INFO] Size:   {w} x {h}")
    print(f"[INFO] Output: {out_dir}")

    # ---------- 1) BASIC WHITE MASK FROM ORIGINAL ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, white_mask = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    white_mask_path = out_dir / "white_mask.png"
    cv2.imwrite(str(white_mask_path), white_mask)
    print(f"[INFO] Wrote white_mask to: {white_mask_path}")

    # ---------- 2) FIND BIG WHITESPACE BLOBS ----------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )
    print(f"[INFO] white components (excluding background): {num_labels - 1}")

    min_whitespace_area = int(MIN_WHITESPACE_FRACTION * total_pixels)
    print(
        f"[INFO] Big whitespace threshold: {min_whitespace_area} pixels "
        f"(~{MIN_WHITESPACE_FRACTION * 100:.1f}%)"
    )

    big_whitespace_mask = np.zeros_like(white_mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_whitespace_area:
            big_whitespace_mask[labels == label_id] = 255
            print(f"  - big whitespace component {label_id}: area={area} pixels")

    big_ws_path = out_dir / "big_whitespace_mask.png"
    cv2.imwrite(str(big_ws_path), big_whitespace_mask)
    print(f"[INFO] Wrote big_whitespace_mask to: {big_ws_path}")

    # ---------- 3) APPLY A *UNIFORM* TINT TO BIG WHITESPACE ----------
    tinted = img.copy().astype(np.float32)
    tint_vec = np.array(TINT_COLOR_BGR, dtype=np.float32)

    ws_mask_bool = big_whitespace_mask == 255
    region = tinted[ws_mask_bool]
    tinted_region = (1.0 - TINT_ALPHA) * region + TINT_ALPHA * tint_vec
    tinted[ws_mask_bool] = tinted_region

    tinted_uint8 = np.clip(tinted, 0, 255).astype(np.uint8)
    tinted_path = out_dir / "tinted_whitespace.png"
    cv2.imwrite(str(tinted_path), tinted_uint8)
    print(f"[INFO] Wrote tinted_whitespace to: {tinted_path}")

    # ---------- 4) DETECT WHITE TABLES ON THE TINTED IMAGE ----------
    print("[INFO] Detecting white tables on tinted image...")
    hsv = cv2.cvtColor(tinted_uint8, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    tables_white_mask = np.zeros_like(v_ch, dtype=np.uint8)
    tables_white_mask[
        (v_ch >= TABLE_WHITE_V_MIN) & (s_ch <= TABLE_WHITE_S_MAX)
    ] = 255

    tbl_white_path = out_dir / "tables_whitemask.png"
    cv2.imwrite(str(tbl_white_path), tables_white_mask)
    print(f"[INFO] Wrote tables_whitemask to: {tbl_white_path}")

    # Morphologically close to connect table interiors
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, TABLE_CLOSE_KERNEL)
    tables_closed = cv2.morphologyEx(
        tables_white_mask, cv2.MORPH_CLOSE, kernel, iterations=TABLE_CLOSE_ITER
    )

    tbl_closed_path = out_dir / "tables_whitemask_closed.png"
    cv2.imwrite(str(tbl_closed_path), tables_closed)
    print(f"[INFO] Wrote tables_whitemask_closed to: {tbl_closed_path}")

    # ---------- 5) CONNECTED COMPONENTS ON TABLE MASK ----------
    num_labels_tbl, labels_tbl, stats_tbl, _ = cv2.connectedComponentsWithStats(
        tables_closed, connectivity=8
    )
    print(f"[INFO] table components (raw): {num_labels_tbl - 1}")

    min_table_area = int(TABLE_MIN_FRACTION * total_pixels)
    print(
        f"[INFO] Table min area threshold: {min_table_area} pixels "
        f"(~{TABLE_MIN_FRACTION * 100:.1f}%)"
    )

    overlay = img.copy()
    tables_found = 0

    for label_id in range(1, num_labels_tbl):
        x = stats_tbl[label_id, cv2.CC_STAT_LEFT]
        y = stats_tbl[label_id, cv2.CC_STAT_TOP]
        w_box = stats_tbl[label_id, cv2.CC_STAT_WIDTH]
        h_box = stats_tbl[label_id, cv2.CC_STAT_HEIGHT]
        area = stats_tbl[label_id, cv2.CC_STAT_AREA]

        if area < min_table_area:
            continue
        if w_box < TABLE_MIN_WIDTH_PX or h_box < TABLE_MIN_HEIGHT_PX:
            continue

        tables_found += 1
        cv2.rectangle(
            overlay,
            (x, y),
            (x + w_box - 1, y + h_box - 1),
            (255, 0, 255),  # magenta
            3,
        )
        print(
            f"  [TABLE] label={label_id} area={area} "
            f"bbox=({x},{y},{w_box},{h_box})"
        )

    if tables_found == 0:
        print("[INFO] No table-like regions passed the thresholds.")
    else:
        print(f"[INFO] Detected {tables_found} table-like regions.")

    tables_overlay_path = out_dir / "tables_overlay.png"
    cv2.imwrite(str(tables_overlay_path), overlay)
    print(f"[INFO] Wrote tables_overlay to: {tables_overlay_path}")


if __name__ == "__main__":
    main()
