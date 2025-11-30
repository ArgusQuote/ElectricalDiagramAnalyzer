#!/usr/bin/env python3
# DevEnvFindHeader6.py

import os
import cv2
import numpy as np
import sys

# ---------------- PATH SETUP (optional) ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from OcrLibrary.BreakerTableAnalyzer6 import (
    BreakerTableAnalyzer,
    _norm_token,
)

# -----------------------
# USER CONFIGURATION
# -----------------------
SOURCE_INPUT = os.path.expanduser("~/Documents/Diagrams/CaseStudy_VectorCrop_Run15/ELECTRICAL SET (Mark Up)_electrical_filtered_page002_panel02.png")
OUTPUT_DIR   = os.path.expanduser("~/Documents/Diagrams/DebugOutput_FH_CUDA4")
# -----------------------


def _find_vertical_lines(gray: np.ndarray,
                         header_y: int | None,
                         footer_y: int | None) -> list[int]:
    """
    Find vertical grid lines between header_y and footer_y using a binarized mask.

    This mimics the vertical-line logic in BreakerTableAnalyzer._remove_grids_and_save,
    but ONLY returns the column x-positions; it doesn't modify the image.
    """
    if header_y is None or footer_y is None:
        return []

    H, W = gray.shape
    if footer_y <= header_y + 10:
        return []

    y1 = max(0, int(header_y) - 4)
    y2 = min(H - 1, int(footer_y) + 4)
    if y2 <= y1:
        return []

    roi = gray[y1:y2, :]
    if roi.size == 0:
        return []

    # --- binarize ROI ---
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )

    roi_h = roi.shape[0]

    # --- detect candidate vertical lines via morphology ---
    Kv = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(25, int(0.015 * roi_h)))
    )
    v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)

    # --- filter to LONG, SKINNY components (true grid lines) ---
    v_x_centers: list[int] = []
    num_v, labels_v, stats_v, _ = cv2.connectedComponentsWithStats(
        v_candidates, connectivity=8
    )

    if num_v > 1:
        min_vert_len = int(0.40 * roi_h)   # at least 40% of table height
        max_vert_thick = max(2, int(0.02 * W))
        for i in range(1, num_v):
            x, y, w, h, area = stats_v[i]
            if h >= min_vert_len and w <= max_vert_thick and area > 0:
                x_center = x + w // 2
                v_x_centers.append(int(x_center))

    if not v_x_centers:
        return []

    # collapse nearby X's into one per column
    xs = sorted(v_x_centers)
    collapsed: list[int] = []
    for x in xs:
        if not collapsed or abs(x - collapsed[-1]) > 4:  # 4px merge window
            collapsed.append(x)

    return collapsed

def make_overlay_for_image(image_path: str, output_dir: str) -> str:
    """
    For a single breaker-table image:
      - detect header / row structure with BreakerTableAnalyzer
      - find vertical lines that bound the CCT/CKT column
      - crop that column (from first breaker row down) FROM THE SOURCE IMAGE
      - remove grid lines in the crop
      - run OCR on the cropped, line-free image
      - draw OCR boxes + raw text on the crop
      - save ONE image per column: <base>_colXX.png

    Assumes helpers:
        _find_vertical_lines(gray, header_y, footer_y)
        _degrid_crop(src_bgr)  # returns grayscale, lines removed
    are defined elsewhere in this module.
    """
    image_path = os.path.abspath(os.path.expanduser(image_path))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    os.makedirs(output_dir, exist_ok=True)

    # ---------- load & analyze ----------
    analyzer = BreakerTableAnalyzer(debug=False)

    orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig is None:
        raise RuntimeError(f"Cannot read file (not an image?): {image_path}")

    # detection happens on resized gray image
    gray = analyzer._prep(orig)
    centers, dbg, header_y = analyzer._row_centers_from_lines(gray)
    first_breaker_line = dbg.lines[0] if dbg.lines else None
    footer_y = getattr(analyzer, "_last_footer_y", None)

    Hg, Wg = gray.shape
    Ho, Wo = orig.shape[:2]

    # map gray -> source coordinates
    scale_y = Hg / float(Ho)
    scale_x = Wg / float(Wo)

    # optional page overlay for debugging header/lines
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if header_y is not None:
        y = int(header_y)
        cv2.line(vis, (0, y), (Wg - 1, y), (0, 255, 0), 2)
        cv2.putText(vis, "HEADER", (10, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if first_breaker_line is not None:
        yb = int(first_breaker_line)
        cv2.line(vis, (0, yb), (Wg - 1, yb), (255, 0, 0), 2)
        cv2.putText(vis, "FIRST BREAKER ROW", (10, yb + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    # ---------- find vertical bounds around CCT/CKT ----------
    v_xs = _find_vertical_lines(gray, header_y, footer_y)
    col_spans_gray = set()

    for item in getattr(analyzer, "_ocr_dbg_items", []):
        txt = item.get("text", "")
        norm = _norm_token(txt)
        if norm not in ("CCT", "CKT"):
            continue

        x1 = int(item["x1"])
        x2 = int(item["x2"])

        # nearest vertical to left and right of token
        left  = max([v for v in v_xs if v <= x1], default=None)
        right = min([v for v in v_xs if v >= x2], default=None)

        if left is not None and right is not None and right > left:
            col_spans_gray.add((left, right))

    base = os.path.splitext(os.path.basename(image_path))[0]

    # save the page-level overlay (not per-crop)
    overlay_page_path = os.path.join(output_dir, f"{base}_overlay.png")
    cv2.imwrite(overlay_page_path, vis)

    # ---------- crop FROM SOURCE & overlay OCR on each crop ----------
    # y-range: from first breaker row down to bottom
    if first_breaker_line is not None:
        y_top_g = int(first_breaker_line)
    else:
        y_top_g = 0
    y_bot_g = Hg

    y_top_o = max(0, min(Ho - 1, int(round(y_top_g / scale_y))))
    y_bot_o = max(y_top_o + 1, min(Ho, int(round(y_bot_g / scale_y))))

    col_spans_sorted = sorted(col_spans_gray, key=lambda p: p[0])

    for idx, (xl_g, xr_g) in enumerate(col_spans_sorted, start=1):
        # map x-bounds to source image
        x_left_o = max(0, min(Wo - 1, int(round((xl_g - 1) / scale_x))))
        x_right_o = max(x_left_o + 1, min(Wo, int(round((xr_g + 1) / scale_x))))

        # crop from source
        src_crop = orig[y_top_o:y_bot_o, x_left_o:x_right_o]
        if src_crop.size == 0:
            continue

        # remove grid lines in the crop (returns GRAY)
        clean_gray = _degrid_crop(src_crop)

        # ---------- OCR + overlay on the SAME crop ----------
        if analyzer.reader is not None:
            overlay = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)
            try:
                detections = analyzer.reader.readtext(
                    clean_gray,
                    detail=1,
                    paragraph=False,  # raw text chunks
                    # no allowlist: we want raw values (letters, numbers, etc.)
                    # tighten vertical grouping so stacked rows don't merge
                    ycenter_ths=0.3,   # default is ~0.5
                    height_ths=0.3,    # default is ~0.5
                )
            except Exception:
                detections = []

            for box, text, conf in detections:
                if not text:
                    continue

                pts = np.array(box, dtype=np.int32)
                cv2.polylines(overlay, [pts], isClosed=True,
                              color=(0, 0, 255), thickness=1)

                x0 = int(min(p[0] for p in box))
                y0 = int(min(p[1] for p in box))
                label = str(text)

                cv2.putText(
                    overlay, label,
                    (x0, max(10, y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
        else:
            # fallback: no OCR, just use the clean grayscale crop
            overlay = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)

        # save ONE image per crop, with OCR overlay
        col_path = os.path.join(output_dir, f"{base}_col{idx:02d}.png")
        cv2.imwrite(col_path, overlay)
        print(f"[CROP+OCR] saved: {col_path}")

    return overlay_page_path

def _degrid_crop(src_bgr: np.ndarray) -> np.ndarray:
    """
    Remove vertical & horizontal table lines from a cropped BGR image.
    - binarize
    - detect long vertical + horizontal strokes
    - inpaint them out
    Returns a grayscale image with lines removed.
    """
    if src_bgr.size == 0:
        return src_bgr

    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # binarize
    bw = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )

    H, W = bw.shape

    # vertical lines
    Kv = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(18, int(0.12 * H)))
    )
    v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)

    # horizontal lines
    Kh = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(25, int(0.35 * W)), 1)
    )
    h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

    grid = cv2.bitwise_or(v, h)

    # slightly thicken the mask so we cover the whole stroke
    mask = cv2.dilate(
        grid,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    # inpaint on grayscale; result is 1-channel, no lines
    clean = cv2.inpaint(gray, mask, 2, cv2.INPAINT_TELEA)
    return clean

def run():
    src = os.path.abspath(os.path.expanduser(SOURCE_INPUT))
    out_dir = os.path.abspath(os.path.expanduser(OUTPUT_DIR))

    if os.path.isdir(src):
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        images = [
            os.path.join(src, f)
            for f in sorted(os.listdir(src))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not images:
            raise RuntimeError(f"No images found in directory: {src}")

        print(f"Found {len(images)} images in {src}")
        for img in images:
            out = make_overlay_for_image(img, out_dir)
            print("  overlay ->", out)
    else:
        out = make_overlay_for_image(src, out_dir)
        print("Overlay saved to:", out)


if __name__ == "__main__":
    run()
