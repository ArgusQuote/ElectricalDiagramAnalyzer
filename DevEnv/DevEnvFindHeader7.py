#!/usr/bin/env python3
# DevEnvFindHeader6.py

import os
import cv2
import numpy as np
import sys
import re

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

BOTTOM_TRIM_FRAC = 0.15   # fraction of vertical span (between first row and bottom) removed from bottom
TOP_TRIM_FRAC    = 0.50   # fraction of EACH column crop removed from top before OCR (0.5 = keep bottom half)
UPSCALE_FACTOR   = 1.0    # 1.0 = no resize, 2.0 = 2x upscaling before OCR

FOOTER_TOKEN_VALUES = {17, 18, 29, 30, 41, 42, 53, 54, 65, 66, 71, 72, 83, 84}

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
      - find vertical lines that bound the CCT/CKT column(s)
      - draw HEADER, FIRST BREAKER ROW, CCT/CKT boxes + vertical lines on a page overlay
      - crop each such column (from first breaker row down) FROM THE SOURCE IMAGE
      - trim bottom BOTTOM_TRIM_FRAC of that vertical span
      - within each column crop, trim TOP_TRIM_FRAC from the top (keep bottom portion)
      - remove grid lines in the crop
      - optionally upscale by UPSCALE_FACTOR
      - run OCR on the processed crop and draw boxes + raw text on the crop
      - find footer token among FOOTER_TOKEN_VALUES
      - use the bottom of that token as a FOOTER_TEXT_LINE seed,
        then snap to the structural footer line using a horizontal-line mask
      - save ONE image per column: <base>_colXX.png (with OCR overlay)
      - also save a page-level overlay: <base>_overlay.png
    """
    image_path = os.path.abspath(os.path.expanduser(image_path))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    os.makedirs(output_dir, exist_ok=True)

    analyzer = BreakerTableAnalyzer(debug=False)

    # ----- load source image -----
    orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig is None:
        raise RuntimeError(f"Cannot read file (not an image?): {image_path}")

    # ----- detection space (resized gray) -----
    gray = analyzer._prep(orig)
    centers, dbg, header_y = analyzer._row_centers_from_lines(gray)
    first_breaker_line = dbg.lines[0] if dbg.lines else None
    footer_y = getattr(analyzer, "_last_footer_y", None)

    Hg, Wg = gray.shape
    Ho, Wo = orig.shape[:2]

    # map orig <-> gray
    scale_y = Hg / float(Ho)
    scale_x = Wg / float(Wo)

    # ----- page-level debug overlay -----
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # HEADER (green)
    if header_y is not None:
        y = int(header_y)
        cv2.line(vis, (0, y), (Wg - 1, y), (0, 255, 0), 2)
        cv2.putText(
            vis, "HEADER",
            (10, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2, cv2.LINE_AA
        )

    # FIRST BREAKER ROW (blue)
    if first_breaker_line is not None:
        yb = int(first_breaker_line)
        cv2.line(vis, (0, yb), (Wg - 1, yb), (255, 0, 0), 2)
        cv2.putText(
            vis, "FIRST BREAKER ROW",
            (10, yb + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 0, 0), 2, cv2.LINE_AA
        )

    # ----- vertical bounds around CCT/CKT tokens -----
    v_xs = _find_vertical_lines(gray, header_y, footer_y)
    col_spans_gray = set()
    drawn_vs = set()

    for item in getattr(analyzer, "_ocr_dbg_items", []):
        txt = item.get("text", "")
        norm = _norm_token(txt)
        if norm not in ("CCT", "CKT"):
            continue

        x1 = int(item["x1"])
        y1 = int(item["y1"])
        x2 = int(item["x2"])
        y2 = int(item["y2"])

        # draw CCT/CKT box on page overlay
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis, norm,
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 1, cv2.LINE_AA
        )

        # nearest vertical left of token
        left_candidates = [v for v in v_xs if v <= x1]
        left = max(left_candidates) if left_candidates else None
        if left is not None and left not in drawn_vs:
            drawn_vs.add(left)
            cv2.line(vis, (left, 0), (left, Hg - 1), (255, 0, 255), 2)
            cv2.putText(
                vis, "V_L",
                (left + 2, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 0, 255), 1, cv2.LINE_AA
            )

        # nearest vertical right of token
        right_candidates = [v for v in v_xs if v >= x2]
        right = min(right_candidates) if right_candidates else None
        if right is not None and right not in drawn_vs:
            drawn_vs.add(right)
            cv2.line(vis, (right, 0), (right, Hg - 1), (255, 0, 255), 2)
            cv2.putText(
                vis, "V_R",
                (right + 2, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 0, 255), 1, cv2.LINE_AA
            )

        if left is not None and right is not None and right > left:
            col_spans_gray.add((left, right))

    base = os.path.splitext(os.path.basename(image_path))[0]

    # ----- crop FROM SOURCE & OCR each column -----

    # vertical range in gray-space: from FIRST BREAKER ROW down
    if first_breaker_line is not None:
        y_top_g = int(first_breaker_line)
    else:
        y_top_g = 0
    y_bot_g = Hg

    # map to source coords
    y_top_o = max(0, min(Ho - 1, int(round(y_top_g / scale_y))))
    y_bot_o = max(y_top_o + 1, min(Ho, int(round(y_bot_g / scale_y))))

    # apply bottom trim on global span
    trim_frac = float(BOTTOM_TRIM_FRAC)
    if trim_frac > 0.0:
        span_h = y_bot_o - y_top_o
        cut = int(span_h * trim_frac)
        if cut > 0:
            y_bot_o = max(y_top_o + 1, y_bot_o - cut)

    col_spans_sorted = sorted(col_spans_gray, key=lambda p: p[0])

    # collect footer token candidates in page (gray) coords
    footer_candidates = []

    for idx, (xl_g, xr_g) in enumerate(col_spans_sorted, start=1):
        # map x-bounds to source image
        x_left_o = max(0, min(Wo - 1, int(round((xl_g - 1) / scale_x))))
        x_right_o = max(x_left_o + 1, min(Wo, int(round((xr_g + 1) / scale_x))))

        # crop from source (already vertically trimmed at bottom)
        src_crop = orig[y_top_o:y_bot_o, x_left_o:x_right_o]
        if src_crop.size == 0:
            continue

        # remove grid lines (expects src_crop BGR, returns clean grayscale)
        clean_gray = _degrid_crop(src_crop)
        Hc, Wc = clean_gray.shape

        # trim top portion of EACH column crop (keep bottom part only)
        cut_top = 0
        top_frac = float(TOP_TRIM_FRAC)
        if 0.0 < top_frac < 1.0:
            cut_top = int(Hc * top_frac)
            if cut_top > 0 and cut_top < Hc:
                clean_gray = clean_gray[cut_top:, :]
                Hc, Wc = clean_gray.shape
            else:
                cut_top = 0  # degenerate, ignore trim

        # optional upscaling
        eff_scale = float(UPSCALE_FACTOR) if UPSCALE_FACTOR > 0 else 1.0
        if eff_scale != 1.0:
            clean_gray = cv2.resize(
                clean_gray,
                None,
                fx=eff_scale,
                fy=eff_scale,
                interpolation=cv2.INTER_CUBIC
            )

        overlay_col = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)

        # OCR on the processed crop
        dets = []
        if analyzer.reader is not None:
            try:
                dets = analyzer.reader.readtext(
                    clean_gray,
                    detail=1,
                    paragraph=False,
                )
            except Exception:
                dets = []

        # draw raw OCR results on the crop AND collect footer candidates
        for box, txt, conf in dets:
            if not txt:
                continue

            # draw on crop overlay
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(
                overlay_col, [pts], isClosed=True,
                color=(0, 0, 255), thickness=1
            )

            x0 = int(min(p[0] for p in box))
            y0 = int(min(p[1] for p in box))
            cv2.putText(
                overlay_col, str(txt),
                (x0, max(10, y0 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            # ----- footer token candidate detection -----
            m = re.search(r"\d+", str(txt))
            if not m:
                continue
            try:
                val = int(m.group(0))
            except ValueError:
                continue
            if val not in FOOTER_TOKEN_VALUES:
                continue

            # map this box back to page (gray) coordinates
            box_gray = []
            for bx, by in box:
                # undo upscaling
                by_unscaled = by / eff_scale
                bx_unscaled = bx / eff_scale

                # position within original src_crop BEFORE top trim
                by_in_src_crop = cut_top + by_unscaled
                bx_in_src_crop = bx_unscaled

                # orig image coords
                y_orig = y_top_o + by_in_src_crop
                x_orig = x_left_o + bx_in_src_crop

                # map to gray coords
                y_gray = int(round(y_orig * scale_y))
                x_gray = int(round(x_orig * scale_x))

                box_gray.append((x_gray, y_gray))

            if not box_gray:
                continue

            y_bottom_gray = max(p[1] for p in box_gray)

            footer_candidates.append({
                "val": val,
                "text": str(txt),
                "y_bottom": y_bottom_gray,
                "box_gray": box_gray,
            })

        # save ONE image per crop, with OCR overlay
        col_path = os.path.join(output_dir, f"{base}_col{idx:02d}.png")
        cv2.imwrite(col_path, overlay_col)
        print(f"[CROP+OCR] saved: {col_path}")

    # ----- choose footer token and draw on page overlay -----
    footer_struct_y = None

    if footer_candidates:
        # highest value wins; tie-breaker by lower position on page (larger y_bottom)
        best = max(footer_candidates, key=lambda d: (d["val"], d["y_bottom"]))

        pts_gray = np.array(best["box_gray"], dtype=np.int32)
        # highlight footer token box in yellow
        cv2.polylines(
            vis, [pts_gray], isClosed=True,
            color=(0, 255, 255), thickness=2
        )

        # label the footer token
        x_min = int(min(p[0] for p in best["box_gray"]))
        y_min = int(min(p[1] for p in best["box_gray"]))
        footer_label = f"FOOTER_TOKEN {best['val']}"
        cv2.putText(
            vis, footer_label,
            (x_min, max(18, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # footer text line at bottom of token (seed)
        y_footer = int(best["y_bottom"])
        cv2.line(
            vis, (0, y_footer), (Wg - 1, y_footer),
            (0, 128, 255), 2
        )
        cv2.putText(
            vis, "FOOTER_TEXT_LINE",
            (10, y_footer + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 128, 255),
            2,
            cv2.LINE_AA
        )

        # ----- SNAP TO STRUCTURAL FOOTER LINE NEAR FOOTER_TEXT_LINE -----
        band_top = max(0, y_footer - 80)
        band_bot = min(Hg, y_footer + 20)
        band = gray[band_top:band_bot, :]
        if band.size > 0:
            blur = cv2.GaussianBlur(band, (3, 3), 0)
            bw = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                21, 10
            )
            # horizontal-line kernel
            Kh = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (max(40, int(0.12 * Wg)), 1)
            )
            h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)
            proj = h_candidates.sum(axis=1)
            if proj.size > 0 and proj.max() > 0:
                # row with strongest horizontal response in this band
                rel_y = int(np.argmax(proj))
                footer_struct_y = band_top + rel_y

    # ----- draw structural footer line if found -----
    if footer_struct_y is not None:
        cv2.line(
            vis, (0, int(footer_struct_y)), (Wg - 1, int(footer_struct_y)),
            (255, 255, 0), 2
        )
        cv2.putText(
            vis, "FOOTER_LINE_STRUCT",
            (10, int(footer_struct_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    # finally save page overlay (with everything)
    overlay_page_path = os.path.join(output_dir, f"{base}_overlay.png")
    cv2.imwrite(overlay_page_path, vis)

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
