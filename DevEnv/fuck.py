#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import re

import cv2
import easyocr


# --------- CONFIG ---------
IMAGE_PATH = "/home/marco/Documents/Diagrams/CaseStudy_VectorCrop_Run15/ELECTRICAL SET (Mark Up)_electrical_filtered_page002_panel02.png"
OUT_DIR = "/home/marco/Documents/Diagrams/HeaderCropTestTokens2"
# --------------------------


def N(s: str) -> str:
    """Normalize text: uppercase, 1->I, 0->O, strip non-alnum."""
    return re.sub(
        r"[^A-Z0-9]",
        "",
        (s or "").upper().replace("1", "I").replace("0", "O"),
    )


CATEGORY_ALIASES = {
    "ckt": {"CKT", "CCT"},
    "description": {
        "CIRCUITDESCRIPTION", "DESCRIPTION", "LOADDESCRIPTION",
        "DESIGNATION", "LOADDESIGNATION", "NAME",
    },
    "trip": {"TRIP", "AMPS", "AMP", "BREAKER", "BKR", "SIZE"},
    "poles": {"POLES", "POLE", "P"},
}
EXCLUDE = {"LOADCLASSIFICATION", "CLASSIFICATION"}
HEADER_TOKEN_SET = set().union(*CATEGORY_ALIASES.values())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read file: {IMAGE_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # --- CROP: TOP 50% minus 15% of that half from top ---
    half_h = int(0.50 * H)      # top half of page
    y1 = int(0.15 * half_h)     # remove 15% of top half -> ~0.075 * H
    y2 = half_h                 # 0.50 * H

    crop_gray = gray[y1:y2, :]
    crop_color = img[y1:y2, :].copy()

    base = Path(IMAGE_PATH).stem

    # ---- OCR on the CROP only ----
    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(
        crop_gray,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/",
    )

    items = []
    for box, text, conf in results:
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        x1b, y1b = min(xs), min(ys)
        x2b, y2b = max(xs), max(ys)
        norm = N(text)

        items.append(
            {
                "box": box,
                "text": text,
                "conf": float(conf or 0.0),
                "x1": x1b,
                "y1": y1b,
                "x2": x2b,
                "y2": y2b,
                "norm": norm,
            }
        )

    # ---- find line with most header tokens (not all tokens) ----
    lines = {}
    for it in items:
        yc = 0.5 * (it["y1"] + it["y2"])
        ybin = int(yc // 14) * 14
        lines.setdefault(ybin, []).append(it)

    header_anchor = None
    header_text_y = None

    best_ybin = None
    best_hdr_count = 0

    for ybin, line_items in lines.items():
        hdr_count = sum(
            any(tok in it["norm"] for tok in HEADER_TOKEN_SET)
            for it in line_items
        )
        if hdr_count > 0:
            if hdr_count > best_hdr_count or (
                hdr_count == best_hdr_count and (best_ybin is None or ybin < best_ybin)
            ):
                best_hdr_count = hdr_count
                best_ybin = ybin

    if best_ybin is not None:
        line_items = lines[best_ybin]
        hdr_items = [
            it for it in line_items
            if any(tok in it["norm"] for tok in HEADER_TOKEN_SET)
        ]
        hdr_items_sorted = sorted(hdr_items, key=lambda it: it["x1"])
        header_anchor = hdr_items_sorted[0]
        header_anchor["is_header_anchor"] = True
        header_text_y = header_anchor["y1"]

    # ---- draw overlay ONLY for header tokens ----
    for it in items:
        norm = it["norm"]
        is_header_token = any(tok in norm for tok in HEADER_TOKEN_SET)
        is_anchor = it.get("is_header_anchor", False)

        # skip non-header tokens entirely
        if not is_header_token and not is_anchor:
            continue

        x1b, y1b, x2b, y2b = it["x1"], it["y1"], it["x2"], it["y2"]

        if is_anchor:
            box_color = (255, 0, 0)   # blue
            text_color = (255, 0, 0)
        else:
            box_color = (255, 0, 255) # magenta
            text_color = (255, 0, 255)

        cv2.rectangle(crop_color, (x1b, y1b), (x2b, y2b), box_color, 1)

        # label with token text
        label = it["text"][:20]
        ty_label = max(0, y1b - 3)
        cv2.putText(
            crop_color,
            label,
            (x1b, ty_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            text_color,
            1,
            cv2.LINE_AA,
        )

        # for magenta header tokens (non-anchor), also write Y value
        if not is_anchor:
            y_center = int(0.5 * (y1b + y2b))
            y_text = f"{y_center}"
            cv2.putText(
                crop_color,
                y_text,
                (x2b + 3, y_center + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                text_color,
                1,
                cv2.LINE_AA,
            )

    # optional: draw header text line across crop
    if header_text_y is not None:
        cv2.line(
            crop_color,
            (0, header_text_y),
            (crop_color.shape[1] - 1, header_text_y),
            (255, 0, 0),
            2,
        )

    overlay_path = os.path.join(OUT_DIR, f"{base}_crop_tokens_overlay.png")
    cv2.imwrite(overlay_path, crop_color)
    print("Saved cropped overlay to:", overlay_path)


if __name__ == "__main__":
    main()
