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
from AnchoringClasses.BreakerFooterFinder import BreakerFooterFinder, FooterResult


def prep_gray(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    H, W = g.shape
    if H < 1600:
        s = 1600.0 / H
        g = cv2.resize(g, (int(W * s), int(H * s)), interpolation=cv2.INTER_CUBIC)
    return g


def save_anchor_overlay(
    gray: np.ndarray,
    header_res: HeaderResult,
    footer_res: FooterResult,
    out_path: str,
):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape

    borders = header_res.dbg.lines
    centers = header_res.centers
    header_y = header_res.header_y
    struct_footer_y = header_res.footer_struct_y

    # ----------------------------------------------------------------------
    # NEW: draw all vertical lines + CCT/CKT column boundaries
    # ----------------------------------------------------------------------
    vlines_x = getattr(footer_res, "vlines_x", []) or []
    cct_cols = getattr(footer_res, "cct_cols", []) or []

    # all vertical candidates (light bluish)
    for x in vlines_x:
        x = int(x)
        cv2.line(vis, (x, 0), (x, H - 1), (180, 180, 255), 1)

    # CCT/CKT column spans (magenta) + labels near header
    for idx, (xl, xr) in enumerate(cct_cols, start=1):
        xl = int(xl)
        xr = int(xr)
        cv2.line(vis, (xl, 0), (xl, H - 1), (255, 0, 255), 2)
        cv2.line(vis, (xr, 0), (xr, H - 1), (255, 0, 255), 2)

        if header_y is not None:
            hy = int(header_y)
            cv2.putText(
                vis,
                f"CCT_COL{idx}",
                (xl + 2, max(16, hy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # ----------------------------------------------------------------------
    # draw borders lightly for context
    # ----------------------------------------------------------------------
    for y in borders:
        cv2.line(vis, (0, int(y)), (W - 1, int(y)), (220, 220, 220), 1)

    # row centers, small green dots
    for y in centers:
        cv2.circle(vis, (24, int(y)), 4, (0, 255, 0), -1)

    # ----- HEADER + FIRST BREAKER LINE -----
    first_breaker_line_y = None

    if header_y is not None:
        hy = int(header_y)

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

    # ----- FOOTER TEXT + FOOTER LINE -----
    footer_y = footer_res.footer_y
    footer_token_y = getattr(footer_res, "token_y", None)
    footer_token_val = getattr(footer_res, "token_val", None)

    if footer_token_y is not None:
        fty = int(footer_token_y)
        cv2.line(vis, (0, fty), (W - 1, fty), (200, 50, 200), 1)
        label = (
            f"FOOTER_TEXT ({footer_token_val})"
            if footer_token_val is not None
            else "FOOTER_TEXT"
        )
        cv2.putText(
            vis,
            label,
            (12, max(16, fty - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 50, 200),
            1,
            cv2.LINE_AA,
        )

        # BIG token label so you can confirm it's using the right one
        if footer_token_val is not None:
            cv2.putText(
                vis,
                f"FOOTER_TOKEN = {footer_token_val}",
                (W - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    if footer_y is not None:
        fy = int(footer_y)
        cv2.line(vis, (0, fy), (W - 1, fy), (0, 0, 255), 2)
        cv2.putText(
            vis,
            "FOOTER",
            (12, max(16, fy + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if footer_token_val is None and struct_footer_y is not None:
        ys = int(struct_footer_y)
        cv2.line(vis, (0, ys), (W - 1, ys), (180, 0, 0), 2)
        cv2.putText(
            vis,
            "FOOTER_STRUCT (FALLBACK)",
            (12, max(16, ys - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for y_mark, label in getattr(footer_res, "dbg_marks", []):
        y = int(y_mark)
        cv2.line(vis, (0, y), (W - 1, y), (160, 160, 255), 1)
        cv2.putText(
            vis,
            label,
            (240, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 255),
            1,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def main():
    SOURCE_IMAGE = "/home/marco/Documents/Diagrams/CaseStudy_VectorCrop_Run15/ELECTRICAL SET (Mark Up)_electrical_filtered_page002_panel02.png"
    OUT_DIR = "/home/marco/Documents/Diagrams/AnchorDebug1"

    os.makedirs(OUT_DIR, exist_ok=True)

    img = cv2.imread(SOURCE_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read file: {SOURCE_IMAGE}")

    gray = prep_gray(img)

    reader = easyocr.Reader(["en"], gpu=True)

    header_finder = BreakerHeaderFinder(reader, debug=True)
    footer_finder = BreakerFooterFinder(
        reader,
        bottom_trim_frac=0.15,
        top_trim_frac=0.50,
        upscale_factor=1.0,
        debug=True,
    )
    footer_finder.debug_dir = OUT_DIR

    # 1) HEADER FIRST
    header_res: HeaderResult = header_finder.analyze_rows(gray)

    # 2) FOOTER USING **NEW CROP METHOD** (orig_bgr passed in)
    footer_res: FooterResult = footer_finder.find_footer(
        gray=gray,
        header_y=header_res.header_y,
        centers=header_res.centers,
        dbg=header_res.dbg,
        ocr_dbg_items=header_finder.ocr_dbg_items,
        footer_struct_y=header_res.footer_struct_y,
        orig_bgr=img,
    )

    print("Header Y:", header_res.header_y)
    print("Row count:", len(header_res.centers))
    print("Footer struct (fallback):", header_res.footer_struct_y)
    print(
        "Footer final:",
        footer_res.footer_y,
        "token:",
        footer_res.token_val,
        "token_y:",
        footer_res.token_y,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(SOURCE_IMAGE).stem
    overlay_path = os.path.join(OUT_DIR, f"{base}_anchors_{ts}.png")
    save_anchor_overlay(gray, header_res, footer_res, overlay_path)
    print("Overlay:", overlay_path)


if __name__ == "__main__":
    main()
