#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# -------------------------------------------------------------
# Debug structure
# -------------------------------------------------------------
@dataclass
class HeaderDbg:
    lines: List[int]
    centers: List[int]


# -------------------------------------------------------------
# Header result bundle
# -------------------------------------------------------------
@dataclass
class HeaderResult:
    centers: List[int]
    dbg: HeaderDbg
    header_y: Optional[int]
    footer_struct_y: Optional[int]
    last_footer_y: Optional[int]
    spaces_detected: int
    spaces_corrected: int
    footer_snapped: bool
    snap_note: Optional[str]
    bottom_border_y: Optional[int]
    bottom_row_center_y: Optional[int]


# -------------------------------------------------------------
# Main Class
# -------------------------------------------------------------
class BreakerHeaderFinder:
    def __init__(self, reader, debug: bool = False):
        self.reader = reader
        self.debug = debug

        # OCR debug state
        self.ocr_dbg_items: List[dict] = []
        self.ocr_dbg_rois: List[Tuple[int, int, int, int]] = []

        # structural/footer attributes
        self.footer_struct_y: Optional[int] = None
        self.last_footer_y: Optional[int] = None
        self.bottom_border_y: Optional[int] = None
        self.bottom_row_center_y: Optional[int] = None

        self.spaces_detected: int = 0
        self.spaces_corrected: int = 0
        self.footer_snapped: bool = False
        self.snap_note: Optional[str] = None

        # NEW: optional debug output folder
        self.debug_dir: Optional[str] = None

    # ---------------------------------------------------------
    # INTERNAL: simple cadence detection using horizontal lines
    # ---------------------------------------------------------
    def _row_centers_from_lines(self, gray: np.ndarray):
        """
        VERY simple placeholder logic:
        1) Adaptive threshold
        2) Morph-open horizontally
        3) Connected components → horizontal lines
        4) Return their centers
        """

        H, W = gray.shape
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21, 10
        )

        # horizontal structuring element
        Kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W // 25), 1))
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            h_candidates, connectivity=8
        )

        centers = []
        lines = []

        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if w > 0.35 * W and h <= 6:
                cy = y + h // 2
                centers.append(int(cy))
                lines.append(int(cy))

        centers_sorted = sorted(centers)
        lines_sorted = sorted(lines)

        # crude header position: first line
        header_y = centers_sorted[0] if centers_sorted else None

        dbg = HeaderDbg(lines=lines_sorted, centers=centers_sorted)
        return centers_sorted, dbg, header_y

    # ---------------------------------------------------------
    # PUBLIC: main row / header analysis
    # ---------------------------------------------------------
    def analyze_rows(self, gray: np.ndarray) -> HeaderResult:
        """
        New RULE:
         • Start with full image.
         • Crop to TOP 50% of image.
         • Discard top 15% of that half.
         • Analyze the remaining band.
         • Shift results back to full-image coordinates.
        """

        H_full, W_full = gray.shape

        # ------------ define analysis band -------------
        half_end = int(0.50 * H_full)                     # first 50% of page
        crop_top = int(0.15 * half_end)                  # remove top 15% of that
        band_start = crop_top                            # ~7.5% of full height
        band_end = half_end                              # 50% of full height

        # safety
        if band_end - band_start < 30:
            band_start = 0
            band_end = H_full

        band = gray[band_start:band_end, :]

        # ------------ debug: save band crop -------------
        if self.debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            path = os.path.join(self.debug_dir, "header_band.png")
            cv2.imwrite(path, band)

        # ------------ run row detection on cropped band -------------
        centers_band, dbg_band, header_y_band = self._row_centers_from_lines(band)

        # ------------ shift results back to full-image coords -------------
        centers_full = [c + band_start for c in centers_band]

        dbg_full = HeaderDbg(
            lines=[y + band_start for y in dbg_band.lines],
            centers=[c + band_start for c in dbg_band.centers],
        )

        header_y_full = header_y_band + band_start if header_y_band is not None else None

        # (struct footer fallback not detected here; footer finder supplies it)
        return HeaderResult(
            centers=centers_full,
            dbg=dbg_full,
            header_y=header_y_full,
            footer_struct_y=self.footer_struct_y,
            last_footer_y=self.last_footer_y,
            spaces_detected=self.spaces_detected,
            spaces_corrected=self.spaces_corrected,
            footer_snapped=self.footer_snapped,
            snap_note=self.snap_note,
            bottom_border_y=self.bottom_border_y,
            bottom_row_center_y=self.bottom_row_center_y,
        )
