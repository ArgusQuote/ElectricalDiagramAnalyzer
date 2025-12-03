from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import re
import cv2
import numpy as np


@dataclass
class HeaderDbg:
    lines: List[int]
    centers: List[int]


Dbg = HeaderDbg


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


class BreakerHeaderFinder:
    """
    Simplified header finder.

    - Crops the top 50% of the page.
    - Removes the top 15% of that half.
      => band ~7.5%H to 50%H.
    - Runs OCR on that band.
    - Groups tokens into horizontal lines (y-bins).
    - Chooses the line with the MOST tokens.
    - The FIRST token on that line (left-most) is the header anchor.
      - Its top y is used to define the header text line / header_y.

    Debug overlay (only if debug=True):
      - Blue   : header anchor token + header text line.
      - Magenta: other header-ish tokens (CKT, DESCRIPTION, TRIP, POLES, ...).
      - Red    : excluded tokens ("LOAD CLASSIFICATION" family).
      - Green  : everything else.
    """

    def __init__(self, reader, debug: bool = False, debug_dir: Optional[str] = None):
        self.reader = reader
        self.debug = debug
        self.debug_dir: Optional[str] = debug_dir

        self.ocr_dbg_items: List[dict] = []
        self.ocr_dbg_rois: List[Tuple[int, int, int, int]] = []

        self.footer_struct_y: Optional[int] = None
        self.last_footer_y: Optional[int] = None
        self.bottom_border_y: Optional[int] = None
        self.bottom_row_center_y: Optional[int] = None

        self.spaces_detected: int = 0
        self.spaces_corrected: int = 0
        self.footer_snapped: bool = False
        self.snap_note: Optional[str] = None

        self.header_text_line_y: Optional[int] = None

        self.CATEGORY_ALIASES = {
            "ckt": {"CKT", "CCT"},
            "description": {
                "CIRCUITDESCRIPTION", "DESCRIPTION", "LOADDESCRIPTION",
                "DESIGNATION", "LOADDESIGNATION", "NAME",
            },
            "trip": {"TRIP", "AMPS", "AMP", "BREAKER", "BKR", "SIZE"},
            "poles": {"POLES", "POLE", "P"},
        }
        self.EXCLUDE = {"LOADCLASSIFICATION", "CLASSIFICATION"}
        self.HEADER_TOKEN_SET = set().union(*self.CATEGORY_ALIASES.values())

    # ---------- public ----------

    def analyze_rows(self, gray: np.ndarray) -> HeaderResult:
        """
        Main entry point.

        For now, this ONLY finds header_y using OCR tokens.
        Row centers / footer info are not computed (kept for API compatibility).
        """
        self.ocr_dbg_items = []
        self.ocr_dbg_rois = []
        self.footer_struct_y = None
        self.last_footer_y = None
        self.bottom_border_y = None
        self.bottom_row_center_y = None
        self.spaces_detected = 0
        self.spaces_corrected = 0
        self.footer_snapped = False
        self.snap_note = None
        self.header_text_line_y = None

        header_y = self._find_header_by_tokens(gray)

        dbg = HeaderDbg(lines=[], centers=[])
        centers: List[int] = []

        return HeaderResult(
            centers=centers,
            dbg=dbg,
            header_y=header_y,
            footer_struct_y=self.footer_struct_y,
            last_footer_y=self.last_footer_y,
            spaces_detected=self.spaces_detected,
            spaces_corrected=self.spaces_corrected,
            footer_snapped=self.footer_snapped,
            snap_note=self.snap_note,
            bottom_border_y=self.bottom_border_y,
            bottom_row_center_y=self.bottom_row_center_y,
        )

    # ---------- debug overlay ----------

    def draw_ocr_overlay(self, base_img: np.ndarray) -> np.ndarray:
        """
        Draw OCR debug ROIs and boxes onto base_img.

        Only active when self.debug is True.
        Overlay is full-size (same size as base_img).

        Colors:
          - Blue   : header anchor token + header text line
          - Magenta: other header tokens
          - Red    : excluded tokens (optional)
        """
        # always return a BGR image
        if base_img.ndim == 2:
            overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            overlay = base_img.copy()

        if not self.debug:
            return overlay

        # ROI rectangles
        for (x1, y1, x2, y2) in self.ocr_dbg_rois:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # header / anchor tokens only
        for item in self.ocr_dbg_items:
            is_anchor  = item.get("is_header_anchor", False)
            is_header  = item.get("is_header_token", False)
            is_excl    = item.get("is_excluded_token", False)

            # skip non-header junk entirely
            if not (is_anchor or is_header or is_excl):
                continue

            x1, y1 = item["x1"], item["y1"]
            x2, y2 = item["x2"], item["y2"]
            txt = item.get("text", "")

            if is_anchor:
                box_color  = (255, 0, 0)   # blue
                text_color = (255, 0, 0)
            elif is_header:
                box_color  = (255, 0, 255) # magenta
                text_color = (255, 0, 255)
            else:  # excluded
                box_color  = (0, 0, 255)   # red
                text_color = (0, 0, 255)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 1)
            if txt:
                cv2.putText(
                    overlay,
                    txt[:20],
                    (x1, max(y1 - 2, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        # header text line (blue) across full image
        if self.header_text_line_y is not None:
            y = int(self.header_text_line_y)
            H, W = overlay.shape[:2]
            cv2.line(overlay, (0, y), (W - 1, y), (255, 0, 0), 2)
            cv2.putText(
                overlay,
                "HEADER_TEXT_LINE",
                (10, max(10, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return overlay

    # ---------- helpers ----------

    @staticmethod
    def _normalize_text(s: str) -> str:
        return re.sub(
            r"[^A-Z0-9]",
            "",
            (s or "").upper().replace("1", "I").replace("0", "O"),
        )

    def _run_ocr(self, img, mag: float):
        try:
            return self.reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/",
                mag_ratio=mag,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.4,
                low_text=0.25,
            )
        except Exception:
            return []

    # ---------- header finder ----------

    def _find_header_by_tokens(self, gray: np.ndarray) -> Optional[int]:
        """
        Use OCR header tokens to find the header row.

        - Crop to top 50% of page.
        - Remove top 15% of that half.  (~7.5%H..50%H)
        - OCR on that band.
        - Group tokens by y-bin.
        - Pick the line with the MOST *header* tokens.
        - Anchor is the left-most header token on that line.
        """
        if self.reader is None:
            return None

        H, W = gray.shape

        # --- header band (same as simple script) ---
        half_h = int(0.50 * H)
        y1_band = int(0.15 * half_h)   # ~0.075 * H
        y2_band = half_h               # 0.50 * H
        x1_band, x2_band = 0, W

        roi = gray[y1_band:y2_band, x1_band:x2_band]
        if roi.size == 0:
            self.header_text_line_y = None
            return None

        # optional: save band for debug
        if self.debug and self.debug_dir:
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
                dbg_name = (
                    f"header_band_y{y1_band}_{y2_band}_"
                    f"h{roi.shape[0]}_w{roi.shape[1]}.png"
                )
                dbg_path = os.path.join(self.debug_dir, dbg_name)
                cv2.imwrite(dbg_path, roi)
                print(f"[BreakerHeaderFinder] Saved OCR header band: {dbg_path}")
            except Exception as e:
                print(f"[BreakerHeaderFinder] Failed to save header band: {e}")

        # reset debug collectors and record ROI in ABSOLUTE coords
        self.ocr_dbg_items = []
        self.ocr_dbg_rois = [(x1_band, y1_band, x2_band, y2_band)]

        # OCR passes
        det = self._run_ocr(roi, 1.6) + self._run_ocr(roi, 2.0)

        items: List[dict] = []
        for box, text, conf in det:
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x1b, y1b = min(xs), min(ys)
            x2b, y2b = max(xs), max(ys)

            # shift to full-image coordinates
            x1_abs = x1_band + x1b
            x2_abs = x1_band + x2b
            y1_abs = y1_band + y1b
            y2_abs = y1_band + y2b

            norm = self._normalize_text(text)
            item = {
                "box": box,
                "text": text,
                "conf": float(conf or 0.0),
                "x1": x1_abs,
                "y1": y1_abs,
                "x2": x2_abs,
                "y2": y2_abs,
                "norm": norm,
            }
            items.append(item)
            self.ocr_dbg_items.append(item)

        if not items:
            self.header_text_line_y = None
            return None

        # ---- group by y-bin ----
        lines: dict[int, list[dict]] = {}
        for it in items:
            yc = 0.5 * (it["y1"] + it["y2"])
            ybin = int(yc // 14) * 14
            lines.setdefault(ybin, []).append(it)

        # ---- choose line with MOST HEADER TOKENS ----
        best_ybin: Optional[int] = None
        best_hdr_count = 0

        for ybin, line_items in lines.items():
            hdr_count = 0
            for it in line_items:
                if any(tok in it["norm"] for tok in self.HEADER_TOKEN_SET):
                    hdr_count += 1
            if hdr_count > 0:
                if (
                    hdr_count > best_hdr_count
                    or (hdr_count == best_hdr_count and (best_ybin is None or ybin < best_ybin))
                ):
                    best_hdr_count = hdr_count
                    best_ybin = ybin

        if best_ybin is None:
            # no header-like tokens anywhere in band
            self.header_text_line_y = None
            return None

        # tokens on the winning line
        line_items = lines[best_ybin]
        hdr_items = [
            it for it in line_items
            if any(tok in it["norm"] for tok in self.HEADER_TOKEN_SET)
        ]
        if not hdr_items:
            self.header_text_line_y = None
            return None

        # left-most header token is anchor
        hdr_items_sorted = sorted(hdr_items, key=lambda it: it["x1"])
        anchor = hdr_items_sorted[0]
        anchor["is_header_anchor"] = True

        # mark all header/excluded tokens for overlay
        for it in items:
            norm = it["norm"]
            if any(tok in norm for tok in self.HEADER_TOKEN_SET):
                it["is_header_token"] = True
            if any(excl in norm for excl in self.EXCLUDE):
                it["is_excluded_token"] = True

        header_text_y = int(anchor["y1"])
        self.header_text_line_y = header_text_y

        return header_text_y
