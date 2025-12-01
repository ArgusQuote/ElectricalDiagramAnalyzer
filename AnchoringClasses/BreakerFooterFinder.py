# OcrLibrary/BreakerFooterFinder.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
import cv2
import numpy as np

from .BreakerHeaderFinder import Dbg  # reuse the small dbg dataclass

FOOTER_TOKEN_VALUES = {17, 18, 29, 30, 41, 42, 53, 54, 65, 66, 71, 72, 83, 84}


@dataclass
class FooterResult:
    footer_y: Optional[int]         # final snapped footer line (gray space)
    token_y_bottom: Optional[int]   # bottom of footer token text (gray space)
    token_val: Optional[int]        # 17, 18, ..., 84
    used_new_method: bool           # True if token/line method used
    dbg_marks: List[Tuple[int, str]]  # (y, label) for overlays


class BreakerFooterFinder:
    """
    Encapsulates footer detection logic.

    Strategy:
      1) New primary:
         - starting from first breaker line (from centers/header)
         - use CCT/CKT OCR debug items to find bounding vertical lines
         - crop each CCT column
         - trim top/bottom per the knobs
         - OCR and find FOOTER_TOKEN_VALUES
         - best token => footer text line seed
         - snap to structural horizontal line near that seed
      2) Fallback:
         - structural footer_y from header finder (footer_struct_y)
    """

    def __init__(
        self,
        reader,
        bottom_trim_frac: float = 0.15,
        top_trim_frac: float = 0.50,
        upscale_factor: float = 1.0,
        debug: bool = False,
    ):
        self.reader = reader
        self.bottom_trim_frac = float(bottom_trim_frac)
        self.top_trim_frac = float(top_trim_frac)
        self.upscale_factor = float(upscale_factor)
        self.debug = debug

    # ---------- public API ----------
    def find_footer(
        self,
        gray: np.ndarray,
        header_y: Optional[int],
        centers: List[int],
        dbg: Dbg,
        ocr_dbg_items: List[dict],
        footer_struct_y: Optional[int],
    ) -> FooterResult:
        """
        Try new token+line method; if it fails, fall back to footer_struct_y.
        """
        H, W = gray.shape
        dbg_marks: List[Tuple[int, str]] = []

        # Try new method
        new_footer = self._footer_by_tokens_and_lines(
            gray,
            header_y,
            centers,
            ocr_dbg_items,
            dbg_marks,
        )

        if new_footer.footer_y is not None:
            # success
            return FooterResult(
                footer_y=new_footer.footer_y,
                token_y_bottom=new_footer.token_y_bottom,
                token_val=new_footer.token_val,
                used_new_method=True,
                dbg_marks=dbg_marks,
            )

        # Fallback to structural footer
        return FooterResult(
            footer_y=footer_struct_y,
            token_y_bottom=None,
            token_val=None,
            used_new_method=False,
            dbg_marks=dbg_marks,
        )

    # ---------- internals ----------

    @dataclass
    class _TokenFooter:
        footer_y: Optional[int]
        token_y_bottom: Optional[int]
        token_val: Optional[int]

    def _footer_by_tokens_and_lines(
        self,
        gray: np.ndarray,
        header_y: Optional[int],
        centers: List[int],
        ocr_dbg_items: List[dict],
        dbg_marks: List[Tuple[int, str]],
    ) -> _TokenFooter:
        H, W = gray.shape

        # Need OCR to do anything here
        if self.reader is None:
            return self._TokenFooter(None, None, None)

        # 1) Estimate first breaker line from centers/header
        if centers:
            centers_sorted = sorted(centers)
            if len(centers_sorted) >= 2:
                med_row_h = float(np.median(np.diff(centers_sorted)))
            else:
                med_row_h = 18.0
            first_center = centers_sorted[0]
            first_breaker_line = int(first_center - 0.5 * med_row_h)
            if header_y is not None:
                first_breaker_line = max(first_breaker_line, header_y + 4)
        else:
            # no centers at all, fall back to header or top-band
            first_breaker_line = (header_y or int(0.15 * H)) + 8

        y_top = max(0, first_breaker_line)
        y_bot = H

        # 2) bottom trim on the global vertical span
        if self.bottom_trim_frac > 0.0 and y_bot > y_top + 1:
            span = y_bot - y_top
            cut = int(span * self.bottom_trim_frac)
            if cut > 0:
                y_bot = max(y_top + 1, y_bot - cut)

        if y_bot <= y_top + 8:
            return self._TokenFooter(None, None, None)

        # 3) detect vertical lines in that band
        v_xs = self._find_vertical_lines(gray, y_top, y_bot)
        if not v_xs:
            return self._TokenFooter(None, None, None)

        # 4) use CCT/CKT debug items to get column spans
        col_spans = self._column_spans_from_ckt_tokens(v_xs, ocr_dbg_items)
        if not col_spans:
            return self._TokenFooter(None, None, None)

        # 5) OCR each CCT column crop & hunt for footer tokens
        footer_candidates = []  # dicts with {val, y_bottom, pts}

        for (xL, xR) in sorted(col_spans, key=lambda p: p[0]):
            xL = max(0, min(W - 1, int(xL)))
            xR = max(xL + 1, min(W, int(xR)))
            col_crop = gray[y_top:y_bot, xL:xR]
            if col_crop.size == 0:
                continue

            Hc, Wc = col_crop.shape
            cut_top = 0
            if 0.0 < self.top_trim_frac < 1.0:
                cut_top = int(Hc * self.top_trim_frac)
                if cut_top > 0 and cut_top < Hc:
                    col_crop = col_crop[cut_top:, :]
                    Hc, Wc = col_crop.shape
                else:
                    cut_top = 0

            if Hc < 10 or Wc < 4:
                continue

            eff_scale = self.upscale_factor if self.upscale_factor > 0 else 1.0
            if eff_scale != 1.0:
                col_up = cv2.resize(
                    col_crop,
                    None,
                    fx=eff_scale,
                    fy=eff_scale,
                    interpolation=cv2.INTER_CUBIC
                )
            else:
                col_up = col_crop

            try:
                dets = self.reader.readtext(
                    col_up,
                    detail=1,
                    paragraph=False,
                )
            except Exception:
                dets = []

            for box, txt, conf in dets:
                if not txt:
                    continue
                m = re.search(r"\d+", str(txt))
                if not m:
                    continue
                try:
                    val = int(m.group(0))
                except ValueError:
                    continue
                if val not in FOOTER_TOKEN_VALUES:
                    continue

                mapped_pts = []
                for bx, by in box:
                    by_unscaled = by / eff_scale
                    bx_unscaled = bx / eff_scale
                    by_in_col = cut_top + by_unscaled
                    bx_in_col = bx_unscaled
                    y_gray = int(round(y_top + by_in_col))
                    x_gray = int(round(xL + bx_in_col))
                    mapped_pts.append((x_gray, y_gray))

                if not mapped_pts:
                    continue

                y_bottom = max(p[1] for p in mapped_pts)
                footer_candidates.append({
                    "val": val,
                    "y_bottom": y_bottom,
                    "pts": mapped_pts,
                    "text": str(txt),
                })

        if not footer_candidates:
            return self._TokenFooter(None, None, None)

        # Best candidate by (value, y_bottom)
        best = max(footer_candidates, key=lambda d: (d["val"], d["y_bottom"]))
        token_val = best["val"]
        token_y_bottom = int(best["y_bottom"])

        dbg_marks.append((token_y_bottom, f"FOOTER_TOKEN_{token_val}"))

        # 6) Snap to structural horizontal line near token_y_bottom
        footer_struct_y = self._snap_footer_line(gray, token_y_bottom)
        if footer_struct_y is not None:
            dbg_marks.append((footer_struct_y, "FOOTER_LINE_STRUCT"))
            return self._TokenFooter(int(footer_struct_y), token_y_bottom, token_val)

        # No structural line: return token baseline as best guess
        return self._TokenFooter(token_y_bottom, token_y_bottom, token_val)

    # --- helpers ---

    def _find_vertical_lines(self, gray: np.ndarray, y_top: int, y_bot: int) -> List[int]:
        H, W = gray.shape
        y1 = max(0, int(y_top))
        y2 = min(H, int(y_bot))
        roi = gray[y1:y2, :]
        if roi.size == 0:
            return []

        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21, 10
        )

        roi_h = roi.shape[0]
        Kv = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(25, int(0.015 * roi_h)))
        )
        v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)

        xs = []
        num_v, labels_v, stats_v, _ = cv2.connectedComponentsWithStats(v_candidates, connectivity=8)
        if num_v > 1:
            min_vert_len = int(0.40 * roi_h)
            max_vert_thick = max(2, int(0.02 * W))
            for i in range(1, num_v):
                x, y, w, h, area = stats_v[i]
                if h >= min_vert_len and w <= max_vert_thick and area > 0:
                    x_center = x + w // 2
                    xs.append(int(x_center))

        if not xs:
            return []

        xs = sorted(xs)
        collapsed = []
        for x in xs:
            if not collapsed or abs(x - collapsed[-1]) > 4:
                collapsed.append(x)
        return collapsed

    def _column_spans_from_ckt_tokens(
        self,
        v_xs: List[int],
        ocr_dbg_items: List[dict],
    ) -> List[Tuple[int, int]]:
        spans = []
        for item in ocr_dbg_items:
            txt = item.get("text", "")
            if not txt:
                continue
            n = re.sub(r"[^A-Z]", "", str(txt).upper().replace("1", "I"))
            if n not in ("CCT", "CKT"):
                continue

            x1 = int(item["x1"])
            x2 = int(item["x2"])

            left_candidates = [v for v in v_xs if v <= x1]
            right_candidates = [v for v in v_xs if v >= x2]
            if not left_candidates or not right_candidates:
                continue
            xL = max(left_candidates)
            xR = min(right_candidates)
            if xR <= xL:
                continue
            spans.append((xL, xR))
        return spans

    def _snap_footer_line(self, gray: np.ndarray, y_seed: int) -> Optional[int]:
        H, W = gray.shape
        band_top = max(0, y_seed - 80)
        band_bot = min(H, y_seed + 20)
        band = gray[band_top:band_bot, :]
        if band.size == 0:
            return None

        blur = cv2.GaussianBlur(band, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21, 10
        )

        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(40, int(0.12 * W)), 1)
        )
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)
        proj = h_candidates.sum(axis=1)
        if proj.size == 0 or proj.max() == 0:
            return None

        rel_y = int(np.argmax(proj))
        return int(band_top + rel_y)
