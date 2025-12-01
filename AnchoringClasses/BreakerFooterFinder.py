# AnchoringClasses/BreakerFooterFinder.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import os
import cv2
import numpy as np
import re


@dataclass
class FooterResult:
    footer_y: Optional[int]
    token_y: Optional[int]
    token_val: Optional[int]
    dbg_marks: List[Tuple[int, str]]  # (y, label) for overlays

    # NEW: for visualization / debugging
    vlines_x: List[int] = field(default_factory=list)              # all vertical-line centers (gray coords)
    cct_cols: List[Tuple[int, int]] = field(default_factory=list)  # [(xl, xr), ...] in gray coords


class BreakerFooterFinder:
    """
    Footer finder that mirrors the logic from DevEnvFindHeader6.py:

      - Builds a vertical-line mask between HEADER and PAGE BOTTOM
      - Uses CCT/CKT header tokens to find left/right CCT columns
      - Crops from FIRST BREAKER LINE down from the SOURCE image
      - Trims bottom & top (knobs)
      - Degrids each crop, optionally upscales, then OCRs
      - Looks for FOOTER_TOKEN_VALUES; highest value wins
      - Uses the bottom of that token as FOOTER_TEXT seed
      - Snaps to strongest horizontal line cluster near that seed
      - If anything fails, falls back to footer_struct_y
    """

    FOOTER_TOKEN_VALUES = {
        17, 18, 29, 30, 41, 42, 53, 54,
        65, 66, 71, 72, 83, 84,
    }

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
        # optional: where to dump vertical mask + column crops
        self.debug_dir: Optional[str] = None

    # ======================================================================
    # PUBLIC ENTRY
    # ======================================================================
    def find_footer(
        self,
        gray: np.ndarray,
        header_y: Optional[int],
        centers: List[int],
        dbg,  # has .lines (row borders)
        ocr_dbg_items: List[dict],  # header OCR items from header finder
        footer_struct_y: Optional[int],
        orig_bgr: Optional[np.ndarray] = None,
    ) -> FooterResult:
        """
        gray       : preprocessed grayscale page (BreakerTableAnalyzer._prep)
        header_y   : header rule y in gray coords
        centers    : row centers (unused directly, but kept for future)
        dbg        : header debug (_Dbg) with .lines (row borders)
        ocr_dbg_items : [{'text','x1','y1','x2','y2'}, ...] including CCT/CKT
        footer_struct_y : structural footer line (old method) used ONLY as fallback
        orig_bgr   : ORIGINAL page image; if None, synthesize from gray
        """
        H, W = gray.shape
        dbg_marks: List[Tuple[int, str]] = []

        # ------------------------------------------------------------------
        # choose orig_bgr + scaling between orig and gray
        # ------------------------------------------------------------------
        if orig_bgr is None:
            # fallback: treat gray as the source
            orig_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        Ho, Wo = orig_bgr.shape[:2]
        Hg, Wg = H, W

        scale_y = Hg / float(Ho)
        scale_x = Wg / float(Wo)

        # ------------------------------------------------------------------
        # locate FIRST BREAKER LINE (same as script: first dbg.lines entry)
        # ------------------------------------------------------------------
        if dbg.lines:
            first_breaker_line = int(dbg.lines[0])
        else:
            first_breaker_line = None

        # ==================================================================
        # 1) VERTICAL LINE MASK + VERTICAL POSITIONS
        # ==================================================================
        vmask_path = None
        if self.debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            vmask_path = os.path.join(self.debug_dir, "vertical_mask.png")

        # IMPORTANT: header_y -> PAGE BOTTOM; NO dependency on footer_struct_y
        v_xs = self._find_vertical_lines(
            gray,
            header_y,
            H - 1,
            mask_out_path=vmask_path,
        )

        # ==================================================================
        # 2) FIND CCT/CKT TOKENS IN HEADER
        # ==================================================================
        def _norm_token(s: str) -> str:
            return re.sub(
                r"\s+",
                " ",
                re.sub(r"[^A-Z0-9/\[\] \-]", "", (s or "").upper()),
            ).strip()

        col_spans_gray: set[Tuple[int, int]] = set()

        for item in ocr_dbg_items:
            txt = item.get("text", "")
            norm = _norm_token(txt)
            if norm not in ("CCT", "CKT"):
                continue

            x1 = int(item["x1"])
            x2 = int(item["x2"])

            # nearest vertical left / right of token
            left_candidates = [vx for vx in v_xs if vx <= x1]
            right_candidates = [vx for vx in v_xs if vx >= x2]
            if not left_candidates or not right_candidates:
                continue
            xl = max(left_candidates)
            xr = min(right_candidates)
            if xr <= xl:
                continue

            col_spans_gray.add((xl, xr))

        col_spans_sorted: List[Tuple[int, int]] = sorted(col_spans_gray, key=lambda p: p[0])

        if not col_spans_sorted or first_breaker_line is None:
            # cannot use new method -> bail to struct footer
            return self._fallback_footer(
                footer_struct_y=footer_struct_y,
                dbg_marks=dbg_marks,
                vlines_x=v_xs,
                cct_cols=col_spans_sorted,
            )

        # ==================================================================
        # 3) VERTICAL LIMITS FROM FIRST BREAKER DOWN + BOTTOM TRIM
        # ==================================================================
        y_top_g = int(first_breaker_line)
        y_bot_g = Hg

        # map vertical gray → orig
        y_top_o = max(0, min(Ho - 1, int(round(y_top_g / scale_y))))
        y_bot_o = max(y_top_o + 1, min(Ho, int(round(y_bot_g / scale_y))))

        # bottom trim (global)
        if self.bottom_trim_frac > 0.0:
            span_h = y_bot_o - y_top_o
            cut = int(span_h * self.bottom_trim_frac)
            if cut > 0:
                y_bot_o = max(y_top_o + 1, y_bot_o - cut)

        # ==================================================================
        # 4) CROP FROM SOURCE, DEGRID, OCR EACH COLUMN
        # ==================================================================
        footer_candidates = []

        for idx, (xl_g, xr_g) in enumerate(col_spans_sorted, start=1):
            # map horizontal gray -> orig
            x_left_o = max(0, min(Wo - 1, int(round((xl_g - 1) / scale_x))))
            x_right_o = max(x_left_o + 1, min(Wo, int(round((xr_g + 1) / scale_x))))

            src_crop = orig_bgr[y_top_o:y_bot_o, x_left_o:x_right_o]
            if src_crop.size == 0:
                continue

            # degrid in the crop (returns grayscale)
            clean_gray = self._degrid_crop(src_crop)
            Hc, Wc = clean_gray.shape

            # trim top of each crop
            cut_top = 0
            if 0.0 < self.top_trim_frac < 1.0:
                cut_top = int(Hc * self.top_trim_frac)
                if cut_top > 0 and cut_top < Hc:
                    clean_gray = clean_gray[cut_top:, :]
                    Hc, Wc = clean_gray.shape
                else:
                    cut_top = 0

            # optional upscaling
            eff_scale = float(self.upscale_factor) if self.upscale_factor > 0 else 1.0
            if eff_scale != 1.0:
                clean_gray = cv2.resize(
                    clean_gray,
                    None,
                    fx=eff_scale,
                    fy=eff_scale,
                    interpolation=cv2.INTER_CUBIC,
                )

            overlay_col = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)

            # OCR on processed crop
            dets = []
            if self.reader is not None:
                try:
                    dets = self.reader.readtext(
                        clean_gray,
                        detail=1,
                        paragraph=False,
                    )
                except Exception:
                    dets = []

            # draw OCR boxes + collect footer tokens
            for box, txt, conf in dets:
                if not txt:
                    continue

                pts = np.array(box, dtype=np.int32)
                cv2.polylines(overlay_col, [pts], True, (0, 0, 255), 1)

                x0 = int(min(p[0] for p in box))
                y0 = int(min(p[1] for p in box))
                cv2.putText(
                    overlay_col,
                    str(txt),
                    (x0, max(10, y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # footer token detection
                m = re.search(r"\d+", str(txt))
                if not m:
                    continue
                try:
                    val = int(m.group(0))
                except ValueError:
                    continue
                if val not in self.FOOTER_TOKEN_VALUES:
                    continue

                # map OCR box -> gray page coords
                box_gray = []
                for bx, by in box:
                    by_unscaled = by / eff_scale
                    bx_unscaled = bx / eff_scale

                    by_in_src_crop = cut_top + by_unscaled
                    bx_in_src_crop = bx_unscaled

                    y_orig = y_top_o + by_in_src_crop
                    x_orig = x_left_o + bx_in_src_crop

                    y_gray = int(round(y_orig * scale_y))
                    x_gray = int(round(x_orig * scale_x))
                    box_gray.append((x_gray, y_gray))

                if not box_gray:
                    continue

                y_bottom_gray = max(p[1] for p in box_gray)

                footer_candidates.append(
                    {
                        "val": val,
                        "text": str(txt),
                        "y_bottom": y_bottom_gray,
                        "box_gray": box_gray,
                        "col_idx": idx,
                    }
                )

            # save the crop overlay if debugging
            if self.debug and self.debug_dir:
                col_path = os.path.join(self.debug_dir, f"footer_col{idx:02d}_ocr.png")
                cv2.imwrite(col_path, overlay_col)

        # ==================================================================
        # 5) PICK BEST FOOTER TOKEN & SNAP TO HORIZONTAL STRUCTURE
        # ==================================================================
        if not footer_candidates:
            return self._fallback_footer(
                footer_struct_y=footer_struct_y,
                dbg_marks=dbg_marks,
                vlines_x=v_xs,
                cct_cols=col_spans_sorted,
            )

        best = max(footer_candidates, key=lambda d: (d["val"], d["y_bottom"]))
        token_val = int(best["val"])
        token_y = int(best["y_bottom"])

        dbg_marks.append((token_y, f"FOOTER_TOKEN_{token_val}"))

        # snap to structural line near token_y (within this page band)
        band_top = max(0, token_y - 80)
        band_bot = min(Hg, token_y + 20)
        band = gray[band_top:band_bot, :]
        if band.size == 0:
            return FooterResult(
                footer_y=token_y,
                token_y=token_y,
                token_val=token_val,
                dbg_marks=dbg_marks,
                vlines_x=v_xs,
                cct_cols=col_spans_sorted,
            )

        blur = cv2.GaussianBlur(band, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )
        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(40, int(0.12 * Wg)), 1),
        )
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)
        proj = h_candidates.sum(axis=1)

        footer_y = token_y
        if proj.size > 0 and proj.max() > 0:
            rel_y = int(np.argmax(proj))
            footer_y = int(band_top + rel_y)
            dbg_marks.append((footer_y, "FOOTER_LINE_STRUCT"))

        return FooterResult(
            footer_y=footer_y,
            token_y=token_y,
            token_val=token_val,
            dbg_marks=dbg_marks,
            vlines_x=v_xs,
            cct_cols=col_spans_sorted,
        )

    # ======================================================================
    # HELPERS
    # ======================================================================
    def _fallback_footer(
        self,
        footer_struct_y: Optional[int],
        dbg_marks: List[Tuple[int, str]],
        vlines_x: Optional[List[int]] = None,
        cct_cols: Optional[List[Tuple[int, int]]] = None,
    ) -> FooterResult:
        """Return structural footer as fallback."""
        if footer_struct_y is not None:
            dbg_marks.append((int(footer_struct_y), "FOOTER_STRUCT_ONLY"))
        return FooterResult(
            footer_y=int(footer_struct_y) if footer_struct_y is not None else None,
            token_y=None,
            token_val=None,
            dbg_marks=dbg_marks,
            vlines_x=vlines_x or [],
            cct_cols=cct_cols or [],
        )

    def _find_vertical_lines(
        self,
        gray: np.ndarray,
        header_y: Optional[int],
        footer_y: Optional[int],
        mask_out_path: Optional[str] = None,
    ) -> List[int]:
        """Exact same vertical-line logic you had in DevEnvFindHeader6."""
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

        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        roi_h = roi.shape[0]
        Kv = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(25, int(0.015 * roi_h))),
        )
        v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)

        # optional debug mask
        if mask_out_path is not None:
            try:
                os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
                cv2.imwrite(mask_out_path, v_candidates)
            except Exception:
                pass

        v_x_centers: List[int] = []
        num_v, labels_v, stats_v, _ = cv2.connectedComponentsWithStats(
            v_candidates,
            connectivity=8,
        )
        if num_v > 1:
            min_vert_len = int(0.40 * roi_h)
            max_vert_thick = max(2, int(0.02 * W))
            for i in range(1, num_v):
                x, y, w, h, area = stats_v[i]
                if h >= min_vert_len and w <= max_vert_thick and area > 0:
                    x_center = x + w // 2
                    v_x_centers.append(int(x_center))

        if not v_x_centers:
            return []

        xs = sorted(v_x_centers)
        collapsed: List[int] = []
        for x in xs:
            if not collapsed or abs(x - collapsed[-1]) > 4:
                collapsed.append(x)
        return collapsed

    def _degrid_crop(self, src_bgr: np.ndarray) -> np.ndarray:
        """Same degridding logic as _degrid_crop in DevEnvFindHeader6."""
        if src_bgr.size == 0:
            return src_bgr

        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        H, W = bw.shape

        Kv = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(18, int(0.12 * H))),
        )
        v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)

        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(25, int(0.35 * W)), 1),
        )
        h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

        grid = cv2.bitwise_or(v, h)
        mask = cv2.dilate(
            grid,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        clean = cv2.inpaint(gray, mask, 2, cv2.INPAINT_TELEA)
        return clean
