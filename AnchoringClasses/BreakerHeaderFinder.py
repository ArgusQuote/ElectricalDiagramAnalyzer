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
    header_bottom_y: Optional[int]
    footer_struct_y: Optional[int]
    last_footer_y: Optional[int]
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

        self.footer_snapped: bool = False
        self.snap_note: Optional[str] = None

        self.header_text_line_y: Optional[int] = None   # blue text line
        self.header_y_abs: Optional[int] = None         # snapped structural line
        self.header_bottom_y_abs: Optional[int] = None

        self.header_text_line_y: Optional[int] = None

        # snapping debug
        self.snap_steps_up: Optional[int] = None
        self.last_horizontal_mask_path: Optional[str] = None

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
        self.footer_snapped = False
        self.snap_note = None
        self.header_text_line_y = None
        self.header_text_line_y = None
        self.header_y_abs = None
        self.snap_steps_up = None
        self.last_horizontal_mask_path = None

        header_y = self._find_header_by_tokens(gray)

        dbg = HeaderDbg(lines=[], centers=[])
        centers: List[int] = []

        return HeaderResult(
            centers=centers,
            dbg=dbg,
            header_y=header_y,
            header_bottom_y=self.header_bottom_y_abs,
            footer_struct_y=self.footer_struct_y,
            last_footer_y=self.last_footer_y,
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
          - Blue   : header anchor token + HEADER_TEXT_LINE (text baseline)
          - Cyan   : HEADER_Y (snapped structural line above)
          - Orange : HEADER_BOTTOM_Y (nearest horizontal line below)
          - Magenta: other header tokens
          - Red    : excluded tokens
        """
        # always return a BGR image
        if base_img.ndim == 2:
            overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            overlay = base_img.copy()

        if not self.debug:
            return overlay

        H, W = overlay.shape[:2]

        # ROI rectangles (blue box around scanned header band)
        for (x1, y1, x2, y2) in self.ocr_dbg_rois:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # header / anchor / excluded tokens only
        for item in self.ocr_dbg_items:
            is_anchor = item.get("is_header_anchor", False)
            is_header = item.get("is_header_token", False)
            is_excl   = item.get("is_excluded_token", False)

            # skip non-header junk entirely
            if not (is_anchor or is_header or is_excl):
                continue

            x1, y1 = item["x1"], item["y1"]
            x2, y2 = item["x2"], item["y2"]
            txt = item.get("text", "")

            if is_anchor:
                box_color  = (255, 0, 0)    # blue
                text_color = (255, 0, 0)
            elif is_header:
                box_color  = (255, 0, 255)  # magenta
                text_color = (255, 0, 255)
            else:  # excluded
                box_color  = (0, 0, 255)    # red
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

        # HEADER_TEXT_LINE (blue) at text baseline
        if self.header_text_line_y is not None:
            y = int(self.header_text_line_y)
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

        # HEADER_Y snapped structural line (cyan) + numeric value
        if self.header_y_abs is not None:
            y = int(self.header_y_abs)
            cv2.line(overlay, (0, y), (W - 1, y), (0, 255, 255), 2)
            label = f"HEADER_Y={y}"
            cv2.putText(
                overlay,
                label,
                (10, min(H - 10, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # HEADER_BOTTOM_Y (nearest horizontal line below, orange-ish)
        if getattr(self, "header_bottom_y_abs", None) is not None:
            yb = int(self.header_bottom_y_abs)
            cv2.line(overlay, (0, yb), (W - 1, yb), (0, 165, 255), 2)
            label = f"HEADER_BOTTOM_Y={yb}"
            cv2.putText(
                overlay,
                label,
                (10, min(H - 10, yb + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

        # how many rows we moved upward during snapping
        if self.snap_steps_up is not None:
            txt = f"snap_up_steps={self.snap_steps_up}"
            cv2.putText(
                overlay,
                txt,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return overlay

    # ---------- helpers ----------

    @staticmethod
    def _normalize_text(s: str) -> str:
        """Strip non-alphanumeric chars, uppercase, and fix digit confusions (1->I, 0->O) for header token matching."""
        return re.sub(
            r"[^A-Z0-9]",
            "",
            (s or "").upper().replace("1", "I").replace("0", "O"),
        )

    def _run_ocr(self, img, mag: float):
        """Run EasyOCR readtext on *img* with the given magnification ratio and a restricted alphanumeric allowlist."""
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

    def _snap_header_to_horizontal_line(
        self,
        gray: np.ndarray,
        y1_band: int,
        y2_band: int,
        header_text_y: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Build a horizontal-line mask on the band [y1_band:y2_band, :]
        and find the closest white line ABOVE and BELOW header_text_y.

        Returns:
            (header_y_above, header_bottom_y_below)  as absolute y coords.
        """
        H, W = gray.shape
        y1_band = max(0, min(H, y1_band))
        y2_band = max(y1_band + 1, min(H, y2_band))

        band = gray[y1_band:y2_band, :]
        if band.size == 0:
            self.snap_steps_up = None
            return None, None

        # --- build horizontal-line mask ---
        blur = cv2.GaussianBlur(band, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        # ------------------------------------------------------------
        # Center-only gap bridge (ONLY inside middle band of the page)
        # ------------------------------------------------------------
        center_lo = int(W * 0.35)
        center_hi = int(W * 0.65)

        gap_bridge = int(max(15, min(90, W * 0.04)))  # ~4% width, clamped
        Kgap = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_bridge, 1))

        bw2 = bw.copy()
        center = bw[:, center_lo:center_hi]
        center_closed = cv2.morphologyEx(center, cv2.MORPH_CLOSE, Kgap, iterations=1)
        bw2[:, center_lo:center_hi] = center_closed
        bw = bw2

        # emphasize HORIZONTAL structures: wide kernel, height=1
        klen_target = int(W * 0.70)          # ~70% of page width
        klen = int(min(W - 2, max(70, klen_target)))  # clamp to [70, W-2]
        K = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, K, iterations=1)

        # optional debug: save the mask so you can *see* it
        self.last_horizontal_mask_path = None
        if self.debug and self.debug_dir:
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
                mask_name = f"header_horiz_mask_y{y1_band}_{y2_band}.png"
                mask_path = os.path.join(self.debug_dir, mask_name)
                cv2.imwrite(mask_path, horiz)
                self.last_horizontal_mask_path = mask_path
                print(f"[BreakerHeaderFinder] Saved horizontal mask: {mask_path}")
            except Exception as e:
                print(f"[BreakerHeaderFinder] Failed to save horiz mask: {e}")

        band_h = horiz.shape[0]

        # convert absolute header_text_y to band-relative index
        y_rel_start = header_text_y - y1_band
        if y_rel_start < 0 or y_rel_start >= band_h:
            self.snap_steps_up = None
            return None, None

        # helper: does a small window around y_idx contain any white pixels?
        def has_white_at(y_idx: int) -> bool:
            y0 = max(0, y_idx - 2)
            y1 = min(band_h, y_idx + 3)
            slice_ = horiz[y0:y1, :]
            return bool(slice_.any())

        max_up = 250   # max pixels to search up
        max_down = 250 # max pixels to search down

        # ----- search UPWARDS for nearest white line -----
        steps_up = 0
        best_up_rel = None
        cur = int(y_rel_start)
        while cur >= 0 and steps_up <= max_up:
            if has_white_at(cur):
                best_up_rel = cur
                break
            cur -= 1          # *** MOVE WINDOW UP ***
            steps_up += 1

        self.snap_steps_up = steps_up if best_up_rel is not None else None

        # ----- search DOWNWARDS for nearest white line -----
        steps_down = 0
        best_down_rel = None
        cur = int(y_rel_start)
        while cur < band_h and steps_down <= max_down:
            if has_white_at(cur):
                best_down_rel = cur
                break
            cur += 1          # move window down
            steps_down += 1

        header_y_above = y1_band + best_up_rel if best_up_rel is not None else None
        header_bottom_y_below = (
            y1_band + best_down_rel if best_down_rel is not None else None
        )

        # --- debug: save horizontal mask with snap lines drawn on it ---
        if self.debug and self.debug_dir is not None:
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
                # rebuild a BGR version of the horiz mask to draw on
                vis = cv2.cvtColor(horiz, cv2.COLOR_GRAY2BGR)
                Hb, Wb = vis.shape[:2]

                # header_text_y in band-relative coords
                y_rel = int(header_text_y - y1_band)
                y_rel = max(0, min(Hb - 1, y_rel))

                # draw header_text baseline (blue)
                cv2.line(vis, (0, y_rel), (Wb - 1, y_rel), (255, 0, 0), 1)

                # draw snapped HEADER_Y (cyan)
                if best_up_rel is not None:
                    cv2.line(
                        vis,
                        (0, int(best_up_rel)),
                        (Wb - 1, int(best_up_rel)),
                        (0, 255, 255),
                        1,
                    )

                # draw snapped HEADER_BOTTOM_Y (orange)
                if best_down_rel is not None:
                    cv2.line(
                        vis,
                        (0, int(best_down_rel)),
                        (Wb - 1, int(best_down_rel)),
                        (0, 165, 255),
                        1,
                    )

                dbg_name = f"header_horiz_mask_overlay_y{y1_band}_{y2_band}.png"
                dbg_path = os.path.join(self.debug_dir, dbg_name)
                cv2.imwrite(dbg_path, vis)
                print(f"[BreakerHeaderFinder] Saved horiz mask overlay: {dbg_path}")
            except Exception as e:
                print(f"[BreakerHeaderFinder] Failed to save horiz mask overlay: {e}")

        return (
            int(header_y_above) if header_y_above is not None else None,
            int(header_bottom_y_below) if header_bottom_y_below is not None else None,
        )

    # ---------- header finder ----------

    def _find_header_by_tokens(self, gray: np.ndarray) -> Optional[int]:
        """
        Use OCR header tokens to find the header row.

        - Crop to top 50% of page.
        - Remove top 15% of that half  (~7.5%H..50%H).
        - OCR on that band.
        - Group tokens by y-bin.
        - Pick the line with the MOST *header* tokens.
        - Anchor is the left-most header token on that line.
        - header_text_line_y = anchor top y (used for blue overlay).
        - header_y_abs      = nearest horizontal line ABOVE.
        - header_bottom_y_abs = nearest horizontal line BELOW.
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
            self.header_y_abs = None
            self.header_bottom_y_abs = None
            self.snap_steps_up = None
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
        self.snap_steps_up = None
        self.header_text_line_y = None
        self.header_y_abs = None
        self.header_bottom_y_abs = None

        # OCR passes on the band
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
            return None

        # ---- group by y-bin ----
        lines: dict[int, list[dict]] = {}
        for it in items:
            yc = 0.5 * (it["y1"] + it["y2"])
            ybin = int(yc // 14) * 14
            lines.setdefault(ybin, []).append(it)

        # ---- choose line with MOST HEADER TOKENS ----
        # ---- choose line with BEST HEADER SCORE (weighted by category) ----
        best_ybin: Optional[int] = None
        best_score = -1.0

        # weights: CKT & DESCRIPTION are strongest; POLES strong; TRIP/AMPS weakest
        cat_weights = {
            "ckt": 4.0,
            "description": 4.0,
            "poles": 3.0,
            "trip": 1.0,
        }

        for ybin, line_items in lines.items():
            # track how many of each category we see on this line
            cat_counts = {k: 0 for k in self.CATEGORY_ALIASES.keys()}
            cats_present = set()

            for it in line_items:
                norm = it["norm"]

                # --- strict for CKT-like tokens (must be exact & digit-free) ---
                if (
                    norm in self.CATEGORY_ALIASES["ckt"]
                    and not any(ch.isdigit() for ch in norm)
                ):
                    cat_counts["ckt"] += 1
                    cats_present.add("ckt")

                # --- other categories still use substring match ---
                for cat, aliases in self.CATEGORY_ALIASES.items():
                    if cat == "ckt":
                        continue
                    if any(alias in norm for alias in aliases):
                        cat_counts[cat] += 1
                        cats_present.add(cat)

            if not cats_present:
                continue  # no header-ish tokens at all on this line

            # score = big bonus for multiple distinct categories
            #       + weighted count per category
            distinct_bonus = 5.0 * len(cats_present)
            token_score = 0.0
            for cat in cats_present:
                w = cat_weights.get(cat, 1.0)
                token_score += w * cat_counts[cat]

            score = distinct_bonus + token_score

            if (
                score > best_score
                or (score == best_score and (best_ybin is None or ybin < best_ybin))
            ):
                best_score = score
                best_ybin = ybin

        if best_ybin is None:
            # no header-like tokens anywhere in band
            return None

        # tokens on the winning line, using the same category logic
        line_items = lines[best_ybin]
        hdr_items: List[dict] = []

        for it in line_items:
            norm = it["norm"]
            is_header = False

            # strict CKT / CCT
            if (
                norm in self.CATEGORY_ALIASES["ckt"]
                and not any(ch.isdigit() for ch in norm)
            ):
                is_header = True
            else:
                # other categories
                for cat, aliases in self.CATEGORY_ALIASES.items():
                    if cat == "ckt":
                        continue
                    if any(alias in norm for alias in aliases):
                        is_header = True
                        break

            if is_header:
                hdr_items.append(it)

        if not hdr_items:
            return None

        # left-most header token is anchor
        hdr_items_sorted = sorted(hdr_items, key=lambda it: it["x1"])
        anchor = hdr_items_sorted[0]
        anchor["is_header_anchor"] = True

        # mark all header/excluded tokens for overlay using same rules
        for it in items:
            norm = it["norm"]
            is_header = False

            # strict CKT / CCT
            if (
                norm in self.CATEGORY_ALIASES["ckt"]
                and not any(ch.isdigit() for ch in norm)
            ):
                is_header = True
            else:
                # other categories
                for cat, aliases in self.CATEGORY_ALIASES.items():
                    if cat == "ckt":
                        continue
                    if any(alias in norm for alias in aliases):
                        is_header = True
                        break

            if is_header:
                it["is_header_token"] = True

            if any(excl in norm for excl in self.EXCLUDE):
                it["is_excluded_token"] = True

        # header text line = average center of all header tokens on this line
        centers = [
            0.5 * (it["y1"] + it["y2"])
            for it in hdr_items
        ]
        header_text_y = int(sum(centers) / len(centers))
        self.header_text_line_y = header_text_y

        # ---- FIND NEAREST LINES ABOVE & BELOW ON HORIZONTAL MASK ----
        header_y_above, header_bottom_y = self._snap_header_to_horizontal_line(
            gray=gray,
            y1_band=y1_band,
            y2_band=y2_band,
            header_text_y=header_text_y,
        )

        # ------------------------------------------------------------------
        # Decide HEADER TOP (self.header_y_abs)
        #   1) Prefer snapped line above if present.
        #   2) If NO top line but bottom line exists below header_text_y,
        #      synthesize a top line at the same distance above as bottom is below:
        #         dy = header_bottom_y - header_text_y
        #         header_top = header_text_y - dy
        #   3) If we have neither, fall back to header_text_y.
        # ------------------------------------------------------------------
        header_top: int

        if header_y_above is not None:
            # normal case: use real structural line above
            header_top = int(header_y_above)
            self.snap_note = self.snap_note or "header_top_from_snap_above"
        else:
            # no top line; can we mirror from a good bottom line?
            if header_bottom_y is not None and header_bottom_y > header_text_y:
                dy = header_bottom_y - header_text_y
                mirrored_top = header_text_y - dy

                # clamp to page
                if mirrored_top < 0:
                    mirrored_top = 0

                header_top = int(mirrored_top)
                self.snap_note = self.snap_note or "header_top_symmetric_from_bottom"
            else:
                # nothing to snap to – just use the text line
                header_top = int(header_text_y)
                self.snap_note = self.snap_note or "header_top_from_text_line"

        self.header_y_abs = header_top

        # ------------------------------------------------------------------
        # Decide HEADER BOTTOM (self.header_bottom_y_abs)
        #   - Prefer snapped line below if it's meaningfully below top.
        #   - Otherwise synthesize a minimum-height band under header_top.
        # ------------------------------------------------------------------
        min_band_height = max(40, int(0.03 * H))  # ~3% of page height, at least 40 px

        if (
            header_bottom_y is not None
            and header_bottom_y > self.header_y_abs + 4
        ):
            # good structural line below header
            self.header_bottom_y_abs = int(header_bottom_y)
        else:
            # no reliable line below → synthesize one a bit under the header
            self.header_bottom_y_abs = int(
                min(H - 1, self.header_y_abs + min_band_height)
            )

        return self.header_y_abs
 