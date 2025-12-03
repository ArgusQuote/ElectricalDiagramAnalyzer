# AnchoringClasses/BreakerFooterFinder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import re


@dataclass
class FooterResult:
    footer_y: Optional[int]
    token_y: Optional[int]
    token_val: Optional[str]
    dbg_marks: List[Tuple[int, str]]
    vlines_x: List[int]
    cct_cols: List[Tuple[int, int]]


class BreakerFooterFinder:
    """
    CLEAN FOOTER FINDER — ONLY the logic Marco requested.

    Steps:
      1) Use ALREADY FOUND CCT/CKT tokens (from header_finder.ocr_dbg_items)
         → Determine left/right vertical boundaries.

      2) Crop columns bound by those vertical lines.

      3) Remove top 50% of the crop and a small bottom trim.

      4) OCR inside this region for numeric tokens.

      5) Largest numeric token (max width) on each side = footer text lines.

      6) From the footer token Y, find nearest long horizontal white-pixel run.
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
        self.debug = debug
        self.bottom_trim_frac = bottom_trim_frac
        self.top_trim_frac = top_trim_frac
        self.upscale_factor = upscale_factor
        self.debug_dir = None

    # ----------------------------------------------------------------------
    # MAIN ENTRY
    # ----------------------------------------------------------------------
    def find_footer(
        self,
        gray: np.ndarray,
        header_y: Optional[int],
        centers: List[int],
        dbg,
        ocr_dbg_items: List[dict],
        orig_bgr: np.ndarray,
    ) -> FooterResult:

        dbg_marks = []

        # ---------------------------------------------------------------
        # STEP 1 — Extract CCT/CKT token boxes
        # ---------------------------------------------------------------
        cct_boxes = []
        for it in ocr_dbg_items:
            t = (it.get("text") or "").upper()
            t = re.sub(r"[^A-Z]", "", t)
            if t in ("CCT", "CKT"):
                cct_boxes.append((it["x1"], it["y1"], it["x2"], it["y2"]))

        if len(cct_boxes) < 1:
            return FooterResult(None, None, None, [], [], [])

        # bounding column extents
        xs = []
        for (x1, y1, x2, y2) in cct_boxes:
            xs.append(x1)
            xs.append(x2)

        xs = sorted(xs)
        xl = xs[0]
        xr = xs[-1]

        vlines_x = [xl, xr]
        cct_cols = [(xl, xr)]

        # ---------------------------------------------------------------
        # STEP 2 — Crop these columns from ORIGINAL BGR
        # ---------------------------------------------------------------
        H, W = gray.shape
        x1 = max(0, xl - 3)
        x2 = min(W - 1, xr + 3)
        col_crop = orig_bgr[:, x1:x2]

        # ---------------------------------------------------------------
        # STEP 3 — cut top 50% and bottom fraction
        # ---------------------------------------------------------------
        h2 = col_crop.shape[0]
        top_cut = int(h2 * self.top_trim_frac)
        bot_cut = int(h2 * (1.0 - self.bottom_trim_frac))

        crop2 = col_crop[top_cut:bot_cut, :]
        crop2_gray = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

        # OPTIONAL DEBUG VIEW
        if self.debug and self.debug_dir:
            cv2.imwrite(f"{self.debug_dir}/footer_crop_columns.png", col_crop)
            cv2.imwrite(f"{self.debug_dir}/footer_crop_final.png", crop2_gray)

        # ---------------------------------------------------------------
        # STEP 4 — OCR for numeric tokens only
        # ---------------------------------------------------------------
        nums = []
        try:
            det = self.reader.readtext(
                crop2_gray,
                detail=1,
                paragraph=False,
                allowlist="0123456789.",
                mag_ratio=self.upscale_factor,
            )
        except Exception:
            det = []

        for box, txt, conf in det:
            if not txt:
                continue
            t = re.sub(r"[^0-9]", "", txt)
            if not t:
                continue

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]

            x1b = min(xs)
            x2b = max(xs)
            y1b = min(ys)
            y2b = max(ys)

            width = x2b - x1b
            nums.append({
                "val": t,
                "x1": x1b,
                "x2": x2b,
                "y1": y1b,
                "y2": y2b,
                "w": width,
            })

        if len(nums) == 0:
            return FooterResult(None, None, None, dbg_marks, vlines_x, cct_cols)

        # ---------------------------------------------------------------
        # STEP 5 — largest numeric token = footer text line
        # ---------------------------------------------------------------
        nums_sorted = sorted(nums, key=lambda x: x["w"], reverse=True)
        best = nums_sorted[0]

        token_val = best["val"]
        token_y = best["y1"] + top_cut

        dbg_marks.append((token_y, f"FOOTER_TOKEN({token_val})"))

        # ---------------------------------------------------------------
        # STEP 6 — find horizontal rule near token_y
        # ---------------------------------------------------------------
        window = orig_bgr[max(0, token_y - 40):min(H, token_y + 40), :]
        if window.size == 0:
            return FooterResult(None, token_y, token_val, dbg_marks, vlines_x, cct_cols)

        win_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

        bw = cv2.adaptiveThreshold(
            win_gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            8,
        )

        proj = bw.sum(axis=1)
        thr = 0.45 * float(proj.max() or 1)

        ys = np.where(proj >= thr)[0]
        if len(ys) == 0:
            footer_y = token_y
        else:
            footer_y = int(ys[len(ys)//2] + (token_y - 40))
            dbg_marks.append((footer_y, "FOOTER_LINE"))

        return FooterResult(
            footer_y=footer_y,
            token_y=token_y,
            token_val=token_val,
            dbg_marks=dbg_marks,
            vlines_x=vlines_x,
            cct_cols=cct_cols,
        )
