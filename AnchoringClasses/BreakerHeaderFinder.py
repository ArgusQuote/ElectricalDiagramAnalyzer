# AnchoringClasses/BreakerHeaderFinder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import re
import cv2
import numpy as np


@dataclass
class HeaderDbg:
    lines: List[int]
    centers: List[int]


# alias so other code can import `Dbg` if it wants
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
    Header + row-structure + structural footer.
    This is the logic that used to live in BreakerTableAnalyzer6:
      - _find_cct_header_y
      - _find_header_by_tokens
      - _row_centers_from_lines
    """

    def __init__(self, reader, debug: bool = False):
        self.reader = reader
        self.debug = debug

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

    # ---------- public ----------

    def analyze_rows(self, gray: np.ndarray) -> HeaderResult:
        centers, dbg, header_y = self._row_centers_from_lines(gray)

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

    # ---------- internals (migrated from BreakerTableAnalyzer6) ----------

    def _find_cct_header_y(self, gray: np.ndarray) -> Optional[int]:
        if self.reader is None:
            return None
        self.ocr_dbg_items, self.ocr_dbg_rois = [], []
        H, W = gray.shape

        def norm(s: str) -> str:
            return re.sub(r"[^A-Z]", "", s.upper().replace("1", "I"))

        def has_long_line_below(y_abs: int) -> bool:
            y1 = max(0, y_abs - 6)
            y2 = min(H, y_abs + 28)
            band = gray[y1:y2, :]
            if band.size == 0:
                return False
            g = cv2.GaussianBlur(band, (3, 3), 0)
            bw = cv2.adaptiveThreshold(
                g, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                21, 10
            )
            klen = max(70, int(W * 0.22))
            K = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, K, iterations=1)
            return lines.sum() > 0

        def scan_roi(y1f, y2f, x2f) -> List[int]:
            y1, y2 = int(H * y1f), int(H * y2f)
            x1, x2 = 0, int(W * x2f)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                return []

            self.ocr_dbg_rois.append((x1, y1, x2, y2))

            def pass_once(mag: float):
                try:
                    return self.reader.readtext(
                        roi, detail=1, paragraph=False,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
                        mag_ratio=mag, contrast_ths=0.05, adjust_contrast=0.7,
                        text_threshold=0.4, low_text=0.25
                    )
                except Exception:
                    return []

            det = pass_once(1.4) + pass_once(1.9)

            for box, txt, conf in det:
                xs = [p[0] + x1 for p in box]
                ys = [p[1] + y1 for p in box]
                self.ocr_dbg_items.append({
                    "text": txt,
                    "conf": float(conf or 0.0),
                    "x1": int(min(xs)),
                    "y1": int(min(ys)),
                    "x2": int(max(xs)),
                    "y2": int(max(ys)),
                })

            lines = {}
            for box, txt, _ in det:
                if not txt:
                    continue
                yc = int(sum(p[1] for p in box) / 4)
                key = (yc // 14) * 14
                lines.setdefault(key, []).append((box, txt))

            cands: List[int] = []

            for box, txt, _ in det:
                t = norm(txt)
                if t in ("CCT", "CKT"):
                    y_abs = y1 + int(min(p[1] for p in box))
                    if has_long_line_below(y_abs):
                        cands.append(y_abs)

            for _, items in lines.items():
                raw = " ".join(t for _, t in items)
                if "HEADER" in raw.upper():
                    continue
                if ("CCT" in norm(raw)) or ("CKT" in norm(raw)):
                    y_abs = y1 + min(int(min(p[1] for p in b)) for b, _ in items)
                    if has_long_line_below(y_abs):
                        cands.append(y_abs)

            return sorted(set(int(y) for y in cands))

        all_hits: List[int] = []

        primary_windows = [
            (0.08, 0.48, 0.28),
            (0.10, 0.50, 0.32),
            (0.08, 0.52, 0.40),
            (0.12, 0.58, 0.45),
        ]
        for y1f, y2f, x2f in primary_windows:
            all_hits += scan_roi(y1f, y2f, x2f)

        fallback_candidates = [
            (0.08, 0.52, 0.65),
            (0.10, 0.56, 0.80),
            (0.08, 0.60, 1.00),
        ]
        for y1f, y2f, x2f in fallback_candidates:
            all_hits += scan_roi(y1f, y2f, x2f)

        x1, x2 = 0, int(0.48 * W)
        y1, y2 = int(0.08 * H), int(0.68 * H)
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0 and self.reader is not None:
            det = self.reader.readtext(
                roi, detail=1, paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -",
                mag_ratio=1.6, contrast_ths=0.05, adjust_contrast=0.7,
                text_threshold=0.4, low_text=0.25
            )
            cands = []
            for box, txt, _ in det:
                t = re.sub(
                    r"[^A-Z]",
                    "",
                    (txt or "").upper().replace("1", "I")
                )
                if t in ("CCT", "CKT"):
                    cands.append(y1 + int(min(p[1] for p in box)))
            all_hits += cands

        cap = int(0.65 * H)
        all_hits = [y for y in all_hits if y <= cap]

        if not all_hits:
            return None
        return int(min(all_hits))

    def _find_header_by_tokens(self, gray: np.ndarray) -> Optional[int]:
        if self.reader is None:
            return None

        H, W = gray.shape
        y1, y2 = int(0.08 * H), int(0.65 * H)
        x1, x2 = 0, W
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        def N(s: str) -> str:
            return re.sub(
                r"[^A-Z0-9]", "",
                (s or "").upper().replace("1", "I").replace("0", "O")
            )

        CATEGORY_ALIASES = {
            "ckt": {"CKT", "CCT"},
            "description": {
                "CIRCUITDESCRIPTION", "DESCRIPTION", "LOADDESCRIPTION",
                "DESIGNATION", "LOADDESIGNATION", "NAME"
            },
            "trip": {"TRIP", "AMPS", "AMP", "BREAKER", "BKR", "SIZE"},
            "poles": {"POLES", "POLE", "P"},
        }
        EXCLUDE = {"LOADCLASSIFICATION", "CLASSIFICATION"}

        def ocr(img, mag):
            try:
                return self.reader.readtext(
                    img, detail=1, paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/",
                    mag_ratio=mag, contrast_ths=0.05, adjust_contrast=0.7,
                    text_threshold=0.4, low_text=0.25
                )
            except Exception:
                return []

        det = ocr(roi, 1.6) + ocr(roi, 2.0)

        lines = {}
        for box, txt, _ in det:
            if not txt:
                continue
            y_abs = y1 + int(min(p[1] for p in box))
            ybin = (y_abs // 14) * 14
            lines.setdefault(ybin, []).append((box, txt))

        def has_long_rule_below(y_abs: int) -> bool:
            band_top = max(0, y_abs - 6)
            band_bot = min(H, y_abs + 28)
            band = gray[band_top:band_bot, :]
            if band.size == 0:
                return False
            g = cv2.GaussianBlur(band, (3, 3), 0)
            bw = cv2.adaptiveThreshold(
                g, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                21, 10
            )
            klen = max(70, int(W * 0.22))
            K = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            longlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, K, iterations=1)
            return longlines.sum() > 0

        best = None
        mid = W * 0.5

        for ybin, items in lines.items():
            categories_all = set()
            left_has = set()
            right_has = set()
            bad = False

            for box, txt in items:
                n = N(txt)
                if not n:
                    continue
                if any(ex in n for ex in EXCLUDE):
                    bad = True
                    break

                matched_cats = set()
                for cat, aliases in CATEGORY_ALIASES.items():
                    if "P" in aliases and n == "P":
                        matched_cats.add("poles")
                        continue
                    if any(alias in n for alias in aliases if alias != "P"):
                        matched_cats.add(cat)

                if matched_cats:
                    xs = [p[0] for p in box]
                    x_center = (min(xs) + max(xs)) / 2.0
                    categories_all |= matched_cats
                    if x_center < mid:
                        left_has |= matched_cats
                    else:
                        right_has |= matched_cats

            if bad or not categories_all:
                continue

            score = len(categories_all)
            if left_has and right_has:
                score += 1
            if len(categories_all) >= 2:
                score += 1

            y_abs = int(ybin)
            if score >= 3 and has_long_rule_below(y_abs):
                cand = (score, y_abs)
                if (
                    best is None
                    or (cand[0] > best[0])
                    or (cand[0] == best[0] and cand[1] < best[1])
                ):
                    best = cand

        if best is None:
            return None
        return int(best[1])

    def _row_centers_from_lines(self, gray: np.ndarray):
        H, W = gray.shape
        y_top = int(H * 0.12)
        y_bot = int(H * 0.95)
        roi = gray[y_top:y_bot, :]

        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21, 10
        )

        def borders_from(bw_src, k_frac, thr_frac):
            if bw_src.size == 0:
                return []
            klen = int(max(70, min(W * k_frac, 320)))
            K = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            lines = cv2.morphologyEx(bw_src, cv2.MORPH_OPEN, K, iterations=1)
            proj = lines.sum(axis=1)
            if proj.size == 0 or proj.max() == 0:
                return []
            need = max(80, int(0.18 * len(proj)))
            low = proj < (0.08 * proj.max())
            run = 0
            cut = None
            for i, is_low in enumerate(low):
                run = run + 1 if is_low else 0
                if run >= need:
                    cut = i - run + 1
                    break
            if cut is not None and cut > 10:
                proj = proj[:cut]
                if proj.size == 0 or proj.max() == 0:
                    return []
            thr = float(thr_frac * proj.max())
            ys = np.where(proj >= thr)[0].astype(int)
            if ys.size == 0:
                return []
            segs = []
            s = ys[0]
            p = ys[0]
            for y in ys[1:]:
                if y - p > 2:
                    segs.append((s, p))
                    s = y
                p = y
            segs.append((s, p))
            return [int((a + b) // 2) for a, b in segs]

        b1 = borders_from(bw, 0.35, 0.35)
        b2 = borders_from(bw, 0.25, 0.30)
        borders = b1 if len(b1) >= len(b2) else b2

        if not borders:
            edges = cv2.Canny(blur, 60, 160)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=140,
                minLineLength=int(W * 0.30),
                maxLineGap=6
            )
            if lines is None:
                self.last_footer_y = None
                self.footer_struct_y = None
                self.bottom_border_y = None
                self.bottom_row_center_y = None
                self.spaces_detected = 0
                self.spaces_corrected = 0
                self.footer_snapped = False
                self.snap_note = None
                return [], HeaderDbg(lines=[], centers=[]), None
            ys = sorted(
                int((y1 + y2) // 2)
                for x1, y1, x2, y2 in lines[:, 0]
                if abs(y2 - y1) <= 2
            )
            borders = [ys[0]]
            for y in ys[1:]:
                if y - borders[-1] > 3:
                    borders.append(y)

        header_y_abs = self._find_header_by_tokens(gray)
        if header_y_abs is None:
            header_y_abs = self._find_cct_header_y(gray)

        if header_y_abs is not None and header_y_abs > int(0.62 * H):
            header_y_abs = None

        header_loc = None if header_y_abs is None else max(0, header_y_abs - y_top)

        start_idx = 0
        if header_loc is not None:
            for i, b in enumerate(borders):
                if b > header_loc + 5:
                    start_idx = i
                    break
            if len(borders) - start_idx < 2:
                start_idx = 0
                header_y_abs = None
        else:
            gaps = np.diff(borders)
            if len(gaps) >= 6:
                for i in range(len(gaps) - 6):
                    w = gaps[i : i + 6]
                    gmed = float(np.median(w))
                    if gmed > 0 and np.all(
                        (w >= 0.7 * gmed) & (w <= 1.3 * gmed)
                    ):
                        start_idx = i
                        break
        tail = borders[start_idx:]

        if len(tail) >= 2:
            gaps = np.diff(tail)
            med = float(np.median(gaps)) if len(gaps) else 0
            stop_rel = None
            for j, g in enumerate(gaps):
                if med > 0 and g > max(1.9 * med, med + 18):
                    stop_rel = j
                    break
            if stop_rel is not None:
                tail = tail[: stop_rel + 1]

        if len(tail) >= 3:
            gaps = np.diff(tail)
            med = float(np.median(gaps)) if len(gaps) else 0.0
            for j, g in enumerate(gaps):
                if med > 0 and g >= max(1.6 * med, med + 14):
                    remaining_borders = len(tail) - (j + 1)
                    if remaining_borders <= 3:
                        tail = tail[: j + 1]
                    break

        if len(tail) >= 3:
            gaps = np.diff(tail)
            med_row_h = float(np.median(gaps)) if len(gaps) else 0.0
            if med_row_h > 0:
                xL = int(0.25 * W)
                xR = int(0.75 * W)
                central = bw[:, xL:xR]
                if central.size > 0:
                    central_denoised = cv2.morphologyEx(
                        central, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(
                            cv2.MORPH_RECT, (3, 1)
                        ),
                        1
                    )
                    central_denoised = cv2.morphologyEx(
                        central_denoised, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(
                            cv2.MORPH_RECT, (1, 3)
                        ),
                        1
                    )
                    row_ink = central_denoised.sum(axis=1).astype(
                        np.float32
                    ) / (255.0 * (xR - xL))
                    win = int(round(1.8 * med_row_h))
                    win = max(9, min(win, int(0.06 * H)))
                    if win % 2 == 0:
                        win += 1
                    smooth = np.convolve(
                        row_ink,
                        np.ones(win, dtype=np.float32) / float(win),
                        mode="same",
                    )
                    LOW_INK = 0.06
                    y_scan_start = int(
                        min(tail[0] + 2 * med_row_h, len(smooth) - 1)
                    )
                    below = np.where(smooth < LOW_INK)[0]
                    if below.size:
                        after = below[below >= y_scan_start]
                        if after.size:
                            cut_abs = y_top + int(after[0])
                            tail = [
                                b for b in tail
                                if (b + y_top) < cut_abs - 2
                            ]

        if len(tail) >= 3:
            gaps_all = np.diff(tail).astype(np.float32)
            if gaps_all.size:
                seed_n = int(min(10, len(gaps_all)))
                if seed_n > 0:
                    global_med = float(
                        np.median(gaps_all[:seed_n])
                    )
                else:
                    global_med = float(np.median(gaps_all))
                MIN_GAP = 12.0
                LO = 0.60
                HI = 1.45
                WIN = 5
                best_run_len = 0
                best_run_end = None
                i = 0
                while i < len(gaps_all):
                    if gaps_all[i] < MIN_GAP:
                        i += 1
                        continue
                    run_start = i
                    local_vals = []
                    local_med = (
                        global_med if global_med > 0 else gaps_all[i]
                    )
                    while i < len(gaps_all):
                        g = gaps_all[i]
                        local_vals.append(float(g))
                        if len(local_vals) > WIN:
                            local_vals.pop(0)
                        local_med = (
                            float(np.median(local_vals))
                            if local_vals
                            else local_med
                        )
                        if not (
                            (g >= MIN_GAP)
                            and (LO * local_med <= g <= HI * local_med)
                        ):
                            break
                        i += 1
                    run_end = i - 1
                    run_len = run_end - run_start + 1
                    if run_len > best_run_len:
                        best_run_len = run_len
                        best_run_end = run_end
                    i = max(i, run_end + 1)
                if best_run_len >= 2:
                    tail = tail[: int(best_run_end + 1) + 1]

        if len(tail) >= 3:
            gaps = np.diff(tail)
            med_row_h = float(np.median(gaps)) if len(gaps) else 0.0
            if med_row_h > 0:
                roi_h = bw.shape[0]
                Kv = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (1, max(18, int(0.05 * roi_h)))
                )
                vlines = cv2.morphologyEx(
                    bw, cv2.MORPH_OPEN, Kv, iterations=1
                )
                xL = int(0.22 * W)
                xR = int(0.78 * W)
                v_central = vlines[:, xL:xR] if xR > xL else vlines
                if v_central.size > 0:
                    v_row = v_central.sum(axis=1).astype(
                        np.float32
                    ) / (255.0 * (xR - xL))
                    win = int(round(1.2 * med_row_h))
                    win = max(7, min(win, int(0.05 * H)))
                    if win % 2 == 0:
                        win += 1
                    v_smooth = np.convolve(
                        v_row,
                        np.ones(win, dtype=np.float32) / float(win),
                        mode="same",
                    )
                    thr = max(0.02, 0.12 * float(v_smooth.max() or 1.0))
                    y_scan_start = int(
                        min(
                            tail[0] + 2 * med_row_h,
                            len(v_smooth) - 1
                        )
                    )
                    need = int(max(1.2 * med_row_h, 12))
                    run = 0
                    cut_rel = None
                    for y in range(y_scan_start, len(v_smooth)):
                        if v_smooth[y] < thr:
                            run += 1
                            if run >= need:
                                cut_rel = y - run + 1
                                break
                        else:
                            run = 0
                    if cut_rel is not None:
                        cut_abs = y_top + int(cut_rel)
                        tail = [
                            b for b in tail
                            if (b + y_top) < cut_abs - 2
                        ]

        borders_abs = [b + y_top for b in tail]
        if header_y_abs is not None and len(borders_abs) >= 2:
            gaps_abs = np.diff(borders_abs)
            med_gap = float(np.median(gaps_abs)) if len(
                gaps_abs
            ) else 0.0
            if med_gap > 0:
                d0 = float(borders_abs[0] - header_y_abs)
                if (
                    d0 > max(1.80 * med_gap, med_gap + 16)
                    and d0 < 3.2 * med_gap
                ):
                    y_expected = int(round(header_y_abs + med_gap))
                    if (
                        y_expected < borders_abs[0] - 8
                        and y_expected > header_y_abs + 6
                    ):
                        borders_abs.insert(0, y_expected)

        borders_rel = [y - y_top for y in borders_abs]
        if len(borders_rel) < 2:
            self.last_footer_y = borders_abs[-1] if borders_abs else None
            self.footer_struct_y = self.last_footer_y
            self.bottom_border_y = self.last_footer_y
            self.bottom_row_center_y = None
            self.spaces_detected = 0
            self.spaces_corrected = 0
            self.footer_snapped = False
            self.snap_note = None
            dbg = HeaderDbg(lines=[b + y_top for b in borders_rel], centers=[])
            return [], dbg, header_y_abs

        centers = []
        for i in range(len(borders_rel) - 1):
            gap = borders_rel[i + 1] - borders_rel[i]
            if gap >= 12:
                centers.append(
                    int((borders_rel[i] + borders_rel[i + 1]) // 2) + y_top
                )
        out = []
        for y in centers:
            if out and abs(y - out[-1]) < 8:
                continue
            out.append(y)

        footer_struct_y = borders_abs[-1] if len(borders_abs) >= 1 else None
        self.footer_struct_y = footer_struct_y

        gaps_abs = np.diff(borders_abs)
        med_row_h = float(np.median(gaps_abs)) if gaps_abs.size else 18.0
        last_center = out[-1] if out else None

        final_footer = footer_struct_y

        if (
            (final_footer is not None)
            and (header_y_abs is not None)
            and (final_footer <= header_y_abs + 6)
        ):
            final_footer = None

        detected_spaces = len(out) * 2
        self.spaces_detected = detected_spaces
        self.spaces_corrected = detected_spaces
        self.footer_snapped = False
        self.snap_note = None

        snap_map = {
            16: 18, 20: 18,
            28: 30, 32: 30,
            40: 42, 44: 42,
            52: 54, 56: 54,
            64: 66, 68: 66,
            70: 72, 74: 72,
            82: 84, 86: 84,
        }

        if (
            (final_footer is not None)
            and (detected_spaces in snap_map)
            and (med_row_h > 0)
        ):
            target_spaces = snap_map[detected_spaces]
            delta_rows = int((target_spaces - detected_spaces) // 2)
            if delta_rows != 0:
                shift = int(round(delta_rows * med_row_h))
                lo = int(
                    (header_y_abs + 8)
                    if header_y_abs is not None
                    else 0
                )
                hi = H - 5
                snapped = int(np.clip(final_footer + shift, lo, hi))
                if (
                    (last_center is not None)
                    and (snapped < last_center + 0.5 * med_row_h)
                ):
                    snapped = int(last_center + 0.5 * med_row_h)
                self.footer_snapped = True
                self.spaces_corrected = target_spaces
                self.snap_note = (
                    f"SNAP spaces {detected_spaces}→{target_spaces}; "
                    f"footer {final_footer}→{snapped}"
                )
                final_footer = snapped

        self.last_footer_y = int(final_footer) if final_footer is not None else None
        self.bottom_border_y = footer_struct_y
        self.bottom_row_center_y = last_center

        dbg_lines_abs = [b + y_top for b in borders_rel]
        dbg = HeaderDbg(lines=dbg_lines_abs, centers=out)
        return out, dbg, header_y_abs
