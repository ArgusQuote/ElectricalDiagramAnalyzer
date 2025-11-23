# OcrLibrary/BreakerTableAnalyzer6.py
from __future__ import annotations
import os, re, json, difflib, cv2, numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

# Version: TP band location removed. Hybrid footer logic (Total Load/Load + structural).
# Header detection updated: header text via OCR, then nearest strong horizontal band (prefer above).
ANALYZER_VERSION = "Analyzer6"
ANALYZER_ORIGIN  = __file__

@dataclass
class _Dbg:
    lines: List[int]
    centers: List[int]

def _abs_repo_cfg(default_name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, default_name)
    return cand if os.path.exists(cand) else default_name

def _load_jsonc(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        return json.loads(s)
    except Exception:
        return None

def _norm_token(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9/\[\] \-]", "", (s or "").upper())).strip()

def _str_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_norm_token(a), b=_norm_token(b)).ratio()


class BreakerTableAnalyzer:
    """
    Stage 1 analyzer (TP bands removed):
      - image prep
      - find header (NEW: OCR header text -> nearest strong horizontal band, prefer above)
      - find row centers & row_count -> spaces
      - HYBRID footer:
          Pass A: OCR 'TOTAL LOAD' (fuzzy) -> else 'LOAD' (guarded) -> snap to top rule
          Pass B: structural (row-cadence + whitespace + vertical grid) w/ final big-gap guard
        Arbitration picks the safer footer.
      - remove gridlines (Gridless_Images/)
      - optional overlay to <src_dir>/debug

    Outputs a dict payload consumed by the parser (second stage).
    """

    # ----- Header rule finder knobs (no disk writes; mask only in-memory) -----
    _HDR_SEARCH_UP_FIRST   = True     # prefer band above header text
    _HDR_MAX_DELTA_PX      = 80       # search radius (+/-) around header text Y
    _HDR_MIN_COVERAGE_FRAC = 0.25     # required white coverage across row to call it a rule
    _HDR_MERGE_Y_TOL       = 3        # merge contiguous rows to bands
    _HDR_HKERNEL_FRAC      = 0.035    # horizontal kernel fraction (multi-scale around this)
    _HDR_MIN_WIDTH_FRAC    = 0.25     # min width for a horizontal rule candidate (used in contour pass)

    def __init__(self, debug: bool = False, config_path: str = "breaker_labels_config.jsonc"):
        self.debug = debug
        self.reader = None
        self._ocr_dbg_items: List[dict] = []
        self._ocr_dbg_rois: List[Tuple[int,int,int,int]] = []
        self._last_footer_y: Optional[int] = None
        self._footer_struct_y: Optional[int] = None
        self._footer_ocr_y: Optional[int] = None
        self._last_gridless_path: Optional[str] = None
        # overlay annotations for OCR footer cues: list of (y_abs, label)
        self._ocr_footer_marks: List[Tuple[int, str]] = []
        self._v_grid_xs: List[int] = []

        # debug taps for header
        self._hdr_text_y: Optional[int] = None
        self._hdr_rule_y: Optional[int] = None

        cfg_path = _abs_repo_cfg(config_path)
        self._cfg = _load_jsonc(cfg_path) or {}

        if debug:
            print(f"[BreakerTableAnalyzer] config_path = {os.path.abspath(cfg_path)} "
                  f"(loaded={self._cfg is not None})")

        if _HAS_OCR:
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)

    # ============== public ==============
    def analyze(self, image_path: str) -> Dict:
        src_path = os.path.abspath(os.path.expanduser(image_path))
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        src_dir  = os.path.dirname(src_path)
        debug_dir = os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src_path)
        gray = self._prep(img)
        # reset per-run OCR footer marks
        self._ocr_footer_marks = []
        self._hdr_text_y = None
        self._hdr_rule_y = None

        centers, dbg, header_y = self._row_centers_from_lines(gray)

        # prefer corrected spaces if available
        spaces_detected  = getattr(self, "_spaces_detected", len(centers) * 2)
        spaces_corrected = getattr(self, "_spaces_corrected", spaces_detected)
        spaces = spaces_corrected

        footer_y = self._last_footer_y  # this is the (possibly snapped) footer

        # overlay (before early-exit)
        page_overlay_path = None
        column_overlay_path = None

        if not centers or header_y is None or footer_y is None:
            if self.debug:
                print(f"[ANALYZER] Early return: centers={len(centers)} header={header_y} footer={footer_y}")
            return {
                "src_path": src_path, "src_dir": src_dir, "debug_dir": debug_dir,
                "gray": gray, "centers": centers, "row_count": len(centers), "spaces": spaces,
                "header_y": header_y, "footer_y": footer_y,
                "tp_cols": {}, "tp_combined": {"left": False, "right": False},
                "gridless_gray": gray, "gridless_path": None, "page_overlay_path": page_overlay_path,
                "spaces_detected": spaces_detected,
                "spaces_corrected": spaces_corrected,
                "footer_snapped": getattr(self, "_footer_snapped", False),
                "snap_note": getattr(self, "_snap_note", None),
                "v_grid_xs": getattr(self, "_v_grid_xs", []),
                "column_overlay_path": column_overlay_path,
                # debug taps
                "header_text_y": self._hdr_text_y,
                "header_rule_y": self._hdr_rule_y,
            }

        gridless_gray, gridless_path = self._remove_grids_and_save(gray, header_y, footer_y, src_path)
        if self.debug:
            # Existing page overlay (rows/header/footer/etc.)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(src_path))[0]
            page_overlay_path = os.path.join(debug_dir, f"{base}_page_overlay_{ts}.png")
            self._save_debug(gray, centers, dbg.lines, page_overlay_path, header_y, footer_y)

            # NEW: columns-only overlay
            column_overlay_path = os.path.join(debug_dir, f"{base}_columns_{ts}.png")
            self._save_column_overlay(gray, column_overlay_path)

        if self.debug:
            print(f"[ANALYZER] Row count: {len(centers)} | Spaces: {spaces}")
            print(f"[ANALYZER] Header TEXT Y: {self._hdr_text_y} | Header RULE Y: {self._hdr_rule_y}")
            print(f"[ANALYZER] Footer Y: {footer_y}")
            if getattr(self, "_snap_note", None):
                print(f"[ANALYZER] {self._snap_note}")
            print(f"[ANALYZER] Gridless : {gridless_path}")
            if page_overlay_path:
                print(f"[ANALYZER] Overlay  : {page_overlay_path}")

        return {
            "src_path": src_path, "src_dir": src_dir, "debug_dir": debug_dir,
            "gray": gray, "centers": centers, "row_count": len(centers), "spaces": spaces,
            "header_y": header_y, "footer_y": footer_y,
            "tp_cols": {}, "tp_combined": {"left": False, "right": False},
            "gridless_gray": gridless_gray, "gridless_path": gridless_path, 
            "page_overlay_path": page_overlay_path,
            "spaces_detected": spaces_detected,
            "spaces_corrected": spaces_corrected,
            "footer_snapped": getattr(self, "_footer_snapped", False),
            "snap_note": getattr(self, "_snap_note", None),
            "v_grid_xs": getattr(self, "_v_grid_xs", []),
            "column_overlay_path": column_overlay_path,
            # debug taps
            "header_text_y": self._hdr_text_y,
            "header_rule_y": self._hdr_rule_y,
        }

    # ============== internals ==============
    def _prep(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        H, W = g.shape
        if H < 1600:
            s = 1600.0 / H
            g = cv2.resize(g, (int(W*s), int(H*s)), interpolation=cv2.INTER_CUBIC)
        return g

    # ========== NEW HEADER PIPELINE ==========
    def _find_header_rule_y(self, gray: np.ndarray) -> Optional[int]:
        """
        1) Locate header **text** row using existing OCR heuristics.
        2) Build a horizontal-only morphology mask (in-memory).
        3) From the text row, search up (then down) within a window for a strong horizontal band.
        4) Return that band's Y as the header line. Nothing is written to disk.
        """
        H, W = gray.shape

        # Step 1: header text y (prefer tokens heuristic; fallback CCT/CKT)
        header_text_y = self._find_header_by_tokens(gray)
        if header_text_y is None:
            header_text_y = self._find_cct_header_y(gray)

        # safety cap (ignore anything too low)
        if header_text_y is not None and header_text_y > int(0.62 * H):
            header_text_y = None

        self._hdr_text_y = header_text_y
        if header_text_y is None:
            return None

        # Step 2: horizontal-only mask (WHITE=horizontal lines), multi-scale around HKERNEL_FRAC
        bw_inv = self._bin_ink(gray)               # white=ink
        hmask  = self._horiz_mask_from_ink(bw_inv, W, self._HDR_HKERNEL_FRAC)

        # Step 3: find band near header_text_y
        header_rule_y = self._find_near_header_band(
            hmask,
            header_text_y,
            search_up_first=self._HDR_SEARCH_UP_FIRST,
            max_delta=self._HDR_MAX_DELTA_PX,
            min_cov_frac=self._HDR_MIN_COVERAGE_FRAC,
            merge_tol=self._HDR_MERGE_Y_TOL
        )
        self._hdr_rule_y = header_rule_y
        return header_rule_y

    def _bin_ink(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bw_inv  # WHITE=ink

    def _horiz_mask_from_ink(self, ink_white: np.ndarray, W: int, horiz_kernel_frac: float, border_trim_px: int = 2) -> np.ndarray:
        m = ink_white.copy()
        if border_trim_px > 0:
            m[:border_trim_px,:] = 0; m[-border_trim_px:,:] = 0
            m[:,:border_trim_px] = 0; m[:,-border_trim_px:] = 0

        fracs = [
            max(0.02, horiz_kernel_frac * 0.7),
            horiz_kernel_frac,
            min(0.10, horiz_kernel_frac * 1.8),
        ]
        out = None
        for f in fracs:
            klen = max(5, int(round(W * f)))
            kh = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            t = cv2.morphologyEx(m, cv2.MORPH_OPEN, kh, iterations=1)
            t = cv2.dilate(t, None, iterations=1)
            t = cv2.erode(t,  None, iterations=1)
            out = t if out is None else cv2.bitwise_or(out, t)
        return out

    def _find_near_header_band(self, hmask: np.ndarray, header_y: int,
                               search_up_first: bool, max_delta: int,
                               min_cov_frac: float, merge_tol: int) -> Optional[int]:
        """
        hmask: WHITE=horizontal lines only.
        Return the representative y (mid) of the first strong band near header_y.
        """
        if header_y is None:
            return None
        H, W = hmask.shape
        ref = int(max(0, min(H - 1, header_y)))

        row_cov = (hmask > 0).sum(axis=1)  # white pixels per row
        thr = int(round(min_cov_frac * W))
        ys = np.where(row_cov >= thr)[0].astype(int)
        if ys.size == 0:
            return None

        # Merge rows into bands
        bands = []
        s = ys[0]; p = ys[0]
        for y in ys[1:]:
            if y - p <= merge_tol:
                p = y
            else:
                bands.append((s, p))
                s = p = y
        bands.append((s, p))
        reps = [int((a + b) // 2) for a, b in bands]

        lo = max(0, ref - max_delta)
        hi = min(H - 1, ref + max_delta)
        near = [y for y in reps if lo <= y <= hi]
        if not near:
            return None

        above = [y for y in near if y <= ref]
        below = [y for y in near if y > ref]

        if search_up_first:
            if above:
                return max(above)
            return min(below, key=lambda y: abs(y - ref)) if below else None
        else:
            if below:
                return min(below)
            return max(above) if above else None

    # ---------- (existing) OCR: find the CCT/CKT header line ----------
    # (unchanged: used as subroutine for header_text_y fallback)
    def _find_cct_header_y(self, gray: np.ndarray) -> Optional[int]:
        if self.reader is None:
            return None
        self._ocr_dbg_items, self._ocr_dbg_rois = [], []
        H, W = gray.shape

        def norm(s: str) -> str:
            return re.sub(r"[^A-Z]", "", s.upper().replace("1","I"))

        def has_long_line_below(y_abs: int) -> bool:
            y1 = max(0, y_abs - 6); y2 = min(H, y_abs + 28)
            band = gray[y1:y2, :]
            if band.size == 0: return False
            g = cv2.GaussianBlur(band, (3,3), 0)
            bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 10)
            klen = max(70, int(W * 0.22))
            K = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, K, iterations=1)
            return lines.sum() > 0

        def scan_roi(y1f, y2f, x2f) -> List[int]:
            y1, y2 = int(H*y1f), int(H*y2f)
            x1, x2 = 0, int(W*x2f)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0: return []
            self._ocr_dbg_rois.append((x1, y1, x2, y2))

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
                xs = [p[0] + x1 for p in box]; ys = [p[1] + y1 for p in box]
                self._ocr_dbg_items.append({
                    "text": txt, "conf": float(conf or 0.0),
                    "x1": int(min(xs)), "y1": int(min(ys)),
                    "x2": int(max(xs)), "y2": int(max(ys)),
                })

            lines = {}
            for box, txt, _ in det:
                if not txt: continue
                yc = int(sum(p[1] for p in box) / 4)
                key = (yc // 14) * 14
                lines.setdefault(key, []).append((box, txt))

            cands = []

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
                    y_abs = y1 + min(int(min(p[1] for p in b)) for b,_ in items)
                    if has_long_line_below(y_abs):
                        cands.append(y_abs)

            return sorted(set(int(y) for y in cands))

        all_hits: List[int] = []
        primary_windows = [(0.08, 0.48, 0.28),(0.10, 0.50, 0.32),(0.08, 0.52, 0.40),(0.12, 0.58, 0.45)]
        for y1f, y2f, x2f in primary_windows:
            all_hits += scan_roi(y1f, y2f, x2f)

        fallback_candidates = [(0.08, 0.52, 0.65),(0.10, 0.56, 0.80),(0.08, 0.60, 1.00)]
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
            cands=[]
            for box, txt, _ in det:
                t = re.sub(r"[^A-Z]", "", (txt or "").upper().replace("1","I"))
                if t in ("CCT","CKT"):
                    cands.append(y1 + int(min(p[1] for p in box)))
            all_hits += cands

        cap = int(0.65 * H)
        all_hits = [y for y in all_hits if y <= cap]

        if not all_hits:
            return None
        return int(min(all_hits))

    def _find_header_by_tokens(self, gray: np.ndarray) -> Optional[int]:
        """
        Fallback header TEXT finder when 'CCT/CKT' isn't present.
        (Unchanged — we use its output as header_text_y in the new method.)
        """
        if self.reader is None:
            return None

        H, W = gray.shape
        y1, y2 = int(0.08 * H), int(0.65 * H)
        x1, x2 = 0, W
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        def N(s: str) -> str:
            return re.sub(r"[^A-Z0-9]", "", (s or "").upper().replace("1", "I").replace("0", "O"))

        CATEGORY_ALIASES = {
            "ckt": {"CKT", "CCT"},
            "description": {"CIRCUITDESCRIPTION", "DESCRIPTION", "LOADDESCRIPTION", "DESIGNATION", "LOADDESIGNATION", "NAME"},
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
            bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 10)
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
                if best is None or (cand[0] > best[0]) or (cand[0] == best[0] and cand[1] < best[1]):
                    best = cand

        if best is None:
            return None
        return int(best[1])

    # ---------- OCR Pass A: find totals footer using 'TOTAL LOAD' (unchanged) ----------
    def _find_totals_footer_from_load(self, gray: np.ndarray,
                                    header_y_abs: Optional[int],
                                    last_center: Optional[int],
                                    med_row_h: float) -> Optional[int]:
        if self.reader is None:
            return None

        H, W = gray.shape
        if not med_row_h or med_row_h <= 0:
            med_row_h = 18.0

        y0 = int(max(0.45 * H, (header_y_abs or int(0.30 * H)) + 40))
        y1 = int(0.95 * H)
        x1, x2 = 0, int(0.92 * W)
        roi = gray[y0:y1, x1:x2]
        if roi.size == 0:
            return None

        def ocr_pass(img, mag):
            try:
                return self.reader.readtext(
                    img, detail=1, paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :/.-",
                    mag_ratio=mag, contrast_ths=0.05, adjust_contrast=0.7,
                    text_threshold=0.4, low_text=0.25,
                )
            except Exception:
                return []

        g = cv2.GaussianBlur(roi, (3,3), 0)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
        inv = cv2.bitwise_not(g)

        dets = ocr_pass(g, 1.6) + ocr_pass(g, 2.0) + ocr_pass(inv, 1.6) + ocr_pass(inv, 2.0)

        def N(s: str) -> str:
            return re.sub(r"[^A-Z]", "", (s or "").upper().replace("1", "I").replace("0", "O"))

        total_hits = []
        load_hits  = []
        for box, txt, _ in dets:
            n = N(txt)
            if not n:
                continue
            y_abs = y0 + int(min(p[1] for p in box))

            sim_total = difflib.SequenceMatcher(a=n, b="TOTALLOAD").ratio()
            is_total = (("TOTAL" in n) and ("LOAD" in n)) or (sim_total >= 0.70)
            if is_total:
                total_hits.append(y_abs)
                self._ocr_footer_marks.append((y_abs, "TOTAL_LOAD_HIT"))
                continue

            if ("LOAD" in n) and ("CLASS" not in n):
                load_hits.append(y_abs)
                self._ocr_footer_marks.append((y_abs, "LOAD_HIT"))

        hits = total_hits if total_hits else load_hits
        if not hits:
            return None

        if last_center is not None:
            lo  = last_center + 0.6 * med_row_h
            hi  = last_center + 3.0 * med_row_h
            near = [h for h in hits if lo <= h <= hi]
            if near:
                hits = near

        y_hit = int(min(hits))
        self._ocr_footer_marks.append((y_hit, "OCR_PICK"))

        band_top = max(0, y_hit - 70)
        band_bot = max(0, y_hit - 4)
        band = gray[band_top:band_bot, :]
        y_final = None

        if band.size > 0:
            b = cv2.GaussianBlur(band, (3,3), 0)
            bw = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 10)
            klen = int(max(70, min(W * 0.30, 320)))
            K = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, K, iterations=1)
            proj  = lines.sum(axis=1)

            if proj.size > 0 and proj.max() > 0:
                ys = np.where(proj >= 0.35 * proj.max())[0]
                if ys.size:
                    y_rule = int(ys[-1])
                    y_final = band_top + y_rule
                    self._ocr_footer_marks.append((y_final, "OCR_RULE"))

        if y_final is None:
            y_final = y_hit - 2

        if header_y_abs is not None:
            min_ok = int(header_y_abs + 8)
            if y_final <= min_ok:
                y_final = min_ok
                self._ocr_footer_marks.append((y_final, "OCR_CLAMP_HEADER"))

        return int(y_final)

    # ---------- Structural row detection + hybrid arbitration ----------
    def _row_centers_from_lines(self, gray: np.ndarray):
        """
        Structural row detection + NEW header rule via in-memory horizontal mask.
        """
        H, W = gray.shape
        y_top = int(H * 0.12)
        y_bot = int(H * 0.95)
        roi = gray[y_top:y_bot, :]

        blur = cv2.GaussianBlur(roi, (3,3), 0)
        bw   = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 10)

        def borders_from(bw_src, k_frac, thr_frac):
            if bw_src.size == 0: return []
            klen = int(max(70, min(W * k_frac, 320)))
            K = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            lines = cv2.morphologyEx(bw_src, cv2.MORPH_OPEN, K, iterations=1)
            proj  = lines.sum(axis=1)
            if proj.size == 0 or proj.max() == 0: return []
            need = max(80, int(0.18 * len(proj)))
            low  = proj < (0.08 * proj.max())
            run = 0; cut = None
            for i, is_low in enumerate(low):
                run = run + 1 if is_low else 0
                if run >= need: cut = i - run + 1; break
            if cut is not None and cut > 10:
                proj = proj[:cut]
                if proj.size == 0 or proj.max() == 0: return []
            thr = float(thr_frac * proj.max())
            ys  = np.where(proj >= thr)[0].astype(int)
            if ys.size == 0: return []
            segs=[]; s=ys[0]; p=ys[0]
            for y in ys[1:]:
                if y-p>2: segs.append((s,p)); s=y
                p=y
            segs.append((s,p))
            return [int((a+b)//2) for a,b in segs]

        b1 = borders_from(bw, 0.35, 0.35)
        b2 = borders_from(bw, 0.25, 0.30)
        borders = b1 if len(b1) >= len(b2) else b2

        if not borders:
            edges = cv2.Canny(blur, 60, 160)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=140,
                                    minLineLength=int(W*0.30), maxLineGap=6)
            if lines is None:
                self._last_footer_y = None
                self._footer_struct_y = None
                self._footer_ocr_y = None
                self._bottom_border_y = None
                self._bottom_row_center_y = None
                return [], _Dbg(lines=[], centers=[]), None
            ys = sorted(int((y1+y2)//2) for x1,y1,x2,y2 in lines[:,0] if abs(y2-y1)<=2)
            borders=[ys[0]]
            for y in ys[1:]:
                if y-borders[-1] > 3: borders.append(y)

        # ======= NEW header detection here =======
        header_y_abs = self._find_header_rule_y(gray)  # << use new method (rule near header text)

        # safety cap (ignore anything too low)
        if header_y_abs is not None and header_y_abs > int(0.62 * H):
            header_y_abs = None

        header_loc = None if header_y_abs is None else max(0, header_y_abs - y_top)

        # PRUNE by header / cadence
        start_idx = 0
        if header_loc is not None:
            for i,b in enumerate(borders):
                if b > header_loc + 5: start_idx = i; break
            if len(borders) - start_idx < 2:
                start_idx = 0; header_y_abs = None
        else:
            gaps = np.diff(borders)
            if len(gaps) >= 6:
                for i in range(len(gaps)-6):
                    w = gaps[i:i+6]; gmed = float(np.median(w))
                    if gmed>0 and np.all((w >= 0.7*gmed) & (w <= 1.3*gmed)):
                        start_idx = i; break
        tail = borders[start_idx:]

        # Large-gap stop
        if len(tail) >= 2:
            gaps = np.diff(tail); med = float(np.median(gaps)) if len(gaps) else 0
            stop_rel = None
            for j,g in enumerate(gaps):
                if med>0 and g > max(1.9*med, med+18):
                    stop_rel = j; break
            if stop_rel is not None:
                tail = tail[: stop_rel + 1]

        # Late tall-band guard
        if len(tail) >= 3:
            gaps = np.diff(tail)
            med = float(np.median(gaps)) if len(gaps) else 0.0
            for j, g in enumerate(gaps):
                if med > 0 and g >= max(1.6 * med, med + 14):
                    remaining_borders = len(tail) - (j + 1)
                    if remaining_borders <= 3:
                        tail = tail[: j + 1]
                    break

        # CENTRAL WHITESPACE CUT
        if len(tail) >= 3:
            gaps = np.diff(tail)
            med_row_h = float(np.median(gaps)) if len(gaps) else 0.0
            if med_row_h > 0:
                xL = int(0.25 * W); xR = int(0.75 * W)
                central = bw[:, xL:xR]
                if central.size > 0:
                    central_denoised = cv2.morphologyEx(central, cv2.MORPH_OPEN,
                                                        cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)), 1)
                    central_denoised = cv2.morphologyEx(central_denoised, cv2.MORPH_OPEN,
                                                        cv2.getStructuringElement(cv2.MORPH_RECT, (1,3)), 1)
                    row_ink = central_denoised.sum(axis=1).astype(np.float32) / (255.0 * (xR - xL))
                    win = int(round(1.8 * med_row_h)); win = max(9, min(win, int(0.06 * H)))
                    if win % 2 == 0: win += 1
                    smooth = np.convolve(row_ink, np.ones(win, dtype=np.float32)/float(win), mode="same")
                    LOW_INK = 0.06
                    y_scan_start = int(min(tail[0] + 2 * med_row_h, len(smooth) - 1))
                    below = np.where(smooth < LOW_INK)[0]
                    if below.size:
                        after = below[below >= y_scan_start]
                        if after.size:
                            cut_abs = y_top + int(after[0])
                            tail = [b for b in tail if (b + y_top) < cut_abs - 2]

        # CADENCE-RUN CUT
        if len(tail) >= 3:
            gaps_all = np.diff(tail).astype(np.float32)
            if gaps_all.size:
                seed_n = int(min(10, len(gaps_all)))
                global_med = float(np.median(gaps_all[:seed_n])) if seed_n > 0 else float(np.median(gaps_all))
                MIN_GAP = 12.0; LO = 0.60; HI = 1.45; WIN = 5
                best_run_len = 0; best_run_end = None
                i = 0
                while i < len(gaps_all):
                    if gaps_all[i] < MIN_GAP:
                        i += 1; continue
                    run_start = i; local_vals = []; local_med = global_med if global_med > 0 else gaps_all[i]
                    while i < len(gaps_all):
                        g = gaps_all[i]; local_vals.append(float(g))
                        if len(local_vals) > WIN: local_vals.pop(0)
                        local_med = float(np.median(local_vals)) if local_vals else local_med
                        if not ((g >= MIN_GAP) and (LO*local_med <= g <= HI*local_med)): break
                        i += 1
                    run_end = i - 1; run_len = run_end - run_start + 1
                    if run_len > best_run_len: best_run_len = run_len; best_run_end = run_end
                    i = max(i, run_end + 1)
                if best_run_len >= 2:
                    tail = tail[: int(best_run_end + 1) + 1]

        # VERTICAL STRUCTURE TAIL CUT
        if len(tail) >= 3:
            gaps = np.diff(tail)
            med_row_h = float(np.median(gaps)) if len(gaps) else 0.0
            if med_row_h > 0:
                roi_h = bw.shape[0]
                Kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, int(0.05*roi_h))))
                vlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)
                xL = int(0.22 * W); xR = int(0.78 * W)
                v_central = vlines[:, xL:xR] if xR > xL else vlines
                if v_central.size > 0:
                    v_row = v_central.sum(axis=1).astype(np.float32) / (255.0 * (xR - xL))
                    win = int(round(1.2 * med_row_h)); win = max(7, min(win, int(0.05 * H)))
                    if win % 2 == 0: win += 1
                    v_smooth = np.convolve(v_row, np.ones(win, dtype=np.float32)/float(win), mode="same")
                    thr = max(0.02, 0.12 * float(v_smooth.max() or 1.0))
                    y_scan_start = int(min(tail[0] + 2 * med_row_h, len(v_smooth) - 1))
                    need = int(max(1.2 * med_row_h, 12)); run = 0; cut_rel = None
                    for y in range(y_scan_start, len(v_smooth)):
                        if v_smooth[y] < thr:
                            run += 1
                            if run >= need:
                                cut_rel = y - run + 1; break
                        else:
                            run = 0
                    if cut_rel is not None:
                        cut_abs = y_top + int(cut_rel)
                        tail = [b for b in tail if (b + y_top) < cut_abs - 2]

        borders_abs = [b + y_top for b in tail]
        if header_y_abs is not None and len(borders_abs) >= 2:
            gaps_abs = np.diff(borders_abs)
            med_gap = float(np.median(gaps_abs)) if len(gaps_abs) else 0.0
            if med_gap > 0:
                d0 = float(borders_abs[0] - header_y_abs)
                if d0 > max(1.80 * med_gap, med_gap + 16) and d0 < 3.2 * med_gap:
                    y_expected = int(round(header_y_abs + med_gap))
                    if y_expected < borders_abs[0] - 8 and y_expected > header_y_abs + 6:
                        borders_abs.insert(0, y_expected)

        borders_rel = [y - y_top for y in borders_abs]
        if len(borders_rel) < 2:
            self._last_footer_y = borders_abs[-1] if borders_abs else None
            self._footer_struct_y = self._last_footer_y
            self._footer_ocr_y = None
            self._bottom_border_y = self._last_footer_y
            self._bottom_row_center_y = None
            self._spaces_detected = 0
            self._spaces_corrected = 0
            self._footer_snapped = False
            self._snap_note = None
            return [], _Dbg(lines=[b + y_top for b in borders_rel], centers=[]), header_y_abs

        centers = []
        for i in range(len(borders_rel)-1):
            gap = borders_rel[i+1] - borders_rel[i]
            if gap >= 12:
                centers.append(int((borders_rel[i] + borders_rel[i+1]) // 2) + y_top)
        out=[]
        for y in centers:
            if out and abs(y-out[-1])<8: continue
            out.append(y)

        footer_struct_y = borders_abs[-1] if len(borders_abs) >= 1 else None
        self._footer_struct_y = footer_struct_y

        gaps_abs = np.diff(borders_abs)
        med_row_h = float(np.median(gaps_abs)) if gaps_abs.size else 18.0
        last_center = out[-1] if out else None

        self._footer_ocr_y = None

        final_footer = footer_struct_y

        if (final_footer is not None) and (header_y_abs is not None) and (final_footer <= header_y_abs + 6):
            final_footer = None

        detected_spaces = len(out) * 2
        self._spaces_detected = detected_spaces
        self._spaces_corrected = detected_spaces
        self._footer_snapped = False
        self._snap_note = None

        snap_map = {
            16: 18, 20: 18,
            28: 30, 32: 30,
            40: 42, 44: 42,
            52: 54, 56: 54,
            64: 66, 68: 66,
            70: 72, 74: 72,
            82: 84, 86: 84,
        }

        if (final_footer is not None) and (detected_spaces in snap_map) and (med_row_h > 0):
            target_spaces = snap_map[detected_spaces]
            delta_rows = int((target_spaces - detected_spaces) // 2)
            if delta_rows != 0:
                shift = int(round(delta_rows * med_row_h))
                lo = int((header_y_abs + 8) if header_y_abs is not None else 0)
                hi = H - 5
                snapped = int(np.clip(final_footer + shift, lo, hi))
                if (last_center is not None) and (snapped < last_center + 0.5 * med_row_h):
                    snapped = int(last_center + 0.5 * med_row_h)
                self._footer_snapped = True
                self._spaces_corrected = target_spaces
                self._snap_note = f"SNAP spaces {detected_spaces}→{target_spaces}; footer {final_footer}→{snapped}"
                final_footer = snapped

        self._last_footer_y = int(final_footer) if final_footer is not None else None
        self._bottom_border_y = footer_struct_y
        self._bottom_row_center_y = last_center

        dbg_lines_abs = [b + y_top for b in borders_rel]
        dbg = _Dbg(lines=dbg_lines_abs, centers=out)
        return out, dbg, header_y_abs

    # ---------- grid removal & debug overlays (unchanged) ----------
    def _remove_grids_and_save(self, gray: np.ndarray, header_y: int, footer_y: int, image_path: str):
        H, W = gray.shape
        work = gray.copy()
        self._v_grid_xs = []

        if header_y is None or footer_y is None or footer_y <= header_y + 10:
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = os.path.dirname(image_path)
            gridless_dir = os.path.join(out_dir, "Gridless_Images")
            os.makedirs(gridless_dir, exist_ok=True)
            gridless_path = os.path.join(gridless_dir, f"{base}_gridless.png")
            cv2.imwrite(gridless_path, work)
            self._last_gridless_path = gridless_path
            return work, gridless_path

        y1 = max(0, int(header_y) - 4)
        y2 = min(H - 1, int(footer_y) + 4)
        if y2 <= y1:
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = os.path.dirname(image_path)
            gridless_dir = os.path.join(out_dir, "Gridless_Images")
            os.makedirs(gridless_dir, exist_ok=True)
            gridless_path = os.path.join(gridless_dir, f"{base}_gridless.png")
            cv2.imwrite(gridless_path, work)
            self._last_gridless_path = gridless_path
            return work, gridless_path

        roi = work[y1:y2, :]

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
        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(40, int(0.12 * W)), 1)
        )
        v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

        v_mask = np.zeros_like(v_candidates, dtype=np.uint8)
        v_x_centers: List[int] = []

        num_v, labels_v, stats_v, _ = cv2.connectedComponentsWithStats(v_candidates, connectivity=8)
        if num_v > 1:
            min_vert_len = int(0.40 * roi_h)
            max_vert_thick = max(2, int(0.02 * W))
            for i in range(1, num_v):
                x, y, w, h, area = stats_v[i]
                if h >= min_vert_len and w <= max_vert_thick and area > 0:
                    v_mask[labels_v == i] = 255
                    x_center = x + w // 2
                    v_x_centers.append(int(x_center))

        if v_x_centers:
            xs = sorted(v_x_centers)
            collapsed = []
            for x in xs:
                if not collapsed or abs(x - collapsed[-1]) > 4:
                    collapsed.append(x)
            self._v_grid_xs = collapsed
        else:
            self._v_grid_xs = []

        h_mask = np.zeros_like(h_candidates, dtype=np.uint8)
        num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(h_candidates, connectivity=8)
        if num_h > 1:
            min_horiz_len = int(0.40 * W)
            max_horiz_thick = max(2, int(0.02 * roi_h))
            for i in range(1, num_h):
                x, y, w, h, area = stats_h[i]
                if w >= min_horiz_len and h <= max_horiz_thick and area > 0:
                    h_mask[labels_h == i] = 255

        grid = cv2.bitwise_or(v_mask, h_mask)

        if grid.sum() == 0:
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = os.path.dirname(image_path)
            gridless_dir = os.path.join(out_dir, "Gridless_Images")
            os.makedirs(gridless_dir, exist_ok=True)
            gridless_path = os.path.join(gridless_dir, f"{base}_gridless.png")
            cv2.imwrite(gridless_path, work)
            self._last_gridless_path = gridless_path
            return work, gridless_path

        grid_mask = cv2.dilate(
            grid,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1
        )

        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        inpainted = cv2.inpaint(roi_bgr, grid_mask, 2, cv2.INPAINT_TELEA)
        clean_roi_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)

        work[y1:y2, :] = clean_roi_gray

        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.dirname(image_path)
        gridless_dir = os.path.join(out_dir, "Gridless_Images")
        os.makedirs(gridless_dir, exist_ok=True)

        gridless_path = os.path.join(gridless_dir, f"{base}_gridless.png")
        ok = cv2.imwrite(gridless_path, work)
        if not ok:
            raise IOError(f"Failed to write gridless image to: {gridless_path}")

        self._last_gridless_path = gridless_path
        return work, gridless_path

    def _save_column_overlay(self, gray: np.ndarray, out_path: str):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        H, W = gray.shape

        v_xs = getattr(self, "_v_grid_xs", [])
        for x in v_xs:
            xi = int(x)
            cv2.line(vis, (xi, 0), (xi, H - 1), (255, 0, 255), 1)
            cv2.putText(vis, "COL", (xi + 2, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, vis)
        except Exception:
            pass

    def _save_debug(self, gray: np.ndarray, centers: List[int], borders: List[int],
                    out_path: str, header_y_abs: Optional[int], footer_y_abs: Optional[int]):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        H, W = gray.shape

        for y in borders:
            cv2.line(vis, (0, int(y)), (W - 1, int(y)), (0, 165, 255), 1)
        for y in centers:
            cv2.circle(vis, (24, int(y)), 4, (0, 255, 0), -1)

        # draw header RULE (new)
        if header_y_abs is not None:
            y = int(header_y_abs)
            cv2.line(vis, (0, y), (W - 1, y), (0, 220, 0), 2)
            cv2.putText(vis, "HEADER_RULE", (12, max(16, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

        if self._footer_struct_y is not None and footer_y_abs is not None and abs(self._footer_struct_y - footer_y_abs) > 2:
            ys = int(self._footer_struct_y)
            cv2.line(vis, (0, ys), (W - 1, ys), (180, 0, 0), 2)
            cv2.putText(vis, "FOOTER_STRUCT", (12, max(16, ys - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1, cv2.LINE_AA)

        if footer_y_abs is not None:
            y = int(footer_y_abs)
            cv2.line(vis, (0, y), (W - 1, y), (0, 0, 255), 2)
            cv2.putText(vis, "FOOTER", (12, max(16, y + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        for y_mark, label in getattr(self, "_ocr_footer_marks", []):
            y = int(y_mark)
            cv2.line(vis, (0, y), (W - 1, y), (200, 50, 200), 1)
            cv2.putText(vis, label, (240, max(14, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 50, 200), 1, cv2.LINE_AA)

        note = getattr(self, "_snap_note", None)
        det  = getattr(self, "_spaces_detected", None)
        cor  = getattr(self, "_spaces_corrected", None)
        trail = []
        if det is not None and cor is not None:
            trail.append(f"spaces {det}→{cor}" if det != cor else f"spaces {det}")
        if note:
            trail.append(note)
        if trail:
            cv2.putText(vis, " | ".join(trail), (12, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, vis)
        except Exception:
            pass
