# OcrLibrary/BreakerTableAnalyzer8.py
from __future__ import annotations
import os, re, json, difflib, cv2, numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime

# -------- OCR availability --------
try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

ANALYZER_VERSION = "Analyzer6_mask_header"
ANALYZER_ORIGIN  = __file__


@dataclass
class _Dbg:
    lines: List[int]
    centers: List[int]


# ----------------- small utils -----------------
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
        import json
        return json.loads(s)
    except Exception:
        return None

def _norm_letters_digits(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9/\[\] \-]", "", (s or "").upper())).strip()

def _str_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_norm_letters_digits(a), b=_norm_letters_digits(b)).ratio()


# ============================================================
#                       MAIN ANALYZER
# ============================================================
class BreakerTableAnalyzer:
    """
    Stage 1 analyzer:
      - image prep
      - **NEW header finding**:
           1) build horizontal-line mask (in memory)
           2) remember header text line via OCR
           3) pick nearest white-pixel cluster in mask as header rule
      - structural rows
      - footer (structural + snapping kept as before)
      - degrade grid (remove only long, skinny lines)

    Debug overlay paths are returned when debug=True.
    """

    def __init__(self, debug: bool = False, config_path: str = "breaker_labels_config.jsonc"):
        self.debug = debug
        self.reader = None
        self._ocr_dbg_items: List[dict] = []
        self._ocr_dbg_rois: List[Tuple[int,int,int,int]] = []
        self._ocr_footer_marks: List[Tuple[int, str]] = []

        self._last_footer_y: Optional[int] = None
        self._footer_struct_y: Optional[int] = None
        self._footer_ocr_y: Optional[int] = None
        self._last_gridless_path: Optional[str] = None
        self._v_grid_xs: List[int] = []

        cfg_path = _abs_repo_cfg(config_path)
        self._cfg = _load_jsonc(cfg_path) or {}
        if debug:
            print(f"[BreakerTableAnalyzer] config_path = {os.path.abspath(cfg_path)} (loaded={self._cfg is not None})")

        # Try GPU → fall back to CPU
        if _HAS_OCR:
            self._init_easyocr()

    # --------------- OCR init ---------------
    def _init_easyocr(self):
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)  # try CUDA
            if self.debug:
                print("[OCR] EasyOCR initialized with GPU=True")
        except Exception as e:
            if self.debug:
                print(f"[OCR] GPU init failed ({e}); falling back to CPU")
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)
                if self.debug:
                    print("[OCR] EasyOCR initialized with GPU=False")
            except Exception as e2:
                if self.debug:
                    print(f"[OCR] CPU init failed ({e2}); OCR disabled")
                self.reader = None

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

        centers, dbg, header_y = self._row_centers_from_lines(gray)

        spaces_detected  = getattr(self, "_spaces_detected", len(centers) * 2)
        spaces_corrected = getattr(self, "_spaces_corrected", spaces_detected)
        spaces = spaces_corrected
        footer_y = self._last_footer_y

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
                "gridless_gray": gray, "gridless_path": None,
                "page_overlay_path": page_overlay_path,
                "spaces_detected": spaces_detected,
                "spaces_corrected": spaces_corrected,
                "footer_snapped": getattr(self, "_footer_snapped", False),
                "snap_note": getattr(self, "_snap_note", None),
                "v_grid_xs": getattr(self, "_v_grid_xs", []),
                "column_overlay_path": column_overlay_path,
            }

        gridless_gray, gridless_path = self._remove_grids_and_save(gray, header_y, footer_y, src_path)

        if self.debug:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(src_path))[0]
            page_overlay_path = os.path.join(debug_dir, f"{base}_page_overlay_{ts}.png")
            self._save_debug(gray, centers, dbg.lines, page_overlay_path, header_y, footer_y)

            column_overlay_path = os.path.join(debug_dir, f"{base}_columns_{ts}.png")
            self._save_column_overlay(gray, column_overlay_path)

            print(f"[ANALYZER] Row count: {len(centers)} | Spaces: {spaces}")
            print(f"[ANALYZER] Header Y: {header_y} | Footer Y: {footer_y}")
            if getattr(self, "_snap_note", None):
                print(f"[ANALYZER] {self._snap_note}")
            print(f"[ANALYZER] Gridless : {gridless_path}")
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

    # ---------------- Header via mask (NEW) ----------------
    def _find_header_via_mask(self, gray: np.ndarray) -> Optional[int]:
        """
        1) Build binarized image (white=ink) and horizontal-line mask (in memory).
        2) OCR to locate header *text line* (CCT/CKT preferred; fallback category header).
        3) From that y_ref, pick the nearest strong white cluster in horizontal mask as header rule.
        """
        H, W = gray.shape

        # --- (1) binarize & horizontal mask (no files are written) ---
        bw_inv = self._bin_white_ink(gray)                       # white=ink
        hmask  = self._horiz_mask_from_bin(bw_inv, W)            # horizontal candidates (white)

        # --- (2) header text line y_ref via OCR ---
        y_ref = self._header_text_y(gray)  # may be None

        # Bound the search window (prefer upper-mid page)
        if y_ref is not None and y_ref > int(0.65 * H):
            y_ref = None

        # If no OCR reference, we cannot anchor to it → no header
        if y_ref is None:
            return None

        # --- (3) nearest strong white cluster in hmask ---
        y_header = self._nearest_white_cluster_y(hmask, y_ref,
                                                 prefer_above=True,
                                                 search_px=int(0.12*H),
                                                 proj_frac=0.35,
                                                 min_width_frac=0.25,
                                                 max_thickness_px=max(2, int(0.02*H)))
        return int(y_header) if y_header is not None else None

    # ---------- Simple binarize & mask helpers ----------
    @staticmethod
    def _bin_white_ink(gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bw_inv  # white = ink

    @staticmethod
    def _horiz_mask_from_bin(bw_inv: np.ndarray, W: int, horiz_kernel_frac: float = 0.035) -> np.ndarray:
        fracs = [
            max(0.02, horiz_kernel_frac * 0.7),
            horiz_kernel_frac,
            min(0.08, horiz_kernel_frac * 1.8),
        ]
        masks = []
        for f in fracs:
            klen = max(5, int(round(W * f)))
            kh = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            m = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, kh, iterations=1)
            m = cv2.dilate(m, None, iterations=1)
            m = cv2.erode(m,  None, iterations=1)
            masks.append(m)
        out = masks[0]
        for m in masks[1:]:
            out = cv2.bitwise_or(out, m)
        return out

    # ---------- OCR for header text Y ----------
    def _header_text_y(self, gray: np.ndarray) -> Optional[int]:
        """
        Prefer explicit CCT/CKT; fallback to category header line.
        Returns a baseline-like Y (min y of that text line).
        """
        if self.reader is None:
            return None

        H, W = gray.shape
        y1, y2 = int(0.08 * H), int(0.65 * H)
        x1, x2 = 0, int(0.92 * W)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None

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

        def N(s: str) -> str:
            return re.sub(r"[^A-Z0-9]", "", (s or "").upper().replace("1","I").replace("0","O"))

        CATEGORY_ALIASES = {
            "ckt": {"CKT", "CCT"},
            "description": {"CIRCUITDESCRIPTION", "DESCRIPTION", "LOADDESCRIPTION", "DESIGNATION", "LOADDESIGNATION", "NAME"},
            "trip": {"TRIP", "AMPS", "AMP", "BREAKER", "BKR", "SIZE"},
            "poles": {"POLES", "POLE", "P"},
        }
        EXCLUDE = {"LOADCLASSIFICATION", "CLASSIFICATION"}  # avoid legends

        # (a) direct CCT/CKT
        cct_hits = []
        for box, txt, _ in det:
            if "CCT" in N(txt) or "CKT" in N(txt):
                ys = [p[1] for p in box]
                cct_hits.append(y1 + int(min(ys)))
        if cct_hits:
            return int(min(cct_hits))

        # (b) category header (score tokens across a y-binned line)
        lines = {}
        for box, txt, _ in det:
            if not txt:
                continue
            y_abs = y1 + int(min(p[1] for p in box))
            ybin = (y_abs // 14) * 14
            lines.setdefault(ybin, []).append((box, txt))

        best = None  # (score, y_abs)
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

                matched = set()
                for cat, aliases in CATEGORY_ALIASES.items():
                    if "P" in aliases and n == "P":
                        matched.add("poles")
                        continue
                    if any(alias in n for alias in (a for a in aliases if a != "P")):
                        matched.add(cat)

                if matched:
                    xs = [p[0] for p in box]
                    x_center = (min(xs) + max(xs)) / 2.0
                    categories_all |= matched
                    (left_has if x_center < mid else right_has).update(matched)

            if bad or not categories_all:
                continue

            score = len(categories_all)
            if left_has and right_has:
                score += 1
            if len(categories_all) >= 2:
                score += 1

            y_abs = int(ybin)
            cand = (score, y_abs)
            if best is None or (cand[0] > best[0]) or (cand[0] == best[0] and cand[1] < best[1]):
                best = cand

        return int(best[1]) if best else None

    # ---------- pick nearest white cluster in horizontal mask ----------
    @staticmethod
    def _nearest_white_cluster_y(hmask: np.ndarray,
                                 y_ref: int,
                                 prefer_above: bool = True,
                                 search_px: int = 160,
                                 proj_frac: float = 0.35,
                                 min_width_frac: float = 0.25,
                                 max_thickness_px: int = 6) -> Optional[int]:
        """
        Given a binary horizontal-line mask (white=lines) and a reference Y,
        choose the cluster center Y that:
          - lies within +/- search_px from y_ref
          - has projection >= proj_frac * max
          - spans at least min_width_frac of width
          - has thickness <= max_thickness_px
        If two are equally close, prefer above when prefer_above=True.
        """
        H, W = hmask.shape[:2]
        proj = hmask.sum(axis=1).astype(np.int64)
        if proj.max() == 0:
            return None

        # Normalize by row width to get a comparable measure
        row_strength = proj / float(255 * max(W, 1))

        thr = proj_frac * float(row_strength.max())
        ys = np.where(row_strength >= thr)[0]
        if ys.size == 0:
            return None

        # Build contiguous bands and enforce width/thickness with a second pass
        segments = []
        s = ys[0]
        p = ys[0]
        for y in ys[1:]:
            if y - p > 1:
                segments.append((s, p))
                s = y
            p = y
        segments.append((s, p))

        # Validate each segment for width/thickness
        candidates = []
        min_w = int(round(W * min_width_frac))
        for a, b in segments:
            y_c = (a + b) // 2
            if (abs(y_c - y_ref) > search_px):
                continue
            # build a 1-pixel horizontal sample at y_c to measure width
            row = hmask[y_c, :]
            xs = np.where(row > 0)[0]
            if xs.size == 0:
                continue
            xL, xR = int(xs.min()), int(xs.max())
            if (xR - xL + 1) < min_w:
                continue
            thickness = (b - a + 1)
            if thickness > max_thickness_px:
                continue
            candidates.append(int(y_c))

        if not candidates:
            # fall back: choose the strongest single row near y_ref
            lo = max(0, int(y_ref - search_px))
            hi = min(H - 1, int(y_ref + search_px))
            sub = np.arange(lo, hi + 1, 1, dtype=int)
            if sub.size == 0:
                return None
            y_best = int(sub[np.argmax(row_strength[lo:hi+1])])
            return y_best

        # nearest to y_ref; tie-break prefer above
        candidates.sort(key=lambda y: (abs(y - y_ref), 0 if (prefer_above and y <= y_ref) else 1, y))
        return candidates[0]

    # ---------------- Row & footer (structural) ----------------
    def _row_centers_from_lines(self, gray: np.ndarray):
        """
        Structural row detection (unchanged) using morphology on the page band,
        but now **header_y_abs** comes from _find_header_via_mask().
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
            # bottom quiet-zone clipping
            need = max(80, int(0.18 * len(proj)))
            low  = proj < (0.08 * proj.max())
            run = 0; cut = None
            for i, is_low in enumerate(low):
                run = run + 1 if is_low else 0
                if run >= need:
                    cut = i - run + 1
                    break
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

        # Fallback Hough
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

        # -------- Header via mask (NEW) --------
        header_y_abs = self._find_header_via_mask(gray)

        # safety cap
        if header_y_abs is not None and header_y_abs > int(0.62 * H):
            header_y_abs = None

        header_loc = None if header_y_abs is None else max(0, header_y_abs - y_top)

        # prune with header / cadence
        start_idx = 0
        if header_loc is not None:
            for i,b in enumerate(borders):
                if b > header_loc + 5:
                    start_idx = i
                    break
            if len(borders) - start_idx < 2:
                start_idx = 0
                header_y_abs = None
        else:
            gaps = np.diff(borders)
            if len(gaps) >= 6:
                for i in range(len(gaps)-6):
                    w = gaps[i:i+6]; gmed = float(np.median(w))
                    if gmed>0 and np.all((w >= 0.7*gmed) & (w <= 1.3*gmed)):
                        start_idx = i; break
        tail = borders[start_idx:]

        # large-gap stop
        if len(tail) >= 2:
            gaps = np.diff(tail); med = float(np.median(gaps)) if len(gaps) else 0
            stop_rel = None
            for j,g in enumerate(gaps):
                if med>0 and g > max(1.9*med, med+18):
                    stop_rel = j; break
            if stop_rel is not None:
                tail = tail[: stop_rel + 1]

        # late tall-band guard
        if len(tail) >= 3:
            gaps = np.diff(tail)
            med = float(np.median(gaps)) if len(gaps) else 0.0
            for j, g in enumerate(gaps):
                if med > 0 and g >= max(1.6 * med, med + 14):
                    remaining_borders = len(tail) - (j + 1)
                    if remaining_borders <= 3:
                        tail = tail[: j + 1]
                    break

        # central whitespace cut
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

        # cadence-run cut
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

        # vertical-structure tail cut
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

        # absolute borders; synthesize initial if header known and gap too large
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

        # row centers
        centers = []
        for i in range(len(borders_rel)-1):
            gap = borders_rel[i+1] - borders_rel[i]
            if gap >= 12:
                centers.append(int((borders_rel[i] + borders_rel[i+1]) // 2) + y_top)
        out=[]
        for y in centers:
            if out and abs(y-out[-1])<8: continue
            out.append(y)

        # footer (structural) + snap (unchanged)
        footer_struct_y = borders_abs[-1] if len(borders_abs) >= 1 else None
        self._footer_struct_y = footer_struct_y

        gaps_abs = np.diff(borders_abs)
        med_row_h = float(np.median(gaps_abs)) if gaps_abs.size else 18.0
        last_center = out[-1] if out else None

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

    # ---------------- Grid removal (unchanged) ----------------
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
        Kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, int(0.015 * roi_h))))
        Kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(0.12 * W)), 1))
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

        grid_mask = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

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

    # ---------------- debug overlays ----------------
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

        # borders
        for y in borders:
            cv2.line(vis, (0, int(y)), (W - 1, int(y)), (0, 165, 255), 1)

        # centers
        for y in centers:
            cv2.circle(vis, (24, int(y)), 4, (0, 255, 0), -1)

        # header line (NEW: explicitly draw it)
        if header_y_abs is not None:
            yh = int(header_y_abs)
            cv2.line(vis, (0, yh), (W - 1, yh), (0, 255, 0), 2)
            cv2.putText(vis, "HEADER", (12, max(16, yh - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # structural footer vs final
        if self._footer_struct_y is not None and footer_y_abs is not None \
           and abs(self._footer_struct_y - footer_y_abs) > 2:
            ys = int(self._footer_struct_y)
            cv2.line(vis, (0, ys), (W - 1, ys), (180, 0, 0), 2)
            cv2.putText(vis, "FOOTER_STRUCT", (12, max(16, ys - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1, cv2.LINE_AA)

        if footer_y_abs is not None:
            y = int(footer_y_abs)
            cv2.line(vis, (0, y), (W - 1, y), (0, 0, 255), 2)
            cv2.putText(vis, "FOOTER", (12, max(16, y + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # OCR footer markers (kept if you later wire OCR footer back)
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
