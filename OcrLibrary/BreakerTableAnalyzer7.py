# OcrLibrary/BreakerTableAnalyzer7.py
from __future__ import annotations
import os, re, json, difflib, cv2, numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime

# ---- Runtime-safe CUDA memory defragmentation for PyTorch/EasyOCR ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Version: TP band location removed. Hybrid footer logic (Total Load/Load + structural).
#          Updated header/footer detection via OCR tokens + nearest horizontal white band.
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
      - find header (OCR tokens → text y → nearest horizontal white band)
      - find footer (OCR footer tokens → text y → nearest horizontal white band)
      - find row centers (structural, unchanged)
      - remove gridlines (Gridless_Images/)
      - optional overlay to <src_dir>/debug

    Outputs a dict payload consumed by the parser (second stage).
    """

    # ----------------------- ctor -----------------------
    def __init__(
        self,
        debug: bool = False,
        config_path: str = "breaker_labels_config.jsonc",
        prefer_gpu: bool = True,
        min_free_mb: int = 600,
    ):
        self.debug = debug
        self.reader = None
        self._ocr_dbg_items: List[dict] = []
        self._ocr_dbg_rois: List[Tuple[int,int,int,int]] = []

        # header/footer state
        self._hdr_text_y: Optional[int] = None
        self._hdr_rule_y: Optional[int] = None
        self._ftr_text_y: Optional[int] = None
        self._ftr_rule_y: Optional[int] = None

        # legacy fields preserved
        self._last_footer_y: Optional[int] = None
        self._footer_struct_y: Optional[int] = None
        self._footer_ocr_y: Optional[int] = None
        self._last_gridless_path: Optional[str] = None
        self._ocr_footer_marks: List[Tuple[int, str]] = []
        self._v_grid_xs: List[int] = []

        cfg_path = _abs_repo_cfg(config_path)
        self._cfg = _load_jsonc(cfg_path) or {}

        if debug:
            print(f"[BreakerTableAnalyzer] config_path = {os.path.abspath(cfg_path)} "
                  f"(loaded={self._cfg is not None})")

        # Build EasyOCR reader with GPU preference and VRAM safety
        self.reader = self._make_easyocr_reader(prefer_gpu=prefer_gpu, min_free_mb=min_free_mb)

    # ---------------- GPU-aware EasyOCR factory ----------------
    @staticmethod
    def _make_easyocr_reader(prefer_gpu: bool = True, min_free_mb: int = 600):
        try:
            import torch
            gpu_ok = False
            if prefer_gpu and torch.cuda.is_available():
                try:
                    free, total = torch.cuda.mem_get_info()  # bytes
                    gpu_ok = (free // (1024 * 1024)) >= int(min_free_mb)
                except Exception:
                    gpu_ok = True  # optimistic try
            import easyocr
            try:
                return easyocr.Reader(['en'], gpu=bool(gpu_ok))
            except Exception:
                return easyocr.Reader(['en'], gpu=False)
        except Exception:
            return None

    # ================= public =================
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

        # reset per-run
        self._ocr_footer_marks = []

        centers, dbg, header_y = self._row_centers_from_lines(gray)

        # prefer corrected spaces if available
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
                "gridless_gray": gray, "gridless_path": None, "page_overlay_path": page_overlay_path,
                "spaces_detected": spaces_detected,
                "spaces_corrected": spaces_corrected,
                "footer_snapped": getattr(self, "_footer_snapped", False),
                "snap_note": getattr(self, "_snap_note", None),
                "v_grid_xs": getattr(self, "_v_grid_xs", []),
                "column_overlay_path": column_overlay_path,
                # expose text/rule y for debugging
                "header_text_y": self._hdr_text_y, "footer_text_y": self._ftr_text_y,
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
            print(f"[ANALYZER] Header text/rule Y: {self._hdr_text_y} / {header_y}")
            print(f"[ANALYZER] Footer text/rule Y: {self._ftr_text_y} / {footer_y}")
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
            "header_text_y": self._hdr_text_y, "footer_text_y": self._ftr_text_y,
        }

    # ================= internals =================
    def _prep(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        H, W = g.shape
        if H < 1600:
            s = 1600.0 / H
            g = cv2.resize(g, (int(W*s), int(H*s)), interpolation=cv2.INTER_CUBIC)
        return g

    # ---- OCR token helpers (shared by header/footer) ----
    @staticmethod
    def _N(s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", (s or "").upper().replace("1", "I").replace("0", "O"))

    def _ocr_tokens_in_band(self, gray: np.ndarray, y1f: float, y2f: float,
                            allowlist: str, mags=(1.6, 2.0)) -> List[Tuple[List[Tuple[int,int]], str, float, int]]:
        """
        Returns a list of (box_pts_abs, text, conf, y_top_abs) for tokens found
        in the [y1f, y2f] vertical fraction of the image.
        """
        if self.reader is None:
            return []
        H, W = gray.shape
        y1, y2 = int(max(0, y1f * H)), int(min(H, y2f * H))
        roi = gray[y1:y2, :]
        if roi.size == 0:
            return []

        det = []
        for m in mags:
            try:
                part = self.reader.readtext(
                    roi, detail=1, paragraph=False,
                    allowlist=allowlist,
                    mag_ratio=m, contrast_ths=0.05, adjust_contrast=0.7,
                    text_threshold=0.4, low_text=0.25
                )
            except Exception:
                part = []
            det += part

        out = []
        for box, txt, conf in det:
            if not txt:
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            box_abs = [(x, y + y1) for x, y in zip(xs, ys)]
            out.append((box_abs, txt, float(conf or 0.0), int(min(ys) + y1)))
        return out

    # ---- binarization + horiz mask (in-memory only; no files) ----
    @staticmethod
    def _binarize_ink(gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bw_inv  # white=ink

    @staticmethod
    def _horiz_mask_from_bin(bw_inv: np.ndarray, W: int, horiz_kernel_frac=0.035) -> np.ndarray:
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

    @staticmethod
    def _extract_horiz_lines_from_mask(hmask: np.ndarray, W: int,
                                       min_width_frac=0.25, max_thickness_px=6,
                                       y_merge_tol_px=4):
        cnts, _ = cv2.findContours(hmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_w = int(round(W * min_width_frac))
        raws=[]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w < min_w or h < 1 or h > max_thickness_px:
                continue
            y_mid = y + h//2
            row = hmask[y_mid, :]
            xs = np.where(row > 0)[0]
            if xs.size == 0: continue
            xL, xR = int(xs.min()), int(xs.max())
            if (xR - xL + 1) < min_w:
                continue
            raws.append((y, y+h-1, xL, xR))

        if not raws:
            return []

        raws.sort(key=lambda r: (r[0]+r[1]) / 2.0)
        merged, cur = [], [raws[0]]
        for r in raws[1:]:
            y0,y1,_,_=r
            py0,py1,_,_=cur[-1]
            if abs(((y0+y1)/2.0) - ((py0+py1)/2.0)) <= y_merge_tol_px:
                cur.append(r)
            else:
                merged.append(cur); cur=[r]
        merged.append(cur)

        lines=[]
        for grp in merged:
            y_t = min(g[0] for g in grp)
            y_b = max(g[1] for g in grp)
            x_l = min(g[2] for g in grp)
            x_r = max(g[3] for g in grp)
            y_c = (y_t + y_b)//2
            lines.append({
                "y_center": int(y_c),
                "y_top":    int(y_t),
                "y_bottom": int(y_b),
                "x_left":   int(x_l),
                "x_right":  int(x_r),
                "width":    int(x_r - x_l + 1),
                "height":   int(max(1, y_b - y_t + 1)),
            })
        lines.sort(key=lambda d: d["y_center"])
        return lines

    @staticmethod
    def _pick_nearest_line(lines: List[dict], ref_y: Optional[int],
                           prefer_above: Optional[bool] = None,
                           gap_min: int = 2, search_px: Optional[int] = None) -> Optional[int]:
        """
        Pick the line closest to ref_y. If prefer_above is True/False,
        restrict to above/below. If None, pick absolute nearest.
        Optionally bound by a search window (±search_px).
        """
        if ref_y is None or not lines:
            return None
        ref_y = int(ref_y)

        cands = lines
        if prefer_above is True:
            cands = [ln for ln in cands if ln["y_center"] <= ref_y - gap_min]
        elif prefer_above is False:
            cands = [ln for ln in cands if ln["y_center"] >= ref_y + gap_min]

        if search_px is not None:
            lo, hi = ref_y - search_px, ref_y + search_px
            cands = [ln for ln in cands if lo <= ln["y_center"] <= hi]

        if not cands:
            return None

        best = min(cands, key=lambda ln: abs(ln["y_center"] - ref_y))
        return int(best["y_center"])

    # ---- HEADER via tokens + nearest white band ----
    def _find_header_rule(self, gray: np.ndarray) -> Optional[int]:
        """
        1) OCR tokens in upper/middle band (0.08..0.65H).
        2) Determine a representative header text y (topmost row of tokens that
           contains typical header categories).
        3) Build binarized IN-MEMORY horiz mask; pick nearest horizontal band to that y.
           We **do not** write any mask images here.
        """
        if self.reader is None:
            return None

        H, W = gray.shape
        # OCR tokens
        allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/"
        det = self._ocr_tokens_in_band(gray, 0.08, 0.65, allow)

        CATEGORY = {
            "CKT","CCT","TRIP","POLES","AMP","AMPS","BREAKER","BKR","SIZE",
            "DESCRIPTION","DESIGNATION","NAME","LOADDESCRIPTION","CIRCUITDESCRIPTION"
        }

        # Gather tokens that look like header labels
        header_like = []
        for box_abs, txt, conf, y_top in det:
            n = self._N(txt)
            if not n: 
                continue
            if any(k in n for k in CATEGORY):
                header_like.append((y_top, txt))

        if not header_like:
            self._hdr_text_y = None
            self._hdr_rule_y = None
            return None

        # Representative header text line: pick the TOPMOST such token row
        hdr_text_y = int(min(y for y,_ in header_like))
        self._hdr_text_y = hdr_text_y

        # Build in-memory horizontal mask
        bw_inv = self._binarize_ink(gray)
        hmask  = self._horiz_mask_from_bin(bw_inv, W, horiz_kernel_frac=0.035)
        lines  = self._extract_horiz_lines_from_mask(hmask, W,
                                                     min_width_frac=0.25,
                                                     max_thickness_px=6,
                                                     y_merge_tol_px=4)

        # Prefer the line **just above** the header text (typical drawings),
        # but if none, allow nearest (either side).
        rule_y = self._pick_nearest_line(lines, hdr_text_y, prefer_above=True, gap_min=2, search_px=120)
        if rule_y is None:
            rule_y = self._pick_nearest_line(lines, hdr_text_y, prefer_above=None, gap_min=2, search_px=200)

        self._hdr_rule_y = rule_y
        return rule_y

    # ---- FOOTER via tokens + nearest white band ----
    def _find_footer_rule(self, gray: np.ndarray) -> Optional[int]:
        """
        Footer OCR cues (prefer 'TOTAL LOAD', else guarded 'LOAD'), then pick the
        nearest horizontal band to the footer text line (usually the rule just above it).
        """
        if self.reader is None:
            return None

        H, W = gray.shape
        y0f, y1f = 0.45, 0.95
        allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :/.-"
        det = self._ocr_tokens_in_band(gray, y0f, y1f, allow, mags=(1.6, 2.0))

        total_hits = []
        load_hits  = []
        for box_abs, txt, conf, y_top in det:
            n = self._N(txt)
            if not n:
                continue
            sim_total = difflib.SequenceMatcher(a=n, b="TOTALLOAD").ratio()
            is_total = (("TOTAL" in n) and ("LOAD" in n)) or (sim_total >= 0.70)
            if is_total:
                total_hits.append(int(y_top))
            elif ("LOAD" in n) and ("CLASS" not in n):
                load_hits.append(int(y_top))

        hits = total_hits if total_hits else load_hits
        if not hits:
            self._ftr_text_y = None
            self._ftr_rule_y = None
            return None

        ftr_text_y = int(min(hits))  # topmost hit near the table body
        self._ftr_text_y = ftr_text_y

        # In-memory horizontal mask (same as header)
        bw_inv = self._binarize_ink(gray)
        hmask  = self._horiz_mask_from_bin(bw_inv, W, horiz_kernel_frac=0.035)
        lines  = self._extract_horiz_lines_from_mask(hmask, W,
                                                     min_width_frac=0.25,
                                                     max_thickness_px=6,
                                                     y_merge_tol_px=4)

        # Prefer the line **just above** the footer text
        rule_y = self._pick_nearest_line(lines, ftr_text_y, prefer_above=True, gap_min=2, search_px=160)
        if rule_y is None:
            rule_y = self._pick_nearest_line(lines, ftr_text_y, prefer_above=None, gap_min=2, search_px=220)

        self._ftr_rule_y = rule_y
        return rule_y

    # ---------- Structural row detection (kept) + header/footer integration ----------
    def _row_centers_from_lines(self, gray: np.ndarray):
        """
        Structural row detection (unchanged core) +
        header/footer now sourced from OCR tokens + nearest horizontal white band.
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
            # clip bottom quiet zone
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

        # Fallback to Hough if needed
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

        # ---- NEW: header/footer via OCR tokens + nearest bands ----
        header_y_abs = self._find_header_rule(gray)
        if header_y_abs is not None and header_y_abs > int(0.65 * H):
            header_y_abs = None  # safety cap

        # PRUNE by header / cadence (unchanged)
        header_loc = None if header_y_abs is None else max(0, header_y_abs - y_top)

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

        # Large-gap stop (unchanged)
        if len(tail) >= 2:
            gaps = np.diff(tail); med = float(np.median(gaps)) if len(gaps) else 0
            stop_rel = None
            for j,g in enumerate(gaps):
                if med>0 and g > max(1.9*med, med+18):
                    stop_rel = j; break
            if stop_rel is not None:
                tail = tail[: stop_rel + 1]

        # Convert to absolute y & build centers (unchanged)
        borders_abs = [b + y_top for b in tail]
        if len(borders_abs) < 2:
            self._last_footer_y = borders_abs[-1] if borders_abs else None
            self._footer_struct_y = self._last_footer_y
            self._footer_ocr_y = None
            self._bottom_border_y = self._last_footer_y
            self._bottom_row_center_y = None
            # spaces trail
            self._spaces_detected = 0
            self._spaces_corrected = 0
            self._footer_snapped = False
            self._snap_note = None
            return [], _Dbg(lines=[b for b in borders_abs], centers=[]), header_y_abs

        centers = []
        for i in range(len(borders_abs)-1):
            a, b = borders_abs[i], borders_abs[i+1]
            if b - a >= 12:
                centers.append(int((a + b) // 2))
        out=[]
        for y in centers:
            if out and abs(y-out[-1])<8: continue
            out.append(y)

        # Structural footer baseline (unchanged)
        footer_struct_y = borders_abs[-1] if len(borders_abs) >= 1 else None
        self._footer_struct_y = footer_struct_y

        # ---- NEW: footer via OCR tokens + nearest white band ----
        footer_ocr_y = self._find_footer_rule(gray)
        self._footer_ocr_y = footer_ocr_y

        # Arbitration: prefer OCR footer when available; else structural
        final_footer = footer_ocr_y if footer_ocr_y is not None else footer_struct_y

        # Guard: footer must be below header
        if (final_footer is not None) and (header_y_abs is not None) and (final_footer <= header_y_abs + 6):
            final_footer = None

        # (Keep prior spaces bookkeeping minimal)
        detected_spaces = len(out) * 2
        self._spaces_detected = detected_spaces
        self._spaces_corrected = detected_spaces
        self._footer_snapped = False
        self._snap_note = None

        self._last_footer_y = int(final_footer) if final_footer is not None else None
        self._bottom_border_y = footer_struct_y
        self._bottom_row_center_y = out[-1] if out else None

        dbg = _Dbg(lines=borders_abs, centers=out)
        return out, dbg, header_y_abs

    # ----------------- grid removal (unchanged behavior) -----------------
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

        # Vertical
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

        # Horizontal
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

    # ----------------- debug overlays (unchanged) -----------------
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

        # header (RULE)
        if header_y_abs is not None:
            y = int(header_y_abs)
            cv2.line(vis, (0, y), (W - 1, y), (0, 220, 0), 2)
            cv2.putText(vis, "HEADER_RULE", (12, max(16, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)
        if self._hdr_text_y is not None:
            y = int(self._hdr_text_y)
            cv2.line(vis, (0, y), (W - 1, y), (0, 150, 0), 1)
            cv2.putText(vis, "HEADER_TEXT", (140, max(16, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1, cv2.LINE_AA)

        # footer (RULE)
        if footer_y_abs is not None:
            y = int(footer_y_abs)
            cv2.line(vis, (0, y), (W - 1, y), (0, 0, 255), 2)
            cv2.putText(vis, "FOOTER_RULE", (12, max(16, y + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if self._ftr_text_y is not None:
            y = int(self._ftr_text_y)
            cv2.line(vis, (0, y), (W - 1, y), (180, 0, 0), 1)
            cv2.putText(vis, "FOOTER_TEXT", (140, max(16, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1, cv2.LINE_AA)

        # OCR cue markers (kept for completeness)
        for y_mark, label in getattr(self, "_ocr_footer_marks", []):
            y = int(y_mark)
            cv2.line(vis, (0, y), (W - 1, y), (200, 50, 200), 1)
            cv2.putText(vis, label, (240, max(14, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 50, 200), 1, cv2.LINE_AA)

        # spaces/snap trail
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
