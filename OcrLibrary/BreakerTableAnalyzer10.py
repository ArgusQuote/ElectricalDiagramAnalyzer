# OcrLibrary/BreakerTableAnalyzer9.py
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

import sys

# ---------------- PATH SETUP (optional) ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# NEW: header/footer anchoring classes
from AnchoringClasses.BreakerHeaderFinder import BreakerHeaderFinder, HeaderResult
from AnchoringClasses.BreakerFooterFinder import BreakerFooterFinder, FooterResult

# Version: TP band location removed. Hybrid footer logic (Total Load/Load + structural).
ANALYZER_VERSION = "Analyzer9"
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
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^A-Z0-9/\[\] \-]", "", (s or "").upper())
    ).strip()


def _str_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_norm_token(a), b=_norm_token(b)).ratio()


class BreakerTableAnalyzer:
    """
    Stage 1 analyzer (TP bands removed):
      - image prep
      - HEADER / FIRST BREAKER row / row centers via BreakerHeaderFinder
      - HYBRID footer:
          primary  : token-based footer via BreakerFooterFinder
          fallback : structural footer from header finder
      - remove gridlines (Gridless_Images/)
      - optional overlay to <debug_root_dir> or <src_dir>/debug

    Outputs a dict payload consumed by the parser (second stage).
    """

    def __init__(
        self,
        debug: bool = False,
        config_path: str = "breaker_labels_config.jsonc",
        # knobs for footer finder
        bottom_trim_frac: float = 0.15,
        top_trim_frac: float = 0.50,
        upscale_factor: float = 1.0,
    ):
        self.debug = debug
        self.reader = None

        # OCR debug state (header/CCT/CKT)
        self._ocr_dbg_items: List[dict] = []
        self._ocr_dbg_rois: List[Tuple[int, int, int, int]] = []

        # footer debug marks (y_abs, label)
        self._ocr_footer_marks: List[Tuple[int, str]] = []

        # last chosen footer (final)
        self._last_footer_y: Optional[int] = None
        # structural footer from row cadence
        self._footer_struct_y: Optional[int] = None

        # row/spacing stats
        self._spaces_detected: int = 0
        self._spaces_corrected: int = 0
        self._footer_snapped: bool = False
        self._snap_note: Optional[str] = None
        self._bottom_border_y: Optional[int] = None
        self._bottom_row_center_y: Optional[int] = None

        # vertical grid x positions (from degridding)
        self._v_grid_xs: List[int] = []

        self._last_gridless_path: Optional[str] = None

        # footer token info (from footer finder)
        self._footer_token_y: Optional[int] = None
        self._footer_token_val: Optional[int] = None

        # External debug root; if set, all overlays go there
        self.debug_root_dir: Optional[str] = None

        cfg_path = _abs_repo_cfg(config_path)
        self._cfg = _load_jsonc(cfg_path) or {}

        if debug:
            print(
                f"[BreakerTableAnalyzer] config_path = {os.path.abspath(cfg_path)} "
                f"(loaded={self._cfg is not None})"
            )

        # OCR reader
        if _HAS_OCR:
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)

        # anchoring classes (header + footer)
        self._header_finder = BreakerHeaderFinder(self.reader, debug=self.debug)
        self._footer_finder = BreakerFooterFinder(
            self.reader,
            bottom_trim_frac=bottom_trim_frac,
            top_trim_frac=top_trim_frac,
            upscale_factor=upscale_factor,
            debug=self.debug,
        )

    # ============== public ==============
    def analyze(self, image_path: str) -> Dict:
        src_path = os.path.abspath(os.path.expanduser(image_path))
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        src_dir = os.path.dirname(src_path)

        # Choose debug directory:
        # - if debug_root_dir is set, use that
        # - otherwise fall back to <src_dir>/debug
        if self.debug_root_dir:
            debug_dir = os.path.abspath(self.debug_root_dir)
        else:
            debug_dir = os.path.join(src_dir, "debug")

        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)
            # footer finder debug output (vertical_mask, col crops) goes here
            self._footer_finder.debug_dir = debug_dir
        else:
            self._footer_finder.debug_dir = None

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src_path)

        gray = self._prep(img)

        # reset per-run debug state
        self._ocr_footer_marks = []
        self._v_grid_xs = []
        self._footer_token_y = None
        self._footer_token_val = None

        # ---------- HEADER + ROW STRUCTURE VIA HEADER FINDER ----------
        header_res: HeaderResult = self._header_finder.analyze_rows(gray)

        centers  = header_res.centers
        dbg      = header_res.dbg
        header_y = header_res.header_y

        # OCR debug items (includes CCT/CKT tokens)
        self._ocr_dbg_items = self._header_finder.ocr_dbg_items
        self._ocr_dbg_rois  = self._header_finder.ocr_dbg_rois

        # prefer corrected spaces if available
        spaces_detected  = header_res.spaces_detected or (len(centers) * 2)
        spaces_corrected = header_res.spaces_corrected or spaces_detected
        spaces           = spaces_corrected

        # structural footer from header finder
        self._footer_struct_y     = header_res.footer_struct_y
        self._bottom_border_y     = header_res.bottom_border_y
        self._bottom_row_center_y = header_res.bottom_row_center_y
        self._spaces_detected     = header_res.spaces_detected
        self._spaces_corrected    = header_res.spaces_corrected
        self._footer_snapped      = header_res.footer_snapped
        self._snap_note           = header_res.snap_note

        # ---------- FOOTER VIA FOOTER FINDER (PRIMARY) ----------
        footer_res: FooterResult = self._footer_finder.find_footer(
            gray=gray,
            header_y=header_y,
            centers=centers,
            dbg=dbg,
            ocr_dbg_items=self._ocr_dbg_items,
            footer_struct_y=self._footer_struct_y,  # fallback only
            orig_bgr=img,                           # source image for crops
        )

        footer_y = footer_res.footer_y
        self._last_footer_y = footer_y
        self._ocr_footer_marks = footer_res.dbg_marks
        self._footer_token_y   = footer_res.token_y
        self._footer_token_val = footer_res.token_val

        # ---------- OVERLAY PATHS ----------
        page_overlay_path = None
        column_overlay_path = None

        # Early-exit safety
        if not centers or header_y is None or footer_y is None:
            if self.debug:
                print(
                    f"[ANALYZER] Early return: centers={len(centers)} "
                    f"header={header_y} footer={footer_y}"
                )
            return {
                "src_path": src_path,
                "src_dir": src_dir,
                "debug_dir": debug_dir,
                "gray": gray,
                "centers": centers,
                "row_count": len(centers),
                "spaces": spaces,
                "header_y": header_y,
                "footer_y": footer_y,
                "tp_cols": {},
                "tp_combined": {"left": False, "right": False},
                "gridless_gray": gray,
                "gridless_path": None,
                "page_overlay_path": page_overlay_path,
                "spaces_detected": spaces_detected,
                "spaces_corrected": spaces_corrected,
                "footer_snapped": getattr(self, "_footer_snapped", False),
                "snap_note": getattr(self, "_snap_note", None),
                "v_grid_xs": getattr(self, "_v_grid_xs", []),
                "column_overlay_path": column_overlay_path,
            }

        # ---------- GRID REMOVAL ----------
        gridless_gray, gridless_path = self._remove_grids_and_save(
            gray, header_y, footer_y, src_path
        )

        # ---------- DEBUG OVERLAYS ----------
        if self.debug:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(src_path))[0]

            # page overlay (rows/header/footer/etc.)
            page_overlay_path = os.path.join(
                debug_dir, f"{base}_page_overlay_{ts}.png"
            )
            self._save_debug(
                gray,
                centers,
                dbg.lines if hasattr(dbg, "lines") else [],
                page_overlay_path,
                header_y,
                footer_y,
            )

            # columns overlay (vertical grid xs from grid removal)
            column_overlay_path = os.path.join(
                debug_dir, f"{base}_columns_{ts}.png"
            )
            self._save_column_overlay(gray, column_overlay_path)

            print(f"[ANALYZER] Row count: {len(centers)} | Spaces: {spaces}")
            print(f"[ANALYZER] Header Y: {header_y} | Footer Y: {footer_y}")
            if getattr(self, "_snap_note", None):
                print(f"[ANALYZER] {self._snap_note}")
            print(f"[ANALYZER] Gridless : {gridless_path}")
            if page_overlay_path:
                print(f"[ANALYZER] Overlay  : {page_overlay_path}")

        # ---------- RETURN PAYLOAD ----------
        return {
            "src_path": src_path,
            "src_dir": src_dir,
            "debug_dir": debug_dir,
            "gray": gray,
            "centers": centers,
            "row_count": len(centers),
            "spaces": spaces,
            "header_y": header_y,
            "footer_y": footer_y,
            "tp_cols": {},
            "tp_combined": {"left": False, "right": False},
            "gridless_gray": gridless_gray,
            "gridless_path": gridless_path,
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
            g = cv2.resize(
                g,
                (int(W * s), int(H * s)),
                interpolation=cv2.INTER_CUBIC,
            )
        return g

    def _remove_grids_and_save(
        self, gray: np.ndarray, header_y: int, footer_y: int, image_path: str
    ):
        """
        Remove only the long structural table lines (horizontal and vertical),
        leaving smaller strokes and character edges intact.

        - Work only between header_y and footer_y (with a small margin).
        - Detect candidate lines with morphology.
        - Run connected-components on those candidates.
        - Keep only components that are long and skinny (true grid lines).
        - Build a thin inpaint mask from those components and inpaint.
        """
        H, W = gray.shape
        work = gray.copy()
        # reset vertical grid cache for this run
        self._v_grid_xs = []

        # Safety guard: if header/footer are weird, just skip degridding
        if header_y is None or footer_y is None or footer_y <= header_y + 10:
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = os.path.dirname(image_path)
            gridless_dir = os.path.join(out_dir, "Gridless_Images")
            os.makedirs(gridless_dir, exist_ok=True)
            gridless_path = os.path.join(
                gridless_dir, f"{base}_gridless.png"
            )
            cv2.imwrite(gridless_path, work)
            self._last_gridless_path = gridless_path
            return work, gridless_path

        # Restrict to the breaker table band
        y1 = max(0, int(header_y) - 4)
        y2 = min(H - 1, int(footer_y) + 4)
        if y2 <= y1:
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = os.path.dirname(image_path)
            gridless_dir = os.path.join(out_dir, "Gridless_Images")
            os.makedirs(gridless_dir, exist_ok=True)
            gridless_path = os.path.join(
                gridless_dir, f"{base}_gridless.png"
            )
            cv2.imwrite(gridless_path, work)
            self._last_gridless_path = gridless_path
            return work, gridless_path

        roi = work[y1:y2, :]

        # --- binarize ROI ---
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

        # --- detect candidate vertical and horizontal lines via morphology ---
        Kv = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(25, int(0.015 * roi_h))),
        )
        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(40, int(0.12 * W)), 1),
        )
        v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

        # --- filter to keep only LONG, SKINNY components (true grid lines) ---

        # Vertical line mask
        v_mask = np.zeros_like(v_candidates, dtype=np.uint8)
        v_x_centers: List[int] = []

        num_v, labels_v, stats_v, _ = cv2.connectedComponentsWithStats(
            v_candidates, connectivity=8
        )
        if num_v > 1:
            min_vert_len = int(0.40 * roi_h)
            max_vert_thick = max(2, int(0.02 * W))
            for i in range(1, num_v):
                x, y, w, h, area = stats_v[i]
                if h >= min_vert_len and w <= max_vert_thick and area > 0:
                    v_mask[labels_v == i] = 255
                    x_center = x + w // 2
                    v_x_centers.append(int(x_center))

        # collapse near-duplicate verticals
        if v_x_centers:
            xs = sorted(v_x_centers)
            collapsed = []
            for x in xs:
                if not collapsed or abs(x - collapsed[-1]) > 4:
                    collapsed.append(x)
            self._v_grid_xs = collapsed
        else:
            self._v_grid_xs = []

        # Horizontal line mask
        h_mask = np.zeros_like(h_candidates, dtype=np.uint8)
        num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(
            h_candidates, connectivity=8
        )
        if num_h > 1:
            min_horiz_len = int(0.40 * W)
            max_horiz_thick = max(2, int(0.02 * roi_h))
            for i in range(1, num_h):
                x, y, w, h, area = stats_h[i]
                if w >= min_horiz_len and h <= max_horiz_thick and area > 0:
                    h_mask[labels_h == i] = 255

        # Combine structural lines
        grid = cv2.bitwise_or(v_mask, h_mask)

        # If nothing was detected, just save original and return
        if grid.sum() == 0:
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_dir = os.path.dirname(image_path)
            gridless_dir = os.path.join(out_dir, "Gridless_Images")
            os.makedirs(gridless_dir, exist_ok=True)
            gridless_path = os.path.join(
                gridless_dir, f"{base}_gridless.png"
            )
            cv2.imwrite(gridless_path, work)
            self._last_gridless_path = gridless_path
            return work, gridless_path

        # Thin dilation: make sure we cover the center of the line,
        # but don't eat letters
        grid_mask = cv2.dilate(
            grid,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        # --- inpaint only those long lines ---
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        inpainted = cv2.inpaint(roi_bgr, grid_mask, 2, cv2.INPAINT_TELEA)
        clean_roi_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)

        # Write cleaned ROI back into the full page
        work[y1:y2, :] = clean_roi_gray

        # Save gridless output
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.dirname(image_path)
        gridless_dir = os.path.join(out_dir, "Gridless_Images")
        os.makedirs(gridless_dir, exist_ok=True)

        gridless_path = os.path.join(
            gridless_dir, f"{base}_gridless.png"
        )
        ok = cv2.imwrite(gridless_path, work)
        if not ok:
            raise IOError(
                f"Failed to write gridless image to: {gridless_path}"
            )

        self._last_gridless_path = gridless_path
        return work, gridless_path

    def _save_column_overlay(self, gray: np.ndarray, out_path: str):
        """
        Debug overlay that ONLY shows inferred column grid lines (v_grid_xs).
        """
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        H, W = gray.shape

        v_xs = getattr(self, "_v_grid_xs", [])
        for x in v_xs:
            xi = int(x)
            cv2.line(vis, (xi, 0), (xi, H - 1), (255, 0, 255), 1)
            cv2.putText(
                vis,
                "COL",
                (xi + 2, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, vis)
        except Exception:
            pass

    # ---------- helper: recompute verticals + CCT/CKT columns for overlay ----------
    def _compute_cct_columns_for_debug(
        self,
        gray: np.ndarray,
        header_y_abs: Optional[int],
        footer_struct_y: Optional[int],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        For debug overlay only:
          - rebuild vertical-line mask between header and structural footer
          - use header OCR items (CCT/CKT) to find column spans

        Returns:
          (cct_column_spans, all_vertical_xs)
        """
        H, W = gray.shape
        if header_y_abs is None or footer_struct_y is None:
            return [], []

        if footer_struct_y <= header_y_abs + 10:
            return [], []

        y1 = max(0, int(header_y_abs) - 4)
        y2 = min(H - 1, int(footer_struct_y) + 4)
        if y2 <= y1:
            return [], []

        roi = gray[y1:y2, :]
        if roi.size == 0:
            return [], []

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

        v_x_centers: List[int] = []
        num_v, labels_v, stats_v, _ = cv2.connectedComponentsWithStats(
            v_candidates, connectivity=8,
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
            return [], []

        # collapse near-duplicates
        xs = sorted(v_x_centers)
        collapsed: List[int] = []
        for x in xs:
            if not collapsed or abs(x - collapsed[-1]) > 4:
                collapsed.append(x)

        vlines_x = collapsed

        # now use OCR CCT/CKT tokens to pick column spans
        def _norm(s: str) -> str:
            return re.sub(
                r"\s+",
                " ",
                re.sub(r"[^A-Z0-9/\[\] \-]", "", (s or "").upper()),
            ).strip()

        col_spans: set[Tuple[int, int]] = set()

        for item in self._ocr_dbg_items:
            txt = item.get("text", "")
            norm = _norm(txt)
            if norm not in ("CCT", "CKT"):
                continue

            x1 = int(item["x1"])
            x2 = int(item["x2"])

            left_candidates  = [vx for vx in vlines_x if vx <= x1]
            right_candidates = [vx for vx in vlines_x if vx >= x2]
            if not left_candidates or not right_candidates:
                continue

            xl = max(left_candidates)
            xr = min(right_candidates)
            if xr <= xl:
                continue

            col_spans.add((xl, xr))

        col_spans_sorted = sorted(col_spans, key=lambda p: p[0])
        return col_spans_sorted, vlines_x

    def _save_debug(
        self,
        gray: np.ndarray,
        centers: List[int],
        borders: List[int],
        out_path: str,
        header_y_abs: Optional[int],
        footer_y_abs: Optional[int],
    ):
        """
        Main page overlay with:
          1) Row borders + centers
          2) Header text line, HEADER, FIRST_BREAKER_LINE
          3) CCT/CKT column spans (vertical lines around tokens)
          4) Footer token line + footer line + structural footer (if different)
          5) Footer dbg marks (e.g. FOOTER_TOKEN_XX, FOOTER_LINE_STRUCT)
          6) Snap info / spaces info
        """
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        H, W = gray.shape

        # ------------------------------------------------------------------
        # 1) Row borders + centers
        # ------------------------------------------------------------------
        for y in borders:
            cv2.line(vis, (0, int(y)), (W - 1, int(y)), (0, 165, 255), 1)

        for y in centers:
            cv2.circle(vis, (24, int(y)), 4, (0, 255, 0), -1)

        # ------------------------------------------------------------------
        # 2) HEADER TEXT LINE, HEADER LINE, FIRST BREAKER LINE
        # ------------------------------------------------------------------
        first_breaker_line_y = None
        if header_y_abs is not None:
            hy = int(header_y_abs)

            # "header text" baseline (just above header rule)
            header_text_y = max(0, hy - 8)
            cv2.line(vis, (0, header_text_y), (W - 1, header_text_y), (0, 255, 255), 1)
            cv2.putText(
                vis,
                "HEADER_TEXT",
                (12, max(16, header_text_y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # actual header rule
            cv2.line(vis, (0, hy), (W - 1, hy), (255, 255, 0), 2)
            cv2.putText(
                vis,
                "HEADER",
                (12, max(16, hy + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # FIRST_BREAKER_LINE = first border > header
            for b in sorted(borders):
                if b > hy + 2:
                    first_breaker_line_y = int(b)
                    break

        if first_breaker_line_y is not None:
            y = int(first_breaker_line_y)
            cv2.line(vis, (0, y), (W - 1, y), (0, 165, 255), 2)
            cv2.putText(
                vis,
                "FIRST_BREAKER_LINE",
                (12, max(16, y + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------------
        # 3) CCT/CKT COLUMN SPANS (recomputed from OCR + verticals)
        # ------------------------------------------------------------------
        footer_struct = self._footer_struct_y if self._footer_struct_y is not None else footer_y_abs
        cct_cols, vlines_x = self._compute_cct_columns_for_debug(
            gray,
            header_y_abs,
            footer_struct,
        )

        # Show ALL vertical candidates (light bluish) for context
        for x in vlines_x:
            x = int(x)
            cv2.line(vis, (x, 0), (x, H - 1), (180, 180, 255), 1)

        # Highlight the specific CCT/CKT columns (magenta pairs)
        for idx, (xl, xr) in enumerate(cct_cols, start=1):
            xl = int(xl)
            xr = int(xr)
            cv2.line(vis, (xl, 0), (xl, H - 1), (255, 0, 255), 2)
            cv2.line(vis, (xr, 0), (xr, H - 1), (255, 0, 255), 2)

            label_y = 16
            if header_y_abs is not None:
                label_y = max(16, int(header_y_abs) - 6)

            cv2.putText(
                vis,
                f"CCT_COL{idx}",
                (xl + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------------
        # 4) FOOTER TEXT LINE (TOKEN) + FOOTER LINE + STRUCTURAL FOOTER
        # ------------------------------------------------------------------
        footer_token_y   = self._footer_token_y
        footer_token_val = self._footer_token_val

        # Token text baseline (footer text)
        if footer_token_y is not None:
            fty = int(footer_token_y)
            cv2.line(vis, (0, fty), (W - 1, fty), (200, 50, 200), 1)
            label = (
                f"FOOTER_TEXT ({footer_token_val})"
                if footer_token_val is not None
                else "FOOTER_TEXT"
            )
            cv2.putText(
                vis,
                label,
                (12, max(16, fty - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 50, 200),
                1,
                cv2.LINE_AA,
            )

            # Big token label in corner
            if footer_token_val is not None:
                cv2.putText(
                    vis,
                    f"FOOTER_TOKEN = {footer_token_val}",
                    (W - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # structural footer (if different from final, show in dark blue)
        if (
            self._footer_struct_y is not None
            and footer_y_abs is not None
            and abs(self._footer_struct_y - footer_y_abs) > 2
        ):
            ys = int(self._footer_struct_y)
            cv2.line(vis, (0, ys), (W - 1, ys), (180, 0, 0), 2)
            cv2.putText(
                vis,
                "FOOTER_STRUCT",
                (12, max(16, ys - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # final footer line
        if footer_y_abs is not None:
            y = int(footer_y_abs)
            cv2.line(vis, (0, y), (W - 1, y), (0, 0, 255), 2)
            cv2.putText(
                vis,
                "FOOTER",
                (12, max(16, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------------
        # 5) OCR cue markers from footer finder (dbg_marks)
        # ------------------------------------------------------------------
        for y_mark, label in getattr(self, "_ocr_footer_marks", []):
            y = int(y_mark)
            cv2.line(vis, (0, y), (W - 1, y), (160, 160, 255), 1)
            cv2.putText(
                vis,
                label,
                (240, max(14, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 160, 255),
                1,
                cv2.LINE_AA,
            )

        # ------------------------------------------------------------------
        # 6) snap trail + counts
        # ------------------------------------------------------------------
        note = getattr(self, "_snap_note", None)
        det = getattr(self, "_spaces_detected", None)
        cor = getattr(self, "_spaces_corrected", None)
        trail = []
        if det is not None and cor is not None:
            trail.append(
                f"spaces {det}→{cor}"
                if det != cor
                else f"spaces {det}"
            )
        if note:
            trail.append(note)
        if trail:
            cv2.putText(
                vis,
                " | ".join(trail),
                (12, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, vis)
        except Exception:
            pass
