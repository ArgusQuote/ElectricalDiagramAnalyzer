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
      - optional overlay to <src_dir>/debug

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
        debug_dir = os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)
            # only enable footer debug output when analyzer debug is on
            self._footer_finder.debug_dir = debug_dir
        else:
            # ensure we don't accidentally dump footer debug images
            self._footer_finder.debug_dir = None

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src_path)

        gray = self._prep(img)

        # reset per-run debug state
        self._ocr_footer_marks = []
        self._v_grid_xs = []

        # ---------- HEADER + ROW STRUCTURE VIA HEADER FINDER ----------
        header_res: HeaderResult = self._header_finder.analyze_rows(gray)

        centers  = header_res.centers
        dbg      = header_res.dbg
        header_y = header_res.header_y

        # expose OCR debug items to external code/dev tools
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
            orig_bgr=img,                           # NEW: pass original BGR for crops
        )

        footer_y = footer_res.footer_y
        self._last_footer_y = footer_y
        self._ocr_footer_marks = footer_res.dbg_marks

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

    def _save_debug(
        self,
        gray: np.ndarray,
        centers: List[int],
        borders: List[int],
        out_path: str,
        header_y_abs: Optional[int],
        footer_y_abs: Optional[int],
    ):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        H, W = gray.shape

        # borders
        for y in borders:
            cv2.line(vis, (0, int(y)), (W - 1, int(y)), (0, 165, 255), 1)

        # centers
        for y in centers:
            cv2.circle(vis, (24, int(y)), 4, (0, 255, 0), -1)

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

        # final footer
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

        # OCR cue markers (from footer finder)
        for y_mark, label in getattr(self, "_ocr_footer_marks", []):
            y = int(y_mark)
            cv2.line(vis, (0, y), (W - 1, y), (200, 50, 200), 1)
            cv2.putText(
                vis,
                label,
                (240, max(14, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 50, 200),
                1,
                cv2.LINE_AA,
            )

        # snap trail + counts
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
