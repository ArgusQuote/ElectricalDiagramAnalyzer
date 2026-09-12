# OcrLibrary/BreakerTableParser13.py
from __future__ import annotations
import os, re, cv2, numpy as np
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

PARSER_VERSION = "BreakerParser13"

_HDR_OCR_SCALE        = 2.0
_HDR_OCR_ALLOWLIST    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/().#"
_HDR_MIN_CONF         = 0.40
_OCR_TIMEOUT_SEC      = 30

# --- Body cell OCR prep (grid-line removal) tuning ---------------------------
#
# Body column crops are bounded by the table's own grid: the column x-bounds sit
# on vertical line centers and the row bands are inflated to the horizontal
# dividers, so border ink lands on the crop edges and gets fed to EasyOCR as if
# it were glyph ink ('1' vs '|', '0' vs '8').
#
# A body cell is a single column wide by one row tall, so any full-width
# horizontal run or edge-hugging full-height vertical run is table structure,
# never a character.
_CELL_PREP_MIN_SIDE_PX      = 6      # crops smaller than this are left untouched
_CELL_PREP_H_KERNEL_FR      = 0.55   # horizontal line kernel, fraction of cell width
_CELL_PREP_V_KERNEL_FR      = 0.65   # vertical line kernel, fraction of cell height
_CELL_PREP_H_MIN_LEN_FR     = 0.60   # keep horizontal runs at least this wide
_CELL_PREP_H_MAX_THICK_FR   = 0.22   # ...and no thicker than this (else it is a glyph)
_CELL_PREP_V_MIN_LEN_FR     = 0.75   # keep vertical runs at least this tall
_CELL_PREP_V_MAX_THICK_FR   = 0.10   # ...and no wider than this (digit strokes are wider)
_CELL_PREP_V_EDGE_ZONE_FR   = 0.15   # only near the left/right edges, where borders live
_CELL_PREP_INPAINT_RADIUS   = 2
_CELL_PREP_EDGE_INSET_X_FR  = 0.03   # residual border fringe shaved from left/right
_CELL_PREP_EDGE_INSET_Y_FR  = 0.04   # residual border fringe shaved from top/bottom
_CELL_PREP_BG_PERCENTILE    = 90     # background level used to fill the shaved fringe

from OcrLibrary.ocr_timeout import readtext_with_timeout as _readtext_with_timeout
 
def _prep_gray_like_analyzer12(src_path: str) -> Optional[np.ndarray]:
    """
    Recreate the same gray image that BreakerTableAnalyzer12 uses:

      - BGR -> gray
      - CLAHE
      - upscale to min height 1600 px

    This ensures header_y / footer_y coordinates still line up.
    """
    if not src_path:
        return None

    try:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

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
    except Exception:
        return None

def _ensure_gray(analyzer_result: Dict) -> Optional[np.ndarray]:
    """
    Make sure analyzer_result has a 'gray' image.

    - If 'gray' already exists, return it.
    - Otherwise, reconstruct it from src_path using the same prep as Analyzer12,
      store it back into analyzer_result['gray'], and return it.
    """
    gray = analyzer_result.get("gray")
    if gray is not None:
        return gray

    src_path = analyzer_result.get("src_path")
    gray = _prep_gray_like_analyzer12(src_path)
    if gray is not None:
        analyzer_result["gray"] = gray
    return gray

_ALLOWED_GFI_TOKENS = {"G", "GF", "GFI", "GFCI", "GFIC"}

def normalize_gfi_modifier(cell_text: str) -> str:
    """
    Returns:
      - "GFI" if cell_text contains a valid token: g, gf, gfi, gfci, gfic
      - "" otherwise

    Token-based (won't match substrings).
    """
    if not cell_text:
        return ""

    txt = str(cell_text).strip().upper()
    if not txt:
        return ""

    # tokens = contiguous letters only (splits on non-letters)
    tokens = re.findall(r"[A-Z]+", txt)
    for t in tokens:
        if t in _ALLOWED_GFI_TOKENS:
            return "GFI"

    return ""

def _is_blank_placeholder_text(text: str) -> bool:
    """
    Returns True when OCR text is only a blank-cell placeholder such as:

        -
        --
        ---
        _
        __
        —
        ––
        - - -
        .-
        -.

    Does not treat text containing letters or digits as blank.
    """
    if not text:
        return False

    value = str(text).strip()
    if not value:
        return False

    # Normalize common dash-like Unicode characters.
    for ch in (
        "\u2010",  # hyphen
        "\u2011",  # non-breaking hyphen
        "\u2012",  # figure dash
        "\u2013",  # en dash
        "\u2014",  # em dash
        "\u2015",  # horizontal bar
        "\u2212",  # minus sign
    ):
        value = value.replace(ch, "-")

    # Remove whitespace and harmless punctuation that commonly appears
    # when OCR reads a blank-cell dash.
    compact = re.sub(r"[\s.,:;]+", "", value)

    # OCR may read a dash as punctuation only.
    if not compact:
        return True

    # Only dash/underscore/pipe/slash-like placeholder marks remain.
    return bool(re.fullmatch(r"[-_/\\|]+", compact))

def _prep_cell_for_ocr(gray_crop: np.ndarray) -> np.ndarray:
    """
    Remove table grid lines from a single body cell crop before EasyOCR reads it.

    Body columns are cropped on the grid itself -- column x-bounds sit on
    vertical line centers and row bands are inflated out to the horizontal
    dividers -- so border ink lands at the crop edges and reaches EasyOCR as if
    it were glyph ink, which is the leading suspect for '1'/'|' and '0'/'8'
    confusions in trip and pole cells.

    Structural lines are inpainted rather than painted over so the local paper
    background is reconstructed and glyphs that touch a border keep their shape.

    The crop's width and height are deliberately never changed: the caller maps
    OCR boxes back to page coordinates through the crop origin and
    _HDR_OCR_SCALE, so a resize or offset here would corrupt that mapping. The
    residual edge fringe is filled with the background level instead of cropped
    for the same reason.

    Args:
        gray_crop: single-channel cell crop (one column wide, one row tall).

    Returns:
        A cleaned copy of the crop, or the input unchanged when it is too small
        to analyze or not a usable 2-D grayscale image.
    """
    if gray_crop is None or not hasattr(gray_crop, "shape"):
        return gray_crop

    if gray_crop.ndim != 2 or gray_crop.size == 0:
        return gray_crop

    h, w = gray_crop.shape[:2]
    if h < _CELL_PREP_MIN_SIDE_PX or w < _CELL_PREP_MIN_SIDE_PX:
        return gray_crop

    if gray_crop.dtype != np.uint8:
        return gray_crop

    blur = cv2.GaussianBlur(gray_crop, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(w * _CELL_PREP_H_KERNEL_FR)), 1),
    )
    horizontal_candidates = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(3, int(h * _CELL_PREP_V_KERNEL_FR))),
    )
    vertical_candidates = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )

    grid = np.zeros_like(bw, dtype=np.uint8)

    # Keep only long, skinny horizontal runs. A run spanning most of a cell's
    # width is a row divider; no character in a trip/poles cell is that wide.
    h_min_len = max(3, int(w * _CELL_PREP_H_MIN_LEN_FR))
    h_max_thick = max(1, int(h * _CELL_PREP_H_MAX_THICK_FR))

    num_h, labels_h, stats_h, _centroids_h = cv2.connectedComponentsWithStats(
        horizontal_candidates,
        connectivity=8,
    )
    for i in range(1, num_h):
        comp_w = int(stats_h[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats_h[i, cv2.CC_STAT_HEIGHT])
        if comp_w >= h_min_len and comp_h <= h_max_thick:
            grid[labels_h == i] = 255

    # Keep only tall, thin vertical runs that hug the left or right edge, where
    # the column borders are. Restricting by position protects centered digit
    # strokes ('1' in a tight row band is tall and thin too).
    v_min_len = max(3, int(h * _CELL_PREP_V_MIN_LEN_FR))
    v_max_thick = max(1, int(w * _CELL_PREP_V_MAX_THICK_FR))
    edge_zone = max(1, int(w * _CELL_PREP_V_EDGE_ZONE_FR))

    num_v, labels_v, stats_v, centroids_v = cv2.connectedComponentsWithStats(
        vertical_candidates,
        connectivity=8,
    )
    for i in range(1, num_v):
        comp_w = int(stats_v[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats_v[i, cv2.CC_STAT_HEIGHT])
        if comp_h < v_min_len or comp_w > v_max_thick:
            continue

        center_x = float(centroids_v[i][0])
        if center_x <= edge_zone or center_x >= (w - 1 - edge_zone):
            grid[labels_v == i] = 255

    if int(np.count_nonzero(grid)) > 0:
        # Thin dilation so the mask covers the line core and its anti-aliased
        # shoulders without eating neighbouring glyph pixels.
        grid_mask = cv2.dilate(
            grid,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        crop_bgr = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
        inpainted = cv2.inpaint(
            crop_bgr,
            grid_mask,
            _CELL_PREP_INPAINT_RADIUS,
            cv2.INPAINT_TELEA,
        )
        cleaned = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    else:
        cleaned = gray_crop.copy()

    # Flatten the outermost pixels, where sub-threshold border fringe survives
    # both the morphology filter and the inpaint.
    inset_x = max(1, int(round(w * _CELL_PREP_EDGE_INSET_X_FR)))
    inset_y = max(1, int(round(h * _CELL_PREP_EDGE_INSET_Y_FR)))

    if w > (inset_x * 2) + 2 and h > (inset_y * 2) + 2:
        background = int(np.percentile(cleaned, _CELL_PREP_BG_PERCENTILE))
        cleaned[:inset_y, :] = background
        cleaned[h - inset_y:, :] = background
        cleaned[:, :inset_x] = background
        cleaned[:, w - inset_x:] = background

    return cleaned

def _save_cell_prep_comparison(
    before_gray: np.ndarray,
    after_gray: np.ndarray,
    out_path: str,
) -> bool:
    """
    Write a BEFORE | AFTER image of one body column's cell prep so grid-line
    removal can be checked visually next to the token overlay for that column.

    Args:
        before_gray: the column strip as cropped from the page.
        after_gray:  the same strip with each row band's prepped cell written back.
        out_path:    destination PNG path.

    Returns:
        True when the file was written.
    """
    if before_gray is None or after_gray is None or not out_path:
        return False

    if before_gray.shape != after_gray.shape:
        return False

    try:
        left = cv2.cvtColor(before_gray, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(after_gray, cv2.COLOR_GRAY2BGR)

        separator = np.zeros((left.shape[0], 3, 3), dtype=np.uint8)
        separator[:, :, 1] = 255
        separator[:, :, 2] = 255

        vis = cv2.hconcat([left, separator, right])

        cv2.putText(vis, "BEFORE", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            "AFTER",
            (left.shape[1] + 7, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 160, 0),
            1,
            cv2.LINE_AA,
        )

        return bool(cv2.imwrite(out_path, vis))
    except Exception as exc:
        print(f"[{PARSER_VERSION}] Failed to write cell-prep comparison {out_path}: {exc}")
        return False

def _cell_has_visual_content(
    gray: np.ndarray,
    x_left: int,
    x_right: int,
    row_top: int,
    row_bottom: int,
) -> bool:
    """
    Decide whether a breaker cell contains meaningful visible marks even when
    OCR returns no text.

    This is used to distinguish:

      - genuinely empty cells -> no review overlay
      - obscured / marked-up / unreadable cells -> red review overlay

    Long horizontal and vertical grid lines are removed before measuring ink,
    so the table borders alone should not make an empty cell look occupied.
    """
    if gray is None or not hasattr(gray, "shape"):
        return False

    H, W = gray.shape[:2]

    x1 = max(0, min(W - 1, int(x_left)))
    x2 = max(x1 + 1, min(W, int(x_right)))
    y1 = max(0, min(H - 1, int(row_top)))
    y2 = max(y1 + 1, min(H, int(row_bottom)))

    if x2 <= x1 or y2 <= y1:
        return False

    crop = gray[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return False

    blur = cv2.GaussianBlur(crop, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )

    # Remove horizontal table/grid lines.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(w * 0.55)), 1),
    )
    horizontal_lines = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    # Remove vertical table/grid lines.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(3, int(h * 0.65))),
    )
    vertical_lines = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )

    cleaned = cv2.subtract(bw, horizontal_lines)
    cleaned = cv2.subtract(cleaned, vertical_lines)

    # Ignore a tiny edge margin where table borders may remain.
    edge_x = max(1, int(w * 0.04))
    edge_y = max(1, int(h * 0.06))

    if w > edge_x * 2 and h > edge_y * 2:
        cleaned = cleaned[
            edge_y:h - edge_y,
            edge_x:w - edge_x,
        ]

    if cleaned.size == 0:
        return False

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8,
    )

    meaningful_area = 0
    meaningful_components = 0

    # Skip component 0, which is the background.
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Ignore isolated noise pixels.
        if area < 3:
            continue

        # Ignore extremely thin leftover grid fragments.
        if comp_w <= 1 and comp_h >= max(4, int(h * 0.40)):
            continue
        if comp_h <= 1 and comp_w >= max(4, int(w * 0.40)):
            continue

        meaningful_area += area
        meaningful_components += 1

    crop_area = max(1, cleaned.shape[0] * cleaned.shape[1])
    ink_ratio = float(meaningful_area) / float(crop_area)

    # Either several visible fragments or enough total ink indicates that
    # something occupies the cell.
    return (
        meaningful_components >= 2
        or meaningful_area >= 8
        or ink_ratio >= 0.008
    )

def _detect_visual_blank_placeholder(
    gray: np.ndarray,
    x_left: int,
    x_right: int,
    row_top: int,
    row_bottom: int,
) -> Optional[str]:
    """
    Conservatively detect intentional blank-cell placeholder marks when OCR
    returned no usable text.

    Returns:
      - "dash" for clean dash / underscore-like placeholders
      - "dot" for clean dot placeholders
      - None for anything ambiguous, dense, tall, crossed, obscured, or noisy

    Safety rule:
      Anything that is not clearly a simple placeholder remains eligible for
      the existing red review overlay.
    """
    if gray is None or not hasattr(gray, "shape"):
        return None

    H, W = gray.shape[:2]

    x1 = max(0, min(W - 1, int(x_left)))
    x2 = max(x1 + 1, min(W, int(x_right)))
    y1 = max(0, min(H - 1, int(row_top)))
    y2 = max(y1 + 1, min(H, int(row_bottom)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = gray[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 5 or w < 5:
        return None

    blur = cv2.GaussianBlur(crop, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )

    # Remove full-width horizontal table lines.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, int(w * 0.65)), 1),
    )
    horizontal_lines = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1,
    )

    # Remove full-height vertical table lines.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(3, int(h * 0.70))),
    )
    vertical_lines = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1,
    )

    cleaned = cv2.subtract(bw, horizontal_lines)
    cleaned = cv2.subtract(cleaned, vertical_lines)

    # Ignore cell borders and residual grid fragments.
    edge_x = max(1, int(w * 0.06))
    edge_y = max(1, int(h * 0.08))

    if w <= edge_x * 2 or h <= edge_y * 2:
        return None

    cleaned = cleaned[
        edge_y:h - edge_y,
        edge_x:w - edge_x,
    ]

    if cleaned.size == 0:
        return None

    inner_h, inner_w = cleaned.shape[:2]
    inner_area = max(1, inner_h * inner_w)

    # Reconnect tiny horizontal gaps inside printed dashes.
    #
    # PDF rasterization, CLAHE, resizing, blur, and adaptive thresholding can
    # split one intentional dash into several nearby fragments. A narrow
    # horizontal closing kernel reconnects those fragments without broadly
    # joining tall, crossed, handwritten, or irregular marks.
    dash_join_width = max(
        3,
        min(7, int(round(inner_w * 0.025))),
    )

    dash_join_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (dash_join_width, 1),
    )

    cleaned_for_components = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        dash_join_kernel,
        iterations=1,
    )

    num_labels, _labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned_for_components,
            connectivity=8,
        )
    )

    components = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp_x = int(stats[i, cv2.CC_STAT_LEFT])
        comp_y = int(stats[i, cv2.CC_STAT_TOP])
        comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Ignore isolated threshold noise.
        if area < 2:
            continue

        # Ignore obvious remaining full grid fragments.
        if comp_w <= 1 and comp_h >= max(4, int(inner_h * 0.45)):
            continue

        if comp_h <= 1 and comp_w >= max(4, int(inner_w * 0.80)):
            continue

        center_x = float(centroids[i][0])
        center_y = float(centroids[i][1])

        components.append(
            {
                "area": area,
                "x": comp_x,
                "y": comp_y,
                "w": comp_w,
                "h": comp_h,
                "cx": center_x,
                "cy": center_y,
            }
        )

    # A placeholder should contain very few simple marks.
    if not components or len(components) > 3:
        return None

    total_area = sum(c["area"] for c in components)
    ink_ratio = float(total_area) / float(inner_area)

    # Anything dense remains a red-review candidate.
    if ink_ratio > 0.06:
        return None

    # Reject tall or large components. These are more likely to be numbers,
    # letters, handwriting, smudges, or an obstruction.
    max_component_height = max(3, int(inner_h * 0.25))
    max_component_area = max(8, int(inner_area * 0.035))

    for c in components:
        if c["h"] > max_component_height:
            return None

        if c["area"] > max_component_area:
            return None

    y_centers = [c["cy"] for c in components]

    # Multiple placeholder marks should sit on approximately one baseline.
    if len(y_centers) > 1:
        if max(y_centers) - min(y_centers) > max(3.0, inner_h * 0.18):
            return None

    # ------------------------------------------------------------
    # DOT PLACEHOLDER
    # Examples:
    #   .
    #   ..
    #   ...
    # ------------------------------------------------------------
    dots_only = True

    for c in components:
        max_dot_w = max(4, int(inner_w * 0.14))
        max_dot_h = max(4, int(inner_h * 0.20))

        aspect = float(c["w"]) / float(max(1, c["h"]))

        if c["w"] > max_dot_w:
            dots_only = False
            break

        if c["h"] > max_dot_h:
            dots_only = False
            break

        if aspect < 0.40 or aspect > 2.50:
            dots_only = False
            break

    if dots_only:
        # Require the marks to remain inside the main body of the cell.
        for c in components:
            if c["cx"] < inner_w * 0.08:
                return None

            if c["cx"] > inner_w * 0.92:
                return None

            if c["cy"] < inner_h * 0.20:
                return None

            if c["cy"] > inner_h * 0.90:
                return None

        return "dot"

    # ------------------------------------------------------------
    # DASH / UNDERSCORE PLACEHOLDER
    # Examples:
    #   -
    #   --
    #   ___
    # ------------------------------------------------------------
    dashes_only = True

    for c in components:
        aspect = float(c["w"]) / float(max(1, c["h"]))

        min_dash_width = max(3, int(inner_w * 0.05))
        max_dash_width = max(min_dash_width, int(inner_w * 0.60))
        max_dash_height = max(3, int(inner_h * 0.16))

        if c["w"] < min_dash_width:
            dashes_only = False
            break

        if c["w"] > max_dash_width:
            dashes_only = False
            break

        if c["h"] > max_dash_height:
            dashes_only = False
            break

        if aspect < 2.25:
            dashes_only = False
            break

    if dashes_only:
        left_edge = min(c["x"] for c in components)
        right_edge = max(c["x"] + c["w"] for c in components)
        total_span = right_edge - left_edge

        # A placeholder should not cover most of the cell.
        if total_span > inner_w * 0.75:
            return None

        # Reject marks pressed directly against the cell borders.
        if left_edge < inner_w * 0.03:
            return None

        if right_edge > inner_w * 0.97:
            return None

        return "dash"

    # Ambiguous shapes remain review-worthy.
    return None

class HeaderBandScanner:
    """
    First-stage helper: given analyzer_result with header_y, crop a tight band
    around the header line, up-res it, OCR everything in that band, and emit:
      - RAW cropped band image (no overlays) saved to debug/
      - overlay image with OCR boxes + text + confidence saved to debug/
      - raw OCR tokens (no decisions, no ranking)
    """

    def __init__(self, *, debug: bool = False, reader=None):
        self.debug = bool(debug)
        if reader is not None:
            self.reader = reader
        elif _HAS_OCR:
            try:
                self.reader = easyocr.Reader(["en"], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(["en"], gpu=False)
        else:
            self.reader = None

    def _ensure_debug_dir(self, analyzer_result: Dict) -> str:
        """Resolve and create the debug output directory from analyzer_result."""
        src_dir = analyzer_result.get("src_dir") or os.path.dirname(
            analyzer_result.get("src_path", "") or "."
        )
        debug_dir = analyzer_result.get("debug_dir") or os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)
        return debug_dir

    def scan(self, analyzer_result: Dict) -> Dict:
        """
        Inputs:
          analyzer_result: dict from BreakerTableAnalyzer.analyze()
            - expects keys: src_path, header_y, header_bottom_y
            - if 'gray' is missing, we will rebuild it from src_path
        """
        # --- get gray for OCR + line detection ---
        raw_gray = analyzer_result.get("gray")
        if raw_gray is None:
            raw_gray = _ensure_gray(analyzer_result)

        gray_ocr   = raw_gray
        gray_lines = raw_gray

        header_y        = analyzer_result.get("header_y")
        src_path        = analyzer_result.get("src_path")
        header_bottom_y = analyzer_result.get("header_bottom_y")

        H = W = None
        if gray_ocr is not None:
            H, W = gray_ocr.shape

        debug_dir = self._ensure_debug_dir(analyzer_result)
        debug_img_raw_path = None
        debug_img_overlay_path = None

        # If we don't have the basics, bail gracefully
        if gray_ocr is None or gray_lines is None or header_y is None or H is None:
            if self.debug:
                print("[HeaderBandScanner] Missing gray/header_y; skipping header scan.")
            return {
                "band_y1": None,
                "band_y2": None,
                "tokens": [],
                "debugImageRaw": None,
                "debugImageOverlay": None,
                "columnGroups": [],
                "normalizedColumns": {
                    "roles": {},
                    "columns": [],
                    "layout": "unknown",
                },
                "error": "Missing gray image or header_y; cannot scan header band.",
            }

        # --- 1) define the header band in page coordinates (MUST have valid header_bottom_y) ---
        if not isinstance(header_bottom_y, (int, float)) or not isinstance(header_y, (int, float)):
            if self.debug:
                print(
                    "[HeaderBandScanner] Missing header_bottom_y/header_y; "
                    "cannot determine header band. Bailing."
                )
                print(f"  header_y={header_y!r}, header_bottom_y={header_bottom_y!r}")
            return {
                "band_y1": None,
                "band_y2": None,
                "tokens": [],
                "debugImageRaw": None,
                "debugImageOverlay": None,
                "columnGroups": [],
                "normalizedColumns": {
                    "roles": {},
                    "columns": [],
                    "layout": "unknown",
                },
                "error": "Missing header_bottom_y from analyzer; cannot determine header band.",
            }

        y1 = max(0, int(header_y))
        y2 = min(H, int(header_bottom_y))

        # Require a non-trivial band height
        if y2 <= y1 + 4:
            if self.debug:
                print(
                    "[HeaderBandScanner] Header band too small or invalid: "
                    f"y1={y1}, y2={y2}, H={H}. Bailing."
                )
            return {
                "band_y1": None,
                "band_y2": None,
                "tokens": [],
                "debugImageRaw": None,
                "debugImageOverlay": None,
                "columnGroups": [],
                "normalizedColumns": {
                    "roles": {},
                    "columns": [],
                    "layout": "unknown",
                },
                "error": "Invalid header band height from analyzer; cannot scan header.",
            }

        band_ocr   = gray_ocr[y1:y2, :]
        band_lines = gray_lines[y1:y2, :]

        # --- 1b) save RAW cropped band to debug folder ---
        if self.debug:
            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]
            debug_img_raw_path = os.path.join(debug_dir, f"{base}_parser_header_band_raw.png")
            try:
                cv2.imwrite(debug_img_raw_path, band_ocr)
            except Exception as e:
                print(f"[HeaderBandScanner] Failed to write raw header band image: {e}")
                debug_img_raw_path = None

        # Always define these so there are no "referenced before assignment" issues
        tokens: List[Dict] = []
        column_groups: List[Dict] = []

        # --- 2) find header-local vertical lines (column dividers) ---
        header_v_cols = self._find_header_verticals(band_lines)
        v_cols = sorted(int(x) for x in header_v_cols)

        if self.debug:
            print(f"[HeaderBandScanner] header_v_cols (lines band) = {v_cols}")

        if self.reader is not None:
            H_band, W_band = band_ocr.shape[:2]

            if v_cols and len(v_cols) >= 2:
                # --- 2a) OCR per column strip between successive vertical lines ---
                for i in range(len(v_cols) - 1):
                    x_left = v_cols[i]
                    x_right = v_cols[i + 1]

                    # guard against degenerate / reversed intervals
                    if x_right <= x_left or x_left < 0 or x_right > W_band:
                        continue

                    # Initialize this column group up front
                    col_group = {
                        "index": len(column_groups) + 1,
                        "x_left": int(x_left),
                        "x_right": int(x_right),
                        "items": [],
                    }
                    column_groups.append(col_group)

                    # Crop this column strip from the OCR band
                    col_band = band_ocr[:, x_left:x_right]

                    if col_band.size == 0:
                        continue

                    # Up-res the column band
                    col_band_up = cv2.resize(
                        col_band,
                        None,
                        fx=_HDR_OCR_SCALE,
                        fy=_HDR_OCR_SCALE,
                        interpolation=cv2.INTER_CUBIC,
                    )

                    try:
                        dets = _readtext_with_timeout(
                            self.reader,
                            col_band_up,
                            detail=1,
                            paragraph=False,
                            allowlist=_HDR_OCR_ALLOWLIST,
                            mag_ratio=1.0,
                            contrast_ths=0.05,
                            adjust_contrast=0.7,
                            text_threshold=0.4,
                            low_text=0.25,
                        )
                    except Exception as e:
                        if self.debug:
                            print(f"[HeaderBandScanner] OCR failed on header column {i}: {e}")
                        dets = []

                    # Map OCR boxes back to *full band* coordinates, and lock them to this column
                    for box, txt, conf in dets:
                        try:
                            conf_f = float(conf or 0.0)
                        except Exception:
                            conf_f = 0.0

                        # box in upscaled column space -> downscale back to column-band coords
                        pts_band_local = [
                            (
                                int(p[0] / _HDR_OCR_SCALE),
                                int(p[1] / _HDR_OCR_SCALE),
                            )
                            for p in box
                        ]
                        xs = [p[0] for p in pts_band_local]
                        ys = [p[1] for p in pts_band_local]

                        # local coords within this column strip
                        x1_local = max(0, min(xs))
                        x2_local = min(col_band.shape[1] - 1, max(xs))
                        y1b = max(0, min(ys))
                        y2b = min(H_band - 1, max(ys))

                        # convert to *band* coords by offsetting with x_left
                        x1b = x_left + x1_local
                        x2b = x_left + x2_local

                        # page-coordinates (add vertical offset)
                        y1_abs = y1 + y1b
                        y2_abs = y1 + y2b

                        tok = {
                            "text": str(txt or "").strip(),
                            "conf": conf_f,
                            "box_band": [int(x1b), int(y1b), int(x2b), int(y2b)],
                            "box_page": [int(x1b), int(y1_abs), int(x2b), int(y2_abs)],
                        }
                        tokens.append(tok)
                        col_group["items"].append(tok)
            else:
                if self.debug:
                    print("[HeaderBandScanner] No full-height verticals; skipping header OCR.")

        # --- 3) normalize & score columns into semantic roles (ckt/desc/trip/poles) ---
        normalized_columns = self._score_header_columns(column_groups)

        # Debug printout to terminal
        if self.debug:
            print("[HeaderBandScanner] Column grouping in header band:")
            cols_summary = normalized_columns.get("columns", [])
            if not cols_summary:
                print("  (no vertical grid columns or no header tokens)")
            for col in cols_summary:
                role = col.get("role")
                ignored_reason = col.get("ignoredReason")
                if role:
                    role_str = role
                elif ignored_reason:
                    role_str = f"IGNORED({ignored_reason})"
                else:
                    role_str = "unknown"

                print(
                    f"  Column {col['index']} "
                    f"[{col['x_left']},{col['x_right']}] "
                    f"role={role_str}: {col['texts']}"
                )
            layout = normalized_columns.get("layout")
            if layout:
                print(f"[HeaderBandScanner] Panel layout: {layout}")

        # --- 4) build debug overlay for the band (cropped view) ---
        if self.debug:
            vis = cv2.cvtColor(band_lines, cv2.COLOR_GRAY2BGR)

            # draw a thin border around the band
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 255), 1)
            label = f"HEADER BAND  y:[{y1},{y2})"
            cv2.putText(
                vis,
                label,
                (8, max(14, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # OCR token overlays
            for tok in tokens:
                x1b, y1b, x2b, y2b = tok["box_band"]
                text = tok["text"]
                conf_f = tok["conf"]

                cv2.rectangle(vis, (x1b, y1b), (x2b, y2b), (0, 0, 255), 1)
                lbl = f"{text} ({conf_f:.2f})"
                ty = y1b - 4 if y1b - 4 > 10 else y2b + 12
                cv2.putText(
                    vis,
                    lbl,
                    (x1b, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # overlay the column grid lines on the header band too
            H_band, W_band = vis.shape[:2]

            # Use same header-local lines we used for OCR
            header_v_cols_dbg = v_cols

            # Draw header-local lines in magenta
            for x in header_v_cols_dbg:
                xi = int(x)
                if 0 <= xi < W_band:
                    cv2.line(vis, (xi, 0), (xi, H_band - 1), (255, 0, 255), 1)
                    cv2.putText(
                        vis,
                        "HDR",
                        (xi + 2, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]
            debug_img_overlay_path = os.path.join(
                debug_dir,
                f"{base}_parser_header_band_overlay.png",
            )
            try:
                cv2.imwrite(debug_img_overlay_path, vis)
            except Exception as e:
                print(f"[HeaderBandScanner] Failed to write header band overlay image: {e}")
                debug_img_overlay_path = None

        return {
            "band_y1": int(y1),
            "band_y2": int(y2),
            "tokens": tokens,
            "debugImageRaw": debug_img_raw_path,
            "debugImageOverlay": debug_img_overlay_path,
            "columnGroups": column_groups,
            "normalizedColumns": normalized_columns,
        }

    def _find_header_verticals(self, band: np.ndarray) -> List[int]:
        """
        Detect vertical grid lines inside the *header band only*.

        Strategy:
          - Binarize the band.
          - Morphologically open with a tall vertical kernel to keep vertical strokes.
          - Connected-components over the result.
          - Keep only components that:
              * are tall enough (>= ~85% of band height), and
              * start near the top and end near the bottom of the band, and
              * are thin (not blocks of fill).
          - Return sorted, de-duplicated X-center positions for those components.
        """
        if band is None or band.size == 0:
            return []

        H_band, W_band = band.shape

        # 1) binarize
        blur = cv2.GaussianBlur(band, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        # 2) emphasize vertical strokes with a tall, skinny kernel
        Kv = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(15, int(0.40 * H_band)),  # 40% of band height – enough to glue segments
            ),
        )
        v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)

        # 3) connected components -> filter long, skinny, full-height-ish components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            v_candidates,
            connectivity=8,
        )

        if num_labels <= 1:
            return []

        min_full_len = int(0.85 * H_band)             # must span at least 85% of header height
        max_thick    = max(2, int(0.02 * W_band))     # must be thin
        top_margin   = 3                              # must touch near top
        bot_margin   = 3                              # and near bottom

        xs_raw: List[int] = []

        for i in range(1, num_labels):  # label 0 is background
            x, y, w, h, area = stats[i]
            if h < min_full_len:
                continue
            if w > max_thick:
                continue

            # Require line to effectively touch both top and bottom of header band
            if y > top_margin:
                continue
            if (y + h) < (H_band - bot_margin):
                continue

            x_center = x + w // 2
            xs_raw.append(int(x_center))

        if not xs_raw:
            return []

        xs_raw.sort()

        # collapse near-duplicates into one X per visual line
        collapsed: List[int] = []
        MERGE_PX = 4
        for x in xs_raw:
            if not collapsed or abs(x - collapsed[-1]) > MERGE_PX:
                collapsed.append(x)

        return collapsed

    def _score_header_columns(self, column_groups: List[Dict]) -> Dict:
        """
        Given column_groups (each with .items = OCR tokens), assign a semantic
        role per column: 'ckt', 'description', 'trip', 'poles', 'combo', 'specialFeatures', or ignored.
        Also returns a panel-level layout:

        layout: 'combined'   -> trip + poles live in same "hero" column (per side)
                'separated'  -> trip and poles live in different hero columns (per side)
                'unknown'    -> anything else / insufficient signal

        Special-features rule:
        - There is at most ONE optional "special features" column per side.
        - It may be labeled using a SINGLE-WORD token: NOTES / INFO / OPTIONS / TYPE (fuzzy).
        - If we detect exactly one candidate on a side AND it maps to exactly one of those labels
        -> role='specialFeatures'.
        - If we detect multiple candidates on a side OR multiple labels inside the same column
        -> bail safely:
            ignore ALL special-features columns on that side.
        """
        def norm_word(s: str) -> str:
            if not s:
                return ""

            s = str(s).upper()
            out = []

            for ch in s:
                if ch in "0OQ":
                    out.append("O")
                elif ch in "1ILJ!|":
                    out.append("I")
                elif ch in "5S$":
                    out.append("S")
                elif ch in "2Z":
                    out.append("Z")
                elif ch in "8B":
                    out.append("B")
                elif ch.isalnum():
                    out.append(ch)

            return "".join(out)

        def _is_like(word: str, targets: List[str], base_threshold: float = 0.78) -> bool:
            w = norm_word(word)
            if not w:
                return False

            for t in targets:
                t_norm = norm_word(t)
                if not t_norm:
                    continue

                if len(t_norm) <= 3:
                    threshold = max(base_threshold, 0.88)
                else:
                    threshold = base_threshold

                if SequenceMatcher(a=w, b=t_norm).ratio() >= threshold:
                    return True

            return False

        def _hero_match(word: str, targets: List[str]) -> bool:
            w = norm_word(word)
            if not w:
                return False

            w_letters = "".join(ch for ch in w if ch.isalpha())

            for t in targets:
                t_norm = norm_word(t)
                if not t_norm:
                    continue

                t_letters = "".join(ch for ch in t_norm if ch.isalpha())

                if t_letters in ("AMP", "AMPS"):
                    if w_letters.startswith("AMP"):
                        return True
                    
                if t_letters in ("OCP", "OCPD", "OCPI", "IOCP"):
                    if w_letters.startswith("OCP"):
                        return True
                    
                if len(t_norm) <= 4:
                    threshold = 0.90
                else:
                    threshold = 0.75

                if SequenceMatcher(a=w, b=t_norm).ratio() >= threshold:
                    return True

                if w_letters and t_letters:
                    if (
                        (w_letters.startswith(t_letters) or w_letters.endswith(t_letters))
                        and abs(len(w_letters) - len(t_letters)) <= 1
                    ):
                        return True

            return False

        # ---------- First pass: compute scores + hero ranks per column ----------
        col_infos: List[Dict] = []

        for col in column_groups:
            items = col.get("items", []) or []

            texts: List[str] = []
            word_entries: List[Tuple[str, bool]] = []

            for tok in items:
                txt = str(tok.get("text", "")).strip()
                if not txt:
                    continue

                conf = float(tok.get("conf", 0.0))
                if conf < _HDR_MIN_CONF:
                    continue

                pieces = [w for w in re.split(r"[\s/,;-]+", txt) if w.strip()]
                if not pieces:
                    continue

                for raw in pieces:
                    raw_clean = raw.strip()
                    if raw_clean:
                        texts.append(raw_clean)

                is_sentence_like = len(pieces) >= 3

                for raw in pieces:
                    raw = raw.strip()
                    if not raw:
                        continue

                    w_norm = norm_word(raw)
                    if not w_norm:
                        continue

                    if not any(ch.isalpha() for ch in w_norm):
                        continue

                    word_entries.append((raw, is_sentence_like))

            # Strong CKT label assist
            has_ckt_core = any(_is_like(t, ["CKT", "CCT"], base_threshold=0.95) for t in texts)
            has_ckt_number_assoc = False
            for t in texts:
                tu = t.strip().upper()
                if tu in ("#", "NO", "NO.", "NUMBER", "NUM", "NUM.", "NBR", "NBR."):
                    has_ckt_number_assoc = True
                    break

            score = {
                "ckt": 0,
                "description": 0,
            }

            hero_trip_rank = 0
            hero_poles_rank = 0

            # --- Special-features detection state ---
            # We now allow exactly ONE of: notes / info / options / type (single-word triggers)
            special_labels = set()
            special_score = 0
            has_special_signal = False

            if not word_entries:
                col_infos.append(
                    {
                        "index": col["index"],
                        "x_left": col["x_left"],
                        "x_right": col["x_right"],
                        "texts": texts,
                        "ignoredReason": None,
                        "score": score,
                        "hero_trip_rank": hero_trip_rank,
                        "hero_poles_rank": hero_poles_rank,
                        "has_ckt_signal": False,
                        "has_desc_signal": False,
                        "role": None,
                        "special_labels": special_labels,
                        "special_score": special_score,
                        "has_special_signal": has_special_signal,
                    }
                )
                continue

            for w_raw, is_sentence_like in word_entries:
                w_norm = norm_word(w_raw)
                if not w_norm:
                    continue

                # --- Special-features keywords (SINGLE-WORD only) ---
                is_notes_like = _is_like(
                    w_raw,
                    ["NOTE", "NOTES"],
                    base_threshold=0.80,
                )
                is_info_like = _is_like(
                    w_raw,
                    ["INFO"],
                    base_threshold=0.82,
                )
                is_options_like = _is_like(
                    w_raw,
                    ["OPTION", "OPTIONS", "OPT", "OPTS"],
                    base_threshold=0.82,
                )
                is_type_like = _is_like(
                    w_raw,
                    ["TYPE"],
                    base_threshold=0.82,
                )

                def _mark_special(label: str):
                    nonlocal special_score, has_special_signal, special_labels
                    special_score += 3
                    has_special_signal = True
                    special_labels.add(label)

                if is_notes_like:
                    _mark_special("notes")
                if is_info_like:
                    _mark_special("info")
                if is_options_like:
                    _mark_special("options")
                if is_type_like:
                    _mark_special("type")

                # If this word came from sentence-like chunk, don't let it drive role scoring
                if is_sentence_like:
                    continue

                # --- CKT / CCT / NO. (very strict) ---
                if _is_like(w_raw, ["CKT", "CCT", "NO", "NO."], base_threshold=0.95):
                    score["ckt"] += 3

                # --- DESCRIPTION / NAME (looser) ---
                if _is_like(
                    w_raw,
                    [
                        "DESCRIPTION",
                        "CIRCUIT DESCRIPTION",
                        "LOAD DESCRIPTION",
                        "DESIGNATION",
                        "LOAD DESIGNATION",
                        "NAME",
                    ],
                    base_threshold=0.70,
                ):
                    score["description"] += 3

                # --- Trip hero ranks ---
                if _hero_match(w_raw, ["TRIP", "TRIPPING"]):
                    hero_trip_rank = max(hero_trip_rank, 5)
                elif _hero_match(w_raw, ["AMP", "AMPS"]):
                    hero_trip_rank = max(hero_trip_rank, 4)
                elif _hero_match(w_raw, ["OCP", "OCPI", "OCPD", "IOCP", "OOPD", "OPD"]):
                    hero_trip_rank = max(hero_trip_rank, 3)
                elif _hero_match(w_raw, ["SIZE"]):
                    hero_trip_rank = max(hero_trip_rank, 3)
                elif _hero_match(w_raw, ["BREAKER", "BK", "BRK", "BKR", "BRKR", "CB"]):
                    hero_trip_rank = max(hero_trip_rank, 2)

                # --- Poles hero ranks ---
                if _hero_match(w_raw, ["POLE", "POLES", "PO.", "PO"]):
                    hero_poles_rank = max(hero_poles_rank, 4)
                elif w_norm == "P":
                    hero_poles_rank = max(hero_poles_rank, 1)

            if has_ckt_core and has_ckt_number_assoc:
                score["ckt"] += 2

            has_ckt_signal = score["ckt"] > 0
            has_desc_signal = score["description"] > 0

            col_infos.append(
                {
                    "index": col["index"],
                    "x_left": col["x_left"],
                    "x_right": col["x_right"],
                    "texts": texts,
                    "ignoredReason": None,
                    "score": score,
                    "hero_trip_rank": hero_trip_rank,
                    "hero_poles_rank": hero_poles_rank,
                    "has_ckt_signal": has_ckt_signal,
                    "has_desc_signal": has_desc_signal,
                    "role": None,
                    "special_labels": special_labels,
                    "special_score": special_score,
                    "has_special_signal": has_special_signal,
                }
            )

        if not col_infos:
            return {"roles": {}, "columns": [], "layout": "unknown"}

        # ---------- First: assign ckt / description from fuzzy scores ----------
        for info in col_infos:
            s = info["score"]
            best_role = None
            best_score_val = 0

            for r in ("ckt", "description"):
                if s[r] > best_score_val:
                    best_score_val = s[r]
                    best_role = r

            if best_role is not None and best_score_val >= 2:
                info["role"] = best_role
            else:
                info["role"] = None

        # ---------- Normalize multiple CKT columns (one primary per side) ----------
        ckt_candidates = [
            info
            for info in col_infos
            if info["score"]["ckt"] > 0 and (info.get("ignoredReason") is None)
        ]

        if ckt_candidates:
            global_left = min(c["x_left"] for c in ckt_candidates)
            global_right = max(c["x_right"] for c in ckt_candidates)
            center_x = 0.5 * (global_left + global_right)

            for info in ckt_candidates:
                col_center = 0.5 * (info["x_left"] + info["x_right"])
                info["ckt_side"] = "left" if col_center < center_x else "right"

            def _has_number_assoc(texts: List[str]) -> bool:
                for t in texts or []:
                    tu = t.strip().upper()
                    if tu in ("#", "NO", "NO.", "NUMBER", "NUM", "NUM.", "NBR", "NBR."):
                        return True
                return False

            def _rank_ckt(info: Dict) -> tuple:
                has_num = 1 if _has_number_assoc(info["texts"]) else 0
                return (info["score"]["ckt"], has_num, -info["x_left"])

            primary_ckt_indices = set()
            for side in ("left", "right"):
                side_list = [c for c in ckt_candidates if c.get("ckt_side") == side]
                if not side_list:
                    continue
                primary = max(side_list, key=_rank_ckt)
                primary_ckt_indices.add(primary["index"])

            for info in ckt_candidates:
                if info["index"] in primary_ckt_indices:
                    if info["role"] is None:
                        info["role"] = "ckt"
                else:
                    if info["role"] == "ckt":
                        info["role"] = None
                    if not info.get("ignoredReason"):
                        info["ignoredReason"] = "secondaryCkt"

        # ---------- Assign geometric side for ALL columns (for specialFeatures and heroes) ----------
        global_left_all = min(info["x_left"] for info in col_infos)
        global_right_all = max(info["x_right"] for info in col_infos)
        center_x_all = 0.5 * (global_left_all + global_right_all)

        for info in col_infos:
            col_center = 0.5 * (info["x_left"] + info["x_right"])
            info["side"] = "left" if col_center < center_x_all else "right"

        # ---------- Special-features selection per side (QUALITY RANKING) ----------
        # Candidates must NOT be ckt/description and must have a special signal.
        special_candidates = [
            info
            for info in col_infos
            if info.get("role") is None
            and not info.get("has_ckt_signal")
            and not info.get("has_desc_signal")
            and info.get("has_special_signal")
            and info.get("special_labels")
        ]

        # ---- Rank special-features candidates ----
        # BEST: breaker-context + (opt/type/notes/info)
        # GOOD: indicator alone
        # POOR: indicator + junk (ignore)
        _SF_INDICATOR_WORDS = ["NOTE", "NOTES", "INFO", "OPTION", "OPTIONS", "OPT", "OPTS", "TYPE"]
        _SF_BREAKER_WORDS = ["BREAKER", "BRKR", "BKR", "CB"]

        def _has_breaker_context(texts: List[str]) -> bool:
            for t in texts or []:
                if _hero_match(t, _SF_BREAKER_WORDS):
                    return True
            return False

        def _count_special_junk_tokens(texts: List[str]) -> int:
            junk = 0
            for t in texts or []:
                if not t:
                    continue

                ts = str(t).strip()
                if not ts:
                    continue

                # Ignore single-letter stuff (often A/B/C etc)
                if len(ts) == 1:
                    continue

                # Indicator words are not junk
                if _is_like(ts, _SF_INDICATOR_WORDS, base_threshold=0.82):
                    continue

                # Breaker context words are not junk
                if _hero_match(ts, _SF_BREAKER_WORDS):
                    continue

                tn = norm_word(ts)
                if not tn:
                    continue

                # Ignore numeric-only fragments
                if not any(ch.isalpha() for ch in tn):
                    continue

                junk += 1
            return junk

        def _special_candidate_rank(info: Dict) -> Tuple[int, int, int]:
            texts = info.get("texts", []) or []
            has_breaker = _has_breaker_context(texts)
            junk = _count_special_junk_tokens(texts)

            # tier: 3=BEST, 2=GOOD, 0=POOR
            if has_breaker:
                tier = 3
            else:
                tier = 2 if junk == 0 else 0

            # numeric score to break ties within a tier
            score_num = int(info.get("special_score", 0)) + (10 if has_breaker else 0) - junk

            # final tie-breaker: prefer left-most within the side (stable)
            return (tier, score_num, -int(info.get("x_left", 0)))

        # Attach rank + pre-ignore POOR candidates
        for c in special_candidates:
            rank = _special_candidate_rank(c)
            c["_sf_rank"] = rank
            c["_sf_tier"] = rank[0]
            if c["_sf_tier"] <= 0:
                # Poor = indicator + junk (e.g. LOAD TYPE) -> ignore it completely
                if not c.get("ignoredReason"):
                    c["ignoredReason"] = "specialFeaturesJunk"

        if self.debug:
            print("[HeaderBandScanner] special_candidates ranked:")
            for c in special_candidates:
                print(
                    f"  idx={c['index']} side={c.get('side')} "
                    f"tier={c.get('_sf_tier')} rank={c.get('_sf_rank')} "
                    f"labels={sorted(list(c.get('special_labels', set())))} "
                    f"ignored={c.get('ignoredReason')} texts={c.get('texts')}"
                )

        if special_candidates:
            for side in ("left", "right"):
                # only candidates that are not POOR
                side_candidates = [
                    c for c in special_candidates
                    if c.get("side") == side and c.get("_sf_tier", 0) > 0
                ]
                if not side_candidates:
                    continue

                # pick best by rank
                side_candidates.sort(key=lambda c: c.get("_sf_rank", (0, 0, 0)), reverse=True)
                winner = side_candidates[0]

                # if the top two are tied exactly -> ambiguous -> ignore all on that side
                if len(side_candidates) > 1:
                    r0 = side_candidates[0].get("_sf_rank")
                    r1 = side_candidates[1].get("_sf_rank")
                    if r0 == r1:
                        for c in side_candidates:
                            c["ignoredReason"] = "specialFeaturesAmbiguous"
                        continue

                # accept winner
                winner["role"] = "specialFeatures"
                winner["ignoredReason"] = None

                # mark any other non-poor candidates as secondary (not ambiguous)
                for c in side_candidates[1:]:
                    if not c.get("ignoredReason"):
                        c["ignoredReason"] = "specialFeaturesSecondary"

        # ---------- Hero candidates: only pure trip/poles columns ----------
        # Exclude any columns that have ANY ckt/desc signal, and also exclude specialFeatures columns/signals.
        hero_candidates = [
            info
            for info in col_infos
            if info.get("role") is None
            and not info.get("has_ckt_signal")
            and not info.get("has_desc_signal")
            and not info.get("has_special_signal")
            and (info.get("ignoredReason") is None)
            and (
                int(info.get("hero_trip_rank", 0)) > 0
                or int(info.get("hero_poles_rank", 0)) > 0
            )
        ]
 
        if not hero_candidates:
            role_to_index: Dict[str, int] = {}
            summaries: List[Dict] = []
            for info in col_infos:
                role = info.get("role")
                ignored_reason = info.get("ignoredReason")
                if role and role not in role_to_index:
                    role_to_index[role] = info["index"]
                summaries.append(
                    {
                        "index": info["index"],
                        "x_left": info["x_left"],
                        "x_right": info["x_right"],
                        "texts": info["texts"],
                        "role": role,
                        "ignoredReason": ignored_reason,
                    }
                )
            return {"roles": role_to_index, "columns": summaries, "layout": "unknown"}

        # ---------- Determine left/right side using hero candidate geometry ----------
        global_left = min(info["x_left"] for info in hero_candidates)
        global_right = max(info["x_right"] for info in hero_candidates)
        center_x = 0.5 * (global_left + global_right)

        for info in col_infos:
            col_center = 0.5 * (info["x_left"] + info["x_right"])
            info["side"] = "left" if col_center < center_x else "right"

        # ---------- Pick best hero trip/poles per side ----------
        best_trip_col = {"left": None, "right": None}
        best_trip_rank = {"left": 0, "right": 0}
        best_poles_col = {"left": None, "right": None}
        best_poles_rank = {"left": 0, "right": 0}

        for info in hero_candidates:
            side = info["side"]
            ht = info["hero_trip_rank"]
            hp = info["hero_poles_rank"]

            if ht > best_trip_rank[side]:
                best_trip_rank[side] = ht
                best_trip_col[side] = info

            if hp > best_poles_rank[side]:
                best_poles_rank[side] = hp
                best_poles_col[side] = info

        # ---------- Panel-level layout decision ----------
        combo_side_exists = False
        separated_side_exists = False

        for side in ("left", "right"):
            tcol = best_trip_col[side]
            pcol = best_poles_col[side]

            if (
                tcol is not None
                and pcol is not None
                and best_trip_rank[side] > 0
                and best_poles_rank[side] > 0
            ):
                if tcol["index"] == pcol["index"]:
                    combo_side_exists = True
                else:
                    separated_side_exists = True

        any_trip_hero = any(best_trip_rank[side] > 0 for side in ("left", "right"))
        any_poles_hero = any(best_poles_rank[side] > 0 for side in ("left", "right"))

        implied_combo_sides = set()
        for side in ("left", "right"):
            if best_trip_rank[side] > 0 and best_poles_rank[side] == 0:
                implied_combo_sides.add(side)

        if self.debug and implied_combo_sides:
            print("[HeaderBandScanner] Implied combo layout on sides:", sorted(implied_combo_sides))

        if combo_side_exists and separated_side_exists:
            layout = "unknown"
        elif combo_side_exists:
            layout = "combined"
        elif separated_side_exists or (any_trip_hero and any_poles_hero):
            layout = "separated"
        elif implied_combo_sides:
            layout = "combined"
        else:
            layout = "unknown"

        # ---------- Hero-based assignment for trip/poles/combo (per side) ----------
        if layout == "combined":
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                pcol = best_poles_col[side]

                if (
                    tcol is not None
                    and pcol is not None
                    and tcol["index"] == pcol["index"]
                    and best_trip_rank[side] > 0
                    and best_poles_rank[side] > 0
                ):
                    if tcol["role"] is None:
                        tcol["role"] = "combo"
                    continue

                if (
                    best_trip_rank[side] > 0
                    and best_poles_rank[side] == 0
                    and tcol is not None
                    and tcol["role"] is None
                ):
                    tcol["role"] = "combo"

        elif layout == "separated":
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                if tcol is not None and best_trip_rank[side] > 0:
                    if tcol["role"] is None:
                        tcol["role"] = "trip"

                pcol = best_poles_col[side]
                if pcol is not None and best_poles_rank[side] > 0:
                    if pcol["role"] is None and (tcol is None or pcol["index"] != tcol["index"]):
                        pcol["role"] = "poles"

        else:
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                if tcol is not None and best_trip_rank[side] > 0:
                    if tcol["role"] is None:
                        tcol["role"] = "trip"

                pcol = best_poles_col[side]
                if pcol is not None and best_poles_rank[side] > 0:
                    if pcol["role"] is None and (tcol is None or pcol["index"] != tcol["index"]):
                        pcol["role"] = "poles"

        # ---------- Build roles map + summaries ----------
        role_to_index: Dict[str, int] = {}
        summaries: List[Dict] = []

        for info in col_infos:
            role = info.get("role")
            ignored_reason = info.get("ignoredReason")

            if role and role not in role_to_index:
                role_to_index[role] = info["index"]

            summaries.append(
                {
                    "index": info["index"],
                    "x_left": info["x_left"],
                    "x_right": info["x_right"],
                    "texts": info["texts"],
                    "role": role,
                    "ignoredReason": ignored_reason,
                }
            )

        return {"roles": role_to_index, "columns": summaries, "layout": layout}

class SeparatedLayoutParser:
    """
    Handles parsing when the header layout is 'separated':
      - Trip and poles live in different hero columns (left/right).
      - Uses horizontal grid lines (within a reference body strip) to define row bands.
      - OCRs the trip/poles body strips ROW-BY-ROW using those bands.
      - Associates each row's amps + poles and emits per-breaker records
        plus an aggregate (amps/poles) histogram.
    """

    def __init__(self, *, debug: bool = False, reader=None):
        self.debug = bool(debug)
        self.reader = reader

    def _row_text_for_column(self, row_top: int, row_bottom: int, col_tokens):
        """
        For a given column + row band, stitch together all tokens that fall
        in that vertical band, ordered left→right.
        """
        if not col_tokens:
            return ""

        parts = []
        for tok in col_tokens:
            x1, y1, x2, y2 = tok["box_page"]
            y_center = 0.5 * (y1 + y2)
            if y_center < row_top or y_center >= row_bottom:
                continue
            parts.append((x1, tok["text"]))

        if not parts:
            return ""

        parts.sort(key=lambda t: t[0])
        return " ".join(p[1] for p in parts).strip()

    def _detect_row_divider_lines_in_strip(self, strip_gray):
        """
        Given a single body strip in GRAY (gridless),
        detect strong horizontal 'row divider' lines that span
        ~95%+ of the strip width.

        Returns:
            List of y-centers (ints) in STRIP-LOCAL coordinates.
        """
        import cv2

        if strip_gray is None or strip_gray.size == 0:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        if H_strip <= 0 or W_strip <= 0:
            return []

        # 1) Binarize (invert so lines are white on black)
        blur = cv2.GaussianBlur(strip_gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        # 2) Emphasize horizontal strokes with a wide, thin kernel
        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(15, int(0.40 * W_strip)),  # fairly wide
                1,                             # very thin vertically
            ),
        )
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

        # 3) Connected components to find horizontal blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            h_candidates,
            connectivity=8,
        )
        if num_labels <= 1:
            return []

        # Lines must span >= 95% of the strip width and be thin
        min_full_w = int(0.95 * W_strip)
        max_thick = max(2, int(0.03 * H_strip))

        ys_raw = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w < min_full_w:
                continue
            if h > max_thick:
                continue

            yc = y + h // 2
            ys_raw.append(int(yc))

        if not ys_raw:
            return []

        ys_raw.sort()

        # Collapse near-duplicate line centers
        merged = []
        MERGE_PX = 3
        for y in ys_raw:
            if not merged or abs(y - merged[-1]) > MERGE_PX:
                merged.append(y)

        if self.debug:
            print(
                f"[SeparatedLayoutParser] Detected {len(merged)} long horizontal lines "
                f"in strip (H={H_strip}, W={W_strip})."
            )

        return merged

    def _compute_row_bands_in_strip(self, strip_gray):
        """
        Given a body-strip in GRAY (gridless), use the detected long
        horizontal lines to define row bands in STRIP-LOCAL coordinates.

        Behavior:
          - Treat regions:
                top_of_strip → first_line,
                each line_i → line_{i+1},
                last_line → bottom_of_strip
            as potential row bands.
          - Use a small padding inside each segment so we avoid the core of
            the grid line, but do NOT over-trim (to keep single characters
            near the line, like a lone '1').

        Returns:
            List of (row_top, row_bottom) in STRIP-LOCAL coordinates.
        """

        if strip_gray is None or strip_gray.size == 0:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        if H_strip <= 0 or W_strip <= 0:
            return []

        line_ys = self._detect_row_divider_lines_in_strip(strip_gray)
        if not line_ys:
            # No lines at all -> treat whole strip as one band
            return [(0, H_strip)]

        # Sort and clamp
        line_ys = sorted(int(y) for y in line_ys)
        line_ys = [max(0, min(H_strip - 1, y)) for y in line_ys]

        # Build raw segments:
        #   [0 -> first_line], [line_i -> line_{i+1}], [last_line -> H_strip]
        segments = []
        prev = 0
        for y in line_ys:
            if y > prev:
                segments.append((prev, y))
            prev = y
        if H_strip > prev:
            segments.append((prev, H_strip))

        bands = []
        prev_bottom = 0
        for (top, bottom) in segments:
            if bottom <= top + 3:
                continue

            span = bottom - top
            # Smaller padding: 2% of span, min 1 px
            pad = max(1, int(0.02 * span))

            # Slightly "inflate" compared to the raw segment, but keep ordering
            row_top = max(prev_bottom, top + pad - 1)
            row_bottom = min(H_strip, bottom - pad + 1)

            if row_bottom > row_top + 2:
                bands.append((row_top, row_bottom))
                prev_bottom = row_bottom

        if self.debug:
            print(
                f"[SeparatedLayoutParser] _compute_row_bands_in_strip: "
                f"lines={line_ys} → segments={segments} → {len(bands)} row bands"
            )

        if not bands:
            # Ultra-defensive fallback
            return [(0, H_strip)]

        return bands

    @staticmethod
    def _row_tokens_text(row_tokens) -> str:
        """Join a row's OCR tokens left-to-right into one cell string."""
        ordered = sorted(row_tokens, key=lambda t: t[2][0])
        return " ".join(t[0] for t in ordered).strip()

    def _make_row_value_checker(self, role: Optional[str]):
        """
        Build the predicate that decides whether a row's tokens already yield a
        usable value for this column, which is what gates the raw-cell fallback.

        Returns None for roles with no numeric validator (e.g. specialFeatures),
        which leaves those columns on the prepped-only path.
        """
        if role == "trip":
            return lambda toks: self._parse_trip_value(self._row_tokens_text(toks)) is not None
        if role == "poles":
            return lambda toks: self._parse_poles_value(self._row_tokens_text(toks)) is not None
        return None

    def _ocr_row_band_tokens(self, row_img):
        """
        OCR one already-cropped row band.

        Returns a list of (text, conf, (x1, y1, x2, y2)) with the box in
        ROW-LOCAL pixel coordinates, i.e. relative to row_img's own origin.
        """
        import cv2

        if row_img is None or row_img.size == 0 or self.reader is None:
            return []

        h_row, w_row = row_img.shape[:2]

        row_up = cv2.resize(
            row_img,
            None,
            fx=_HDR_OCR_SCALE,
            fy=_HDR_OCR_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        try:
            dets = _readtext_with_timeout(
                self.reader,
                row_up,
                detail=1,
                paragraph=False,
                allowlist=_HDR_OCR_ALLOWLIST,
                mag_ratio=1.0,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.4,
                low_text=0.25,
            )
        except Exception:
            dets = []

        out = []
        for box, txt, conf in dets:
            # length-aware confidence thresholding (same as combined)
            txt_clean = str(txt or "").strip()
            if not txt_clean:
                continue

            try:
                conf_f = float(conf or 0.0)
            except Exception:
                conf_f = 0.0

            min_conf = _HDR_MIN_CONF
            if len(txt_clean) == 1:
                # Relax threshold a bit for single-character tokens (e.g. '1')
                min_conf = _HDR_MIN_CONF * 0.7

            if conf_f < min_conf:
                continue

            # box is in UPSCALED row coords -> map back to row-local
            pts_local = [
                (
                    int(p[0] / _HDR_OCR_SCALE),
                    int(p[1] / _HDR_OCR_SCALE),
                )
                for p in box
            ]
            xs = [p[0] for p in pts_local]
            ys = [p[1] for p in pts_local]

            x1_row = max(0, min(w_row - 1, min(xs)))
            x2_row = max(0, min(w_row - 1, max(xs)))
            y1_row = max(0, min(h_row - 1, min(ys)))
            y2_row = max(0, min(h_row - 1, max(ys)))

            out.append((txt_clean, conf_f, (x1_row, y1_row, x2_row, y2_row)))

        return out

    def _ocr_body_column_by_rows(
        self,
        strip_gray,
        x_left: int,
        body_y_top: int,
        row_bands_strip,
        debug_prep_path: Optional[str] = None,
        row_value_ok=None,
    ):
        """
        OCR a single trip/poles body strip ROW-BY-ROW.

        Inputs:
          - strip_gray: gray crop for this body column, still carrying the table
                        grid; each row band is de-gridded by _prep_cell_for_ocr
                        immediately before OCR
                        shape (H_strip, W_strip)
          - x_left:     column's left X in PAGE coordinates
          - body_y_top: Y at which this strip starts in PAGE coordinates
          - row_bands_strip: list of (row_top, row_bottom) in STRIP-LOCAL coords
          - debug_prep_path: when debug is on, where to write the BEFORE | AFTER
                        cell-prep comparison for this column
          - row_value_ok: optional predicate taking this row's token list and
                        returning whether it yields a usable value for this
                        column's role. Enables the raw-cell fallback below.

        Returns:
          Flat list of OCR tokens, each with box_page coordinates.
        """
        if strip_gray is None or strip_gray.size == 0 or self.reader is None:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        tokens = []

        prep_vis = (
            strip_gray.copy()
            if (self.debug and debug_prep_path)
            else None
        )

        for (row_top, row_bottom) in row_bands_strip:
            row_top = max(0, min(H_strip - 1, int(row_top)))
            row_bottom = max(row_top + 1, min(H_strip, int(row_bottom)))
            if row_bottom <= row_top:
                continue

            row_gray = strip_gray[row_top:row_bottom, :]
            if row_gray.size == 0:
                continue

            # Strip the table borders out of this cell first. The crop keeps its
            # exact dimensions, so the box mapping further down stays valid.
            row_prepped = _prep_cell_for_ocr(row_gray)

            if prep_vis is not None:
                prep_vis[row_top:row_bottom, :] = row_prepped

            row_tokens = self._ocr_row_band_tokens(row_prepped)

            # De-gridding wins on cells whose borders touch the glyphs, but on a
            # minority of cells it drops a trailing digit or leaves a fragment
            # that OCRs as a spurious character. Re-reading the untouched cell
            # whenever the prepped read yields nothing usable keeps this parser
            # from ever seeing less than the pre-cell-prep parser did, while
            # leaving every successful prepped read as the authoritative one.
            if row_value_ok is not None and not row_value_ok(row_tokens):
                raw_tokens = self._ocr_row_band_tokens(row_gray)
                if row_value_ok(raw_tokens):
                    row_tokens = raw_tokens
                    if self.debug:
                        print(
                            f"[{PARSER_VERSION}] raw-cell fallback recovered row "
                            f"y={body_y_top + row_top}-{body_y_top + row_bottom} "
                            f"x={x_left}: {[t[0] for t in raw_tokens]}"
                        )

            for txt_clean, conf_f, (x1_row, y1_row, x2_row, y2_row) in row_tokens:
                # Map row-local -> STRIP-LOCAL -> PAGE
                x1_page = x_left + x1_row
                x2_page = x_left + x2_row
                y1_page = body_y_top + row_top + y1_row
                y2_page = body_y_top + row_top + y2_row

                tokens.append(
                    {
                        "text": txt_clean,
                        "conf": conf_f,
                        "box_page": [int(x1_page), int(y1_page), int(x2_page), int(y2_page)],
                    }
                )

        if prep_vis is not None:
            _save_cell_prep_comparison(strip_gray, prep_vis, debug_prep_path)

        return tokens

    def _parse_trip_value(self, text: str):
        """
        Extract an amp rating (e.g. '20', '20A', '20 AMP') from trip text.
        Returns an int or None.
        """
        import re

        if not text:
            return None

        t = text.upper()
        # If clearly marked SPARE, ignore
        if "SPARE" in t or "SPACE" in t:
            return None

        m = re.search(r"(\d{1,4})", t)
        if not m:
            return None

        try:
            val = int(m.group(1))
        except Exception:
            return None

        if val <= 0:
            return None

        # Match combined-mode amp validation:
        # reject row numbers / OCR junk like 1, 2, 3, 22, 27, etc.
        if val < 10:
            return None

        if val % 5 != 0:
            return None

        return val

    def _parse_poles_value(self, text: str):
        """
        Extract a pole count (1/2/3) from poles text.
        Returns 1,2,3 or None.
        """
        import re

        if not text:
            return None

        t = text.upper()
        if "SPARE" in t or "SPACE" in t:
            return None

        # Strong patterns first
        if "3P" in t or "3-P" in t or "3 POLE" in t:
            return 3
        if "2P" in t or "2-P" in t or "2 POLE" in t:
            return 2
        if "1P" in t or "1-P" in t or "1 POLE" in t:
            return 1
        if "SP" in t:  # SP = single pole
            return 1

        # Fallback: bare digit 1/2/3
        m = re.search(r"\b([123])\b", t)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

        return None

    def parse(self, analyzer_result: dict, header_scan: dict) -> dict:
        """
        Separated layout:
        - Trip and poles are separate columns per side.
        - Optional specialFeatures column per side (NOTES or OPTIONS).
        - If specialFeatures exists on a side, we also extract row text from it
            and attach it to each detected breaker row as 'specialFeaturesText'.
        """
        import os
        import cv2

        raw_gray = analyzer_result.get("gray")
        if raw_gray is None:
            raw_gray = _ensure_gray(analyzer_result)

        gray_body = raw_gray
        gray_lines = raw_gray

        if gray_body is None or gray_lines is None:
            if self.debug:
                print("[SeparatedLayoutParser] Missing gray_body/gray_lines; cannot crop body columns.")
            return {"layout": "separated", "bodyColumns": [], "detected_breakers": [], "breakerCounts": {}}

        H, W = gray_body.shape[:2]

        src_path = analyzer_result.get("src_path")
        src_dir = analyzer_result.get("src_dir") or os.path.dirname(src_path or ".")
        debug_dir = analyzer_result.get("debug_dir") or os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(src_path or "panel"))[0]

        header_bottom_y = analyzer_result.get("header_bottom_y")
        footer_y = analyzer_result.get("footer_y")

        if not isinstance(header_bottom_y, (int, float)):
            if self.debug:
                print("[SeparatedLayoutParser] Missing header_bottom_y; cannot determine body top. Bailing.")
                print(f"  header_bottom_y={header_bottom_y!r}")
            return {
                "layout": "separated",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
                "error": "Missing header_bottom_y from analyzer; cannot determine body band.",
            }

        if not isinstance(footer_y, (int, float)):
            if self.debug:
                print("[SeparatedLayoutParser] Missing footer_y; cannot determine body bottom. Bailing.")
                print(f"  footer_y={footer_y!r}")
            return {
                "layout": "separated",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
                "error": "Missing footer_y from analyzer; cannot determine body band.",
            }

        body_y_top = max(0, int(header_bottom_y))
        body_y_bottom = int(footer_y)

        if self.debug:
            print(
                f"[SeparatedLayoutParser] Body Y-range: [{body_y_top}, {body_y_bottom}) "
                f"(H={H})  (using header_bottom_y→footer_y as body band)"
            )

        if body_y_bottom <= body_y_top + 4:
            if self.debug:
                print("[SeparatedLayoutParser] Body band too small; skipping body columns.")
            return {"layout": "separated", "bodyColumns": [], "detected_breakers": [], "breakerCounts": {}}

        normalized = header_scan.get("normalizedColumns") or {}
        if normalized.get("layout") != "separated":
            if self.debug:
                print(
                    "[SeparatedLayoutParser] Warning: called with layout "
                    f"{normalized.get('layout')}; expected 'separated'."
                )
            return {
                "layout": normalized.get("layout", "unknown"),
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
            }

        cols_summary = normalized.get("columns", []) or []

        # Include optional specialFeatures
        wanted_roles = {"trip", "poles", "specialFeatures"}

        body_columns = []

        for col in cols_summary:
            role = col.get("role")
            if role not in wanted_roles:
                continue

            x_left = max(0, int(col.get("x_left", 0)))
            x_right = min(W - 1, int(col.get("x_right", W - 1)))
            if x_right <= x_left + 1:
                continue

            body_strip = gray_body[body_y_top:body_y_bottom, x_left:x_right]
            if body_strip.size == 0:
                continue

            body_columns.append(
                {
                    "index": col["index"],
                    "role": role,
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_top": body_y_top,
                    "y_bottom": body_y_bottom,
                    "debugImageOverlay": None,
                }
            )

        if self.debug:
            print(f"[SeparatedLayoutParser] Extracted {len(body_columns)} body columns for trip/poles/specialFeatures.")

        if not body_columns:
            return {"layout": "separated", "bodyColumns": body_columns, "detected_breakers": [], "breakerCounts": {}}

        # Assign side based on PRIMARY roles only (trip/poles), so specialFeatures doesn't skew center_x
        primary_cols = [c for c in body_columns if c["role"] in {"trip", "poles"}]
        if primary_cols:
            global_left = min(c["x_left"] for c in primary_cols)
            global_right = max(c["x_right"] for c in primary_cols)
        else:
            global_left = min(c["x_left"] for c in body_columns)
            global_right = max(c["x_right"] for c in body_columns)

        center_x = 0.5 * (global_left + global_right)

        for col in body_columns:
            col_center = 0.5 * (col["x_left"] + col["x_right"])
            col["side"] = "left" if col_center < center_x else "right"

        by_side = {
            "left": {"trip": None, "poles": None, "specialFeatures": None},
            "right": {"trip": None, "poles": None, "specialFeatures": None},
        }

        for col in body_columns:
            side = col["side"]
            role = col["role"]
            if by_side[side][role] is None:
                by_side[side][role] = col["index"]

        if self.debug:
            print("[SeparatedLayoutParser] columns by side:", by_side)

        # Determine row bands per side (same as before)
        role_cols = [c for c in body_columns if c["role"] in {"trip", "poles"}]
        row_bands_by_side = {"left": None, "right": None}
        row_spans_by_side = {"left": [], "right": []}

        for side in ("left", "right"):
            side_cols = [c for c in role_cols if c["side"] == side]
            if not side_cols:
                continue

            ref_col = None
            for c in side_cols:
                if c["role"] == "trip":
                    ref_col = c
                    break
            if ref_col is None:
                for c in side_cols:
                    if c["role"] == "poles":
                        ref_col = c
                        break

            if ref_col is None:
                continue

            ref_strip = gray_body[ref_col["y_top"]:ref_col["y_bottom"], ref_col["x_left"]:ref_col["x_right"]]
            if ref_strip.size == 0:
                row_bands_strip = [(0, ref_col["y_bottom"] - ref_col["y_top"])]
            else:
                row_bands_strip = self._compute_row_bands_in_strip(ref_strip)
                if not row_bands_strip:
                    row_bands_strip = [(0, ref_col["y_bottom"] - ref_col["y_top"])]

            row_bands_by_side[side] = row_bands_strip
            row_spans_by_side[side] = [(body_y_top + top_s, body_y_top + bottom_s) for (top_s, bottom_s) in row_bands_strip]

            if self.debug:
                print(f"[SeparatedLayoutParser] Side='{side}' using {len(row_spans_by_side[side])} row bands from horizontal lines.")

        # OCR each wanted column by rows
        tokens_by_col_index: Dict[int, List[Dict]] = {}

        for col in body_columns:
            idx = col["index"]
            x_l = col["x_left"]
            x_r = col["x_right"]
            y_top = col["y_top"]
            y_bot = col["y_bottom"]
            side = col.get("side", "left")

            row_bands_strip = row_bands_by_side.get(side)
            if not row_bands_strip:
                tokens_by_col_index[idx] = []
                continue

            strip = gray_body[y_top:y_bot, x_l:x_r]
            if strip.size == 0:
                tokens_by_col_index[idx] = []
                continue

            cellprep_path = (
                os.path.join(
                    debug_dir,
                    f"{base}_parser_body_sep_col{idx}_{col['role']}_cellprep.png",
                )
                if self.debug
                else None
            )

            tokens_by_col_index[idx] = self._ocr_body_column_by_rows(
                strip_gray=strip,
                x_left=x_l,
                body_y_top=y_top,
                row_bands_strip=row_bands_strip,
                debug_prep_path=cellprep_path,
                row_value_ok=self._make_row_value_checker(col.get("role")),
            )

        # Debug overlays unchanged (works for specialFeatures too)
        if self.debug:
            for col in body_columns:
                idx = col["index"]
                role = col["role"]
                x_l = col["x_left"]
                x_r = col["x_right"]
                y_top = col["y_top"]
                y_bot = col["y_bottom"]

                col_tokens = tokens_by_col_index.get(idx, [])
                if not col_tokens:
                    continue

                strip = gray_body[y_top:y_bot, x_l:x_r]
                if strip.size == 0:
                    continue

                H_strip, W_strip = strip.shape[:2]
                vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

                cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 255), 1)
                lbl = f"BODY {role.upper()}  col={idx}"
                cv2.putText(vis, lbl, (8, max(16, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                line_ys = self._detect_row_divider_lines_in_strip(strip)
                for yc in line_ys:
                    yc_int = max(0, min(H_strip - 1, int(yc)))
                    cv2.line(vis, (0, yc_int), (W_strip - 1, yc_int), (255, 0, 0), 1)
                    cv2.putText(vis, "ROW", (4, max(10, yc_int - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

                for tok in col_tokens:
                    x1p, y1p, x2p, y2p = tok["box_page"]
                    text = tok.get("text", "")
                    conf = tok.get("conf", 0.0)

                    x1 = max(0, min(W_strip - 1, x1p - x_l))
                    x2 = max(0, min(W_strip - 1, x2p - x_l))
                    y1 = max(0, min(H_strip - 1, y1p - y_top))
                    y2 = max(0, min(H_strip - 1, y2p - y_top))

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    label = f"{text} ({conf:.2f})"
                    ty = y1 - 4 if y1 - 4 > 10 else y2 + 12
                    cv2.putText(vis, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                overlay_path = os.path.join(debug_dir, f"{base}_parser_body_sep_col{idx}_{role}_overlay.png")
                try:
                    cv2.imwrite(overlay_path, vis)
                    col["debugImageOverlay"] = overlay_path
                except Exception as e:
                    print(f"[SeparatedLayoutParser] Failed to write body overlay col {idx} ({role}): {e}")
                    col["debugImageOverlay"] = None

        detected_breakers = []
        breaker_counts: Dict[str, int] = {}      # NON-GFI only
        gfi_breaker_counts: Dict[str, int] = {}  # GFI only
        review_cells: List[Dict] = []

        body_columns_by_index = {
            int(col["index"]): col
            for col in body_columns
            if col.get("index") is not None
        }

        for side in ("left", "right"):
            trip_idx = by_side[side].get("trip")
            poles_idx = by_side[side].get("poles")
            sf_idx = by_side[side].get("specialFeatures")
            side_row_spans = row_spans_by_side.get(side) or []

            if trip_idx is None or poles_idx is None or not side_row_spans:
                continue

            trip_col = body_columns_by_index.get(int(trip_idx))
            poles_col = body_columns_by_index.get(int(poles_idx))

            if trip_col is None or poles_col is None:
                continue

            trip_tokens = tokens_by_col_index.get(trip_idx, [])
            poles_tokens = tokens_by_col_index.get(poles_idx, [])
            sf_tokens = (
                tokens_by_col_index.get(sf_idx, [])
                if sf_idx is not None
                else []
            )

            for row_idx, (row_top, row_bottom) in enumerate(side_row_spans):
                trip_text = self._row_text_for_column(
                    row_top,
                    row_bottom,
                    trip_tokens,
                )
                poles_text = self._row_text_for_column(
                    row_top,
                    row_bottom,
                    poles_tokens,
                )

                amps = self._parse_trip_value(trip_text)
                poles = self._parse_poles_value(poles_text)

                trip_has_text = bool(str(trip_text or "").strip())
                poles_has_text = bool(str(poles_text or "").strip())

                trip_has_marks = _cell_has_visual_content(
                    gray_body,
                    trip_col["x_left"],
                    trip_col["x_right"],
                    row_top,
                    row_bottom,
                )

                poles_has_marks = _cell_has_visual_content(
                    gray_body,
                    poles_col["x_left"],
                    poles_col["x_right"],
                    row_top,
                    row_bottom,
                )

                combined_row_text = (
                    f"{trip_text or ''} {poles_text or ''}"
                ).strip().upper()

                is_spare_or_space = (
                    "SPARE" in combined_row_text
                    or "SPACE" in combined_row_text
                )

                # First check placeholders successfully returned by OCR.
                trip_text_placeholder = _is_blank_placeholder_text(
                    trip_text
                )
                poles_text_placeholder = _is_blank_placeholder_text(
                    poles_text
                )

                # If OCR returned no text but visible marks exist, perform a
                # very strict visual-placeholder check.
                trip_visual_placeholder = None
                if not trip_has_text and trip_has_marks:
                    trip_visual_placeholder = (
                        _detect_visual_blank_placeholder(
                            gray_body,
                            trip_col["x_left"],
                            trip_col["x_right"],
                            row_top,
                            row_bottom,
                        )
                    )

                poles_visual_placeholder = None
                if not poles_has_text and poles_has_marks:
                    poles_visual_placeholder = (
                        _detect_visual_blank_placeholder(
                            gray_body,
                            poles_col["x_left"],
                            poles_col["x_right"],
                            row_top,
                            row_bottom,
                        )
                    )

                # These variables are effectively the explicit cell tags:
                #
                #   text placeholder:
                #       OCR read "--", "_", "..", etc.
                #
                #   visual placeholder:
                #       OCR returned nothing, but the image contains a clean,
                #       narrowly defined dash/dot placeholder.
                trip_is_placeholder = (
                    trip_text_placeholder
                    or trip_visual_placeholder is not None
                )

                poles_is_placeholder = (
                    poles_text_placeholder
                    or poles_visual_placeholder is not None
                )

                if self.debug:
                    if trip_is_placeholder:
                        print(
                            "[SeparatedLayoutParser] "
                            f"side={side} row={row_idx} trip placeholder "
                            f"text={trip_text!r} "
                            f"visual={trip_visual_placeholder!r}"
                        )

                    if poles_is_placeholder:
                        print(
                            "[SeparatedLayoutParser] "
                            f"side={side} row={row_idx} poles placeholder "
                            f"text={poles_text!r} "
                            f"visual={poles_visual_placeholder!r}"
                        )

                # Placeholder cells may contain visible ink, but that ink is
                # intentional blank notation and must not make the row occupied.
                trip_cell_has_content = (
                    not trip_is_placeholder
                    and (trip_has_text or trip_has_marks)
                )

                poles_cell_has_content = (
                    not poles_is_placeholder
                    and (poles_has_text or poles_has_marks)
                )

                row_has_content = (
                    trip_cell_has_content
                    or poles_cell_has_content
                )

                # Entirely empty row:
                # no green, blue, or red overlays.
                if not row_has_content:
                    continue

                # Intentionally unused breaker position:
                # no overlay and no breaker.
                if is_spare_or_space:
                    continue

                # Amperage cell:
                #   valid -> green
                #   invalid/missing within an occupied row -> red
                if trip_is_placeholder:
                    pass

                elif amps is not None:
                    review_cells.append(
                        {
                            "side": side,
                            "rowIndex": int(row_idx),
                            "rowTop": int(row_top),
                            "rowBottom": int(row_bottom),
                            "xLeft": int(trip_col["x_left"]),
                            "xRight": int(trip_col["x_right"]),
                            "role": "trip",
                            "status": "good",
                            "text": trip_text,
                            "value": int(amps),
                        }
                    )
                else:
                    review_cells.append(
                        {
                            "side": side,
                            "rowIndex": int(row_idx),
                            "rowTop": int(row_top),
                            "rowBottom": int(row_bottom),
                            "xLeft": int(trip_col["x_left"]),
                            "xRight": int(trip_col["x_right"]),
                            "role": "trip",
                            "status": "issue",
                            "text": trip_text,
                            "value": None,
                            "reason": (
                                "Amperage was missing or could not be parsed "
                                "in an occupied breaker row."
                            ),
                        }
                    )

                # Poles cell:
                #   valid -> blue
                #   invalid/missing within an occupied row -> red
                if poles_is_placeholder:
                    pass

                elif poles is not None:
                    review_cells.append(
                        {
                            "side": side,
                            "rowIndex": int(row_idx),
                            "rowTop": int(row_top),
                            "rowBottom": int(row_bottom),
                            "xLeft": int(poles_col["x_left"]),
                            "xRight": int(poles_col["x_right"]),
                            "role": "poles",
                            "status": "good",
                            "text": poles_text,
                            "value": int(poles),
                        }
                    )
                else:
                    review_cells.append(
                        {
                            "side": side,
                            "rowIndex": int(row_idx),
                            "rowTop": int(row_top),
                            "rowBottom": int(row_bottom),
                            "xLeft": int(poles_col["x_left"]),
                            "xRight": int(poles_col["x_right"]),
                            "role": "poles",
                            "status": "issue",
                            "text": poles_text,
                            "value": None,
                            "reason": (
                                "Poles were missing or could not be parsed "
                                "in an occupied breaker row."
                            ),
                        }
                    )

                # A breaker is included only when both values are valid.
                if amps is None or poles is None:
                    continue

                sf_text = ""
                if sf_idx is not None and sf_tokens:
                    sf_text = self._row_text_for_column(
                        row_top,
                        row_bottom,
                        sf_tokens,
                    )

                gfi_flag = normalize_gfi_modifier(sf_text)
                key = f"{poles}P_{amps}A"

                if gfi_flag:
                    gfi_breaker_counts[key] = (
                        gfi_breaker_counts.get(key, 0) + 1
                    )
                else:
                    breaker_counts[key] = breaker_counts.get(key, 0) + 1

                rec = {
                    "side": side,
                    "rowIndex": int(row_idx),
                    "rowTop": int(row_top),
                    "rowBottom": int(row_bottom),
                    "amperage": int(amps),
                    "poles": int(poles),
                    "tripText": trip_text,
                    "polesText": poles_text,
                }

                if gfi_flag:
                    rec["specialFeatures"] = gfi_flag

                if sf_text:
                    rec["specialFeaturesText"] = sf_text

                detected_breakers.append(rec)

        if self.debug:
            print(
                f"[SeparatedLayoutParser] detected_breakers={len(detected_breakers)}, "
                f"unique combos={len(breaker_counts)} (non-GFI), "
                f"gfi combos={len(gfi_breaker_counts)}"
            )

        return {
            "layout": "separated",
            "bodyColumns": body_columns,
            "detected_breakers": detected_breakers,
            "breakerCounts": breaker_counts,
            "gfiBreakerCounts": gfi_breaker_counts,
            "reviewCells": review_cells,
        }

class CombinedLayoutParser:
    """
    Handles parsing when the header layout is 'combined':
      - Trip + poles live together in a single combo column (per side).
      - Identifies body rows from the grid lines.
      - OCRs the combo body strips (split into top/bottom halves).
      - For each row, parses a combined "amps/poles" text like:
            20A 1P
            20A1P
            20 A 1 P
            20A/1P
            20/1
            20-1
            1 P 20 A
        into (amps, poles), with guards:
          * poles ∈ {1,2,3}
          * amps > 0 and amps % 5 == 0   (amps always end in 0 or 5)
    """

    def __init__(self, *, debug: bool = False, reader=None):
        self.debug = bool(debug)
        self.reader = reader  # shared EasyOCR instance (same as header)

    def _row_text_for_column(self, row_top: int, row_bottom: int, col_tokens):
        """
        For a given column + row band, stitch together all tokens that fall
        in that vertical band, ordered left→right.
        """
        if not col_tokens:
            return ""

        parts = []
        for tok in col_tokens:
            x1, y1, x2, y2 = tok["box_page"]
            y_center = 0.5 * (y1 + y2)
            if y_center < row_top or y_center >= row_bottom:
                continue
            parts.append((x1, tok["text"]))

        if not parts:
            return ""

        parts.sort(key=lambda t: t[0])
        return " ".join(p[1] for p in parts).strip()

    def _detect_row_divider_lines_in_strip(self, strip_gray):
        """
        Given a single body combo-strip in GRAY (gridless),
        detect strong horizontal 'row divider' lines that span
        ~95%+ of the strip width.

        Returns:
            List of y-centers (ints) in STRIP-LOCAL coordinates.
        """
        import cv2

        if strip_gray is None or strip_gray.size == 0:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        if H_strip <= 0 or W_strip <= 0:
            return []

        # 1) Binarize (invert so lines are white on black)
        blur = cv2.GaussianBlur(strip_gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        # 2) Emphasize horizontal strokes with a wide, thin kernel
        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(15, int(0.40 * W_strip)),  # fairly wide
                1,                             # very thin vertically
            ),
        )
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

        # 3) Connected components to find horizontal blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            h_candidates,
            connectivity=8,
        )
        if num_labels <= 1:
            return []

        # Lines must span >= 95% of the strip width and be thin
        min_full_w = int(0.95 * W_strip)
        max_thick = max(2, int(0.03 * H_strip))

        ys_raw = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w < min_full_w:
                continue
            if h > max_thick:
                continue

            yc = y + h // 2
            ys_raw.append(int(yc))

        if not ys_raw:
            return []

        ys_raw.sort()

        # Collapse near-duplicate line centers
        merged = []
        MERGE_PX = 3
        for y in ys_raw:
            if not merged or abs(y - merged[-1]) > MERGE_PX:
                merged.append(y)

        if self.debug:
            print(
                f"[CombinedLayoutParser] Detected {len(merged)} long horizontal lines "
                f"in strip (H={H_strip}, W={W_strip})."
            )

        return merged

    def _compute_row_bands_in_strip(self, strip_gray):
        """
        Given a combo body-strip in GRAY (gridless), use the detected long
        horizontal lines to define row bands in STRIP-LOCAL coordinates.

          - Regions:
                top_of_strip → first_line,
                each line_i → line_{i+1},
                last_line → bottom_of_strip
            as potential row bands.
          - This way, we still get a top row and bottom row even if there
            is no grid line at the very top or very bottom of the body.

        Returns:
            List of (row_top, row_bottom) in STRIP-LOCAL coordinates.
        """

        if strip_gray is None or strip_gray.size == 0:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        if H_strip <= 0 or W_strip <= 0:
            return []

        line_ys = self._detect_row_divider_lines_in_strip(strip_gray)
        if not line_ys:
            # No lines at all -> treat whole strip as one band
            return [(0, H_strip)]

        # Sort and clamp
        line_ys = sorted(int(y) for y in line_ys)
        line_ys = [max(0, min(H_strip - 1, y)) for y in line_ys]

        # Build raw segments:
        # [0 -> first_line], [line_i -> line_{i+1}], [last_line -> H_strip]
        segments = []
        prev = 0
        for y in line_ys:
            if y > prev:
                segments.append((prev, y))
            prev = y

        if H_strip > prev:
            segments.append((prev, H_strip))

        bands = []
        prev_bottom = 0
        for (top, bottom) in segments:
            if bottom <= top + 3:
                continue

            span = bottom - top
            # Smaller padding than before:2% of span, min 1 px
            pad = max(1, int(0.02 * span))

            # Slightly "inflate" compared to the raw segment, but keep ordering
            row_top = max(prev_bottom, top + pad - 1)
            row_bottom = min(H_strip, bottom - pad + 1)

            if row_bottom > row_top + 2:
                bands.append((row_top, row_bottom))
                prev_bottom = row_bottom

        if self.debug:
            print(
                f"[CombinedLayoutParser] _compute_row_bands_in_strip: "
                f"lines={line_ys} → segments={segments} → {len(bands)} row bands"
            )

        if not bands:
            # Ultra-defensive fallback
            return [(0, H_strip)]

        return bands

    def _ocr_body_column_by_rows(
        self,
        strip_gray,
        x_left: int,
        body_y_top: int,
        row_bands_strip,
        debug_prep_path: Optional[str] = None,
        row_value_ok=None,
    ):
        """
        OCR a single combo body strip ROW-BY-ROW.

        Inputs:
          - strip_gray: gray crop for this combo column, still carrying the table
                        grid; each row band is de-gridded by _prep_cell_for_ocr
                        immediately before OCR
                        shape (H_strip, W_strip)
          - x_left:     column's left X in PAGE coordinates
          - body_y_top: Y at which this strip starts in PAGE coordinates
          - row_bands_strip: list of (row_top, row_bottom) in STRIP-LOCAL coords
          - debug_prep_path: when debug is on, where to write the BEFORE | AFTER
                        cell-prep comparison for this column
          - row_value_ok: optional predicate taking this row's token list and
                        returning whether it yields a usable amps/poles pair.
                        Enables the raw-cell fallback below.

        Returns:
          Flat list of OCR tokens, each with box_page coordinates.
        """
        if strip_gray is None or strip_gray.size == 0 or self.reader is None:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        tokens = []

        prep_vis = (
            strip_gray.copy()
            if (self.debug and debug_prep_path)
            else None
        )

        for (row_top, row_bottom) in row_bands_strip:
            row_top = max(0, min(H_strip - 1, int(row_top)))
            row_bottom = max(row_top + 1, min(H_strip, int(row_bottom)))
            if row_bottom <= row_top:
                continue

            row_gray = strip_gray[row_top:row_bottom, :]
            if row_gray.size == 0:
                continue

            # Strip the table borders out of this cell first. The crop keeps its
            # exact dimensions, so the box mapping further down stays valid.
            row_prepped = _prep_cell_for_ocr(row_gray)

            if prep_vis is not None:
                prep_vis[row_top:row_bottom, :] = row_prepped

            row_tokens = self._ocr_row_band_tokens(row_prepped)

            # See the note on the separated-layout parser: a successful prepped
            # read always wins, but a cell the de-gridding damaged is re-read
            # untouched so this parser can never resolve fewer rows than the
            # pre-cell-prep parser did.
            if row_value_ok is not None and not row_value_ok(row_tokens):
                raw_tokens = self._ocr_row_band_tokens(row_gray)
                if row_value_ok(raw_tokens):
                    row_tokens = raw_tokens
                    if self.debug:
                        print(
                            f"[{PARSER_VERSION}] raw-cell fallback recovered combo row "
                            f"y={body_y_top + row_top}-{body_y_top + row_bottom} "
                            f"x={x_left}: {[t[0] for t in raw_tokens]}"
                        )

            for txt_clean, conf_f, (x1_row, y1_row, x2_row, y2_row) in row_tokens:
                # Map row-local -> STRIP-LOCAL -> PAGE
                x1_page = x_left + x1_row
                x2_page = x_left + x2_row
                y1_page = body_y_top + row_top + y1_row
                y2_page = body_y_top + row_top + y2_row

                tokens.append(
                    {
                        "text": txt_clean,
                        "conf": conf_f,
                        "box_page": [int(x1_page), int(y1_page), int(x2_page), int(y2_page)],
                    }
                )

        if prep_vis is not None:
            _save_cell_prep_comparison(strip_gray, prep_vis, debug_prep_path)

        return tokens

    @staticmethod
    def _row_tokens_text(row_tokens) -> str:
        """Join a row's OCR tokens left-to-right into one cell string."""
        ordered = sorted(row_tokens, key=lambda t: t[2][0])
        return " ".join(t[0] for t in ordered).strip()

    def _combo_row_value_ok(self, row_tokens) -> bool:
        """A combo row is usable once it yields at least one (amps, poles) pair."""
        return bool(self._parse_combo_cell(self._row_tokens_text(row_tokens)))

    def _ocr_row_band_tokens(self, row_img):
        """
        OCR one already-cropped row band.

        Returns a list of (text, conf, (x1, y1, x2, y2)) with the box in
        ROW-LOCAL pixel coordinates, i.e. relative to row_img's own origin.
        """
        import cv2

        if row_img is None or row_img.size == 0 or self.reader is None:
            return []

        h_row, w_row = row_img.shape[:2]

        row_up = cv2.resize(
            row_img,
            None,
            fx=_HDR_OCR_SCALE,
            fy=_HDR_OCR_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        try:
            dets = _readtext_with_timeout(
                self.reader,
                row_up,
                detail=1,
                paragraph=False,
                allowlist=_HDR_OCR_ALLOWLIST,
                mag_ratio=1.0,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.4,
                low_text=0.25,
            )
        except Exception:
            dets = []

        out = []
        for box, txt, conf in dets:
            txt_clean = str(txt or "").strip()
            if not txt_clean:
                continue

            try:
                conf_f = float(conf or 0.0)
            except Exception:
                conf_f = 0.0

            min_conf = _HDR_MIN_CONF
            # Single characters (e.g. a lone pole digit) get a lower floor
            if len(txt_clean) == 1:
                min_conf = _HDR_MIN_CONF * 0.7

            if conf_f < min_conf:
                continue

            # box is in UPSCALED row coords -> map back to row-local
            pts_local = [
                (
                    int(p[0] / _HDR_OCR_SCALE),
                    int(p[1] / _HDR_OCR_SCALE),
                )
                for p in box
            ]
            xs = [p[0] for p in pts_local]
            ys = [p[1] for p in pts_local]

            x1_row = max(0, min(w_row - 1, min(xs)))
            x2_row = max(0, min(w_row - 1, max(xs)))
            y1_row = max(0, min(h_row - 1, min(ys)))
            y2_row = max(0, min(h_row - 1, max(ys)))

            out.append((txt_clean, conf_f, (x1_row, y1_row, x2_row, y2_row)))

        return out

    def _normalize_digitish_chars(self, s: str) -> str:
        """
        Normalize characters commonly mis-OCR'd inside numeric runs:

          - S / Z / 5 / $ -> '5'
          - 0 / O / D / Q / G -> '0'
          - 1 / I / L / J / ! / | -> '1'

        We apply this only to the combo text, and SPARE/SPACE detection is
        done on the raw string *before* normalization.
        """
        mapping_5 = set("5S$Z")
        mapping_0 = set("0ODQG")
        mapping_1 = set("1ILJ!|")

        out = []
        for ch in s:
            cu = ch.upper()
            if cu in mapping_5:
                out.append("5")
            elif cu in mapping_0:
                out.append("0")
            elif cu in mapping_1:
                out.append("1")
            else:
                out.append(cu)
        return "".join(out)

    def _is_valid_amps(self, val: int) -> bool:
        """
        Guard for valid amps:
          - > 0
          - ends in 0 or 5
          - optionally, >= 10 (to avoid silly 5A parses)
        """
        if val is None:
            return False
        if val <= 0:
            return False
        if val < 10:
            return False
        if val % 5 != 0:
            return False
        return True
    
    def _decode_amp_digits(self, digits: str) -> Optional[int]:
        """
        Normalize amperage digit strings coming from combo cells.

        Handles:
          - '20'   -> 20
          - '201'  -> 20   (slash misread as '1')
          - '20/'  -> 20   (after char normalization -> '201')
          - '20I'/'20L'/'20J'/'20T' -> 20  (via normalization → '201')
          - '251'  -> 25   (one junk digit inside)
        """
        if not digits:
            return None

        # First try straight parse
        try:
            val = int(digits)
        except ValueError:
            val = None

        if val is not None and self._is_valid_amps(val):
            return val

        # Classic "lost slash" pattern at the end, e.g.:
        #   '201' -> '20', '151' -> '15', '251' -> '25'
        if len(digits) >= 3 and digits.endswith("1"):
            try:
                val2 = int(digits[:-1])
            except ValueError:
                val2 = None
            if val2 is not None and self._is_valid_amps(val2):
                return val2

        # General: remove exactly one junk digit and see if we get a valid amp
        for i in range(len(digits)):
            candidate = digits[:i] + digits[i + 1 :]
            if not candidate:
                continue
            try:
                val2 = int(candidate)
            except ValueError:
                continue
            if self._is_valid_amps(val2):
                return val2

        return None

    def _decode_pole_digits(self, digits: str) -> Optional[int]:
        """
        Normalize pole digit strings.

        Handles ONLY:
        - '1'    -> 1
        - '2'    -> 2
        - '3'    -> 3
        - '11'   -> 1  (leading '1' is misread slash)
        - '12'   -> 2
        - '13'   -> 3

        Anything else ('20', '15', '101', etc.) is rejected.
        """
        if not digits:
            return None

        # Single digit is simple
        if len(digits) == 1:
            try:
                d = int(digits)
            except ValueError:
                return None
            return d if d in (1, 2, 3) else None

        # 2-digit pattern where leading '1' is our fake slash
        if len(digits) == 2 and digits[0] == "1" and digits[1] in "123":
            return int(digits[1])

        # All other multi-digit cases are considered invalid for poles
        return None

    def _validate_amp_pole_pair(self, amp_str: str, pole_str: str):
        """
        Given raw amp/pole substrings (already isolated by regex),
        normalize to digits and enforce:
          - poles ∈ {1,2,3} (with 11/12/13 cleanup)
          - amps % 5 == 0 and > 0 (with '201' / extra-digit cleanup)

        Always returns a 2-tuple: (amps or None, poles or None).
        """
        amp_digits = "".join(ch for ch in amp_str if ch.isdigit())
        pole_digits = "".join(ch for ch in pole_str if ch.isdigit())

        if not amp_digits or not pole_digits:
            return None, None

        amps = self._decode_amp_digits(amp_digits)
        if amps is None:
            return None, None

        poles = self._decode_pole_digits(pole_digits)
        if poles is None:
            return None, None

        return amps, poles

    def _parse_combo_cell(self, text: str):
        """
        Parse a combined 'amps/poles' text blob into a list of (amps, poles) pairs.

        Handles:
          - 20A 1P
          - 20A1P
          - 20 A 1 P
          - 20A/1P
          - 20/1
          - 20-1
          - 1 P 20 A
          - 20 13  (→ 20 A / 3 P)
          - 20 12  (→ 20 A / 2 P)
          - 20 1 1 (→ 20 A / 1 P)

        Plus "lost slash" dense-digit cases like:
          - 2071  → 20 / 1
          - 25012 → 250 / 2

        Returns:
          list of (amps:int, poles:int). Empty list if nothing valid.
        """
        import re

        if not text:
            return []

        raw = text.upper().strip()

        # Ignore clearly marked spares/spaces
        if "SPARE" in raw or "SPACE" in raw:
            return []

        # Normalize mis-OCR'd chars (S/Z→5, O/Q/D/G→0, I/L/J/|/!→1)
        norm = self._normalize_digitish_chars(raw)

        pairs = []

        # --- 1) Explicit amp/pole regex patterns (amps-first and poles-first) ---

        amp_first_re = re.compile(
            r"""
            (?P<amp>\d{1,4})      # 1-4 digits (amps)
            \s*A?                 # optional 'A'
            \s*[/\-]?\s*          # optional separator '/', '-'
            (?P<pole>\d{1,2})     # 1-2 digits (poles: 1, 2, 3, 11,12,13, etc.)
            \s*P?                 # optional 'P'
            """,
            re.VERBOSE,
        )

        pole_first_re = re.compile(
            r"""
            (?P<pole>\d{1,2})     # 1-2 digits (poles)
            \s*P?                 # optional 'P'
            \s*[/\-]?\s*          # optional separator '/', '-'
            (?P<amp>\d{1,4})      # 1-4 digits (amps)
            \s*A?                 # optional 'A'
            """,
            re.VERBOSE,
        )

        for pattern in (amp_first_re, pole_first_re):
            for m in pattern.finditer(norm):
                amp_str = m.group("amp")
                pole_str = m.group("pole")
                amps, poles = self._validate_amp_pole_pair(amp_str, pole_str)
                if amps is not None and poles is not None:
                    pairs.append((amps, poles))

        if pairs:
            # Found at least one explicit combo; don't run other heuristics.
            return pairs

        # --- 2) Dense-digit "lost slash" heuristic (single pair max) ---

        digits = "".join(ch for ch in norm if ch.isdigit())
        if len(digits) >= 2:
            # Last digit is candidate pole; rest is amps-ish blob
            pole_candidate = self._decode_pole_digits(digits[-1:])
            left = digits[:-1]
            amp_candidate = self._decode_amp_digits(left)

            if pole_candidate is not None and amp_candidate is not None:
                return [(amp_candidate, pole_candidate)]

        # --- 3) Token-level heuristic for split tokens like '20 13', '20 1 1' ---

        tokens = [t for t in re.split(r"\s+", norm) if t]

        amp_candidates = []
        pole_candidates = []

        for tok in tokens:
            # Skip obvious junk like bare '/', '-', etc.
            if not any(ch.isdigit() for ch in tok):
                continue

            digit_str = "".join(ch for ch in tok if ch.isdigit())
            if not digit_str:
                continue

            amp_val = self._decode_amp_digits(digit_str)
            if amp_val is not None:
                amp_candidates.append(amp_val)

            pole_val = self._decode_pole_digits(digit_str)
            if pole_val is not None:
                pole_candidates.append(pole_val)

        if amp_candidates and pole_candidates:
            # Heuristic:
            #   - Use the *largest* amp (usually the multi-digit one like 20, 30, 50)
            #   - Use the *last* pole candidate (often from '13' or trailing '1')
            best_amp = max(amp_candidates)
            best_pole = pole_candidates[-1]
            return [(best_amp, best_pole)]

        # Nothing safely parsed
        return []

    def parse(self, analyzer_result: dict, header_scan: dict) -> dict:
        """
        Combined layout:
        - Combo column per side (amps+poles).
        - Optional specialFeatures column per side (NOTES or OPTIONS).
        - If specialFeatures exists on a side, we extract row text from it
            and attach it to each detected breaker record as 'specialFeaturesText'.
        """
        import os
        import cv2

        raw_gray = analyzer_result.get("gray")
        if raw_gray is None:
            raw_gray = _ensure_gray(analyzer_result)

        gray_body = raw_gray
        gray_lines = raw_gray

        if gray_body is None or gray_lines is None:
            if self.debug:
                print("[CombinedLayoutParser] Missing gray_body/gray_lines; cannot crop body columns.")
            return {"layout": "combined", "bodyColumns": [], "detected_breakers": [], "breakerCounts": {}}

        H, W = gray_body.shape[:2]

        src_path = analyzer_result.get("src_path")
        src_dir = analyzer_result.get("src_dir") or os.path.dirname(src_path or ".")
        debug_dir = analyzer_result.get("debug_dir") or os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(src_path or "panel"))[0]

        header_bottom_y = analyzer_result.get("header_bottom_y")
        footer_y = analyzer_result.get("footer_y")

        if not isinstance(header_bottom_y, (int, float)):
            if self.debug:
                print("[CombinedLayoutParser] Missing header_bottom_y; cannot determine body top. Bailing.")
                print(f"  header_bottom_y={header_bottom_y!r}")
            return {
                "layout": "combined",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
                "error": "Missing header_bottom_y from analyzer; cannot determine body band.",
            }

        if not isinstance(footer_y, (int, float)):
            if self.debug:
                print("[CombinedLayoutParser] Missing footer_y; cannot determine body bottom. Bailing.")
                print(f"  footer_y={footer_y!r}")
            return {
                "layout": "combined",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
                "error": "Missing footer_y from analyzer; cannot determine body band.",
            }

        body_y_top = max(0, int(header_bottom_y))
        body_y_bottom = int(footer_y)

        if self.debug:
            print(
                f"[CombinedLayoutParser] Body Y-range: [{body_y_top}, {body_y_bottom}) "
                f"(H={H})  (using header_bottom_y→footer_y as body band)"
            )

        if body_y_bottom <= body_y_top + 4:
            if self.debug:
                print("[CombinedLayoutParser] Body band too small; skipping body columns.")
            return {"layout": "combined", "bodyColumns": [], "detected_breakers": [], "breakerCounts": {}}

        normalized = header_scan.get("normalizedColumns") or {}
        if normalized.get("layout") != "combined":
            if self.debug:
                print(
                    "[CombinedLayoutParser] Warning: called with layout "
                    f"{normalized.get('layout')}; expected 'combined'."
                )
            return {
                "layout": normalized.get("layout", "unknown"),
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
            }

        cols_summary = normalized.get("columns", []) or []

        # Include optional specialFeatures
        wanted_roles = {"combo", "specialFeatures"}

        body_columns = []

        for col in cols_summary:
            role = col.get("role")
            if role not in wanted_roles:
                continue

            x_left = max(0, int(col.get("x_left", 0)))
            x_right = min(W - 1, int(col.get("x_right", W - 1)))
            if x_right <= x_left + 1:
                continue

            body_strip = gray_body[body_y_top:body_y_bottom, x_left:x_right]
            if body_strip.size == 0:
                continue

            body_columns.append(
                {
                    "index": col["index"],
                    "role": role,
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_top": body_y_top,
                    "y_bottom": body_y_bottom,
                    "debugImageOverlay": None,
                }
            )

        if self.debug:
            print(f"[CombinedLayoutParser] Extracted {len(body_columns)} body columns for combo/specialFeatures.")

        if not body_columns:
            return {"layout": "combined", "bodyColumns": body_columns, "detected_breakers": [], "breakerCounts": {}}

        # Assign side based on PRIMARY role only (combo), so specialFeatures doesn't skew center_x
        primary_cols = [c for c in body_columns if c["role"] == "combo"]
        if primary_cols:
            global_left = min(c["x_left"] for c in primary_cols)
            global_right = max(c["x_right"] for c in primary_cols)
        else:
            global_left = min(c["x_left"] for c in body_columns)
            global_right = max(c["x_right"] for c in body_columns)

        center_x = 0.5 * (global_left + global_right)

        for col in body_columns:
            col_center = 0.5 * (col["x_left"] + col["x_right"])
            col["side"] = "left" if col_center < center_x else "right"

        by_side: Dict[str, Dict[str, Optional[int]]] = {
            "left": {"combo": None, "specialFeatures": None},
            "right": {"combo": None, "specialFeatures": None},
        }

        for col in body_columns:
            side = col["side"]
            role = col["role"]
            if by_side[side][role] is None:
                by_side[side][role] = col["index"]

        if self.debug:
            print("[CombinedLayoutParser] columns by side:", by_side)

        # Determine row bands per side from combo reference
        combo_cols = [c for c in body_columns if c["role"] == "combo"]
        row_bands_by_side = {"left": None, "right": None}
        row_spans_by_side = {"left": [], "right": []}

        for side in ("left", "right"):
            side_cols = [c for c in combo_cols if c.get("side") == side]
            if not side_cols:
                continue

            ref_col = side_cols[0]
            ref_strip = gray_body[ref_col["y_top"]:ref_col["y_bottom"], ref_col["x_left"]:ref_col["x_right"]]
            if ref_strip.size == 0:
                row_bands_strip = [(0, ref_col["y_bottom"] - ref_col["y_top"])]
            else:
                row_bands_strip = self._compute_row_bands_in_strip(ref_strip)
                if not row_bands_strip:
                    row_bands_strip = [(0, ref_col["y_bottom"] - ref_col["y_top"])]

            row_bands_by_side[side] = row_bands_strip
            row_spans_by_side[side] = [(body_y_top + top_s, body_y_top + bottom_s) for (top_s, bottom_s) in row_bands_strip]

            if self.debug:
                print(f"[CombinedLayoutParser] Side='{side}' using {len(row_spans_by_side[side])} row bands from horizontal lines.")

        # OCR each wanted column by rows
        tokens_by_col_index: Dict[int, list] = {}

        for col in body_columns:
            idx = col["index"]
            x_l = col["x_left"]
            x_r = col["x_right"]
            y_top = col["y_top"]
            y_bot = col["y_bottom"]
            side = col.get("side", "left")

            row_bands_strip = row_bands_by_side.get(side)
            if not row_bands_strip:
                tokens_by_col_index[idx] = []
                continue

            strip = gray_body[y_top:y_bot, x_l:x_r]
            if strip.size == 0:
                tokens_by_col_index[idx] = []
                continue

            cellprep_path = (
                os.path.join(
                    debug_dir,
                    f"{base}_parser_body_combined_col{idx}_cellprep.png",
                )
                if self.debug
                else None
            )

            tokens_by_col_index[idx] = self._ocr_body_column_by_rows(
                strip_gray=strip,
                x_left=x_l,
                body_y_top=y_top,
                row_bands_strip=row_bands_strip,
                debug_prep_path=cellprep_path,
                row_value_ok=self._combo_row_value_ok,
            )

        # Debug overlays unchanged (works for specialFeatures too)
        if self.debug:
            for col in body_columns:
                idx = col["index"]
                role = col["role"]
                x_l = col["x_left"]
                x_r = col["x_right"]
                y_top = col["y_top"]
                y_bot = col["y_bottom"]

                col_tokens = tokens_by_col_index.get(idx, [])
                if not col_tokens:
                    continue

                strip = gray_body[y_top:y_bot, x_l:x_r]
                if strip.size == 0:
                    continue

                H_strip, W_strip = strip.shape[:2]
                vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

                cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 255), 1)
                lbl = f"BODY {role.upper()}  col={idx}"
                cv2.putText(vis, lbl, (8, max(16, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                line_ys = self._detect_row_divider_lines_in_strip(strip)
                for yc in line_ys:
                    yc_int = max(0, min(H_strip - 1, int(yc)))
                    cv2.line(vis, (0, yc_int), (W_strip - 1, yc_int), (255, 0, 0), 1)
                    cv2.putText(vis, "ROW", (4, max(10, yc_int - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

                for tok in col_tokens:
                    x1p, y1p, x2p, y2p = tok["box_page"]
                    text = tok.get("text", "")
                    conf = tok.get("conf", 0.0)

                    x1 = max(0, min(W_strip - 1, x1p - x_l))
                    x2 = max(0, min(W_strip - 1, x2p - x_l))
                    y1 = max(0, min(H_strip - 1, y1p - y_top))
                    y2 = max(0, min(H_strip - 1, y2p - y_top))

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    label = f"{text} ({conf:.2f})"
                    ty = y1 - 4 if y1 - 4 > 10 else y2 + 12
                    cv2.putText(vis, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                overlay_path = os.path.join(debug_dir, f"{base}_parser_body_combined_col{idx}_overlay.png")
                try:
                    cv2.imwrite(overlay_path, vis)
                    col["debugImageOverlay"] = overlay_path
                except Exception as e:
                    print(f"[CombinedLayoutParser] Failed to write body overlay col {idx} ({role}): {e}")
                    col["debugImageOverlay"] = None

        detected_breakers = []
        breaker_counts: Dict[str, int] = {}      # NON-GFI only
        gfi_breaker_counts: Dict[str, int] = {}  # GFI only
        review_cells: List[Dict] = []

        body_columns_by_index = {
            int(col["index"]): col
            for col in body_columns
            if col.get("index") is not None
        }

        for side in ("left", "right"):
            combo_idx = by_side[side].get("combo")
            sf_idx = by_side[side].get("specialFeatures")
            side_row_spans = row_spans_by_side.get(side) or []

            if combo_idx is None or not side_row_spans:
                continue

            combo_col = body_columns_by_index.get(int(combo_idx))
            if combo_col is None:
                continue

            combo_tokens = tokens_by_col_index.get(combo_idx, [])
            sf_tokens = (
                tokens_by_col_index.get(sf_idx, [])
                if sf_idx is not None
                else []
            )

            for row_idx, (row_top, row_bottom) in enumerate(side_row_spans):
                combo_text = self._row_text_for_column(
                    row_top,
                    row_bottom,
                    combo_tokens,
                )

                combo_pairs = self._parse_combo_cell(combo_text)

                combo_text_normalized = (
                    str(combo_text or "").strip().upper()
                )

                combo_has_text = bool(combo_text_normalized)

                combo_has_marks = _cell_has_visual_content(
                    gray_body,
                    combo_col["x_left"],
                    combo_col["x_right"],
                    row_top,
                    row_bottom,
                )

                # Placeholder recognized directly from OCR text.
                combo_text_placeholder = (
                    _is_blank_placeholder_text(
                        combo_text_normalized
                    )
                )

                # Placeholder recognized visually only when OCR returned no
                # usable text but the image still contains visible marks.
                combo_visual_placeholder = None

                if not combo_has_text and combo_has_marks:
                    combo_visual_placeholder = (
                        _detect_visual_blank_placeholder(
                            gray_body,
                            combo_col["x_left"],
                            combo_col["x_right"],
                            row_top,
                            row_bottom,
                        )
                    )

                combo_is_placeholder = (
                    combo_text_placeholder
                    or combo_visual_placeholder is not None
                )

                if self.debug and combo_is_placeholder:
                    print(
                        "[CombinedLayoutParser] "
                        f"side={side} row={row_idx} combo placeholder "
                        f"text={combo_text!r} "
                        f"visual={combo_visual_placeholder!r}"
                    )

                is_spare_or_space = (
                    "SPARE" in combo_text_normalized
                    or "SPACE" in combo_text_normalized
                )

                # Empty rows, dash placeholders, and intentional spare/space rows
                # receive no overlay.
                if not combo_has_text and not combo_has_marks:
                    continue

                if combo_is_placeholder:
                    continue

                if is_spare_or_space:
                    continue

                if combo_pairs:
                    review_cells.append(
                        {
                            "side": side,
                            "rowIndex": int(row_idx),
                            "rowTop": int(row_top),
                            "rowBottom": int(row_bottom),
                            "xLeft": int(combo_col["x_left"]),
                            "xRight": int(combo_col["x_right"]),
                            "role": "combo",
                            "status": "good",
                            "text": combo_text,
                            "pairs": [
                                {
                                    "amperage": int(amps),
                                    "poles": int(poles),
                                }
                                for amps, poles in combo_pairs
                            ],
                        }
                    )
                elif combo_has_text or combo_has_marks:
                    review_cells.append(
                        {
                            "side": side,
                            "rowIndex": int(row_idx),
                            "rowTop": int(row_top),
                            "rowBottom": int(row_bottom),
                            "xLeft": int(combo_col["x_left"]),
                            "xRight": int(combo_col["x_right"]),
                            "role": "combo",
                            "status": "issue",
                            "text": combo_text,
                            "pairs": [],
                            "reason": (
                                "Combined breaker cell contained marks "
                                "but amperage/poles could not be parsed."
                            ),
                        }
                    )

                # Empty cells produce no review record and therefore no overlay.
                if not combo_pairs:
                    continue

                sf_text = ""
                if sf_idx is not None and sf_tokens:
                    sf_text = self._row_text_for_column(
                        row_top,
                        row_bottom,
                        sf_tokens,
                    )

                gfi_flag = normalize_gfi_modifier(sf_text)

                if self.debug:
                    print(
                        f"[CombinedLayoutParser] side={side} "
                        f"row={row_idx} text='{combo_text}' -> {combo_pairs}"
                    )

                for amps, poles in combo_pairs:
                    key = f"{poles}P_{amps}A"

                    if gfi_flag:
                        gfi_breaker_counts[key] = (
                            gfi_breaker_counts.get(key, 0) + 1
                        )
                    else:
                        breaker_counts[key] = breaker_counts.get(key, 0) + 1

                    rec = {
                        "side": side,
                        "rowIndex": int(row_idx),
                        "rowTop": int(row_top),
                        "rowBottom": int(row_bottom),
                        "amperage": int(amps),
                        "poles": int(poles),
                        "comboText": combo_text,
                    }

                    if gfi_flag:
                        rec["specialFeatures"] = gfi_flag

                    if sf_text:
                        rec["specialFeaturesText"] = sf_text

                    detected_breakers.append(rec)

        if self.debug:
            print(
                f"[CombinedLayoutParser] detected_breakers={len(detected_breakers)}, "
                f"unique combos={len(breaker_counts)} (non-GFI), "
                f"gfi combos={len(gfi_breaker_counts)}"
            )

        return {
            "layout": "combined",
            "bodyColumns": body_columns,
            "detected_breakers": detected_breakers,
            "breakerCounts": breaker_counts,
            "gfiBreakerCounts": gfi_breaker_counts,
            "reviewCells": review_cells,
        }

class BreakerTableParser:
    """
    Parser6 orchestrator.
    For now:
      - treats analyzer header/footer as hard anchors
      - runs HeaderBandScanner to OCR the header band and emit a debug overlay
      - returns a minimal legacy-compatible payload:
          spaces: from analyzer
          detected_breakers: empty
    """

    def __init__(self, *, debug: bool = False, reader=None):
        self.debug = bool(debug)
        # Share OCR reader with analyzer when possible
        if reader is not None:
            self.reader = reader
        elif _HAS_OCR:
            try:
                self.reader = easyocr.Reader(["en"], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(["en"], gpu=False)
        else:
            self.reader = None

        self._header_scanner = HeaderBandScanner(debug=self.debug, reader=self.reader)
        self._separated_parser = SeparatedLayoutParser(debug=self.debug, reader=self.reader)        
        self._combined_parser  = CombinedLayoutParser(debug=self.debug, reader=self.reader)

    @staticmethod
    def _ckt_number(text):
        """
        Return one positive circuit number from a CKT cell.

        Multiple different numbers are treated as unreadable so the
        physical odd/even sequence can infer the correct value.
        """
        matches = re.findall(r"\d{1,3}", str(text or ""))

        if len(matches) != 1:
            return None

        try:
            value = int(matches[0])
        except (TypeError, ValueError):
            return None

        return value if value > 0 else None


    def _attach_circuit_positions(
        self,
        analyzer_result,
        header_scan,
        layout,
        spaces,
        detected_breakers,
        review_cells,
    ):
        """
        Add display-only circuit positions without changing breaker values.

        The CKT column supplies the physical one-row grid. Printed circuit
        numbers validate that grid; missing numbers are inferred from the
        required odd/even sequence.

        Left:  1, 3, 5, 7...
        Right: 2, 4, 6, 8...
        """
        issues = []
        gray = (analyzer_result or {}).get("gray")

        if gray is None or not hasattr(gray, "shape") or not spaces:
            return issues

        try:
            body_top = int(
                analyzer_result.get("header_bottom_y")
            )
            body_bottom = int(
                analyzer_result.get("footer_y")
            )
            spaces = int(spaces)

        except (TypeError, ValueError):
            return issues

        if body_bottom <= body_top or spaces <= 0:
            return issues

        normalized = (
            (header_scan or {})
            .get("normalizedColumns")
            or {}
        )

        columns = normalized.get("columns") or []

        parser = (
            self._combined_parser
            if layout == "combined"
            else self._separated_parser
        )

        image_width = int(gray.shape[1])

        expected_rows = max(
            1,
            (spaces + 1) // 2,
        )

        rows_by_side = {
            "left": [],
            "right": [],
        }

        grid_uncertain = {
            "left": False,
            "right": False,
        }

        ckt_columns = {
            "left": None,
            "right": None,
        }

        # Find the primary CKT column on each side.
        for column in columns:
            if (
                not isinstance(column, dict)
                or column.get("role") != "ckt"
            ):
                continue

            side = str(
                column.get("side")
                or ""
            ).strip().lower()

            if side not in ckt_columns:
                try:
                    center_x = (
                        float(column.get("x_left"))
                        + float(column.get("x_right"))
                    ) / 2.0

                    side = (
                        "left"
                        if center_x < image_width / 2.0
                        else "right"
                    )

                except (TypeError, ValueError):
                    continue

            if ckt_columns[side] is None:
                ckt_columns[side] = column

        # Build the complete physical CKT-row grid for each side.
        for side in ("left", "right"):
            column = ckt_columns.get(side)

            if not isinstance(column, dict):
                continue

            try:
                x_left = max(
                    0,
                    int(column.get("x_left")),
                )

                x_right = min(
                    image_width,
                    int(column.get("x_right")),
                )

            except (TypeError, ValueError):
                continue

            if x_right <= x_left:
                continue

            strip = gray[
                body_top:body_bottom,
                x_left:x_right,
            ]

            if strip is None or strip.size == 0:
                continue

            bands = list(
                parser._compute_row_bands_in_strip(
                    strip
                )
                or []
            )

            # If structural line detection did not return exactly one row
            # per circuit position, retain a complete evenly-spaced grid
            # and flag its positions for review.
            if len(bands) != expected_rows:
                grid_uncertain[side] = True
                strip_height = int(strip.shape[0])

                bands = [
                    (
                        round(
                            row_index
                            * strip_height
                            / expected_rows
                        ),
                        round(
                            (row_index + 1)
                            * strip_height
                            / expected_rows
                        ),
                    )
                    for row_index in range(
                        expected_rows
                    )
                ]

            def _valid_ckt_tokens(tokens):
                text = parser._row_tokens_text(
                    tokens
                )

                return (
                    self._ckt_number(text)
                    is not None
                )

            tokens = (
                parser._ocr_body_column_by_rows(
                    strip_gray=strip,
                    x_left=x_left,
                    body_y_top=body_top,
                    row_bands_strip=bands,
                    row_value_ok=_valid_ckt_tokens,
                )
            )

            first_circuit = (
                1 if side == "left" else 2
            )

            for (
                row_index,
                (local_top, local_bottom),
            ) in enumerate(bands):

                row_top = (
                    body_top
                    + int(local_top)
                )

                row_bottom = (
                    body_top
                    + int(local_bottom)
                )

                circuit = (
                    first_circuit
                    + row_index * 2
                )

                text = (
                    parser._row_text_for_column(
                        row_top,
                        row_bottom,
                        tokens,
                    )
                )

                observed = self._ckt_number(
                    text
                )

                conflict = (
                    observed is not None
                    and observed != circuit
                )

                rows_by_side[side].append(
                    {
                        "top": row_top,
                        "bottom": row_bottom,
                        "circuit": circuit,
                        "observed": observed,
                        "conflict": conflict,
                    }
                )

        def _map_span(
            side,
            row_top,
            row_bottom,
            span,
        ):
            """
            Select the contiguous CKT rows having the greatest physical
            overlap with this breaker cell.
            """
            rows = rows_by_side.get(side) or []

            try:
                span = max(
                    1,
                    int(span or 1),
                )

                row_top = float(row_top)
                row_bottom = float(row_bottom)

            except (TypeError, ValueError):
                return None, True

            if (
                not rows
                or row_bottom <= row_top
                or span > len(rows)
            ):
                return None, True

            breaker_center = (
                row_top + row_bottom
            ) / 2.0

            candidates = []

            for start in range(
                len(rows) - span + 1
            ):
                window = rows[
                    start:start + span
                ]

                overlap = sum(
                    max(
                        0.0,
                        min(
                            row_bottom,
                            row["bottom"],
                        )
                        - max(
                            row_top,
                            row["top"],
                        ),
                    )
                    for row in window
                )

                window_center = (
                    window[0]["top"]
                    + window[-1]["bottom"]
                ) / 2.0

                candidates.append(
                    (
                        overlap,
                        -abs(
                            breaker_center
                            - window_center
                        ),
                        -start,
                        start,
                    )
                )

            candidates.sort(
                reverse=True
            )

            best = candidates[0]

            ambiguous = (
                best[0] <= 0
            )

            if (
                len(candidates) > 1
                and abs(
                    best[0]
                    - candidates[1][0]
                ) <= 1.0
            ):
                ambiguous = True

            start = best[3]

            return (
                rows[start:start + span],
                ambiguous,
            )

        def _display_row(
            mapped_rows,
            poles,
        ):
            # 2P always displays on its top circuit.
            if poles == 2:
                return mapped_rows[0]

            # 3P always displays on its middle circuit.
            if poles >= 3:
                return mapped_rows[
                    len(mapped_rows) // 2
                ]

            return mapped_rows[0]

        seen_issues = {
            (
                issue["side"],
                issue["circuitNumber"],
                issue["reason"],
            )
            for issue in issues
        }

        def _add_issue(
            side,
            circuit,
            reason,
        ):
            key = (
                side,
                int(circuit),
                str(reason),
            )

            if key in seen_issues:
                return

            seen_issues.add(key)

            issues.append(
                {
                    "side": side,
                    "circuitNumber": int(
                        circuit
                    ),
                    "reason": str(reason),
                }
            )

        # Add circuit metadata to valid breaker records.
        for breaker in (
            detected_breakers or []
        ):
            if not isinstance(
                breaker,
                dict,
            ):
                continue

            side = str(
                breaker.get("side")
                or ""
            ).strip().lower()

            try:
                poles = int(
                    breaker.get("poles")
                    or 1
                )
            except (TypeError, ValueError):
                poles = 1

            mapped_rows, ambiguous = (
                _map_span(
                    side,
                    breaker.get("rowTop"),
                    breaker.get("rowBottom"),
                    poles,
                )
            )

            if not mapped_rows:
                breaker[
                    "circuitPositionIssue"
                ] = True

                continue

            display_row = _display_row(
                mapped_rows,
                poles,
            )

            circuit_numbers = [
                row["circuit"]
                for row in mapped_rows
            ]

            conflict = any(
                row["conflict"]
                for row in mapped_rows
            )

            uncertain = bool(
                grid_uncertain.get(side)
            )

            has_issue = (
                ambiguous
                or conflict
                or uncertain
            )

            breaker[
                "circuitNumbers"
            ] = circuit_numbers

            breaker[
                "startCircuitNumber"
            ] = circuit_numbers[0]

            # This is the exact circuit row where 2B should display
            # the breaker label.
            breaker[
                "circuitNumber"
            ] = display_row["circuit"]

            breaker[
                "circuitPositionSource"
            ] = (
                "ckt_ocr"
                if all(
                    row["observed"]
                    == row["circuit"]
                    for row in mapped_rows
                )
                else "ckt_sequence"
            )

            breaker[
                "circuitPositionIssue"
            ] = has_issue

        # Find a known pole count for existing red review cells.
        poles_by_cell = {}

        for cell in review_cells or []:
            if not isinstance(cell, dict):
                continue

            key = (
                str(
                    cell.get("side")
                    or ""
                ).strip().lower(),
                cell.get("rowIndex"),
                cell.get("rowTop"),
                cell.get("rowBottom"),
            )

            if cell.get("role") == "poles":
                try:
                    poles_by_cell[key] = int(
                        cell.get("value")
                    )
                except (TypeError, ValueError):
                    pass

        # Convert existing red overlay cells into output markers.
        for cell in review_cells or []:
            if (
                not isinstance(cell, dict)
                or cell.get("status") != "issue"
            ):
                continue

            side = str(
                cell.get("side")
                or ""
            ).strip().lower()

            key = (
                side,
                cell.get("rowIndex"),
                cell.get("rowTop"),
                cell.get("rowBottom"),
            )

            poles = poles_by_cell.get(
                key,
                1,
            )

            mapped_rows, _ambiguous = (
                _map_span(
                    side,
                    cell.get("rowTop"),
                    cell.get("rowBottom"),
                    poles,
                )
            )

            if not mapped_rows:
                continue

            circuit = _display_row(
                mapped_rows,
                poles,
            )["circuit"]

            cell["circuitNumber"] = (
                circuit
            )

            _add_issue(
                side,
                circuit,
                cell.get("reason")
                or (
                    "Breaker value could not "
                    "be read confidently."
                ),
            )

        return issues

    def parse_from_analyzer(self, analyzer_result: Dict) -> Dict:
        """
        Entry point used by the API.

        Right now:
          - Reads spaces from analyzer_result (corrected if available)
          - Runs the header-band OCR scan
          - If header layout is 'separated', runs SeparatedLayoutParser
          - Aggregates detected breakers into breakerCounts
          - Prints a human-readable summary to the terminal (when debug=True)
          - Returns JSON-safe parser result
        """
        if not isinstance(analyzer_result, dict):
            analyzer_result = {}

        # --- spaces: from analyzer 'spaces', or fall back to 'panel_size' (Analyzer12) ---
        raw_spaces = analyzer_result.get("spaces")
        if raw_spaces is None:
            raw_spaces = analyzer_result.get("panel_size")

        if raw_spaces is None:
            spaces = 0
        else:
            try:
                spaces = int(raw_spaces)
            except (TypeError, ValueError):
                spaces = 0

        header_scan = self._header_scanner.scan(analyzer_result)

        normalized = header_scan.get("normalizedColumns") or {}
        layout = normalized.get("layout", "unknown")

        separated_scan: Optional[Dict] = None
        combined_scan: Optional[Dict] = None
        detected_breakers: List[Dict] = []

        if layout == "separated":
            if self.debug:
                print("[BreakerTableParser] Layout 'separated' → using SeparatedLayoutParser.")
            separated_scan = self._separated_parser.parse(analyzer_result, header_scan)
            detected_breakers = separated_scan.get("detected_breakers", []) or []
        elif layout == "combined":
            if self.debug:
                print("[BreakerTableParser] Layout 'combined' → using CombinedLayoutParser.")
            combined_scan = self._combined_parser.parse(analyzer_result, header_scan)
            detected_breakers = combined_scan.get("detected_breakers", []) or []
        else:
            if self.debug:
                print(f"[BreakerTableParser] Layout '{layout}' → no body parser yet.")

        # --- Use counts from the layout parser ---
        breaker_counts: Dict[str, int] = {}
        gfi_breaker_counts: Dict[str, int] = {}
        review_cells: List[Dict] = []

        if layout == "separated" and separated_scan:
            breaker_counts = separated_scan.get("breakerCounts") or {}
            gfi_breaker_counts = separated_scan.get("gfiBreakerCounts") or {}
            review_cells = separated_scan.get("reviewCells") or []

        elif layout == "combined" and combined_scan:
            breaker_counts = combined_scan.get("breakerCounts") or {}
            gfi_breaker_counts = combined_scan.get("gfiBreakerCounts") or {}
            review_cells = combined_scan.get("reviewCells") or []

        breaker_position_issues = (
            self._attach_circuit_positions(
                analyzer_result=analyzer_result,
                header_scan=header_scan,
                layout=layout,
                spaces=spaces,
                detected_breakers=detected_breakers,
                review_cells=review_cells,
            )
        )

        if self.debug:
            print(
                f"[BreakerTableParser] Header band y:[{header_scan.get('band_y1')},"
                f"{header_scan.get('band_y2')}) tokens={len(header_scan.get('tokens', []))}"
            )

        # --- Build final result ---
        result = {
            "parserVersion": PARSER_VERSION,
            "name": None,
            "spaces": spaces,
            "detected_breakers": detected_breakers,
            "breakerCounts": breaker_counts,
            "gfiBreakerCounts": gfi_breaker_counts,
            "reviewCells": review_cells,
            "breakerPositionIssues": breaker_position_issues,
            "headerScan": header_scan,
            "separatedScan": separated_scan,
            "combinedScan": combined_scan,
        }

        # --- Terminal summary (only when debug=True) ---
        if self.debug:
            self._print_terminal_summary(analyzer_result, result)

        return result

    def _print_terminal_summary(self, analyzer_result: Dict, result: Dict) -> None:
        """
        Print a simple summary that mirrors the final JSON, per panel:

          Panel name - LIA
          Amps - 125A, main breaker amps - Unknown, volts - 480V, spaces - 42
          Breakers
            1 P, 20 A, count - 30
            3 P, 100 A, count - 4
            ...

        No per-breaker rows are printed here; only the aggregated tally.
        """

        # ---- Source image / crop ----
        src_path = analyzer_result.get("src_path") or analyzer_result.get("image_path")
        if src_path:
            src_name = os.path.basename(src_path)
        else:
            src_name = "Unknown source"

        # ---- Panel-level metadata ----

        # Name: prefer parser result, then analyzer, then header attrs if present
        name = (
            result.get("name")
            or analyzer_result.get("panel_name")
            or analyzer_result.get("name")
        )

        header_attrs = analyzer_result.get("header_attrs") or {}
        if not name:
            name = header_attrs.get("name")

        if not name:
            name = "Unknown"

        # Amps / main breaker amps / voltage:

        # Panel bus amps
        panel_amps = (
            analyzer_result.get("panel_amps")
            or analyzer_result.get("amps")
            or analyzer_result.get("rating_amps")
            or header_attrs.get("amperage")
        )

        # Main breaker amps
        main_breaker_amps = (
            analyzer_result.get("main_breaker_amps")
            or analyzer_result.get("main_amps")
            or analyzer_result.get("main_rating_amps")
        )

        # Voltage
        volts = (
            analyzer_result.get("voltage")
            or analyzer_result.get("volts")
            or analyzer_result.get("system_voltage")
            or header_attrs.get("voltage")
        )

        spaces = result.get("spaces")

        # Convert missing values to "Unknown" strings for nice printing
        def _fmt(v: Optional[object], suffix: str = "") -> str:
            if v is None:
                return "Unknown"
            try:
                if suffix:
                    return f"{int(v)}{suffix}"
                return str(v)
            except Exception:
                return str(v)

        breaker_counts: Dict[str, int] = result.get("breakerCounts") or {}
        gfi_counts: Dict[str, int] = result.get("gfiBreakerCounts") or {}

        # ---- Print ----
        print()
        print("====================================")
        print("Breakers")

        if not breaker_counts:
            print("  (none detected)")
            print("====================================")
            return

        # sort by poles, then amps numerically if possible
        def _sort_key(item):
            key, _count = item
            m = re.match(r"(\d+)P_(\d+)A", key)
            if not m:
                return (9999, 9999, key)
            p = int(m.group(1))
            a = int(m.group(2))
            return (p, a, key)

        def _print_section(title: str, counts: Dict[str, int]) -> None:
            print(title)
            if not counts:
                print("  (none detected)")
                return

            for key, count in sorted(counts.items(), key=_sort_key):
                m = re.match(r"(\d+)P_(\d+)A", key)
                if m:
                    poles = int(m.group(1))
                    amps = int(m.group(2))
                    print(f"  {poles} P, {amps} A, count - {count}")
                else:
                    # Fallback if key format ever changes
                    print(f"  {key}, count - {count}")

        _print_section("Breakers (non-GFI)", breaker_counts)
        print("------------------------------------")
        _print_section("Breakers (GFI)", gfi_counts)

        print("====================================")
