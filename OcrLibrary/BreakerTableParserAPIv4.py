# OcrLibrary/BreakerTableParserAPIv4.py
import sys, os, inspect
import cv2
import numpy as np
import re

# ----- ensure repo root on sys.path -----
_THIS_FILE  = os.path.abspath(__file__)
_OCRLIB_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT  = os.path.dirname(_OCRLIB_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

API_VERSION = "API_4"
API_ORIGIN  = __file__

# Panel validation snap map for spaces normalization
SNAP_MAP = {
    16: 18, 20: 18,
    28: 30, 32: 30,
    40: 42, 44: 42,
    52: 54, 56: 54,
    64: 66, 68: 66,
    70: 72, 74: 72,
    82: 84, 86: 84,
}
# Header-level validation (pre-table-parse)
VALID_VOLTAGES = {120, 208, 240, 480, 600}
AMP_MIN = 100
AMP_MAX = 1200
 
# --- Header band refinement (from dev "find header" env) ---
SEARCH_UP_FIRST     = True   # try above header first when picking band
MAX_DELTA_PX        = 80     # max distance from OCR header row
MIN_COVERAGE_FRAC   = 0.60   # row coverage threshold
MERGE_Y_TOLERANCE   = 3      # merge contiguous rows into bands

def ocr_tokens_in_header_band(gray, analyzer):
    """
    Run EasyOCR only in a top header band and return tokens with a flag
    for 'header-like' words.

    gray: single-channel uint8 image
    analyzer: BreakerTableAnalyzer instance (must have .reader)
    """
    if not hasattr(analyzer, "reader") or analyzer.reader is None:
        return []

    H, W = gray.shape
    # Focus on upper half-ish where header normally lives
    y1, y2 = int(0.08 * H), int(0.50 * H)
    roi = gray[y1:y2, :]

    def _ocr_pass(mag):
        return analyzer.reader.readtext(
            roi,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/",
            mag_ratio=mag,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.4,
            low_text=0.25,
        )

    try:
        det = _ocr_pass(1.6) + _ocr_pass(2.0)
    except Exception:
        det = []

    def N(s: str) -> str:
        s = (s or "").upper().replace("1", "I").replace("0", "O")
        return re.sub(r"[^A-Z0-9]", "", s)

    CATEGORY = {
        "CKT", "CCT", "TRIP", "POLES", "AMP", "AMPS", "BREAKER", "BKR", "SIZE",
        "DESCRIPTION", "DESIGNATION", "NAME", "LOADDESCRIPTION", "CIRCUITDESCRIPTION",
    }

    toks = []
    for box, txt, _conf in det:
        if not txt:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        norm = N(txt)
        toks.append({
            "text": txt,
            "is_header_word": any(k in norm for k in CATEGORY),
            "x1": int(min(xs)),
            "y1": int(min(ys)) + y1,
            "x2": int(max(xs)),
            "y2": int(max(ys)) + y1,
        })
    return toks

def infer_header_y_from_tokens(tokens, img_height: int, max_rel_height: float = 0.65) -> int | None:
    """
    Infer a single header_y from already OCR'd tokens, without additional OCR.

    Strategy:
      - Group tokens into ~16px horizontal bands.
      - Score each band by how many 'header words' it has.
      - Pick the band with the highest score.
      - Use the topmost y1 within that band as header_y.
      - Reject if it's too low on the page (> max_rel_height * H).
    """
    if not tokens:
        return None

    rows = {}  # band_y -> [tokens]
    for t in tokens:
        cy = (t["y1"] + t["y2"]) // 2
        band = (cy // 16) * 16
        rows.setdefault(band, []).append(t)

    best_band = None
    best_score = 0
    for band_y, row_tokens in rows.items():
        score = sum(1 for t in row_tokens if t.get("is_header_word"))
        if score > best_score:
            best_score = score
            best_band = band_y

    if best_band is None or best_score == 0:
        return None

    header_y = min(t["y1"] for t in rows[best_band])
    if header_y > int(max_rel_height * img_height):
        return None

    return int(header_y)

def _binarize_for_horiz(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw_inv = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return bw_inv  # white = ink


def _horiz_mask_from_bin(bw_inv: np.ndarray, W: int,
                         horiz_kernel_frac: float = 0.035,
                         border_trim_px: int = 2) -> np.ndarray:
    m = bw_inv.copy()
    if border_trim_px > 0:
        m[:border_trim_px, :] = 0
        m[-border_trim_px:, :] = 0
        m[:, :border_trim_px] = 0
        m[:, -border_trim_px:] = 0

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


def extract_horiz_lines(gray: np.ndarray,
                        horiz_kernel_frac: float = 0.035,
                        min_width_frac: float = 0.60,
                        max_thickness_px: int = 6,
                        y_merge_tol_px: int = 4):
    """
    Returns (lines, horiz_mask). For header-band search we mostly care about horiz_mask.
    """
    H, W = gray.shape
    bw_inv = _binarize_for_horiz(gray)
    horiz = _horiz_mask_from_bin(bw_inv, W, horiz_kernel_frac)

    cnts, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w = int(round(W * min_width_frac))
    raws = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < 1 or h > max_thickness_px:
            continue
        y_mid = y + h // 2
        xs = np.where(horiz[y_mid, :] > 0)[0]
        if xs.size == 0:
            continue
        xL, xR = int(xs.min()), int(xs.max())
        if (xR - xL + 1) < min_w:
            continue
        raws.append((y, y + h - 1, xL, xR))

    if not raws:
        return [], horiz

    raws.sort(key=lambda r: (r[0] + r[1]) / 2.0)
    merged, cur = [], [raws[0]]
    for r in raws[1:]:
        y0, y1, _, _ = r
        py0, py1, _, _ = cur[-1]
        if abs(((y0 + y1) / 2.0) - ((py0 + py1) / 2.0)) <= y_merge_tol_px:
            cur.append(r)
        else:
            merged.append(cur)
            cur = [r]
    merged.append(cur)

    lines = []
    for grp in merged:
        y_t = min(g[0] for g in grp)
        y_b = max(g[1] for g in grp)
        x_l = min(g[2] for g in grp)
        x_r = max(g[3] for g in grp)
        y_c = (y_t + y_b) // 2
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
    return lines, horiz

def find_near_header_band(hmask: np.ndarray,
                          header_y: int | None,
                          search_up_first: bool = SEARCH_UP_FIRST,
                          max_delta: int = MAX_DELTA_PX,
                          min_cov_frac: float = MIN_COVERAGE_FRAC,
                          merge_tol: int = MERGE_Y_TOLERANCE) -> int | None:
    """
    hmask: horizontal-only mask (WHITE=horizontal lines)
    Returns y (int) for the first strong white band near header_y.
    """
    if header_y is None:
        return None

    H, W = hmask.shape
    ref = int(max(0, min(H - 1, int(header_y))))

    row_cov = (hmask > 0).sum(axis=1)
    thr = int(round(min_cov_frac * W))
    ys = np.where(row_cov >= thr)[0].astype(int)
    if ys.size == 0:
        return None

    # Merge contiguous rows into bands
    bands = []
    s = ys[0]
    p = ys[0]
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

def find_header_bottom_band(hmask: np.ndarray,
                            header_token_y: int | None,
                            used_header_y: int | None = None,
                            min_cov_frac: float = MIN_COVERAGE_FRAC,
                            merge_tol: int = MERGE_Y_TOLERANCE) -> int | None:
    """
    Find the first strong horizontal band *below* the header tokens.

    - header_token_y: median/header-row y from tokens (must be non-None)
    - used_header_y : the y we already chose as header_y (avoid reusing that band)
    - Returns an absolute y (int) or None.
    """
    if header_token_y is None:
        return None

    H, W = hmask.shape
    row_cov = (hmask > 0).sum(axis=1)
    thr = int(round(min_cov_frac * W))
    ys = np.where(row_cov >= thr)[0].astype(int)
    if ys.size == 0:
        return None

    # Merge contiguous rows into bands
    bands = []
    s = ys[0]
    p = ys[0]
    for y in ys[1:]:
        if y - p <= merge_tol:
            p = y
        else:
            bands.append((s, p))
            s = p = y
    bands.append((s, p))

    reps = [int((a + b) // 2) for a, b in bands]

    # Cutoff = must be below the token line AND below the header_y we chose,
    # so we never pick the same band.
    cutoff = int(header_token_y)
    if used_header_y is not None:
        cutoff = max(cutoff, int(used_header_y))

    below = [y for y in reps if y > cutoff]
    if not below:
        return None

    # Closest band just below the cutoff
    return int(min(below))

# --- Panel name de-duper (module-scope; persists for this process/job) ---
_NAME_COUNTS = {}

def _norm_name(s):
    return str(s or "").strip().upper()

def _dedupe_name(raw_name: str | None) -> str:
    base = (str(raw_name or "").strip()) or "(unnamed)"
    key = _norm_name(base)
    cnt = _NAME_COUNTS.get(key, 0) + 1
    _NAME_COUNTS[key] = cnt
    return base if cnt == 1 else f"{base} ({cnt})"

# Optional: call from tests to reset between runs in the same process
def reset_name_deduper():
    _NAME_COUNTS.clear()

# ----- STRICT imports: Analyzer + Header + Parser5 only -----
from OcrLibrary.BreakerTableAnalyzer6 import BreakerTableAnalyzer, ANALYZER_VERSION
from OcrLibrary.PanelHeaderParserV4   import PanelParser as PanelHeaderParser
from OcrLibrary.BreakerTableParser6   import BreakerTableParser, PARSER_VERSION

class BreakerTablePipeline:

    def __init__(self, *, debug: bool = True):
        self.debug = bool(debug)
        self._analyzer = None
        self._header_parser = None

    # ---- numeric helpers ----
    def _to_int_or_none(self, v):
        if v is None:
            return None
        s = ''.join(ch for ch in str(v) if ch.isdigit())
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    # ---- validation helpers ----
    def _mask_header_non_name(self, header_result: dict | None, *, detected_name):
        out = dict(header_result) if isinstance(header_result, dict) else {}
        out["name"] = detected_name
        attrs = out.get("attrs")
        if isinstance(attrs, dict):
            masked = {}
            for k in attrs.keys():
                if k == "detected_breakers":
                    masked[k] = []
                else:
                    masked[k] = "x"
            out["attrs"] = masked
        else:
            out["attrs"] = {}
        for k in list(out.keys()):
            if k not in ("name", "attrs"):
                out[k] = "x"
        return out

    def _mask_parser_non_name(self, parser_result: dict | None, *, detected_name):
        out = dict(parser_result) if isinstance(parser_result, dict) else {}
        out["name"] = detected_name
        out["spaces"] = "x"
        out["detected_breakers"] = []
        return out or {"name": detected_name, "spaces": "x", "detected_breakers": []}

    def _extract_panel_keys(self, analyzer_result: dict | None, header_result: dict | None, parser_result: dict | None):
        ar = analyzer_result or {}
        hdr = header_result or {}
        attrs = hdr.get("attrs") if isinstance(hdr.get("attrs"), dict) else {}
        prs = parser_result or {}

        ah = {}
        nh = ar.get("normalized_header")
        if isinstance(nh, dict):
            ah = {
                "name": nh.get("name"),
                "voltage": nh.get("voltage"),
                "bus_amps": nh.get("bus"),
                "main_amps": nh.get("main"),
            }

        def pick(d: dict, *keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None

        name = pick(hdr, "name") or ah.get("name")
        volts = pick(hdr, "volts", "voltage", "v") or pick(attrs, "volts", "voltage", "v") or ah.get("voltage")
        bus_amps = pick(hdr, "bus_amps", "busamps", "bus", "amperage") \
                   or pick(attrs, "bus_amps", "busamps", "bus", "amperage") \
                   or ah.get("bus_amps")
        main_amps = pick(hdr, "main_amps", "main", "main_rating", "main_breaker_amps", "mainBreakerAmperage") \
                    or pick(attrs, "main_amps", "main", "main_rating", "main_breaker_amps", "mainBreakerAmperage") \
                    or ah.get("main_amps")
        spaces = prs.get("spaces")
        return name, volts, bus_amps, main_amps, spaces

    def _parse_voltage(self, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v if v in VALID_VOLTAGES else None
        import re
        s = str(v)
        m_pair = re.search(r'(?<!\d)(120|208|240|277|480|600)\s*[Y/]\s*(120|208|240|277|480|600)(?!\d)', s, flags=re.I)
        if m_pair:
            hi = max(int(m_pair.group(1)), int(m_pair.group(2)))
            return hi if hi in VALID_VOLTAGES else None
        m_single = re.search(r'(?<!\d)(120|208|240|480|600)(?!\d)', s)
        return int(m_single.group(1)) if m_single and int(m_single.group(1)) in VALID_VOLTAGES else None

    def _is_valid_amp(self, val) -> bool:
        n = self._to_int_or_none(val)
        if n is None:
            return False
        if n < AMP_MIN or n > AMP_MAX:
            return False
        return (n % 10) in (0, 5)

    # ---- helpers ----
    def _ensure_analyzer(self):
        if self._analyzer is None:
            self._analyzer = BreakerTableAnalyzer(debug=self.debug)
        return self._analyzer

    def _ensure_header_parser(self):
        if self._header_parser is None:
            self._header_parser = PanelHeaderParser(debug=self.debug)
        return self._header_parser

    def run(
        self,
        image_path: str,
        *,
        run_analyzer: bool = True,
        run_header:  bool = True,
        run_parser:  bool = True,
    ) -> dict:
        """
        Execute the pipeline in strict order:
          Analyzer → Header Parser → (header check) → ALT Table Parser

        Returns the same dict shape as the legacy parse_image().
        """
        img = os.path.abspath(os.path.expanduser(image_path))

        analyzer_result = None
        header_result   = None
        parser_result   = None

        # --- 1) Analyzer ---
        analyzer = self._ensure_analyzer()
        if run_analyzer:
            try:
                analyzer_result = analyzer.analyze(img)
            except Exception as e:
                analyzer_result = None
                if self.debug:
                    print(f"[WARN] Analyzer failed: {e}")

        # --- 1b) Refine header_y using dev 'find header' env logic + overwrite overlay ---
        try:
            if run_analyzer and isinstance(analyzer_result, dict):
                gray = analyzer_result.get("gray")
                if gray is not None and hasattr(gray, "shape") and gray.size > 0:
                    # 1) OCR tokens in the header band
                    tokens = ocr_tokens_in_header_band(gray, analyzer)
                    if self.debug:
                        print(f"[DEBUG] header-band tokens: {len(tokens)}")

                    header_y_tokens = infer_header_y_from_tokens(tokens, gray.shape[0])
                    if header_y_tokens is None:
                        if self.debug:
                            print("[DEBUG] infer_header_y_from_tokens returned None (no good header band).")
                    else:
                        # 2) Build horizontal-only mask
                        _lines, horiz_only = extract_horiz_lines(
                            gray,
                            horiz_kernel_frac=0.035,
                            min_width_frac=0.60,
                            max_thickness_px=6,
                            y_merge_tol_px=4,
                        )

                        # 3) From OCR header row, find nearest strong band (top-of-header logic)
                        refined_y = find_near_header_band(
                            horiz_only,
                            header_y_tokens,
                            search_up_first=SEARCH_UP_FIRST,
                            max_delta=MAX_DELTA_PX,
                            min_cov_frac=MIN_COVERAGE_FRAC,
                            merge_tol=MERGE_Y_TOLERANCE,
                        )

                        if refined_y is None:
                            if self.debug:
                                print(
                                    f"[DEBUG] find_near_header_band returned None "
                                    f"(header_token_y={header_y_tokens})."
                                )
                        else:
                            # Keep old one for debugging
                            old_hy = analyzer_result.get("header_y")

                            analyzer_result["header_y_original"]   = old_hy
                            analyzer_result["header_token_y"]      = int(header_y_tokens)
                            analyzer_result["header_y"]            = int(refined_y)
                            analyzer_result["header_rule_source"]  = "near_header_band_v1"

                            if self.debug:
                                print(
                                    f"[INFO] Refined header_y via near-header band: "
                                    f"tokens_y={header_y_tokens}, band_y={refined_y}, old_y={old_hy}"
                                )

                            # 3b) NEW: find header "bottom" band strictly below the token band
                            header_bottom_y = find_header_bottom_band(
                                horiz_only,
                                header_token_y=header_y_tokens,
                                used_header_y=refined_y,
                                min_cov_frac=MIN_COVERAGE_FRAC,
                                merge_tol=MERGE_Y_TOLERANCE,
                            )
                            if header_bottom_y is not None:
                                analyzer_result["header_bottom_y"] = int(header_bottom_y)
                                analyzer_result["header_bottom_rule_source"] = "near_header_band_v1"
                                if self.debug:
                                    print(
                                        f"[INFO] Detected header bottom band: "
                                        f"token_y={header_y_tokens}, bottom_y={header_bottom_y}"
                                    )

                            # --- VISUAL: overwrite the existing overlay PNG with the refined header line(s) ---
                            try:
                                overlay_path = analyzer_result.get("page_overlay_path")
                                if overlay_path and os.path.exists(overlay_path):
                                    ov = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
                                    if ov is not None:
                                        H_ov, W_ov = ov.shape[:2]
                                        H_gray = float(gray.shape[0])

                                        def _map_y(y_src: int) -> int:
                                            y_ratio = float(y_src) / H_gray if H_gray > 0 else 0.0
                                            return int(max(0, min(H_ov - 1, round(y_ratio * H_ov))))

                                        # Top-of-header (refined header_y)
                                        y_line_top = _map_y(int(refined_y))
                                        cv2.line(ov, (0, y_line_top), (W_ov - 1, y_line_top), (0, 0, 255), 3)
                                        cv2.putText(
                                            ov, "HDR_TOP",
                                            (8, max(16, y_line_top - 6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 1, cv2.LINE_AA
                                        )

                                        # Bottom-of-header (if found) — draw in cyan
                                        hb_y_val = analyzer_result.get("header_bottom_y")
                                        if isinstance(hb_y_val, (int, float)):
                                            y_line_bottom = _map_y(int(hb_y_val))
                                            cv2.line(
                                                ov, (0, y_line_bottom),
                                                (W_ov - 1, y_line_bottom),
                                                (0, 255, 0), 2
                                            )
                                            cv2.putText(
                                                ov, "HDR_BOTTOM",
                                                (8, max(16, y_line_bottom + 14)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (0, 255, 0), 1, cv2.LINE_AA
                                            )

                                        # Overwrite the original overlay file so there isn't an "old" header debug image
                                        cv2.imwrite(overlay_path, ov)

                                        if self.debug:
                                            print(f"[DEBUG] Overwrote overlay with refined header line(s): {overlay_path}")

                                # OPTIONAL: if there was a dedicated header debug image, remove it
                                old_hdr_dbg = analyzer_result.get("header_debug_path")
                                if old_hdr_dbg and os.path.exists(old_hdr_dbg):
                                    os.remove(old_hdr_dbg)
                                    analyzer_result["header_debug_path_removed"] = old_hdr_dbg
                                    if self.debug:
                                        print(f"[DEBUG] Removed old header debug image: {old_hdr_dbg}")

                            except Exception as e:
                                if self.debug:
                                    print(f"[WARN] Could not update overlay with refined header: {e}")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Header band refinement failed: {e}")

        # --- 2) Header Parser ---
        if run_header:
            try:
                header_parser = self._ensure_header_parser()
                if hasattr(analyzer, "reader"):
                    header_parser.reader = analyzer.reader
                if analyzer_result and isinstance(analyzer_result, dict):
                    hy   = analyzer_result.get("header_y")
                    gray = analyzer_result.get("gray")
                    if isinstance(hy, (int, float)) and hasattr(gray, "shape"):
                        H_ana = float(gray.shape[0])
                        header_ratio = max(0.0, min(1.0, float(hy) / H_ana))
                        header_result = header_parser.parse_panel(img, header_y_ratio=header_ratio)
                    else:
                        header_result = header_parser.parse_panel(img)
                else:
                    header_result = header_parser.parse_panel(img)
            except Exception as e:
                header_result = None
                if self.debug:
                    print(f"[WARN] Header parser failed: {e}")

        # --- 2b) Apply de-duped display name as early as possible ---
        try:
            base_name, _v, _b, _m, _ = self._extract_panel_keys(analyzer_result, header_result, None)
        except Exception:
            base_name = None
        dedup_name = _dedupe_name(base_name)
        if isinstance(header_result, dict):
            header_result["name"] = dedup_name
        try:
            nh = (analyzer_result or {}).get("normalized_header")
            if isinstance(nh, dict):
                nh["name"] = dedup_name
        except Exception:
            pass

        # --- 3) Header validity check (but DO NOT skip the table parser) ---
        should_run_parser = run_parser
        panel_status = None
        try:
            _dn, volts, bus_amps, main_amps, _spaces_unused = self._extract_panel_keys(
                analyzer_result, header_result, None
            )
            volts_i = self._parse_voltage(volts)
            volts_invalid = (volts_i is None) or (volts_i not in VALID_VOLTAGES)
            bus_invalid   = not self._is_valid_amp(bus_amps)                          # REQUIRED
            main_invalid  = (main_amps is not None) and (not self._is_valid_amp(main_amps))  # OPTIONAL

            if volts_invalid or bus_invalid or main_invalid:
                panel_status = f"unable to detect key information on panel ({dedup_name})"
                header_result = self._mask_header_non_name(header_result, detected_name=dedup_name)
                # NOTE: we still run the ALT table parser; just change the message:
                if self.debug:
                    miss = []
                    if volts_invalid: miss.append(f"volts={volts!r}")
                    if bus_invalid:   miss.append(f"bus_amps={bus_amps!r}")
                    if main_invalid:  miss.append(f"main_amps={main_amps!r}")
                    print("[INFO] Header invalid; proceeding to TABLE PARSER anyway "
                          f"(name={dedup_name!r}; {', '.join(miss)})")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Header pre-parse validation failed: {e}")

        # ===>>> 3b) TABLE PARSER (ALT only) — PLACE IS HERE, always after header check
        if should_run_parser:
            try:
                # ALT only, no fallbacks
                parser = BreakerTableParser(debug=self.debug, reader=getattr(analyzer, "reader", None))
                if analyzer_result is not None:
                    parser_result = parser.parse_from_analyzer(analyzer_result)
                else:
                    parser_result = parser.parse_from_analyzer({})
            except Exception as e:
                parser_result = None
                if self.debug:
                    print(f"[WARN] Parser failed: {e}")
        # Ensure the table parser result advertises the deduped name for the UI
        if isinstance(parser_result, dict):
            parser_result["name"] = dedup_name

        # --- optional legacy prints (only if table parser ran) ---
        if self.debug and parser_result is not None:
            try:
                print("\n>>> FINAL (legacy-compatible prints) >>>")
                print(parser_result.get("spaces"))
                print(parser_result.get("detected_breakers"))
            except Exception:
                pass

        # --- 4) Panel validity & masking logic (spaces + header recheck) ---
        try:
            if panel_status is None:
                _dn2, volts, bus_amps, main_amps, spaces = self._extract_panel_keys(
                    analyzer_result, header_result, parser_result
                )
                volts_i = self._parse_voltage(volts)
                volts_invalid  = (volts_i is None) or (volts_i not in VALID_VOLTAGES)
                bus_invalid    = not self._is_valid_amp(bus_amps)
                main_invalid   = (main_amps is not None) and (not self._is_valid_amp(main_amps))
                if isinstance(spaces, int):
                    spaces_norm = SNAP_MAP.get(spaces, spaces)
                    spaces_invalid = spaces_norm is None or spaces_norm <= 0
                else:
                    spaces_invalid = True

                if spaces_invalid or volts_invalid or bus_invalid or main_invalid:
                    panel_status = f"unable to detect key information on panel ({dedup_name})"
                    header_result = self._mask_header_non_name(header_result, detected_name=dedup_name)
                    parser_result = self._mask_parser_non_name(parser_result,  detected_name=dedup_name)
        except Exception as e:
            if self.debug:
                print(f"[WARN] Panel validation/masking failed: {e}")

        # --- combined result ---
        return {
            "apiVersion": API_VERSION,
            "origin": API_ORIGIN,
            "stages": {
                "analyzer": run_analyzer,
                "parser": run_parser,
                "header": run_header,
            },
            "results": {
                "analyzer": analyzer_result,
                "parser": parser_result,
                "header": header_result,
            },
            "panelStatus": panel_status,  # None if OK; message if flagged
        }


# ---- Back-compat: keep the old function name ----
def parse_image(
    image_path: str,
    *,
    run_analyzer: bool = True,
    run_parser: bool = True,
    run_header: bool = True,
    debug: bool = True
):
    pipe = BreakerTablePipeline(debug=debug)
    return pipe.run(
        image_path,
        run_analyzer=run_analyzer,
        run_header=run_header,
        run_parser=run_parser,
    )

if __name__ == "__main__":
    # --- Dev banner ---
    modA = sys.modules[BreakerTableAnalyzer.__module__]
    implA = inspect.getsourcefile(BreakerTableAnalyzer) or inspect.getfile(BreakerTableAnalyzer)
    print(">>> DEV Analyzer version:", getattr(modA, "ANALYZER_VERSION", "unknown"))
    print(">>> DEV Analyzer file:", os.path.abspath(implA))

    # ALT sanity
    modP = sys.modules.get(BreakerTableParser.__module__)
    implP = inspect.getsourcefile(BreakerTableParser) or inspect.getfile(BreakerTableParser)
    print(">>> DEV Parser version:", PARSER_VERSION if 'PARSER_VERSION' in globals() else "unknown")
    print(">>> DEV Parser file:", os.path.abspath(implP))
    print(">>> DEV Parser module:", BreakerTableParser.__module__)

    if len(sys.argv) >= 2:
        img = sys.argv[1]
    else:
        img = "/home/paperspace/ElectricalDiagramAnalyzer/detectron2_training/OutputImages/good_tester_combo_page001_table01_rect.png"

    print(">>> DEV Image:", img)

    # Simple flags: --no-analyzer --no-parser --no-header
    args = sys.argv[2:]
    run_analyzer = "--no-analyzer" not in args
    run_parser = "--no-parser" not in args
    run_header = "--no-header" not in args

    pipeline = BreakerTablePipeline(debug=True)
    result = pipeline.run(
        img,
        run_analyzer=run_analyzer,
        run_parser=run_parser,
        run_header=run_header,
    )

    print("\n>>> Stage Summary:")
    for key, val in result["results"].items():
        print(f"  {key}: {'OK' if val else 'None'}")
    if result.get("panelStatus"):
        print(f"  status: {result['panelStatus']}")
    else:
        print("  status: OK")
