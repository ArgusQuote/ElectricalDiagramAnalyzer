# OcrLibrary/BreakerTableParserAPIv8.py
import sys, os, inspect
import re
import cv2
import copy

_THIS_FILE  = os.path.abspath(__file__)
_OCRLIB_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT  = os.path.dirname(_OCRLIB_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

API_VERSION = "API_8"
API_ORIGIN  = __file__

SNAP_MAP = {
    16: 18, 20: 18,
    28: 30, 32: 30,
    40: 42, 44: 42,
    52: 54, 56: 54,
    64: 66, 68: 66,
    70: 72, 74: 72,
    82: 84, 86: 84,
}

VALID_VOLTAGES = {120, 208, 240, 480, 600}
AMP_MIN = 100
AMP_MAX = 1200

_NAME_COUNTS = {}

def _norm_name(s):
    """Strip and uppercase a panel name for deduplication key comparison."""
    return str(s or "").strip().upper()

def _dedupe_name(raw_name: str | None) -> str:
    """Return a unique display name, appending '(2)', '(3)', etc. when the same base name recurs within a job."""
    base = (str(raw_name or "").strip()) or "(unnamed)"
    key = _norm_name(base)
    cnt = _NAME_COUNTS.get(key, 0) + 1
    _NAME_COUNTS[key] = cnt
    return base if cnt == 1 else f"{base} ({cnt})"

def reset_name_deduper():
    """Clear the per-job panel name deduplication counter (call before each new job)."""
    _NAME_COUNTS.clear()

from OcrLibrary.BreakerTableAnalyzer12 import BreakerTableAnalyzer, ANALYZER_VERSION
from OcrLibrary.PanelHeaderParserV7   import PanelParser as PanelHeaderParser
from OcrLibrary.BreakerTableParser10   import BreakerTableParser, PARSER_VERSION

class BreakerTablePipeline:
    """Three-stage OCR pipeline: Analyzer -> Header Parser -> Table Parser. Orchestrates extraction of panel data from a single panel image."""

    def __init__(self, *, debug: bool = True):
        """Initialize the pipeline with optional debug output."""
        self.debug = bool(debug)
        self._analyzer = None
        self._header_parser = None

    def _to_int_or_none(self, v):
        """Extract digits from *v* and parse as int; return None if no digits or parse fails."""
        if v is None:
            return None
        s = ''.join(ch for ch in str(v) if ch.isdigit())
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def _mask_header_non_name(self, header_result: dict | None, *, detected_name):
        """Replace all header attrs except the panel name with 'x' placeholders (used when header validation fails)."""
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
        """Replace parser result attrs except the panel name with 'x' placeholders (used when validation fails)."""
        out = dict(parser_result) if isinstance(parser_result, dict) else {}
        out["name"] = detected_name
        out["spaces"] = "x"
        out["detected_breakers"] = []
        return out or {"name": detected_name, "spaces": "x", "detected_breakers": []}

    def _extract_panel_keys(self, analyzer_result: dict | None, header_result: dict | None, parser_result: dict | None):
        """Pull (name, volts, bus_amps, main_amps, spaces) from analyzer/header/parser results, preferring header values."""
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
        """Parse a voltage value from string or int; handles pairs like '480Y/277V' and singles. Returns an int in VALID_VOLTAGES or None."""
        if v is None:
            return None
        if isinstance(v, int):
            return v if v in VALID_VOLTAGES else None
        s = str(v)
        m_pair = re.search(r'(?<!\d)(120|208|240|277|480|600)\s*[Y/]\s*(120|208|240|277|480|600)(?!\d)', s, flags=re.I)
        if m_pair:
            hi = max(int(m_pair.group(1)), int(m_pair.group(2)))
            return hi if hi in VALID_VOLTAGES else None
        m_single = re.search(r'(?<!\d)(120|208|240|480|600)(?!\d)', s)
        return int(m_single.group(1)) if m_single and int(m_single.group(1)) in VALID_VOLTAGES else None

    def _is_valid_amp(self, val) -> bool:
        """Return True if *val* parses to an amperage in [100..1200] that is a multiple of 5."""
        n = self._to_int_or_none(val)
        if n is None:
            return False
        if n < AMP_MIN or n > AMP_MAX:
            return False
        return (n % 10) in (0, 5)

    def _ensure_dir(self, p: str) -> str:
        """Create directory *p* if it doesn't exist; return *p*."""
        os.makedirs(p, exist_ok=True)
        return p

    def _scale_box(self, box, src_w, src_h, dst_w, dst_h):
        """Scale a [x1, y1, x2, y2] bounding box from (src_w, src_h) to (dst_w, dst_h) coordinate space."""
        # box = [x1,y1,x2,y2]
        if not box or src_w <= 0 or src_h <= 0:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except Exception:
            return None
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        return [int(round(x1*sx)), int(round(y1*sy)), int(round(x2*sx)), int(round(y2*sy))]

    def _draw_box(self, vis, box, color_bgr, label=None, *, fill_alpha: float = 0.0, thickness: int = 2):
        """Draw a rectangle on *vis* at *box* [x1,y1,x2,y2] with optional translucent fill and text label."""
        if not box or len(box) != 4:
            return
        H, W = vis.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1:
            return

        # optional faint fill
        if fill_alpha and fill_alpha > 0.0:
            a = float(max(0.0, min(1.0, fill_alpha)))
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.addWeighted(overlay, a, vis, 1.0 - a, 0, dst=vis)

        # outline
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, int(thickness))

        if label:
            cv2.putText(
                vis, str(label),
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_bgr,
                1,
                cv2.LINE_AA,
            )

    def _build_review_overlay(self, *, image_path, analyzer_result, header_result, parser_result, dedup_name):
        """
        Writes ONE combined review overlay:
          - Magenta: winning header attrs
          - Green: trip columns (or combo columns if combined)
          - Blue: poles columns (separated)
        Returns absolute filepath or None.
        """
        ar = analyzer_result or {}
        gray = ar.get("gray")
        if gray is None or not hasattr(gray, "shape"):
            return None

        H, W = gray.shape[:2]
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Where to save (job_root/pdf_images/review_overlays/)
        norm_img = os.path.abspath(image_path).replace("\\", "/")

        rel_path = None
        out_path = None

        if "/pdf_images/" in norm_img:
            job_root = norm_img.split("/pdf_images/")[0]
            review_dir = self._ensure_dir(os.path.join(job_root, "pdf_images", "review_overlays"))
            safe_base = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(dedup_name or "panel")).strip("_") or "panel"
            fname = f"{safe_base}_review_overlay.png"
            out_path = os.path.join(review_dir, fname)
            rel_path = f"pdf_images/review_overlays/{fname}"
        else:
            # fallback if we can't infer job_root
            src_dir = os.path.dirname(os.path.abspath(image_path))
            review_dir = self._ensure_dir(os.path.join(src_dir, "review_overlays"))
            safe_base = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(dedup_name or "panel")).strip("_") or "panel"
            fname = f"{safe_base}_review_overlay.png"
            out_path = os.path.join(review_dir, fname)
            rel_path = f"review_overlays/{fname}"

        # --- Magenta header attrs ---
        # EXPECTATION: header parser returns:
        # header_result["winningBoxes"] = {"name":[x1,y1,x2,y2], "voltage":[...], ...}
        # header_result["boxImageShape"] = {"w": <int>, "h": <int>}  (shape used for those coords)
        hdr = header_result if isinstance(header_result, dict) else {}
        winning = hdr.get("winningBoxes") if isinstance(hdr.get("winningBoxes"), dict) else {}
        shape = hdr.get("boxImageShape") if isinstance(hdr.get("boxImageShape"), dict) else {}
        src_w = int(shape.get("w") or W)
        src_h = int(shape.get("h") or H)

        MAGENTA = (255, 0, 255)  # BGR
        for k, box in winning.items():
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            scaled = self._scale_box(box, src_w, src_h, W, H)
            self._draw_box(vis, scaled, MAGENTA, label=k, fill_alpha=0.0, thickness=3)

        # --- Column overlays (from Parser9 normalized columns) ---
        prs = parser_result if isinstance(parser_result, dict) else {}
        hscan = prs.get("headerScan") if isinstance(prs.get("headerScan"), dict) else {}
        norm = hscan.get("normalizedColumns") if isinstance(hscan.get("normalizedColumns"), dict) else {}
        layout = (norm.get("layout") or "unknown").lower()
        cols = norm.get("columns") or []

        header_bottom_y = ar.get("header_bottom_y")
        footer_y = ar.get("footer_y")

        y_top = 0
        y_bot = H - 1
        if isinstance(header_bottom_y, (int, float)):
            y_top = max(0, min(H - 1, int(header_bottom_y)))
        if isinstance(footer_y, (int, float)):
            y_bot = max(y_top + 1, min(H - 1, int(footer_y)))

        GREEN = (0, 255, 0)      # trip OR combined
        BLUE  = (255, 0, 0)      # poles
        ORANGE = (0, 165, 255)   # specialFeatures (CB Info / Notes / Options / Type)

        def draw_col(col, color, label):
            try:
                x1 = int(col.get("x_left", 0))
                x2 = int(col.get("x_right", 0))
            except Exception:
                return
            if x2 <= x1 + 1:
                return
            self._draw_box(
                vis,
                [x1, y_top, x2, y_bot],
                color,
                label=label,
                fill_alpha=0.12,   # faint fill
                thickness=3
            )

        for col in cols:
            role = col.get("role")
            if layout == "combined":
                if role == "combo":
                    draw_col(col, GREEN, "COMBO")
                elif role == "specialFeatures":
                    draw_col(col, ORANGE, "INFO")
            elif layout == "separated":
                if role == "trip":
                    draw_col(col, GREEN, "TRIP")
                elif role == "poles":
                    draw_col(col, BLUE, "POLES")
                elif role == "specialFeatures":
                    draw_col(col, ORANGE, "INFO")
            else:
                # unknown layout: still show anything we have
                if role == "combo":
                    draw_col(col, GREEN, "COMBO")
                elif role == "trip":
                    draw_col(col, GREEN, "TRIP")
                elif role == "poles":
                    draw_col(col, BLUE, "POLES")
                elif role == "specialFeatures":
                    draw_col(col, ORANGE, "INFO")

        # Save one combined file for the UI
        safe_base = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(dedup_name or "panel")).strip("_") or "panel"
        out_path = os.path.join(review_dir, f"{safe_base}_review_overlay.png")
        try:
            ok = cv2.imwrite(out_path, vis)
            if not ok:
                if self.debug:
                    print(f"[WARN] cv2.imwrite returned False for: {out_path}")
                return None
            return rel_path
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to write review overlay: {e}")
            return None

    def _ensure_analyzer(self):
        """Lazily create the BreakerTableAnalyzer instance (shared EasyOCR reader)."""
        if self._analyzer is None:
            self._analyzer = BreakerTableAnalyzer(debug=self.debug)
        return self._analyzer

    def _ensure_header_parser(self):
        """Lazily create the PanelHeaderParser instance."""
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
        header_result_raw = copy.deepcopy(header_result) if isinstance(header_result, dict) else None

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
        header_invalid = False
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
                header_invalid = True
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

        review_overlay_path = None
        try:
            review_overlay_path = self._build_review_overlay(
                image_path=img,
                analyzer_result=analyzer_result,
                header_result=(header_result_raw or header_result),
                parser_result=parser_result,
                dedup_name=dedup_name,
            )
        except Exception as e:
            if self.debug:
                print(f"[WARN] Review overlay generation failed: {e}")

        # expose to UI
        if isinstance(parser_result, dict):
            parser_result["reviewOverlayPath"] = review_overlay_path
        if isinstance(header_result, dict):
            header_result["reviewOverlayPath"] = review_overlay_path
        if header_invalid:
            header_result = self._mask_header_non_name(header_result, detected_name=dedup_name)

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


def parse_image(
    image_path: str,
    *,
    run_analyzer: bool = True,
    run_parser: bool = True,
    run_header: bool = True,
    debug: bool = True
):
    """Convenience wrapper: create a BreakerTablePipeline and run it on a single image."""
    pipe = BreakerTablePipeline(debug=debug)
    return pipe.run(
        image_path,
        run_analyzer=run_analyzer,
        run_header=run_header,
        run_parser=run_parser,
    )

if __name__ == "__main__":
    modA = sys.modules[BreakerTableAnalyzer.__module__]
    implA = inspect.getsourcefile(BreakerTableAnalyzer) or inspect.getfile(BreakerTableAnalyzer)
    print(">>> DEV Analyzer version:", getattr(modA, "ANALYZER_VERSION", "unknown"))
    print(">>> DEV Analyzer file:", os.path.abspath(implA))

    modP = sys.modules.get(BreakerTableParser.__module__)
    implP = inspect.getsourcefile(BreakerTableParser) or inspect.getfile(BreakerTableParser)
    print(">>> DEV Parser version:", PARSER_VERSION if 'PARSER_VERSION' in globals() else "unknown")
    print(">>> DEV Parser file:", os.path.abspath(implP))
    print(">>> DEV Parser module:", BreakerTableParser.__module__)

    # Require an image path; no hardcoded fallback
    if len(sys.argv) < 2:
        print("Usage: python BreakerTableParserAPIv4.py /path/to/image.png [--no-analyzer] [--no-parser] [--no-header]")
        sys.exit(1)

    img = sys.argv[1]
    print(">>> DEV Image:", img)

    # Simple flags: --no-analyzer --no-parser --no-header
    args = sys.argv[2:]
    run_analyzer = "--no-analyzer" not in args
    run_parser   = "--no-parser" not in args
    run_header   = "--no-header" not in args

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
