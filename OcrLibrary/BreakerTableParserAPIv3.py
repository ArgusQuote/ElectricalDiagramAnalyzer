import sys, os, inspect
import re

# ----- ensure repo root on sys.path -----
_THIS_FILE  = os.path.abspath(__file__)
_OCRLIB_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT  = os.path.dirname(_OCRLIB_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

API_VERSION = "V2_HEADER_VALIDATE_SKIP+ALT_ROUTER"
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

# ----- resilient imports (package and fallback local) -----
try:
    from OcrLibrary.BreakerTableAnalyzer4 import BreakerTableAnalyzer, ANALYZER_VERSION
    from OcrLibrary.BreakerTableParser4   import BreakerTableParser,  PARSER_VERSION
    from OcrLibrary.PanelHeaderParserV4   import PanelParser as PanelHeaderParser
    # NEW: ALT parser (package form)
    try:
        from OcrLibrary.BreakerTableParserALT import BreakerTableParser as BreakerTableParserALT
        try:
            from OcrLibrary.BreakerTableParserALT import PARSER_VERSION as PARSER_ALT_VERSION
        except Exception:
            PARSER_ALT_VERSION = "unknown"
    except Exception:
        BreakerTableParserALT = None
        PARSER_ALT_VERSION    = "unknown"

except ModuleNotFoundError:
    from BreakerTableAnalyzer4 import BreakerTableAnalyzer, ANALYZER_VERSION
    from PanelHeaderParserV4   import PanelParser as PanelHeaderParser
    # Table parser may not exist in some light envs; import lazily inside run if needed
    try:
        from BreakerTableParser4 import BreakerTableParser, PARSER_VERSION  # type: ignore
    except Exception:
        BreakerTableParser, PARSER_VERSION = None, "unknown"  # lazy fallback

    # NEW: ALT parser (fallback local)
    try:
        from BreakerTableParserALT import BreakerTableParser as BreakerTableParserALT  # type: ignore
        try:
            from BreakerTableParserALT import PARSER_VERSION as PARSER_ALT_VERSION  # type: ignore
        except Exception:
            PARSER_ALT_VERSION = "unknown"
    except Exception:
        BreakerTableParserALT = None
        PARSER_ALT_VERSION    = "unknown"


class BreakerTablePipeline:
    """
    Class-based, single point of entry that runs:
      1) BreakerTableAnalyzer3.analyze()
      2) PanelHeaderParserV4.PanelParser.parse_panel()
      3) BreakerTableParser4.parse_from_analyzer()

    Mirrors the working Dev test env you shared.
    """

    def __init__(self, *, debug: bool = True):
        self.debug = bool(debug)

        # stage objects (created on demand so we can reuse OCR reader)
        self._analyzer = None
        self._header_parser = None

    # ---- numeric helpers ----
    def _to_int_or_none(self, v):
        """
        Robust converter: pulls digits only (handles '114A', '90 A', '13640 VA').
        Returns int or None.
        """
        if v is None:
            return None
        s = ''.join(ch for ch in str(v) if ch.isdigit())
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    # ---- ALT routing: detect presence of a "Poles" header column ----
    def _has_poles_column(self, analyzer_result: dict | None, header_result: dict | None) -> bool:
        """
        Returns True if the header line appears to include a 'poles' column.
        Strategy:
          1) Inspect header_result text (cheap).
          2) If unknown, OCR a tight band around analyzer_result['header_y'] with analyzer.reader.
        """
        TOKENS = {"POLES", "POLE"}  # Avoid bare 'P' (too noisy)

        # 1) Inspect structured header_result fields
        if isinstance(header_result, dict):
            hay = []
            for k, v in header_result.items():
                if k == "attrs" and isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, str):
                            hay.append(vv)
                elif isinstance(v, str):
                    hay.append(v)
            blob = " ".join(hay).upper()
            if any(t in blob for t in TOKENS):
                return True

        # 2) OCR near header_y with analyzer's reader
        ar = analyzer_result or {}
        gray = ar.get("gray")
        header_y = ar.get("header_y")
        reader = getattr(self._ensure_analyzer(), "reader", None)

        if gray is None or header_y is None or reader is None:
            return False

        try:
            H, W = getattr(gray, "shape", (0, 0))
            if not H or not W:
                return False

            y1 = max(0, int(header_y) - 10)
            y2 = min(H, int(header_y) + 70)
            roi = gray[y1:y2, 0:W]
            if roi.size == 0:
                return False

            def ocr(img, mag):
                try:
                    return reader.readtext(
                        img, detail=1, paragraph=False,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/",
                        mag_ratio=mag, contrast_ths=0.05, adjust_contrast=0.7,
                        text_threshold=0.4, low_text=0.25
                    )
                except Exception:
                    return []

            det = ocr(roi, 1.6) + ocr(roi, 2.0)
            for box, txt, conf in det:
                n = re.sub(r"[^A-Z]", "", (txt or "").upper().replace("1","I").replace("0","O"))
                if any(tok in n for tok in TOKENS):
                    if self.debug:
                        print(f"[ALT_ROUTER] Poles token near header: {txt} (conf={conf:.2f})")
                    return True
            return False
        except Exception as e:
            if self.debug:
                print(f"[ALT_ROUTER] Poles OCR check failed: {e}")
            return False

    # ---- validation helpers ----
    def _mask_header_non_name(self, header_result: dict | None, *, detected_name):
        """
        Preserve header schema. Keep 'name'; set all other scalar attrs to 'x'.
        If attrs dict exists, keep it as a dict and set its keys to 'x'
        (except 'detected_breakers' which stays an empty list for downstream safety).
        """
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
        # Any other top-level non-name fields → "x"
        for k in list(out.keys()):
            if k not in ("name", "attrs"):
                out[k] = "x"
        return out

    def _mask_parser_non_name(self, parser_result: dict | None, *, detected_name):
        """
        Preserve parser schema. Keep 'name'. Set 'spaces' to 'x' and
        'detected_breakers' to [] if present. Any other keys → 'x'.
        """
        out = dict(parser_result) if isinstance(parser_result, dict) else {}
        out["name"] = detected_name
        if "spaces" in out:
            out["spaces"] = "x"
        else:
            out["spaces"] = "x"
        # Always expose detected_breakers for downstream callers
        out["detected_breakers"] = []
        return out or {"name": detected_name, "spaces": "x", "detected_breakers": []}

    def _extract_panel_keys(self, analyzer_result: dict | None, header_result: dict | None, parser_result: dict | None):
        """
        Build a single normalized header view for validation and downstream use.
        Priority:
          1) HeaderParser normalized attrs
          2) Analyzer normalized picks (if exported)
        Also supports both flat and nested (header['attrs']) schemas.
        """
        ar = analyzer_result or {}
        hdr = header_result or {}
        attrs = hdr.get("attrs") if isinstance(hdr.get("attrs"), dict) else {}
        prs = parser_result or {}

        # --- Pull analyzer's normalized header if available ---
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

        # Normalized priority: HeaderParser first, then Analyzer normalized
        name = pick(hdr, "name") or ah.get("name")

        # Voltage may be in header or header['attrs'] or analyzer normalized
        volts = pick(hdr, "volts", "voltage", "v") or pick(attrs, "volts", "voltage", "v") or ah.get("voltage")

        # Bus amps (panel bus rating)
        bus_amps = pick(hdr, "bus_amps", "busamps", "bus", "amperage") \
                   or pick(attrs, "bus_amps", "busamps", "bus", "amperage") \
                   or ah.get("bus_amps")

        # Main amps optional; various spellings including camelCase
        main_amps = pick(hdr, "main_amps", "main", "main_rating", "main_breaker_amps", "mainBreakerAmperage") \
                    or pick(attrs, "main_amps", "main", "main_rating", "main_breaker_amps", "mainBreakerAmperage") \
                    or ah.get("main_amps")

        spaces = prs.get("spaces")
        return name, volts, bus_amps, main_amps, spaces

    def _parse_voltage(self, v):
        """Return system voltage from strings like '208Y/120', '480/277', '120/240', or bare '208', else None."""
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
        """
        Accept ints (or numeric-like strings) in [AMP_MIN, AMP_MAX] whose last digit is 0 or 5.
        """
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
        run_header:  bool  = True,
        run_parser:  bool  = True,
        parser_used        = None,   # NEW: 'default' or 'alt'
    ) -> dict:
        """
        Execute the pipeline in strict order:
          Analyzer → Header Parser → (skip guard) → Table Parser

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

        # --- 2) Header Parser (second) ---
        if run_header:
            try:
                header_parser = self._ensure_header_parser()
                # share OCR reader for consistency
                if hasattr(analyzer, "reader"):
                    header_parser.reader = analyzer.reader

                if analyzer_result and isinstance(analyzer_result, dict):
                    hy = analyzer_result.get("header_y")
                    gray = analyzer_result.get("gray")
                    if isinstance(hy, (int, float)) and hasattr(gray, "shape"):
                        H_ana = float(gray.shape[0])
                        header_ratio = max(0.0, min(1.0, float(hy) / H_ana))
                        header_result = header_parser.parse_panel(
                            img,
                            header_y_ratio=header_ratio
                        )
                    else:
                        header_result = header_parser.parse_panel(img)
                else:
                    header_result = header_parser.parse_panel(img)
            except Exception as e:
                header_result = None
                if self.debug:
                    print(f"[WARN] Header parser failed: {e}")

        # --- 3) Pre-parse validation (header-only); then Table Parser ---
        should_run_parser = run_parser
        # --- Choose parser (ALT if no Poles column) ---
        use_alt = False
        if should_run_parser:
            try:
                use_alt = not self._has_poles_column(analyzer_result, header_result)
            except Exception as e:
                if self.debug:
                    print(f"[ALT_ROUTER] Detection failed (fallback to default): {e}")
                use_alt = False
        
        panel_status = None

        # Header-only checks: volts (required), bus_amps (required), main_amps (optional)
        try:
            detected_name, volts, bus_amps, main_amps, _spaces_unused = self._extract_panel_keys(analyzer_result, header_result, None)

            volts_i = self._parse_voltage(volts)
            volts_invalid = (volts_i is None) or (volts_i not in VALID_VOLTAGES)
            bus_invalid   = not self._is_valid_amp(bus_amps)                          # REQUIRED
            main_invalid  = (main_amps is not None) and (not self._is_valid_amp(main_amps))  # OPTIONAL

            if volts_invalid or bus_invalid or main_invalid:
                panel_status = f"unable to detect key information on panel ({detected_name})"
                # Mask header & parser safely (preserve schema); SKIP heavy parse
                header_result = self._mask_header_non_name(header_result, detected_name=detected_name)
                parser_result = self._mask_parser_non_name(None, detected_name=detected_name)
                should_run_parser = False
                if self.debug:
                    miss = []
                    if volts_invalid: miss.append(f"volts={volts!r}")
                    if bus_invalid:   miss.append(f"bus_amps={bus_amps!r}")
                    if main_invalid:  miss.append(f"main_amps={main_amps!r}")
                    print("[INFO] Skipping TABLE PARSER due to invalid header fields "
                        f"(name={detected_name!r}; {', '.join(miss)})")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Header pre-parse validation failed: {e}")

        if should_run_parser:
            # lazy-import default parser if needed
            global BreakerTableParser
            if BreakerTableParser is None:
                try:
                    from OcrLibrary.BreakerTableParser4 import BreakerTableParser  # type: ignore
                except Exception:
                    try:
                        from BreakerTableParser4 import BreakerTableParser  # type: ignore
                    except Exception as e:
                        if self.debug:
                            print(f"[WARN] Table parser unavailable: {e}")
                        BreakerTableParser = None

            # lazy-import ALT if we intend to use it and it's not loaded yet
            global BreakerTableParserALT
            if use_alt and BreakerTableParserALT is None:
                try:
                    from OcrLibrary.BreakerTableParserALT import BreakerTableParser as BreakerTableParserALT  # type: ignore
                except Exception:
                    try:
                        from BreakerTableParserALT import BreakerTableParser as BreakerTableParserALT  # type: ignore
                    except Exception as e:
                        if self.debug:
                            print(f"[WARN] ALT table parser unavailable: {e}")
                        BreakerTableParserALT = None

            parser_cls = BreakerTableParserALT if (use_alt and BreakerTableParserALT is not None) else BreakerTableParser
            parser_used = "alt" if (parser_cls is not None and parser_cls is BreakerTableParserALT) else "default"
            if self.debug:
                print(f"[ALT_ROUTER] parser_used={parser_used}  (has_poles={not use_alt})")

            if parser_cls is not None:
                try:
                    parser = parser_cls(debug=self.debug, reader=getattr(analyzer, "reader", None))
                    if analyzer_result is not None:
                        parser_result = parser.parse_from_analyzer(analyzer_result)
                    else:
                        parser_result = parser.parse_from_analyzer({})
                except Exception as e:
                    parser_result = None
                    if self.debug:
                        print(f"[WARN] Parser failed: {e}")

        # --- optional legacy prints (only if table parser ran) ---
        if self.debug and parser_result is not None:
            try:
                print("\n>>> FINAL (legacy-compatible prints) >>>")
                print(parser_result.get("spaces"))
                print(parser_result.get("detected_breakers"))
            except Exception:
                pass

        # --- panel validity & masking logic (spaces check; only if not flagged already) ---
        try:
            if panel_status is None:
                detected_name, volts, bus_amps, main_amps, spaces = self._extract_panel_keys(analyzer_result, header_result, parser_result)
                # volts/bus/main re-check for safety + spaces validation
                volts_i = self._parse_voltage(volts)
                volts_invalid  = (volts_i is None) or (volts_i not in VALID_VOLTAGES)
                bus_invalid    = not self._is_valid_amp(bus_amps)                         # REQUIRED
                main_invalid   = (main_amps is not None) and (not self._is_valid_amp(main_amps))  # OPTIONAL
                # Normalize spaces, then allow any positive int (or, if you prefer, only allow the snapped set)
                if isinstance(spaces, int):
                    spaces_norm = SNAP_MAP.get(spaces, spaces)
                    # permissive: any positive int is fine
                    spaces_invalid = spaces_norm is None or spaces_norm <= 0
                    # or strict to your canonical set:
                    # CANON = set(SNAP_MAP.values())
                    # spaces_invalid = spaces_norm not in CANON
                else:
                    spaces_invalid = True

                if spaces_invalid or volts_invalid or bus_invalid or main_invalid:
                    panel_status = f"unable to detect key information on panel ({detected_name})"
                    header_result = self._mask_header_non_name(header_result, detected_name=detected_name)
                    parser_result = self._mask_parser_non_name(parser_result,  detected_name=detected_name)
                    if self.debug:
                        miss = []
                        if volts_invalid: miss.append(f"volts={volts!r}")
                        if bus_invalid:   miss.append(f"bus_amps={bus_amps!r}")
                        if main_invalid:  miss.append(f"main_amps={main_amps!r}")
                        print("[INFO] Skipping TABLE PARSER due to invalid header fields "
                            f"(name={detected_name!r}; {', '.join(miss)})")
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
                "parserUsed": parser_used,  # NEW: 'default' or 'alt'
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
    """Legacy function wrapper; calls the class-based pipeline."""
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

    # Table parser may be unavailable in some envs; guard the banner
    try:
        modP = sys.modules.get(BreakerTableParser.__module__) if BreakerTableParser else None
        implP = None
        if modP:
            implP = inspect.getsourcefile(BreakerTableParser) or inspect.getfile(BreakerTableParser)
        print(">>> DEV Parser version:", PARSER_VERSION if 'PARSER_VERSION' in globals() else "unknown")
        if implP:
            print(">>> DEV Parser file:", os.path.abspath(implP))
    except Exception:
        pass

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

    # Parser4 banner
    try:
        modP = sys.modules.get(BreakerTableParser.__module__) if BreakerTableParser else None
        implP = None
        if modP:
            implP = inspect.getsourcefile(BreakerTableParser) or inspect.getfile(BreakerTableParser)
        print(">>> DEV Parser4 version:", PARSER_VERSION if 'PARSER_VERSION' in globals() else "unknown")
        if implP:
            print(">>> DEV Parser4 file:", os.path.abspath(implP))
    except Exception:
        pass

    # ALT banner
    try:
        if BreakerTableParserALT:
            modALT = sys.modules.get(BreakerTableParserALT.__module__)
            implALT = inspect.getsourcefile(BreakerTableParserALT) or inspect.getfile(BreakerTableParserALT)
            print(">>> DEV ParserALT version:", PARSER_ALT_VERSION)
            print(">>> DEV ParserALT file:", os.path.abspath(implALT))
        else:
            print(">>> DEV ParserALT: not available")
    except Exception:
        pass

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
