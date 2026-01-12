# OcrLibrary/BreakerTableParserAPIv6.py
import sys, os, inspect

_THIS_FILE  = os.path.abspath(__file__)
_OCRLIB_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT  = os.path.dirname(_OCRLIB_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

API_VERSION = "API_4"
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
    return str(s or "").strip().upper()

def _dedupe_name(raw_name: str | None) -> str:
    base = (str(raw_name or "").strip()) or "(unnamed)"
    key = _norm_name(base)
    cnt = _NAME_COUNTS.get(key, 0) + 1
    _NAME_COUNTS[key] = cnt
    return base if cnt == 1 else f"{base} ({cnt})"

def reset_name_deduper():
    _NAME_COUNTS.clear()

from OcrLibrary.BreakerTableAnalyzer12 import BreakerTableAnalyzer, ANALYZER_VERSION
from OcrLibrary.PanelHeaderParserV6   import PanelParser as PanelHeaderParser
from OcrLibrary.BreakerTableParser9   import BreakerTableParser, PARSER_VERSION

class BreakerTablePipeline:
    def __init__(self, *, debug: bool = True):
        self.debug = bool(debug)
        self._analyzer = None
        self._header_parser = None

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
