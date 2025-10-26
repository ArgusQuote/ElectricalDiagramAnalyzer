import sys, os, inspect

# ----- ensure repo root on sys.path -----
_THIS_FILE  = os.path.abspath(__file__)
_OCRLIB_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT  = os.path.dirname(_OCRLIB_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

API_VERSION = "V19_4_SPLIT_WITH_HEADER"
API_ORIGIN  = __file__

# ----- resilient imports (package and fallback local) -----
try:
    from OcrLibrary.BreakerTableAnalyzer3 import BreakerTableAnalyzer, ANALYZER_VERSION
    from OcrLibrary.BreakerTableParser4   import BreakerTableParser,  PARSER_VERSION
    from OcrLibrary.PanelHeaderParserV4   import PanelParser as PanelHeaderParser
except ModuleNotFoundError:
    from BreakerTableAnalyzer3 import BreakerTableAnalyzer, ANALYZER_VERSION
    from PanelHeaderParserV4   import PanelParser as PanelHeaderParser
    # Table parser may not exist in some light envs; import lazily inside run if needed
    try:
        from BreakerTableParser4 import BreakerTableParser, PARSER_VERSION  # type: ignore
    except Exception:
        BreakerTableParser, PARSER_VERSION = None, "unknown"  # lazy fallback


class BreakerTablePipeline:
    """
    Class-based, single point of entry that runs:
      1) BreakerTableAnalyzer3.analyze()
      2) PanelHeaderParserV4.PanelParser.parse_panel()
      3) BreakerTableParser4.parse_from_analyzer()  (unless header says 'false positive')

    Mirrors the working Dev test env you shared.
    """

    def __init__(self, *, debug: bool = True):
        self.debug = bool(debug)

        # stage objects (created on demand so we can reuse OCR reader)
        self._analyzer = None
        self._header_parser = None

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

        # --- 3) Table Parser (last; skip on header false-positive) ---
        should_run_parser = run_parser
        try:
            hdr_name = (header_result or {}).get("name", "")
            if isinstance(hdr_name, str) and hdr_name.strip().lower() == "false positive":
                should_run_parser = False
                if self.debug:
                    print("[INFO] Skipping breaker table parser due to header false positive")
        except Exception:
            pass

        if should_run_parser:
            # lazy-import if we couldn't import above
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

            if BreakerTableParser is not None:
                try:
                    parser = BreakerTableParser(debug=self.debug, reader=getattr(analyzer, "reader", None))
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

        # --- combined result (same shape as before) ---
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
