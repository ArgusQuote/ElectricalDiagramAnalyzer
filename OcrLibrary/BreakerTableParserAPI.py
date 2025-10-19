# OcrLibrary/BreakerTableParserAPI.py
import sys, os, inspect

# ----- make sure the repo root is on sys.path -----
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


def parse_image(
    image_path: str,
    *,
    run_analyzer: bool = True,
    run_parser: bool = True,
    run_header: bool = True,
    debug: bool = True
):
    """
    Unified API with stage toggles.
    - run_analyzer: run BreakerTableAnalyzer3.analyze()
    - run_parser:   run BreakerTableParser3.parse_from_analyzer()
    - run_header:   run PanelHeaderParserV4.PanelParser()

    Returns a dict with results for each stage (any may be None).
    """

    image_path = os.path.abspath(os.path.expanduser(image_path))

    analyzer_result = None
    parser_result = None
    header_result = None

    # --- Analyzer stage (unchanged) ---
    analyzer = BreakerTableAnalyzer(debug=debug)
    if run_analyzer:
        try:
            analyzer_result = analyzer.analyze(image_path)
        except Exception as e:
            if debug:
                print(f"[WARN] Analyzer failed: {e}")

    # --- Header Parser stage (run second) ---
    if run_header:
        try:
            header_parser = PanelHeaderParser(debug=debug)
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
                        image_path,
                        header_y_ratio=header_ratio
                    )
                else:
                    header_result = header_parser.parse_panel(image_path)
            else:
                header_result = header_parser.parse_panel(image_path)
        except Exception as e:
            if debug:
                print(f"[WARN] Header parser failed: {e}")

    # --- Parser stage (run last, only if header didn't false-positive) ---
    should_run_parser = run_parser
    try:
        hdr_name = (header_result or {}).get("name", "")
        if isinstance(hdr_name, str) and hdr_name.strip().lower() == "false positive":
            should_run_parser = False
            if debug:
                print("[INFO] Skipping breaker table parser due to header false positive")
    except Exception:
        pass

    if should_run_parser:
        try:
            parser = BreakerTableParser(debug=debug, reader=getattr(analyzer, "reader", None))
            if analyzer_result is not None:
                parser_result = parser.parse_from_analyzer(analyzer_result)
            else:
                parser_result = parser.parse_from_analyzer({})
        except Exception as e:
            if debug:
                print(f"[WARN] Parser failed: {e}")

    # --- Debug output for legacy behavior ---
    if debug and parser_result is not None:
        print("\n>>> FINAL (legacy-compatible prints) >>>")
        print(parser_result.get("spaces"))
        print(parser_result.get("detected_breakers"))

    # --- Return combined result ---
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


if __name__ == "__main__":
    # --- Dev banner ---
    modA = sys.modules[BreakerTableAnalyzer.__module__]
    implA = inspect.getsourcefile(BreakerTableAnalyzer) or inspect.getfile(BreakerTableAnalyzer)
    print(">>> DEV Analyzer version:", getattr(modA, "ANALYZER_VERSION", "unknown"))
    print(">>> DEV Analyzer file:", os.path.abspath(implA))

    modP = sys.modules[BreakerTableParser.__module__]
    implP = inspect.getsourcefile(BreakerTableParser) or inspect.getfile(BreakerTableParser)
    print(">>> DEV Parser version:", getattr(modP, "PARSER_VERSION", "unknown"))
    print(">>> DEV Parser file:", os.path.abspath(implP))

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

    result = parse_image(
        img,
        run_analyzer=run_analyzer,
        run_parser=run_parser,
        run_header=run_header,
        debug=True,
    )

    print("\n>>> Stage Summary:")
    for key, val in result["results"].items():
        print(f"  {key}: {'OK' if val else 'None'}")
 