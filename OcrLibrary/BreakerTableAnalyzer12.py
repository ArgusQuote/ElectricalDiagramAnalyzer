# OcrLibrary/BreakerTableAnalyzer9.py
from __future__ import annotations
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

import sys

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# NEW: header/footer anchoring classes
from AnchoringClasses.BreakerHeaderFinder import BreakerHeaderFinder, HeaderResult
from AnchoringClasses.BreakerFooterFinder import BreakerFooterFinder, FooterResult

ANALYZER_VERSION = "Analyzer9_header_footer_only"
ANALYZER_ORIGIN  = __file__


@dataclass
class SimpleHFResult:
    """
    Minimal result object for this analyzer:
      - header_y         : top structural line of header band
      - header_bottom_y  : bottom structural line of header band
      - footer_y         : snapped footer line
      - footer_token_val : numeric token used to infer panel size (e.g. 84, 41, ...)
      - panel_size       : canonical panel size (84/72/... per PANEL_FOOTER_MAP)
    """
    src_path: str
    header_y: Optional[int]
    header_bottom_y: Optional[int]
    footer_y: Optional[int]
    footer_token_val: Optional[int]
    panel_size: Optional[int]


class BreakerTableAnalyzer:
    """
    SUPER-SIMPLIFIED analyzer:

      1) Read image and prep gray.
      2) Use BreakerHeaderFinder to get:
           - header_y
           - header_bottom_y
      3) Pass those (and gray) into BreakerFooterFinder.find_footer()
         to get:
           - footer_y
           - footer_token_val
           - panel_size

    Returns ONLY those values in a small dict / dataclass.
    No row centers, no degridding, no column overlays.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.reader = None

        # OCR reader shared by header + footer
        if _HAS_OCR:
            try:
                self.reader = easyocr.Reader(["en"], gpu=False)
            except Exception:
                self.reader = None

        # header / footer classes
        self._header_finder = BreakerHeaderFinder(
            self.reader,
            debug=self.debug,
            debug_dir=None,   # set per image in analyze()
        )
        self._footer_finder = BreakerFooterFinder(
            self.reader,
            debug=self.debug,
        )

        # optional external override for where debug images go
        self.debug_root_dir: Optional[str] = None

    # ============== public ==============

    def analyze(self, image_path: str) -> Dict:
        """
        Main entry:

          res = analyzer.analyze("/path/to/panel.png")

        Returns dict:

          {
            "src_path":         <str>,
            "header_y":         <int or None>,
            "header_bottom_y":  <int or None>,
            "footer_y":         <int or None>,
            "footer_token_val": <int or None>,
            "panel_size":       <int or None>,
            "debug_dir":        <str or None>,
          }
        """
        src_path = os.path.abspath(os.path.expanduser(image_path))
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        src_dir = os.path.dirname(src_path)

        # ---------- debug dir ----------
        if self.debug_root_dir:
            debug_dir = os.path.abspath(self.debug_root_dir)
        else:
            debug_dir = os.path.join(src_dir, "debug") if self.debug else None

        if self.debug and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        # wire debug dir into header/footer
        self._header_finder.debug_dir = debug_dir
        self._footer_finder.debug_dir = debug_dir

        # ---------- load + prep ----------
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src_path)
        gray = self._prep(img)

        # ---------- HEADER ----------
        header_res: HeaderResult = self._header_finder.analyze_rows(gray)

        header_y        = header_res.header_y
        header_bottom_y = header_res.header_bottom_y

        if self.debug:
            print(
                f"[BreakerTableAnalyzer] header_y={header_y}, "
                f"header_bottom_y={header_bottom_y}"
            )

        # Build the payload expected by BreakerFooterFinder.find_footer()
        analyzer_result = {
            "src_path":        src_path,
            "src_dir":         src_dir,
            "debug_dir":       debug_dir,
            "gray":            gray,     # raw gray used for line masks
            "gridless_gray":   None,     # you can plug a degridded image later if you want
            "header_y":        header_y,
            "header_bottom_y": header_bottom_y,
            "footer_struct_y": None,     # not used in this minimal flow
        }

        # ---------- FOOTER ----------
        footer_res: FooterResult = self._footer_finder.find_footer(
            analyzer_result=analyzer_result
        )

        footer_y        = footer_res.footer_y
        footer_token_val = footer_res.token_val
        panel_size      = footer_res.panel_size

        if self.debug:
            print(
                f"[BreakerTableAnalyzer] footer_y={footer_y}, "
                f"footer_token_val={footer_token_val}, panel_size={panel_size}"
            )

        # ---------- minimal payload ----------
        out = {
            "src_path":         src_path,
            "header_y":         header_y,
            "header_bottom_y":  header_bottom_y,
            "footer_y":         footer_y,
            "footer_token_val": footer_token_val,
            "panel_size":       panel_size,
            "debug_dir":        debug_dir,
        }
        return out

    # ============== internals ==============

    def _prep(self, img: np.ndarray) -> np.ndarray:
        """
        Same prep you had before: grayscale + CLAHE + min-height upscale.
        """
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


# Optional: quick test harness
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python BreakerTableAnalyzer9.py /path/to/panel.png")
        sys.exit(1)

    path = sys.argv[1]
    analyzer = BreakerTableAnalyzer(debug=True)
    res = analyzer.analyze(path)

    print("\n=== SIMPLE HEADER / FOOTER RESULT ===")
    for k, v in res.items():
        print(f"{k}: {v}")
