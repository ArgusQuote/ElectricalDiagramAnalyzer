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

    If debug=True, also writes a combined header+footer overlay:
        debug/<base>_hf_overlay.png
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

        If self.debug is True, also writes:
            <debug_dir>/<base>_hf_overlay.png
        showing:
          - header tokens
          - header text line (blue)
          - HEADER_Y (cyan)
          - HEADER_BOTTOM_Y (orange)
          - footer token text line (magenta, with value)
          - FOOTER_Y (green)
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

        footer_y         = footer_res.footer_y
        footer_token_val = footer_res.token_val
        panel_size       = footer_res.panel_size

        if self.debug:
            print(
                f"[BreakerTableAnalyzer] footer_y={footer_y}, "
                f"footer_token_val={footer_token_val}, panel_size={panel_size}"
            )

        # ---------- debug overlay ----------
        if self.debug and debug_dir is not None:
            try:
                self._write_debug_overlay(
                    gray=gray,
                    header_res=header_res,
                    footer_res=footer_res,
                    src_path=src_path,
                    debug_dir=debug_dir,
                )
            except Exception as e:
                # don't kill the pipeline on debug failure
                print(f"[BreakerTableAnalyzer] Failed to write debug overlay: {e}")

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

    def _write_debug_overlay(
        self,
        gray: np.ndarray,
        header_res: HeaderResult,
        footer_res: FooterResult,
        src_path: str,
        debug_dir: Optional[str],
    ) -> None:
        """
        Build a single overlay showing:

          - header tokens (from BreakerHeaderFinder.draw_ocr_overlay)
          - HEADER_TEXT_LINE (blue)
          - HEADER_Y (cyan)
          - HEADER_BOTTOM_Y (orange)
          - footer token baseline (magenta, labeled with value)
          - footer_y (green)
          - footer dbg_marks (teal ticks with labels)
        """
        if debug_dir is None:
            return

        H, W = gray.shape[:2]

        # start from header overlay (already draws header tokens + lines)
        overlay = self._header_finder.draw_ocr_overlay(gray)

        # ensure BGR
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        # ----- footer token: treat token_y as footer_text_line -----
        if footer_res.token_y is not None:
            y_tok = int(footer_res.token_y)
            val   = footer_res.token_val
            cv2.line(overlay, (0, y_tok), (W - 1, y_tok), (255, 0, 255), 1)  # magenta
            label = f"FOOTER_TEXT_LINE (val={val})"
            cv2.putText(
                overlay,
                label,
                (10, max(20, y_tok - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # ----- footer_y (snapped footer line) -----
        if footer_res.footer_y is not None:
            y_footer = int(footer_res.footer_y)
            cv2.line(overlay, (0, y_footer), (W - 1, y_footer), (0, 255, 0), 2)  # green
            cv2.putText(
                overlay,
                "FOOTER_Y",
                (10, min(H - 10, y_footer + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # ----- optional: footer dbg_marks as small ticks -----
        for y_mark, label in footer_res.dbg_marks:
            y = int(y_mark)
            if 0 <= y < H:
                cv2.line(overlay, (0, y), (W - 1, y), (0, 128, 128), 1)
                cv2.putText(
                    overlay,
                    label,
                    (W - 220, max(10, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 128, 128),
                    1,
                    cv2.LINE_AA,
                )

        base = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(debug_dir, f"{base}_hf_overlay.png")

        cv2.imwrite(out_path, overlay)
        print(f"[BreakerTableAnalyzer] Saved header+footer overlay -> {out_path}")


# Optional: quick CLI hook
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python BreakerTableAnalyzer9.py /path/to/panel.png")
        sys.exit(1)

    path = sys.argv[1]
    analyzer = BreakerTableAnalyzer(debug=True)
    res = analyzer.analyze(path)

    print("\n=== SIMPLE HEADER / FOOTER RESULT ===")
    for k, v in res.items():
        print(f"{k}: {v}")
