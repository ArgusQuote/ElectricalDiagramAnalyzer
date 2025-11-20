# OcrLibrary/BreakerTableParser6.py
from __future__ import annotations
import os, re, json, cv2, numpy as np
from typing import Dict, Optional, List, Tuple

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

PARSER_VERSION = "Parser6_headerband_v1"

# --- simple OCR settings for header band ---
_HDR_OCR_SCALE        = 2.0          # up-res factor for header band OCR
_HDR_PAD_ROWS_TOP     = 42           # pixels above header_y
_HDR_PAD_ROWS_BOT     = 50           # pixels below header_y
_HDR_OCR_ALLOWLIST    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/()."


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
        # Prefer reader provided by analyzer (shared EasyOCR instance)
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
            - expects keys: gray or gridless_gray, header_y, src_path, src_dir, debug_dir
        """
        # --- SAFE gray selection (no boolean ops on numpy arrays) ---
        gray = analyzer_result.get("gridless_gray", None)
        if gray is None:
            gray = analyzer_result.get("gray", None)

        header_y = analyzer_result.get("header_y")
        src_path = analyzer_result.get("src_path")

        H = W = None
        if gray is not None:
            H, W = gray.shape

        debug_dir = self._ensure_debug_dir(analyzer_result)
        debug_img_raw_path = None
        debug_img_overlay_path = None

        # If we don't have the basics, bail gracefully
        if gray is None or header_y is None or H is None:
            if self.debug:
                print("[HeaderBandScanner] Missing gray or header_y; skipping header scan.")
            return {
                "band_y1": None,
                "band_y2": None,
                "tokens": [],
                "debugImageRaw": None,
                "debugImageOverlay": None,
            }

        # --- 1) define the header band in page coordinates ---
        y1 = max(0, int(header_y) - _HDR_PAD_ROWS_TOP)
        y2 = min(H, int(header_y) + _HDR_PAD_ROWS_BOT)
        if y2 <= y1:
            if self.debug:
                print(f"[HeaderBandScanner] Invalid band y1={y1}, y2={y2}; skipping.")
            return {
                "band_y1": y1,
                "band_y2": y2,
                "tokens": [],
                "debugImageRaw": None,
                "debugImageOverlay": None,
            }

        band = gray[y1:y2, :]
        if band.size == 0:
            if self.debug:
                print("[HeaderBandScanner] Empty header band after crop; skipping.")
            return {
                "band_y1": y1,
                "band_y2": y2,
                "tokens": [],
                "debugImageRaw": None,
                "debugImageOverlay": None,
            }

        # --- 1b) save RAW cropped band to debug folder so you can see exactly what we used ---
        if self.debug:
            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]
            debug_img_raw_path = os.path.join(debug_dir, f"{base}_parser_header_band_raw.png")
            try:
                cv2.imwrite(debug_img_raw_path, band)
            except Exception as e:
                print(f"[HeaderBandScanner] Failed to write raw header band image: {e}")
                debug_img_raw_path = None

        tokens: List[Dict] = []

        if self.reader is not None:
            # --- 2) up-res the band for OCR ---
            band_up = cv2.resize(
                band,
                None,
                fx=_HDR_OCR_SCALE,
                fy=_HDR_OCR_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )

            try:
                dets = self.reader.readtext(
                    band_up,
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
                    print(f"[HeaderBandScanner] OCR failed on header band: {e}")
                dets = []

            # --- 3) map OCR boxes back to band + page coordinates ---
            for box, txt, conf in dets:
                try:
                    conf_f = float(conf or 0.0)
                except Exception:
                    conf_f = 0.0

                # box in upscaled space -> downscale back to band coords
                pts_band = [
                    (
                        int(p[0] / _HDR_OCR_SCALE),
                        int(p[1] / _HDR_OCR_SCALE),
                    )
                    for p in box
                ]
                xs = [p[0] for p in pts_band]
                ys = [p[1] for p in pts_band]
                x1b, x2b = max(0, min(xs)), min(W - 1, max(xs))
                y1b, y2b = max(0, min(ys)), min(y2 - y1 - 1, max(ys))

                # page-coordinates (add vertical offset)
                y1_abs = y1 + y1b
                y2_abs = y1 + y2b

                tokens.append(
                    {
                        "text": str(txt or "").strip(),
                        "conf": conf_f,
                        "box_band": [int(x1b), int(y1b), int(x2b), int(y2b)],
                        "box_page": [int(x1b), int(y1_abs), int(x2b), int(y2_abs)],
                    }
                )

        # --- 4) build debug overlay for the band (cropped view) ---
        if self.debug:
            vis = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)

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
                # ensure label is visible above the box if possible
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

            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]
            debug_img_overlay_path = os.path.join(debug_dir, f"{base}_parser_header_band_overlay.png")
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
        }

class BreakerTableParser:
    """
    Parser6 orchestrator (work-in-progress).
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

    def parse_from_analyzer(self, analyzer_result: Dict) -> Dict:
        """
        Entry point used by the API.

        Right now:
          - Reads spaces from analyzer_result (corrected if available)
          - Runs the header-band OCR scan
          - Returns skeleton parser result (no breaker parsing yet)
        """
        if not isinstance(analyzer_result, dict):
            analyzer_result = {}

        spaces = analyzer_result.get("spaces_corrected")
        if spaces is None:
            spaces = analyzer_result.get("spaces")
        if spaces is None:
            spaces = 0

        header_scan = self._header_scanner.scan(analyzer_result)

        if self.debug:
            print(
                f"[BreakerTableParser] Header band y:[{header_scan.get('band_y1')},"
                f"{header_scan.get('band_y2')}) tokens={len(header_scan.get('tokens', []))}"
            )
            if header_scan.get("debugImage"):
                print(f"[BreakerTableParser] Header-band debug image: {header_scan['debugImage']}")

        # Minimal, legacy-compatible result
        return {
            "parserVersion": PARSER_VERSION,
            "name": None,  # API will overwrite with deduped panel name
            "spaces": int(spaces),
            "detected_breakers": [],  # filled later as we implement parsing
            "headerScan": header_scan,
        }
