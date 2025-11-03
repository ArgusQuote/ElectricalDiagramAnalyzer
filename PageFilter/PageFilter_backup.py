#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import fitz  # PyMuPDF
import cv2

# ---- EasyOCR (optional) ----
try:
    import easyocr
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


class PageFilter:
    """
    Stage 1 (OCR-first, recall-safe):
      - KEEP if we see an Electrical-looking corner stamp (E* with >=2 digits).
      - DROP only if we confidently see a non-E stamp (P/M/C/A/F/S/X + >=2 digits).
      - KEEP if page contains PANEL SCHEDULE keywords.
      - Otherwise undecided → Stage 2.

    Stage 2 (footprints on undecided pages only):
      - In large whitespace regions, find hole(s) whose bounding-box width/height are
        within given fractions of the page and with sufficient rectangularity.
      - KEEP if >= MIN_RECT_COUNT footprints are found; else DROP.

    Outputs:
      - Filtered PDF with kept pages (same fidelity as original).
      - JSON log (OCR hits, footprint hits, decision reasons) ONLY IF debug=True,
        saved to: <output_dir>/filter_debug/<basename>_filter_log.json

    Public API:
        kept, dropped, out_pdf, log_json = FILTER.readPdf("/path/to.pdf")
    """

    def __init__(
        self,
        output_dir: str,
        # ---- Rendering / performance ----
        dpi: int = 300,                  # raster DPI for undecided pages
        longest_cap_px: Optional[int] = 9000,  # hard-cap the longest side when rasterizing
        proc_scale: float = 0.5,         # process at downscale for footprints
        # ---- OCR settings ----
        use_ocr: bool = True,
        ocr_gpu: bool = False,
        ocr_zoom: float = 2.4,           # zoom for cropped corner OCR
        crop_frac: Tuple[float, float, float, float] = (0.88, 0.92, 0.98, 0.98),  # bottom-right
        panel_keywords: Tuple[str, ...] = ("PANEL SCHEDULE", "PANELBOARD SCHEDULE"),
        non_e_conf_min: float = 0.70,    # min conf to DROP on non-E
        # ---- Regex patterns ----
        pattern_e_loose: str = r"^E[A-Z\-]{0,5}.*\d.*\d",
        pattern_non_e: str = r"^(?:P|M|C|A|F|S|X)[A-Z\-]{0,5}.*\d.*\d",
        # ---- Whitespace/ink (footprint) extraction ----
        dilate_ink_px: int = 3,
        close_ink_px: int = 3,
        margin_shave_px: int = 6,
        min_whitespace_area_fr: float = 0.01,
        border_exclude_px: int = 4,
        # ---- Footprint shape (percent-of-page & rectangularity) ----
        rect_w_fr_range: Tuple[float, float] = (0.10, 0.55),
        rect_h_fr_range: Tuple[float, float] = (0.10, 0.60),
        min_rectangularity: float = 0.70,
        min_rect_count: int = 2,
        # ---- Behavior / logging ----
        verbose: bool = True,
        debug: bool = False,             # << Only when True we write JSON log
        out_pdf_suffix: str = "_electrical_filtered.pdf",
    ):
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # where debug JSON will go (only created/written when debug=True)
        self.debug = bool(debug)
        self.debug_dir = os.path.join(self.output_dir, "filter_debug")
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        # render/perf
        self.dpi = int(dpi)
        self.longest_cap_px = longest_cap_px if (longest_cap_px is None or longest_cap_px > 0) else None
        self.proc_scale = float(proc_scale)

        # OCR
        self.use_ocr = bool(use_ocr and OCR_AVAILABLE)
        self.ocr_gpu = bool(ocr_gpu)
        self.ocr_zoom = float(ocr_zoom)
        self.crop_frac = crop_frac
        self.panel_keywords = tuple(panel_keywords)
        self.non_e_conf_min = float(non_e_conf_min)
        self.reader = None  # lazy init

        # patterns
        self.PATTERN_E_LOOSE = re.compile(pattern_e_loose, re.IGNORECASE)
        self.PATTERN_NON_E = re.compile(pattern_non_e, re.IGNORECASE)

        # whitespace/ink
        self.DILATE_INK_PX = int(dilate_ink_px)
        self.CLOSE_INK_PX = int(close_ink_px)
        self.MARGIN_SHAVE_PX = int(margin_shave_px)
        self.MIN_WHITESPACE_AREA_FR = float(min_whitespace_area_fr)
        self.BORDER_EXCLUDE_PX = int(border_exclude_px)

        # footprints
        self.RECT_W_FR_RANGE = rect_w_fr_range
        self.RECT_H_FR_RANGE = rect_h_fr_range
        self.MIN_RECTANGULARITY = float(min_rectangularity)
        self.MIN_RECT_COUNT = int(min_rect_count)

        # logging/output
        self.verbose = bool(verbose)
        self.out_pdf_suffix = out_pdf_suffix

        if self.use_ocr and not OCR_AVAILABLE and self.verbose:
            print("[WARN] easyocr not available; proceeding without OCR (footprint fallback only)")

    # ----------------- Public API -----------------
    def readPdf(self, pdf_path: str) -> Tuple[List[int], List[int], Optional[str], Optional[str]]:
        """
        Process the PDF page-by-page:
        - OCR-first: keep/drop immediately if decisive.
        - Footprints: only if OCR undecided.
        Returns:
            kept_pages (1-based), dropped_pages (1-based), output_pdf_path (or None), log_json_path (or None)
        """
        pdf_path = os.path.expanduser(pdf_path)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(pdf_path)

        doc = fitz.open(pdf_path)
        out = fitz.open()
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_pdf = os.path.join(self.output_dir, base + self.out_pdf_suffix)

        # debug log path (only used if self.debug)
        log_path = os.path.join(self.debug_dir, f"{base}_filter_log.json") if self.debug else None

        if self.use_ocr and self.reader is None:
            self._init_ocr()

        kept, dropped = [], []
        logs: List[Dict[str, Any]] = []  # only populated if debug=True

        if self.verbose:
            print(f"[INFO] Scanning {len(doc)} page(s): OCR-first, footprints only if undecided...")

        for i, page in enumerate(doc):  # i is 0-based; logs/outputs use 1-based
            page_no = i + 1

            # --- Stage 1: OCR gold-pan (decide immediately if possible) ---
            ocr_hits = self._ocr_corner(page, page_index=i) if (self.use_ocr and self.reader) else []
            e_hits = [h for h in ocr_hits if self._looks_like_e_sheet(h["text"])]
            non_e_hits = [h for h in ocr_hits if h["conf"] >= self.non_e_conf_min and self._looks_like_non_e_sheet(h["text"])]
            has_panel_kw = self._page_has_panel_keywords(page)

            decision: Optional[str] = None
            reason = ""
            matches = {"E": [], "nonE": []}
            fp_summary: Dict[str, Any] = {}  # will be populated only if we run footprints

            if e_hits:
                decision, reason = "KEEP", "OCR: E match (normalized)"
                matches["E"] = [{"text": h["text"], "norm": self._normalize_token(h["text"]), "conf": h["conf"]} for h in e_hits]
            elif non_e_hits:
                decision, reason = "DROP", "OCR: confident non-E match"
                matches["nonE"] = [{"text": h["text"], "norm": self._normalize_token(h["text"]), "conf": h["conf"]} for h in non_e_hits]
            elif has_panel_kw:
                decision, reason = "KEEP", "OCR: panel keyword present"

            # --- Stage 2: rectangular-void footprints ONLY if undecided ---
            if decision is None:
                bgr = self._render_page_bgr(page)  # rasterize only now
                fp_ok, fp_count, fp_boxes = self._has_rectangular_footprints(bgr)
                fp_summary = {
                    "footprints_ok": fp_ok,
                    "footprints_found": fp_count,
                    "footprints": fp_boxes,  # list of dicts (bbox fractions, rectangularity)
                }
                if fp_ok:
                    decision, reason = "KEEP", f"Footprints: {fp_count} rectangular void(s)"
                else:
                    decision, reason = "DROP", "No qualifying footprints"

            # --- Apply decision ---
            if decision == "KEEP":
                out.insert_pdf(doc, from_page=i, to_page=i)
                kept.append(page_no)
            else:
                dropped.append(page_no)

            # --- Build log entry if debug ---
            if self.debug:
                logs.append({
                    "page": page_no,
                    "ocr_hits": ocr_hits,
                    "matches": matches,
                    "panel_keywords": has_panel_kw,
                    "decision": decision,
                    "reason": reason,
                    **({"footprints": fp_summary} if fp_summary else {}),
                })

            if self.verbose:
                print(f"  Page {page_no:03d}: {decision} — {reason}")

        # Save filtered PDF
        out_pdf_path = None
        if len(out) > 0:
            out.save(out_pdf)
            out_pdf_path = out_pdf
            if self.verbose:
                print(f"[OK] Saved filtered PDF → {out_pdf_path}  (kept {len(kept)}/{len(doc)})")
        else:
            if self.verbose:
                print("[WARN] No pages kept — nothing saved")

        # Save JSON log only if debug=True
        log_json_path = None
        if self.debug:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
            log_json_path = log_path
            if self.verbose:
                print(f"[DEBUG] Wrote log JSON → {log_json_path}")

        return kept, dropped, out_pdf_path, log_json_path

    # ----------------- OCR helpers -----------------
    def _init_ocr(self):
        if not self.use_ocr:
            return
        try:
            self.reader = easyocr.Reader(['en'], gpu=self.ocr_gpu, verbose=False)
            if self.verbose:
                print("[INFO] EasyOCR initialized")
        except Exception as e:
            self.reader = None
            self.use_ocr = False
            if self.verbose:
                print(f"[WARN] EasyOCR init failed: {e}. Continuing without OCR.")

    @staticmethod
    def _clip_from_frac(page: fitz.Page, frac: Tuple[float, float, float, float]) -> fitz.Rect:
        r = page.rect
        x0, y0, x1, y1 = frac
        return fitz.Rect(r.x0 + r.width * x0, r.y0 + r.height * y0,
                         r.x0 + r.width * x1, r.y0 + r.height * y1)

    def _ocr_corner(self, page: fitz.Page, page_index: int) -> List[Dict[str, Any]]:
        """OCR the fixed corner crop; returns list of {text, conf} dicts. Empty if OCR disabled."""
        if not (self.use_ocr and self.reader):
            return []
        clip = self._clip_from_frac(page, self.crop_frac)
        pix = page.get_pixmap(matrix=fitz.Matrix(self.ocr_zoom, self.ocr_zoom), clip=clip, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        results = self.reader.readtext(arr, detail=1, paragraph=False,
                                       allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- .")
        return [{"text": (t or "").strip(), "conf": float(c)} for (_b, t, c) in results]

    def _page_has_panel_keywords(self, page: fitz.Page) -> bool:
        text = page.get_text("text") or ""
        up = text.upper()
        return any(kw in up for kw in self.panel_keywords)

    @staticmethod
    def _normalize_token(tok: str) -> str:
        s = (tok or "").upper().strip()
        if not s:
            return ""
        s = s.replace("—", "-").replace("–", "-").replace("_", "").replace(" ", "").replace(".", "")
        out = []
        for i, ch in enumerate(s):
            prev_is_digit = i > 0 and s[i - 1].isdigit()
            next_is_digit = (i + 1 < len(s)) and s[i + 1].isdigit()
            if ch == 'O' and (prev_is_digit or next_is_digit):
                out.append('0')
            elif ch in ('I', 'L', '|') and (prev_is_digit or next_is_digit):
                out.append('1')
            elif ch == 'S' and (prev_is_digit or next_is_digit):
                out.append('5')
            elif ch == 'B' and (prev_is_digit or next_is_digit):
                out.append('8')
            else:
                out.append(ch)
        return "".join(out)

    def _looks_like_e_sheet(self, tok: str) -> bool:
        norm = self._normalize_token(tok)
        if not norm:
            return False
        if norm.endswith("-"):
            norm = norm[:-1]
        return bool(self.PATTERN_E_LOOSE.match(norm))

    def _looks_like_non_e_sheet(self, tok: str) -> bool:
        norm = self._normalize_token(tok)
        if not norm:
            return False
        if norm.endswith("-"):
            norm = norm[:-1]
        return bool(self.PATTERN_NON_E.match(norm))

    # ----------------- Rendering for footprint -----------------
    def _render_page_bgr(self, page: fitz.Page) -> np.ndarray:
        """
        Render page → BGR numpy array at self.dpi, respecting self.longest_cap_px.
        """
        scale = self.dpi / 72.0  # PDF points → inches
        if self.longest_cap_px is not None:
            w_pt, h_pt = page.rect.width, page.rect.height
            w_px, h_px = w_pt * scale, h_pt * scale
            longest = max(w_px, h_px)
            if longest > self.longest_cap_px:
                scale *= (self.longest_cap_px / longest)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr

    # ----------------- Footprint detector -----------------
    def _has_rectangular_footprints(self, img_bgr: np.ndarray) -> Tuple[bool, int, List[Dict[str, float]]]:
        """
        Returns (ok, count, boxes) where:
          ok    = True if count >= MIN_RECT_COUNT
          count = number of qualifying footprints
          boxes = list of dicts with bbox fractions and rectangularity for logging
        """
        Hf, Wf = img_bgr.shape[:2]
        if self.proc_scale < 1.0:
            img = cv2.resize(img_bgr, (int(Wf * self.proc_scale), int(Hf * self.proc_scale)),
                             interpolation=cv2.INTER_AREA)
        else:
            img = img_bgr

        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ink/whitespace at small scale
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 255=ink
        if self.CLOSE_INK_PX > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (2 * max(1, int(self.CLOSE_INK_PX * self.proc_scale)) + 1,
                 2 * max(1, int(self.CLOSE_INK_PX * self.proc_scale)) + 1)
            )
            ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k, iterations=1)
        if self.DILATE_INK_PX > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (2 * max(1, int(self.DILATE_INK_PX * self.proc_scale)) + 1,
                 2 * max(1, int(self.DILATE_INK_PX * self.proc_scale)) + 1)
            )
            ink = cv2.dilate(ink, k, iterations=1)

        whitespace = cv2.bitwise_not(ink)

        # shave margins (small scale)
        mpx = max(1, int(self.MARGIN_SHAVE_PX * self.proc_scale))
        whitespace[:mpx, :] = 0
        whitespace[-mpx:, :] = 0
        whitespace[:, :mpx] = 0
        whitespace[:, -mpx:] = 0

        lab = (whitespace > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(lab, connectivity=8)
        if num <= 1:
            return (False, 0, [])

        page_area = H * W
        keep_ids = [cid for cid in range(1, num)
                    if stats[cid, cv2.CC_STAT_AREA] / float(page_area) >= self.MIN_WHITESPACE_AREA_FR]
        if not keep_ids:
            return (False, 0, [])

        keep_mask = np.zeros((H, W), dtype=np.uint8)
        for cid in keep_ids:
            keep_mask[labels == cid] = 255

        contours, hier = cv2.findContours(keep_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(hier) == 0:
            return (False, 0, [])
        hier = hier[0]

        min_border = max(1, int(self.BORDER_EXCLUDE_PX * self.proc_scale))
        boxes: List[Dict[str, float]] = []
        found = 0

        for idx, cnt in enumerate(contours):
            # we only want holes inside whitespace components
            if hier[idx][3] == -1:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # ignore touching borders
            if x <= min_border or y <= min_border or (x + w) >= (W - min_border) or (y + h) >= (H - min_border):
                continue

            # fractions of page
            w_fr = w / float(W)
            h_fr = h / float(H)
            if not (self.RECT_W_FR_RANGE[0] <= w_fr <= self.RECT_W_FR_RANGE[1]):
                continue
            if not (self.RECT_H_FR_RANGE[0] <= h_fr <= self.RECT_H_FR_RANGE[1]):
                continue

            area_cnt = cv2.contourArea(cnt)
            area_box = w * h
            if area_box <= 0:
                continue

            rectangularity = area_cnt / float(area_box)
            if rectangularity >= self.MIN_RECTANGULARITY:
                found += 1
                boxes.append({
                    "bbox_w_frac": round(w_fr, 4),
                    "bbox_h_frac": round(h_fr, 4),
                    "rectangularity": round(float(rectangularity), 4),
                })
                if found >= self.MIN_RECT_COUNT:
                    return (True, found, boxes)

        return (False, found, boxes)
