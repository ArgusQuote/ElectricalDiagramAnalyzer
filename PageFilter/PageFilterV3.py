#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pypdfium2 as pdfium  # Apache 2.0 - PDF rendering
import pikepdf              # MPL 2.0 - PDF manipulation
from PIL import Image
import cv2

import subprocess, shutil, tempfile
from pathlib import Path

# ---- EasyOCR (optional) ----
try:
    import easyocr
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


class PageFilter:
    """
    Tight-crop pipeline (single crop only):
      - One bottom-right crop (title-block column) that is the same base width, but:
          * grows UP by label_tall_factor
          * grows a little to the RIGHT by crop_expand_right
        We OCR this ONCE per page.
      - Pass A: from that OCR, detect E-sheets via E### tokens.
      - Pass B: for E-sheets, REUSE the SAME OCR TEXT to score label phrases.
        'Panel schedules' has higher priority than generic 'schedules'.

      If there are NO E-sheets at all:
        - Use undecided → footprints logic.
    """

    def __init__(
        self,
        output_dir: str,
        # ---- Rendering / performance ----
        dpi: int = 300,
        longest_cap_px: Optional[int] = 9000,
        proc_scale: float = 0.5,
        # ---- OCR settings ----
        use_ocr: bool = True,
        ocr_gpu: bool = False,
        ocr_zoom: float = 2.4,  # zoom for the (now taller) corner OCR
        crop_frac: Tuple[float, float, float, float] = (0.88, 0.92, 0.98, 0.98),  # base corner column
        non_e_conf_min: float = 0.70,
        # ---- Single-crop growth knobs ----
        label_tall_factor: float = 4.0,   # times taller than the base corner height (UP)
        crop_expand_right: float = 0.02,  # fraction of page width to expand RIGHT
        # scoring thresholds
        hard_hit_score: float = 6.5,     # gold threshold
        min_hit_score: float = 3.0,      # soft threshold
        # weights
        w_strong: float = 4.5,           # single/one-line + panel schedules
        w_riser: float = 3.5,            # riser / riser diagram
        w_diag: float = 2.0,             # optional generic diagram/drawing (weaker)
        w_base: float = 1.0,             # generic schedules (weak)
        w_plural_bonus: float = 0.4,
        conf_mult: float = 1.2,          # confidence multiplier
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
        debug: bool = False,
        save_crop_pngs: bool = True,
        out_pdf_suffix: str = "_electrical_filtered.pdf",
        use_ghostscript_letter: bool = True,
        letter_orientation: str = "landscape",   # or "landscape"
        gs_use_cropbox: bool = True,
        gs_compat: str = "1.7",       
        # Back-compat sink: accept legacy kwargs without error
        **deprecated_kwargs: Any,
    ):
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # where debug JSON & crops will go
        self.debug = bool(debug)
        self.debug_dir = os.path.join(self.output_dir, "filter_debug")
        self.crops_dir = os.path.join(self.debug_dir, "crops")
        self.save_crop_pngs = bool(save_crop_pngs)
        if self.debug or self.save_crop_pngs:
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(self.crops_dir, exist_ok=True)

        # render/perf
        self.dpi = int(dpi)
        self.longest_cap_px = longest_cap_px if (longest_cap_px is None or longest_cap_px > 0) else None
        self.proc_scale = float(proc_scale)

        # OCR
        self.use_ocr = bool(use_ocr and OCR_AVAILABLE)
        self.ocr_gpu = bool(ocr_gpu)
        self.ocr_zoom = float(ocr_zoom)
        self.crop_frac = crop_frac
        self.non_e_conf_min = float(non_e_conf_min)
        self.reader = None  # lazy init

        # patterns
        self.PATTERN_E_LOOSE = re.compile(pattern_e_loose, re.IGNORECASE)
        self.PATTERN_NON_E = re.compile(pattern_non_e, re.IGNORECASE)

        # single-crop config
        self.label_tall_factor = float(label_tall_factor)
        self.crop_expand_right = float(crop_expand_right)

        # back-compat: if legacy label_expand_up was passed, infer tall factor
        if deprecated_kwargs:
            if "label_expand_up" in deprecated_kwargs:
                try:
                    expand_up = float(deprecated_kwargs["label_expand_up"])
                    corner_h_frac = max(1e-6, (self.crop_frac[3] - self.crop_frac[1]))
                    inferred = 1.0 + (expand_up / corner_h_frac)
                    if label_tall_factor == 4.0:  # default implies not explicitly set
                        self.label_tall_factor = max(1.0, inferred)
                    if verbose:
                        print(f"[INFO] Using inferred label_tall_factor={self.label_tall_factor:.2f} "
                              f"from legacy label_expand_up={expand_up:.3f}")
                except Exception:
                    if verbose:
                        print("[INFO] Ignoring legacy label_expand_up (could not infer).")
            # Silently ignore legacy widen params
            for k in ("label_expand_left", "label_expand_left_retry", "label_zoom"):
                if k in deprecated_kwargs and verbose:
                    print(f"[INFO] Ignoring legacy parameter '{k}' (no longer used).")

        self.HARD_HIT_SCORE = float(hard_hit_score)
        self.MIN_HIT_SCORE = float(min_hit_score)
        self.W_STRONG = float(w_strong)
        self.W_RISER = float(w_riser)
        self.W_DIAG = float(w_diag)
        self.W_BASE = float(w_base)
        self.W_PLURAL_BONUS = float(w_plural_bonus)
        self.CONF_MULT = float(conf_mult)

        # compiled label regex buckets (kept) – regex + fuzzy backup in scorer
        self._label_patterns: Dict[str, List[re.Pattern]] = {
            "panel_schedules": [
                re.compile(r"\bpanel\s*schedules?\b", re.IGNORECASE),
                re.compile(r"\bpanelboard\s*schedules?\b", re.IGNORECASE),
                # tolerate OCR spaces/hyphens
                re.compile(r"\bpanel[\s\-]+schedules?\b", re.IGNORECASE),
            ],
            "single_one_line": [
                re.compile(r"\bsingle[\s\-]?line\b", re.IGNORECASE),
                re.compile(r"\bone[\s\-]?line\b", re.IGNORECASE),
                re.compile(r"\b1[\s\-]?line\b", re.IGNORECASE),
                re.compile(r"\b(?:one|single)[\s\-]?line\s*diagram\b", re.IGNORECASE),
            ],
            "riser": [
                re.compile(r"\briser\b", re.IGNORECASE),
                re.compile(r"\briser\s*diagram\b", re.IGNORECASE),
            ],
            "diagram_generic": [
                re.compile(r"\belectrical\s*diagram(s)?\b", re.IGNORECASE),
                re.compile(r"\bdiagram(s)?\b", re.IGNORECASE),
                re.compile(r"\bdrawing(s)?\b", re.IGNORECASE),
            ],
            "schedules_generic": [
                re.compile(r"\bschedules?\b", re.IGNORECASE),
            ],
        }
        self._bucket_weight = {
            "panel_schedules": self.W_STRONG,
            "single_one_line": self.W_STRONG,
            "riser": self.W_RISER,
            "diagram_generic": self.W_DIAG,
            "schedules_generic": self.W_BASE,
        }

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

        # GS post-process config
        self.use_ghostscript_letter = bool(use_ghostscript_letter)
        self.letter_orientation = "landscape" if str(letter_orientation).lower().startswith("land") else "portrait"
        self.gs_use_cropbox = bool(gs_use_cropbox)
        self.gs_compat = str(gs_compat)

        if self.use_ocr and not OCR_AVAILABLE and self.verbose:
            print("[WARN] easyocr not available; proceeding without OCR (footprint fallback only)")

    # ----------------- Public API -----------------
    def readPdf(self, pdf_path: str) -> Tuple[List[int], List[int], Optional[str], Optional[str]]:
        """
        Single-crop flow:
        - Pass A: OCR taller/wider corner crop → detect E-sheets.
        - Pass B: REUSE OCR text to score labels (panel schedules has priority over generic schedules).
        - No gold? Keep all E-sheets.
        - No E-sheets? Use undecided→footprints path.

        Output PDF pages are normalized to **Letter** size (612×792 pt) via Ghostscript
        unless self.use_ghostscript_letter=False.
        """
        pdf_path = os.path.expanduser(pdf_path)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(pdf_path)

        # Use pypdfium2 for rendering, pikepdf for PDF manipulation
        pdfium_doc = pdfium.PdfDocument(pdf_path)
        pikepdf_doc = pikepdf.open(pdf_path)
        out_pdf_obj = pikepdf.new()
        letter_width, letter_height = 612, 792  # US Letter in points

        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_pdf = os.path.join(self.output_dir, base + self.out_pdf_suffix)

        log_path = os.path.join(self.debug_dir, f"{base}_filter_log.json") if self.debug else None

        if self.use_ocr and self.reader is None:
            self._init_ocr()

        kept, dropped = [], []
        logs: List[Dict[str, Any]] = []

        if self.verbose:
            print(f"[INFO] Scanning {len(pdfium_doc)} page(s): TallCorner OCR → Label Score (reuse) → Fallback Footprints...")

        # -------- Pass A: detect E-sheets via one taller/wider corner OCR --------
        e_pages: List[int] = []
        undecided_pages: List[int] = []
        passA_hits: Dict[int, Dict[str, Any]] = {}

        for i in range(len(pdfium_doc)):
            page_no = i + 1
            pdfium_page = pdfium_doc[i]
            page_width = pdfium_page.get_width()
            page_height = pdfium_page.get_height()
            ocr_hits = self._ocr_corner_pdfium(pdfium_page, page_width, page_height, page_index=i) if (self.use_ocr and self.reader) else []
            e_hits = [h for h in ocr_hits if self._looks_like_e_sheet(h["text"])]
            non_e_hits = [h for h in ocr_hits if h["conf"] >= self.non_e_conf_min and self._looks_like_non_e_sheet(h["text"])]

            if self.verbose:
                norm_hits = [self._normalize_token(h["text"]) for h in ocr_hits]
                print(f"  [TallCorner] Page {page_no:03d}: hits={norm_hits}  e={bool(e_hits)}  nonE={(len(non_e_hits)>0)}")

            if e_hits:
                e_pages.append(i)
                passA_hits[i] = {"page": page_no, "corner_hits": ocr_hits, "decision_passA": "E"}
            elif non_e_hits:
                passA_hits[i] = {"page": page_no, "corner_hits": ocr_hits, "decision_passA": "nonE"}
            else:
                undecided_pages.append(i)
                passA_hits[i] = {"page": page_no, "corner_hits": ocr_hits, "decision_passA": "undecided"}

        # -------- Pass B: reuse same OCR to score labels for E-pages --------
        selected_indexes: List[int] = []
        gold_found = False
        label_scores: Dict[int, float] = {}
        label_tags: Dict[int, List[str]] = {}

        if e_pages and self.use_ocr and self.reader:
            for i in e_pages:
                spans = [{"text": (h.get("text") or ""), "conf": float(h.get("conf", 0.9))} for h in passA_hits[i]["corner_hits"]]
                score, tags = (0.0, [])
                if spans:
                    pg = pdfium_doc[i]
                    page_rect = (0, 0, pg.get_width(), pg.get_height())
                    score, tags = self._score_label_spans(page_rect, spans)
                label_scores[i], label_tags[i] = score, tags
                if self.verbose:
                    print(f"  [Labels:reuse] E-page {i+1:03d}: tallx{int(self.label_tall_factor)} score={score:.2f} tags={tags}")

            panel_pages = [i for i, tags in label_tags.items() if "panel_schedules" in tags]
            sched_pages = [i for i, tags in label_tags.items() if ("panel_schedules" not in tags and "schedules_generic" in tags)]
            other_hits = [i for i, tags in label_tags.items()
                        if ("single_one_line" in tags) or ("riser" in tags) or ("diagram_generic" in tags)]

            if panel_pages:
                schedule_selection = sorted(panel_pages)
                if self.verbose:
                    print(f"[INFO] Priority selection → panel schedules pages: {[idx+1 for idx in schedule_selection]}")
            elif sched_pages:
                schedule_selection = sorted(sched_pages)
                if self.verbose:
                    print(f"[INFO] Fallback selection → generic schedules pages: {[idx+1 for idx in schedule_selection]}")
            else:
                schedule_selection = []

            selected_indexes = sorted(set(schedule_selection).union(other_hits))

            if not selected_indexes:
                hard = [i for i, s in label_scores.items() if s >= self.HARD_HIT_SCORE]
                soft = [i for i, s in label_scores.items() if self.MIN_HIT_SCORE <= s < self.HARD_HIT_SCORE]
                if hard:
                    gold_found = True
                    selected_indexes = sorted(hard)
                    if self.verbose:
                        print(f"[INFO] GOLD pages (labels): {[idx+1 for idx in selected_indexes]}")
                elif soft:
                    selected_indexes = sorted(soft)
                    if self.verbose:
                        print(f"[INFO] Soft-hit pages (labels): {[idx+1 for idx in selected_indexes]}")
                else:
                    selected_indexes = sorted(e_pages)
                    if self.verbose:
                        print(f"[INFO] No label hits → keeping all E-pages: {[idx+1 for idx in selected_indexes]}")

            # ---- Write selected pages ----
            for i in range(len(pdfium_doc)):
                page_no = i + 1
                if i in selected_indexes:
                    # Use pikepdf to copy the page
                    out_pdf_obj.pages.append(pikepdf_doc.pages[i])
                    kept.append(page_no)

                    if self.debug:
                        logs.append({
                            "page": page_no,
                            "decision": "KEEP",
                            "reason": "panel schedules priority" if i in panel_pages else
                                    ("generic schedules fallback" if i in sched_pages else
                                    ("E-page labels (gold)" if gold_found else "E-page labels (fallback E-pages)")),
                            "label_score": label_scores.get(i, 0.0),
                            "label_tags": label_tags.get(i, []),
                            "passA": passA_hits.get(i, {}),
                        })
                    if self.verbose:
                        print(f"  Page {page_no:03d}: KEEP — tags={label_tags.get(i, [])} (score={label_scores.get(i,0):.2f})")
                else:
                    dropped.append(page_no)
                    if self.debug:
                        logs.append({
                            "page": page_no,
                            "decision": "DROP",
                            "reason": "Not selected after label prioritization",
                            "label_score": label_scores.get(i, 0.0),
                            "label_tags": label_tags.get(i, []),
                            "passA": passA_hits.get(i, {}),
                        })

            return self._finalize_output(pdfium_doc, pikepdf_doc, out_pdf_obj, kept, dropped, out_pdf, log_path, logs)

        # -------- No E-pages found: fallback undecided→footprints --------
        if self.verbose:
            print("[INFO] No E-sheets detected in Pass A. Running original undecided→footprints flow.")

        for i in range(len(pdfium_doc)):
            page_no = i + 1
            pdfium_page = pdfium_doc[i]
            decision: Optional[str] = None
            reason = ""
            fp_summary: Dict[str, Any] = {}
            ocr_hits = passA_hits.get(i, {}).get("corner_hits", [])

            if passA_hits.get(i, {}).get("decision_passA") == "nonE":
                decision, reason = "DROP", "OCR corner: confident non-E"
            elif passA_hits.get(i, {}).get("decision_passA") == "E":
                decision, reason = "KEEP", "OCR corner: E match"
            else:
                bgr = self._render_page_bgr_pdfium(pdfium_page)
                fp_ok, fp_count, fp_boxes = self._has_rectangular_footprints(bgr)
                fp_summary = {
                    "footprints_ok": fp_ok,
                    "footprints_found": fp_count,
                    "footprints": fp_boxes,
                }
                if fp_ok:
                    decision, reason = "KEEP", f"Footprints: {fp_count} rectangular void(s)"
                else:
                    decision, reason = "DROP", "No qualifying footprints"

            if decision == "KEEP":
                # Use pikepdf to copy the page
                out_pdf_obj.pages.append(pikepdf_doc.pages[i])
                kept.append(page_no)
            else:
                dropped.append(page_no)

            if self.debug:
                logs.append({
                    "page": page_no,
                    "ocr_corner_hits": ocr_hits,
                    "decision": decision,
                    "reason": reason,
                    **({"footprints": fp_summary} if fp_summary else {}),
                })

            if self.verbose:
                print(f"  Page {page_no:03d}: {decision} — {reason}")

        return self._finalize_output(pdfium_doc, pikepdf_doc, out_pdf_obj, kept, dropped, out_pdf, log_path, logs)

    # ----------------- OCR helpers -----------------
    def _init_ocr(self):
        """Lazily initialize the EasyOCR reader; disable OCR on failure."""
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

    def _gs_fit_to_letter(self, in_pdf: str, out_pdf: str) -> str:
        """
        Use Ghostscript to rewrite 'in_pdf' so each page is scaled to US Letter,
        preserving vectors and avoiding clipping. Returns output file path.
        """
        src = Path(in_pdf).expanduser().resolve()
        dst = Path(out_pdf).expanduser().resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        gs = shutil.which("gs")
        if not gs:
            raise RuntimeError("Ghostscript not found. Install with: sudo apt-get install -y ghostscript")

        # US Letter points (portrait or landscape)
        if self.letter_orientation == "landscape":
            w_pt, h_pt = 792, 612
        else:
            w_pt, h_pt = 612, 792

        args = [
            gs, "-q", "-dBATCH", "-dNOPAUSE",
            "-sDEVICE=pdfwrite",
            f"-dCompatibilityLevel={self.gs_compat}",
            "-dFIXEDMEDIA", "-dPDFFitPage",
            "-dAutoRotatePages=/None",
            "-dDEVICEWIDTHPOINTS=" + str(w_pt),
            "-dDEVICEHEIGHTPOINTS=" + str(h_pt),
        ]

        # respect CropBox if requested
        if self.gs_use_cropbox:
            args.append("-dUseCropBox")

        # write to temp, then replace
        tmp_out = Path(tempfile.gettempdir()) / (dst.name + ".tmp")
        args.extend(["-sOutputFile=" + str(tmp_out), str(src)])

        cp = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"Ghostscript failed ({cp.returncode}). Stderr:\n{cp.stderr.strip()}")

        tmp_out.replace(dst)
        return str(dst)

    @staticmethod
    def _clip_from_frac_tuple(page_width: float, page_height: float, frac: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Get clip rect as (x0, y0, x1, y1) from page dimensions and fraction tuple."""
        x0f, y0f, x1f, y1f = frac
        return (page_width * x0f, page_height * y0f,
                page_width * x1f, page_height * y1f)

    def _corner_rect_tuple(self, page_width: float, page_height: float) -> Tuple[float, float, float, float]:
        """
        Single, taller & slightly wider crop:
          - LEFT/BASE TOP/BOTTOM from crop_frac
          - Extends UP by (label_tall_factor - 1) * corner_height
          - Expands RIGHT by (crop_expand_right * page.width)
        Returns (x0, y0, x1, y1) in page points.
        """
        left, top, right, bottom = self.crop_frac
        base_x0, base_y0, base_x1, base_y1 = self._clip_from_frac_tuple(page_width, page_height, (left, top, right, bottom))
        ch = base_y1 - base_y0

        # grow up
        extra_up = max(0.0, (self.label_tall_factor - 1.0) * ch)
        new_top = max(0, base_y0 - extra_up)

        # expand right
        extra_right = max(0.0, self.crop_expand_right * page_width)
        new_right = min(page_width, base_x1 + extra_right)

        return (base_x0, new_top, new_right, base_y0 + ch)

    def _save_crop_png(self, arr: np.ndarray, path: str) -> None:
        """Write a numpy array to disk as a PNG via OpenCV (silently ignores errors)."""
        try:
            if arr.ndim == 2:
                cv2.imwrite(path, arr)
            else:
                cv2.imwrite(path, arr)
        except Exception:
            pass

    def _ocr_corner_pdfium(self, page, page_width: float, page_height: float, page_index: int) -> List[Dict[str, Any]]:
        """
        OCR the single taller/wider crop using pypdfium2; returns list of {text, conf}.
        """
        if not (self.use_ocr and self.reader):
            return []
        clip = self._corner_rect_tuple(page_width, page_height)
        clip_x0, clip_y0, clip_x1, clip_y1 = clip
        
        # Render full page at OCR zoom, then crop
        bitmap = page.render(scale=self.ocr_zoom)
        pil_img = bitmap.to_pil()
        
        # Calculate crop in pixels
        px0 = int(clip_x0 * self.ocr_zoom)
        py0 = int(clip_y0 * self.ocr_zoom)
        px1 = int(clip_x1 * self.ocr_zoom)
        py1 = int(clip_y1 * self.ocr_zoom)
        
        cropped = pil_img.crop((px0, py0, px1, py1))
        arr = np.array(cropped)
        
        # Convert RGB to BGR for OpenCV compatibility
        if len(arr.shape) == 3 and arr.shape[2] >= 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        if self.save_crop_pngs:
            out_path = os.path.join(self.crops_dir, f"page_{page_index+1:03d}_tallcorner_x{int(self.label_tall_factor)}.png")
            self._save_crop_png(arr, out_path)

        results = self.reader.readtext(arr, detail=1, paragraph=False,
                                       allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- .")
        return [{"text": (t or "").strip(), "conf": float(c)} for (_b, t, c) in results]

    # ----------------- Label scoring helpers -----------------
    def _score_label_spans(self, page_rect: Tuple[float, float, float, float], spans: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """
        Score intent phrases using:
        1) existing regex + confidence weighting (unchanged),
        2) lightweight fuzzy token heuristics for ALL buckets:
            - panel schedules   → 'panel' (≤1 edit) + 'schedule/schedules' (≤2 edits)
            - generic schedules → 'schedule/schedules' (≤2 edits)
            - single/one-line   → ('single'≤1 OR 'one'≤1 OR token == '1') + 'line'≤1
            - riser             → 'riser' ≤1
            - diagram generic   → any of {'diagram','diagrams','drawing','drawings','schematic'} ≤2
        """
        # -------- 1) Regex scoring (as before) --------
        text_norm = self._normalize_for_regex(" ".join(s.get("text","") for s in spans))
        total = 0.0
        tags_set: set = set()

        for bucket, patterns in self._label_patterns.items():
            bucket_score = 0.0
            found_any = False
            for pat in patterns:
                for m in pat.finditer(text_norm):
                    found_any = True
                    conf_mean = self._approx_match_conf(spans, m.group(0))
                    hit = self._bucket_weight[bucket]
                    if "schedules" in m.group(0).lower():
                        hit += self.W_PLURAL_BONUS
                    bucket_score += hit * (1.0 + (conf_mean * self.CONF_MULT - 1.0))
            if found_any:
                tags_set.add(bucket)
                total += bucket_score

        # -------- 2) Fuzzy token backup --------
        tokens = self._extract_alpha_tokens(text_norm)  # now alphanum tokens
        has = lambda tgt, d: self._has_token_like(tokens, target=tgt, max_dist=d)

        # schedules (generic + panel)
        schedulish = has("schedule", 2) or has("schedules", 2)
        panelish   = has("panel", 1) or has("panelboard", 2)

        # single/one-line (accept '1' token)
        lineish    = has("line", 1)
        singleish  = has("single", 1) or has("one", 1) or ("1" in tokens)

        # riser
        riserish   = has("riser", 1)

        # diagrams/drawings
        diagramish = has("diagram", 2) or has("diagrams", 2)
        drawingish = has("drawing", 2) or has("drawings", 2)
        schematicish = has("schematic", 2) or has("schematics", 2)

        # Apply fuzzy tags + add weights (one-time boosts; keep regex weight if it already hit)
        if schedulish and panelish:
            if "panel_schedules" not in tags_set:
                tags_set.add("panel_schedules")
                total += self.W_STRONG
        elif schedulish:
            if "schedules_generic" not in tags_set and "panel_schedules" not in tags_set:
                tags_set.add("schedules_generic")
                total += self.W_BASE

        if singleish and lineish:
            if "single_one_line" not in tags_set:
                tags_set.add("single_one_line")
                total += self.W_STRONG

        if riserish:
            if "riser" not in tags_set:
                tags_set.add("riser")
                total += self.W_RISER

        if diagramish or drawingish or schematicish:
            if "diagram_generic" not in tags_set:
                tags_set.add("diagram_generic")
                total += self.W_DIAG

        return round(total, 3), sorted(tags_set)

    @staticmethod
    def _extract_alpha_tokens(s: str) -> List[str]:
        """
        Tokenize into simple alphanum words (lowercased). We keep numbers so '1-line' works.
        """
        return [t for t in re.findall(r"[A-Za-z0-9]+", s.lower()) if t]

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Tiny Levenshtein for short tokens."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0: return lb
        if lb == 0: return la
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, lb + 1):
                cur = dp[j]
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[j] = min(dp[j] + 1,      # deletion
                            dp[j-1] + 1,    # insertion
                            prev + cost)    # substitution
                prev = cur
        return dp[lb]

    def _has_token_like(self, tokens: List[str], target: str, max_dist: int) -> bool:
        """Return True if any token in *tokens* is within *max_dist* Levenshtein edits of *target*."""
        for t in tokens:
            # quick prune by length
            if abs(len(t) - len(target)) > max_dist:
                continue
            if self._levenshtein(t, target) <= max_dist:
                return True
        return False

    @staticmethod
    def _normalize_for_regex(s: str) -> str:
        """Normalize whitespace and dashes in *s* for regex matching."""
        s = s.replace("\n", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.replace("—", "-").replace("–", "-")
        return s.strip()

    @staticmethod
    def _approx_match_conf(spans: List[Dict[str, Any]], needle: str) -> float:
        """
        Heuristic: average confidence of spans whose tokens appear in 'needle'.
        """
        if not spans:
            return 0.9
        needle_l = needle.lower()
        confs: List[float] = []
        for s in spans:
            t = (s.get("text") or "").strip()
            if not t:
                continue
            tokens = [tok for tok in re.split(r"\s+", t.lower()) if tok]
            if any(tok and tok in needle_l for tok in tokens):
                c = float(s.get("conf", 0.9))
                confs.append(c if 0 <= c <= 1 else max(0.0, min(1.0, c/100.0)))
        return (sum(confs) / len(confs)) if confs else 0.9

    # ----------------- Normalization for corner tokens -----------------
    @staticmethod
    def _normalize_token(tok: str) -> str:
        """Normalize an OCR token: uppercase, strip punctuation, fix common digit confusions (O->0, I->1, S->5, B->8)."""
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
        """Return True if *tok* matches the E-sheet pattern (e.g., E1.01, E-201)."""
        norm = self._normalize_token(tok)
        if not norm:
            return False
        if norm.endswith("-"):
            norm = norm[:-1]
        return bool(self.PATTERN_E_LOOSE.match(norm))

    def _looks_like_non_e_sheet(self, tok: str) -> bool:
        """Return True if *tok* matches a non-E sheet pattern (e.g., P1.01, M-201, A-101)."""
        norm = self._normalize_token(tok)
        if not norm:
            return False
        if norm.endswith("-"):
            norm = norm[:-1]
        return bool(self.PATTERN_NON_E.match(norm))

    # ----------------- Rendering for footprint -----------------
    def _render_page_bgr_pdfium(self, page) -> np.ndarray:
        """Render a pypdfium2 page to BGR numpy array."""
        page_width = page.get_width()
        page_height = page.get_height()
        scale = self.dpi / 72.0  # PDF points → pixels
        if self.longest_cap_px is not None:
            w_px, h_px = page_width * scale, page_height * scale
            longest = max(w_px, h_px)
            if longest > self.longest_cap_px:
                scale *= (self.longest_cap_px / longest)
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil()
        arr = np.array(pil_img)
        # Convert RGB to BGR for OpenCV
        if len(arr.shape) == 3 and arr.shape[2] >= 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    # ----------------- Footprint detector -----------------
    def _has_rectangular_footprints(self, img_bgr: np.ndarray) -> Tuple[bool, int, List[Dict[str, float]]]:
        """Detect rectangular whitespace voids (panel-schedule-like regions) in a page image for undecided-page filtering."""
        Hf, Wf = img_bgr.shape[:2]
        if self.proc_scale < 1.0:
            img = cv2.resize(img_bgr, (int(Wf * self.proc_scale), int(Hf * self.proc_scale)),
                             interpolation=cv2.INTER_AREA)
        else:
            img = img_bgr

        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # ----------------- Finalize -----------------
    def _finalize_output(self, pdfium_doc, pikepdf_doc, out_pdf_obj, kept, dropped, out_pdf, log_path, logs):
        """
        Save filtered pages. If Ghostscript is enabled:
        - write an intermediate "<base>_raw.pdf"
        - GS writes the final "<base>.pdf" (no _gs)
        - keep _raw only when debug=True
        If Ghostscript is disabled:
        - write final directly to "<base>.pdf"
        """
        # Nothing to save?
        if len(out_pdf_obj.pages) == 0:
            if self.verbose:
                print("[WARN] No pages kept — nothing saved")
            try:
                pdfium_doc.close()
                pikepdf_doc.close()
            except Exception:
                pass
            return kept, dropped, None, None

        base_noext = os.path.splitext(os.path.basename(out_pdf))[0]  # e.g. "<input>_electrical_filtered"
        out_dir = os.path.dirname(out_pdf)
        raw_pdf = os.path.join(out_dir, f"{base_noext}_raw.pdf")     # intermediate when GS is on
        final_pdf = out_pdf                                          # desired final name (no _gs)

        # ----- Save intermediate / final depending on GS -----
        if self.use_ghostscript_letter:
            # 1) Write raw (vector-preserving, no resize)
            out_pdf_obj.save(raw_pdf)
            out_pdf_obj.close()
            if self.verbose:
                print(f"[OK] Saved intermediate (raw) → {raw_pdf}  (kept {len(kept)}/{len(pdfium_doc)})")

            # 2) Ghostscript → final (no '_gs' in name)
            try:
                final_pdf = self._gs_fit_to_letter(raw_pdf, final_pdf)
                if self.verbose:
                    orient = self.letter_orientation
                    print(f"[OK] Ghostscript Letter ({orient}) → {final_pdf}")
                # 3) Remove raw unless debugging
                if not self.debug:
                    try:
                        os.remove(raw_pdf)
                    except Exception:
                        pass
            except Exception as e:
                # On GS failure, fall back to raw as the output
                if self.verbose:
                    print(f"[WARN] Ghostscript Letter-fit failed: {e}. Using raw output.")
                final_pdf = raw_pdf
        else:
            # GS disabled → write directly to the final path
            out_pdf_obj.save(final_pdf)
            out_pdf_obj.close()
            if self.verbose:
                print(f"[OK] Saved filtered PDF → {final_pdf}  (kept {len(kept)}/{len(pdfium_doc)})")

        # ----- Debug log -----
        log_json_path = None
        if self.debug:
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(logs, f, indent=2)
                log_json_path = log_path
                if self.verbose:
                    print(f"[DEBUG] Wrote log JSON → {log_json_path}")
            except Exception:
                pass

        # Close source docs
        try:
            pdfium_doc.close()
            pikepdf_doc.close()
        except Exception:
            pass

        return kept, dropped, final_pdf, log_json_path
