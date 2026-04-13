#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# V3 - simplified line-window analyzer

import os
import re
import sys
import json
import time
from PIL import Image, ImageDraw
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

import pypdfium2 as pdfium  # Apache 2.0


# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


# ---------- IO PATHS ----------
INPUT_PDF = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/S14.pdf").expanduser()
SPEC_OUTPUT_ROOT = Path("~/Spec_Sheet_Analysis/Results").expanduser()
SPEC_DATA_ROOT = Path("~/Spec_Sheet_Analysis/data").expanduser()

SPEC_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SPEC_DATA_ROOT.mkdir(parents=True, exist_ok=True)


# ---------- DEFAULT OVERRIDES ----------
def build_default_ui_overrides() -> dict:
    return {
        "panelboards": {
            "bussing_material": "ALUMINUM",
            "allow_plug_on_breakers": True,
            "rating_type": "FULLY_RATED",
            "allow_feed_thru_lugs": True,
            "default_trim_style": "FLUSH",
            "enclosure": "NEMA1",
            "allow_square_d_spd": True,
        },
        "transformers": {
            "winding_material": "ALUMINUM",
            "temperature_rating": 150,
            "default_type": "3PHASESTANDARD",
            "weathershield": False,
            "mounting": "FLOOR",
            "resin_enclosure": "3R",
        },
        "disconnects": {
            "allow_littlefuse": True,
            "default_switch_type": "GENERAL_DUTY",
            "default_enclosure": "NEMA1",
            "default_fusible": True,
            "default_ground_required": True,
            "default_solid_neutral": True,
        },
    }


# ============================================================
# Models
# ============================================================

@dataclass
class EvidenceItem:
    page: Optional[int]
    source: str
    snippet: str
    matched_phrase: Optional[str] = None
    matched_line: Optional[str] = None
    image_path: Optional[str] = None
    full_page_image_path: Optional[str] = None
    page_index: Optional[int] = None
    line_index: Optional[int] = None


@dataclass
class DetectedValue:
    value: Any
    matched_phrase: Optional[str] = None
    evidence: List[EvidenceItem] = field(default_factory=list)


@dataclass
class SectionMatch:
    mode: str
    start_page: Optional[int]
    end_page: Optional[int]
    score: float
    title: Optional[str] = None


@dataclass
class SpecAnalysisResult:
    section_match: Optional[SectionMatch]
    panelboards: Dict[str, DetectedValue]
    job_flags: Dict[str, DetectedValue]
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "section_match": asdict(self.section_match) if self.section_match else None,
            "panelboards": {k: asdict(v) for k, v in self.panelboards.items()},
            "job_flags": {k: asdict(v) for k, v in self.job_flags.items()},
            "debug": self.debug,
        }


# ============================================================
# Config
# ============================================================

PANELBOARD_TITLE_PATTERNS = [
    re.compile(r"\bpanelboards?\b", re.IGNORECASE),
    re.compile(r"\bpanelboards?\s+breaker\s+type\b", re.IGNORECASE),
    re.compile(r"\bdistribution\s+panelboards?\b", re.IGNORECASE),
    re.compile(r"\bpower\s+and\s+lighting\s+panelboards?\b", re.IGNORECASE),
    re.compile(r"\blighting\s+and\s+appliance\s+panelboards?\b", re.IGNORECASE),
    re.compile(r"\blow\s+voltage\s+panelboards?\b", re.IGNORECASE),
]

PANELBOARD_SUPPORT_TERMS = [
    "panelboard",
    "panelboards",
    "breaker",
    "breakers",
    "bolt on",
    "bolt in",
    "plug on",
    "plug in",
    "bus",
    "buses",
    "bussing",
    "main breaker",
    "neutral",
    "ground bar",
    "circuit directory",
]

PANELBOARD_STOP_TERMS = [
    "transformers",
    "transformer",
    "disconnect switches",
    "disconnect switch",
    "wiring devices",
    "surge suppressors",
    "lighting",
    "exit signs",
    "grounding",
    "splitters",
    "junction boxes",
    "conduit",
]

PANELBOARD_REFERENCE_PENALTIES = [
    "related sections",
    "related section",
    "references",
    "table of contents",
    "contents",
]

MANUFACTURER_TOKENS = [
    "square d",
    "sqd",
    "schneider",
    "schneider electric",
    "schneidier elec",
]

COMPETITOR_MANUFACTURER_TOKENS = [
    "eaton",
    "cutler hammer",
    "westinghouse",
    "abb",
    "siemens",
    "general electric",
]


# ============================================================
# General Helpers
# ============================================================

def score_strong_top_panelboard_header(page_dict: dict) -> int:
    top_raw = " ".join([
        page_dict["regions"]["top_left"],
        page_dict["regions"]["top_center"],
        page_dict["regions"]["top_right"],
        page_dict["regions"]["top_band"],
    ])

    top_norm = normalize_for_matching(top_raw)
    score = 0

    # Strong title signal
    if re.search(r"\bpanelboards?\b", top_norm, re.IGNORECASE):
        score += 3
    if re.search(r"\bdistribution panelboards?\b", top_norm, re.IGNORECASE):
        score += 2
    if re.search(r"\bpower and lighting panelboards?\b", top_norm, re.IGNORECASE):
        score += 2
    if re.search(r"\blighting and appliance panelboards?\b", top_norm, re.IGNORECASE):
        score += 2
    if re.search(r"\bpanelboards? breaker type\b", top_norm, re.IGNORECASE):
        score += 2

    # Section numbering helps, but should not be mandatory
    if re.search(r"\b26\s*24\b", top_raw, re.IGNORECASE):
        score += 2
    if re.search(r"\b26\s*24\s*\d{2}\b", top_raw, re.IGNORECASE):
        score += 2

    # Extra trust if it looks like an actual section/title line
    if re.search(r"\bsection\b", top_norm, re.IGNORECASE):
        score += 1
    if re.search(r"[-–—]", top_raw):
        score += 1

    return score

def has_inline_panelboard_section_signal(page_dict: dict) -> bool:
    text = normalize_preserve_lines(page_dict["text"])
    lines = split_nonempty_lines(text)

    for line in lines:
        line_norm = normalize_for_matching(line)

        looks_like_section_line = bool(
            re.match(r"^\s*26\s*\d{2}\s*\d{2}(?:\.\d{2})?\s*[-–—]?\s*", line, re.IGNORECASE)
        )

        if looks_like_section_line and "panelboard" in line_norm:
            return True

    return False

def classify_panelboard_page_layout(page_dict: dict) -> str:
    top_score = score_strong_top_panelboard_header(page_dict)

    if top_score >= 5:
        return "strong_top_header"

    if has_inline_panelboard_section_signal(page_dict):
        return "inline_compact"

    return "weak_candidate"

def backtrack_to_inline_panelboard_start(pages: List[dict], idx: int) -> int:
    """
    Compact-doc rescue:
    if the chosen page looks like continuation content, check the previous page
    for the actual inline section start like '26 24 16 - PANELBOARDS'.

    Only looks back one page to stay safe.
    """
    prev_idx = idx - 1
    if prev_idx < 0:
        return idx

    prev_text = normalize_preserve_lines(pages[prev_idx]["text"])
    prev_lines = split_nonempty_lines(prev_text)

    for line in prev_lines:
        line_norm = normalize_for_matching(line)

        looks_like_section_line = bool(
            re.match(r"^\s*26\s*\d{2}\s*\d{2}(?:\.\d{2})?\s*[-–—]?\s*", line, re.IGNORECASE)
        )

        if looks_like_section_line and "panelboard" in line_norm:
            return prev_idx

    return idx


def get_inline_panelboard_title_from_page(page_dict: dict) -> Optional[str]:
    """
    Pull the actual inline compact section label from anywhere on the page,
    e.g. '26 24 16 - PANELBOARDS'
    """
    lines = split_nonempty_lines(normalize_preserve_lines(page_dict["text"]))

    for line in lines:
        line_norm = normalize_for_matching(line)

        looks_like_section_line = bool(
            re.match(r"^\s*26\s*\d{2}\s*\d{2}(?:\.\d{2})?\s*[-–—]?\s*", line, re.IGNORECASE)
        )

        if looks_like_section_line and "panelboard" in line_norm:
            return line.strip()

    return None

def ms_to_readable(ms: int | float | None) -> str:
    if ms is None:
        return "N/A"

    total_ms = int(ms)
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    seconds = remainder // 1_000
    milliseconds = remainder % 1_000

    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"


def _extract_section_code(text: str) -> Optional[Tuple[int, int, int]]:
    """
    Extract section code like:
      262417
      26 24 17
      SECTION 262400
      SECTION 26 24 17

    Returns:
      (26, 24, 17) style tuple if found
      otherwise None
    """
    raw = text or ""

    patterns = [
        r"\bsection\s*(\d{2})\s*(\d{2})\s*(\d{2})\b",
        r"\bsection\s+(\d{2})\s+(\d{2})\s+(\d{2})\b",
        r"\b(\d{2})\s*(\d{2})\s*(\d{2})\b",
        r"\b(\d{2})\s+(\d{2})\s+(\d{2})\b",
    ]

    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            a, b, c = m.groups()
            return (int(a), int(b), int(c))

    return None

def _score_section_code_proximity(page_dict: dict) -> float:
    """
    Strongly prefer Division 26 / Section group 24.
    Use this as a guide, not a hard gate.
    """
    candidate_regions = [
        page_dict["regions"]["top_left"],
        page_dict["regions"]["top_center"],
        page_dict["regions"]["top_right"],
        page_dict["regions"]["top_band"],
        page_dict["text"],
    ]

    best_score = 0.0

    for region_text in candidate_regions:
        code = _extract_section_code(region_text)
        if not code:
            continue

        div, group, item = code

        score = 0.0

        if div == 26:
            score += 8.0
        else:
            score -= 12.0

        if div == 26 and group == 24:
            score += 20.0
        elif div == 26 and group < 24:
            score -= 10.0
        elif div == 26 and group > 24:
            score -= 8.0

        best_score = max(best_score, score)

    return best_score

def now_ts_ms() -> int:
    return int(time.time() * 1000)


def build_job_id(pdf_path: Path) -> str:
    base = pdf_path.stem.upper()
    safe = re.sub(r"[^A-Z0-9]+", "_", base).strip("_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe}__{stamp}"


def ensure_json_safe(value):
    if isinstance(value, dict):
        return {str(k): ensure_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [ensure_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [ensure_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


# ============================================================
# PDF Text Extraction
# ============================================================

def _extract_region_text(textpage, left=None, bottom=None, right=None, top=None) -> str:
    try:
        return textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top) or ""
    except Exception:
        return ""


def extract_pdf_pages_text(pdf_path: str) -> List[dict]:
    pdf = pdfium.PdfDocument(pdf_path)
    pages = []

    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()

            width = float(page.get_width())
            height = float(page.get_height())

            full_text = textpage.get_text_bounded() or ""

            top_cut = height * 0.78
            bottom_cut = height * 0.22

            left_1 = 0.0
            left_2 = width / 3.0
            left_3 = 2.0 * width / 3.0
            right = width

            top_band = _extract_region_text(textpage, left=0, bottom=top_cut, right=width, top=height)
            middle_band = _extract_region_text(textpage, left=0, bottom=bottom_cut, right=width, top=top_cut)
            bottom_band = _extract_region_text(textpage, left=0, bottom=0, right=width, top=bottom_cut)

            top_left = _extract_region_text(textpage, left=left_1, bottom=top_cut, right=left_2, top=height)
            top_center = _extract_region_text(textpage, left=left_2, bottom=top_cut, right=left_3, top=height)
            top_right = _extract_region_text(textpage, left=left_3, bottom=top_cut, right=right, top=height)

            bottom_left = _extract_region_text(textpage, left=left_1, bottom=0, right=left_2, top=bottom_cut)
            bottom_center = _extract_region_text(textpage, left=left_2, bottom=0, right=left_3, top=bottom_cut)
            bottom_right = _extract_region_text(textpage, left=left_3, bottom=0, right=right, top=bottom_cut)

            pages.append({
                "page_num": i + 1,
                "width": width,
                "height": height,
                "text": full_text,
                "regions": {
                    "top_band": top_band,
                    "middle_band": middle_band,
                    "bottom_band": bottom_band,
                    "top_left": top_left,
                    "top_center": top_center,
                    "top_right": top_right,
                    "bottom_left": bottom_left,
                    "bottom_center": bottom_center,
                    "bottom_right": bottom_right,
                }
            })

            textpage.close()
            page.close()
    finally:
        pdf.close()

    return pages

def _get_span_bbox(textpage, start_index: int, count: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Return a single bounding box for a span of characters on the page.
    Coordinates are in PDF page units: (left, bottom, right, top).
    """
    boxes = []

    total_chars = textpage.count_chars()
    end_index = min(start_index + count, total_chars)

    for i in range(start_index, end_index):
        try:
            l, b, r, t = textpage.get_charbox(i)
            if r > l and t > b:
                boxes.append((l, b, r, t))
        except Exception:
            continue

    if not boxes:
        return None

    left = min(bx[0] for bx in boxes)
    bottom = min(bx[1] for bx in boxes)
    right = max(bx[2] for bx in boxes)
    top = max(bx[3] for bx in boxes)

    return (left, bottom, right, top)

def _bboxes_overlap_y(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    tolerance: float = 2.0,
) -> bool:
    a_left, a_bottom, a_right, a_top = a
    b_left, b_bottom, b_right, b_top = b
    return not (a_top < b_bottom - tolerance or b_top < a_bottom - tolerance)


def _pdf_rect_to_pixel_rect(
    rect: Tuple[float, float, float, float],
    page_height_pdf: float,
    scale: float,
) -> Tuple[int, int, int, int]:
    left, bottom, right, top = rect
    x1 = int(left * scale)
    x2 = int(right * scale)
    y1 = int((page_height_pdf - top) * scale)
    y2 = int((page_height_pdf - bottom) * scale)
    return x1, y1, x2, y2

def generate_evidence_image(
    pdf_path: str,
    page_number: int,
    snippet: str,
    matched_phrase: str,
    matched_line: str,
    output_path: str,
    full_page_output_path: str,
    scale: float = 2.0,
    context_band_ratio: float = 0.15,
) -> Optional[dict]:
    """
    Simple page-correct image generator:
    - searches the matched line on the exact page
    - searches the matched phrase on the exact page
    - prefers phrase hits on the same visual line
    - renders full-page and full-width context-band images
    """
    pdf = None
    page = None
    textpage = None

    try:
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[page_number - 1]
        textpage = page.get_textpage()

        page_height_pdf = float(page.get_height())

        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # 1. Find matched line or whole snippet region on this exact page
        line_bbox = None
        use_full_snippet_region = False

        raw_line = (matched_line or "").strip()
        normalized_line = normalize_preserve_lines(raw_line).strip()

        # If the line looks wrapped/incomplete, prefer the whole snippet block
        if raw_line.endswith(("type", "with", "by", "of", "and", "or", "(", "-", "be", "shall")):
            use_full_snippet_region = True

        if not use_full_snippet_region:
            line_variants = [
                raw_line,
                normalized_line,
                normalized_line.replace("", "-"),
                normalized_line.replace("–", "-"),
                normalized_line.replace("—", "-"),
                normalized_line.replace("", " "),
                normalized_line.replace("–", " "),
                normalized_line.replace("—", " "),
                normalized_line.replace("bolt-on", "bolt on"),
                normalized_line.replace("bolt on", "bolt-on"),
                normalized_line.replace("series-rated", "series rated"),
                normalized_line.replace("series rated", "series-rated"),
                normalized_line.rstrip(".:,;"),
            ]

            seen = set()
            cleaned_variants = []
            for v in line_variants:
                v = re.sub(r"\s+", " ", (v or "").strip())
                if not v:
                    continue
                key = v.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned_variants.append(v)

            for candidate_line in cleaned_variants:
                line_search = None
                try:
                    line_search = textpage.search(candidate_line, match_case=False, match_whole_word=False)
                    hit = line_search.get_next()
                    if hit:
                        start_index, count = hit
                        bbox = _get_span_bbox(textpage, start_index, count)
                        if bbox:
                            line_bbox = bbox
                            break
                finally:
                    try:
                        if line_search is not None:
                            line_search.close()
                    except Exception:
                        pass

        if line_bbox is None:
            # Fallback: build one bbox from only the most relevant snippet lines
            snippet_lines = [ln.strip() for ln in normalize_preserve_lines(snippet).splitlines() if ln.strip()]
            phrase_norm = normalize_for_matching(matched_phrase)

            relevant_indices = []
            for i, snippet_line in enumerate(snippet_lines):
                line_norm = normalize_for_matching(snippet_line)
                if phrase_norm and phrase_norm in line_norm:
                    relevant_indices.append(i)

            # If no direct phrase line found, fall back to lines with governing language
            if not relevant_indices:
                for i, snippet_line in enumerate(snippet_lines):
                    line_norm = normalize_for_matching(snippet_line)
                    if any(term in line_norm for term in [
                        "shall",
                        "not acceptable",
                        "not allowed",
                        "not permitted",
                        "approved equal",
                        "manufactured by",
                        "type",
                        "only",
                    ]):
                        relevant_indices.append(i)

            # For manufacturer hits like Square D, only use the exact matching line.
            # For other categories, include one line above/below for context.
            selected_indices = set()

            phrase_norm = normalize_for_matching(matched_phrase)
            manufacturer_like = phrase_norm in {
                "square d",
                "sqd",
                "schneider",
                "schneider electric",
                "schneidier elec",
            }

            for idx in relevant_indices:
                if manufacturer_like:
                    selected_indices.add(idx)
                else:
                    for j in range(max(0, idx - 1), min(len(snippet_lines), idx + 2)):
                        selected_indices.add(j)

            selected_lines = [snippet_lines[i] for i in sorted(selected_indices)]
            snippet_boxes = []

            for snippet_line in selected_lines:
                snippet_variants = [
                    snippet_line,
                    normalize_preserve_lines(snippet_line).strip(),
                    normalize_preserve_lines(snippet_line).replace("bolt-on", "bolt on").strip(),
                    normalize_preserve_lines(snippet_line).replace("bolt on", "bolt-on").strip(),
                    normalize_preserve_lines(snippet_line).replace("series-rated", "series rated").strip(),
                    normalize_preserve_lines(snippet_line).replace("series rated", "series-rated").strip(),
                    normalize_preserve_lines(snippet_line).rstrip(".:,;").strip(),
                ]

                seen = set()
                cleaned_snippet_variants = []
                for v in snippet_variants:
                    v = re.sub(r"\s+", " ", (v or "").strip())
                    if not v:
                        continue
                    k = v.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    cleaned_snippet_variants.append(v)

                found_box = None
                for candidate_variant in cleaned_snippet_variants:
                    line_search = None
                    try:
                        line_search = textpage.search(candidate_variant, match_case=False, match_whole_word=False)
                        hit = line_search.get_next()
                        if hit:
                            start_index, count = hit
                            bbox = _get_span_bbox(textpage, start_index, count)
                            if bbox:
                                found_box = bbox
                                break
                    finally:
                        try:
                            if line_search is not None:
                                line_search.close()
                        except Exception:
                            pass

                if found_box is not None:
                    snippet_boxes.append(found_box)

            if snippet_boxes:
                left = min(b[0] for b in snippet_boxes)
                bottom = min(b[1] for b in snippet_boxes)
                right = max(b[2] for b in snippet_boxes)
                top = max(b[3] for b in snippet_boxes)
                line_bbox = (left, bottom, right, top)

        # Detect manufacturer-like phrases where phrase-only fallback is acceptable
        base_phrase = normalize_preserve_lines(matched_phrase).strip()
        phrase_norm = normalize_for_matching(base_phrase)
        manufacturer_like = phrase_norm in {
            "square d",
            "sqd",
            "schneider",
            "schneider electric",
            "schneidier elec",
        }

        # If we still do not have a line bbox and this is manufacturer evidence,
        # fall back to using the phrase itself as the anchor region.
        if line_bbox is None and manufacturer_like:
            phrase_only_variants = [
                matched_phrase,
                base_phrase,
                base_phrase.replace("", "-"),
                base_phrase.replace("–", "-"),
                base_phrase.replace("—", "-"),
            ]

            seen = set()
            cleaned_phrase_only_variants = []
            for v in phrase_only_variants:
                v = re.sub(r"\s+", " ", (v or "").strip())
                if not v:
                    continue
                k = v.lower()
                if k in seen:
                    continue
                seen.add(k)
                cleaned_phrase_only_variants.append(v)

            for candidate_phrase in cleaned_phrase_only_variants:
                phrase_search = None
                try:
                    phrase_search = textpage.search(candidate_phrase, match_case=False, match_whole_word=False)
                    hit = phrase_search.get_next()
                    if hit:
                        start_index, count = hit
                        bbox = _get_span_bbox(textpage, start_index, count)
                        if bbox:
                            line_bbox = bbox
                            break
                finally:
                    try:
                        if phrase_search is not None:
                            phrase_search.close()
                    except Exception:
                        pass

        if line_bbox is None:
            return None

        # 2. Find matched phrase on the same visual line
        phrase_bbox = None

        phrase_variants = [
            matched_phrase,
            base_phrase,
            base_phrase.replace("", "-"),
            base_phrase.replace("–", "-"),
            base_phrase.replace("—", "-"),
            base_phrase.replace("bolt on", "bolt-on"),
            base_phrase.replace("bolt-on", "bolt on"),
            base_phrase.replace("series rated", "series-rated"),
            base_phrase.replace("series-rated", "series rated"),
            base_phrase.replace("fully rated", "fully-rated"),
            base_phrase.replace("fully-rated", "fully rated"),
        ]

        seen = set()
        cleaned_phrase_variants = []
        for v in phrase_variants:
            v = re.sub(r"\s+", " ", (v or "").strip())
            if not v:
                continue
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_phrase_variants.append(v)

        for candidate_phrase in cleaned_phrase_variants:
            phrase_search = None
            try:
                phrase_search = textpage.search(candidate_phrase, match_case=False, match_whole_word=False)

                while True:
                    hit = phrase_search.get_next()
                    if not hit:
                        break

                    start_index, count = hit
                    bbox = _get_span_bbox(textpage, start_index, count)
                    if not bbox:
                        continue

                    if _bboxes_overlap_y(bbox, line_bbox):
                        phrase_bbox = bbox
                        break

                if phrase_bbox is not None:
                    break

            finally:
                try:
                    if phrase_search is not None:
                        phrase_search.close()
                except Exception:
                    pass

        if phrase_bbox is None:
            phrase_bbox = line_bbox

        # Decide whether to highlight the phrase only or the whole deciding line
        highlight_bbox = phrase_bbox
        line_norm = normalize_for_matching(matched_line)

        qualifier_terms = [
            "not acceptable",
            "not allowed",
            "not permitted",
            "are prohibited",
            "is prohibited",
            "prohibited",
            "shall be",
            "must be",
            "is required",
            "are required",
            "only",
            "type",
        ]

        if any(term in line_norm for term in qualifier_terms):
            highlight_bbox = line_bbox

        # 3. Full page image with highlight
        full_page_img = pil_image.copy()
        full_draw = ImageDraw.Draw(full_page_img, "RGBA")

        fx1, fy1, fx2, fy2 = _pdf_rect_to_pixel_rect(highlight_bbox, page_height_pdf, scale)

        full_draw.rectangle(
            [(fx1 - 4, fy1 - 2), (fx2 + 4, fy2 + 2)],
            fill=(255, 235, 59, 70),
            outline=(255, 193, 7, 140),
            width=1,
        )

        full_page_img.save(full_page_output_path)

        # 4. Full-width context band around matched line
        _, ly1, _, ly2 = _pdf_rect_to_pixel_rect(line_bbox, page_height_pdf, scale)
        line_center_y = (ly1 + ly2) // 2

        band_height_px = max(220, int(full_page_img.size[1] * context_band_ratio))
        band_top = max(0, line_center_y - band_height_px // 2)
        band_bottom = min(full_page_img.size[1], band_top + band_height_px)

        if band_bottom - band_top < band_height_px:
            band_top = max(0, band_bottom - band_height_px)

        cropped = full_page_img.crop((0, band_top, full_page_img.size[0], band_bottom))
        crop_draw = ImageDraw.Draw(cropped, "RGBA")

        cx1 = fx1
        cx2 = fx2
        cy1 = fy1 - band_top
        cy2 = fy2 - band_top

        crop_draw.rectangle(
            [(max(0, cx1 - 4), max(0, cy1 - 2)), (min(cropped.size[0], cx2 + 4), min(cropped.size[1], cy2 + 2))],
            fill=(255, 235, 59, 70),
            outline=(255, 193, 7, 140),
            width=1,
        )

        cropped.save(output_path)

        return {
            "zoom_image_path": output_path,
            "full_page_image_path": full_page_output_path,
        }

    except Exception:
        return None

    finally:
        try:
            if textpage is not None:
                textpage.close()
        except Exception:
            pass
        try:
            if page is not None:
                page.close()
        except Exception:
            pass
        try:
            if pdf is not None:
                pdf.close()
        except Exception:
            pass

# ============================================================
# Text Helpers
# ============================================================

def normalize_preserve_lines(text: str) -> str:
    text = (text or "")
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2012", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2212", "-")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def normalize_for_matching(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2012", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2212", "-")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Section Finding
# ============================================================

def _top_region_text(page_dict: dict) -> str:
    return normalize_for_matching(
        " ".join([
            page_dict["regions"]["top_left"],
            page_dict["regions"]["top_center"],
            page_dict["regions"]["top_right"],
            page_dict["regions"]["top_band"],
        ])
    )


def _full_region_text(page_dict: dict) -> str:
    return normalize_for_matching(page_dict["text"] or "")


def _has_panelboard_title(text: str) -> bool:
    return any(p.search(text) for p in PANELBOARD_TITLE_PATTERNS)


def _panelboard_support_score(text: str) -> int:
    score = 0
    for term in PANELBOARD_SUPPORT_TERMS:
        if term in text:
            score += 1
    return score


def _panelboard_stop_score(text: str) -> int:
    score = 0
    for term in PANELBOARD_STOP_TERMS:
        if term in text:
            score += 1
    return score


def _panelboard_reference_penalty(text: str) -> int:
    score = 0
    for term in PANELBOARD_REFERENCE_PENALTIES:
        if term in text:
            score += 1
    return score


def _looks_like_panelboard_section_start(page_dict: dict) -> bool:
    top_text = _top_region_text(page_dict)
    full_text = _full_region_text(page_dict)

    # Primary signal = panelboard naming
    if _has_panelboard_title(top_text):
        return True

    # Secondary signal = "section ..." plus panelboard title somewhere on page
    has_section_word = "section" in top_text or "section" in full_text
    has_panelboard = _has_panelboard_title(full_text)

    if has_section_word and has_panelboard:
        return True

    return False


def score_page_for_panelboard_relevance(page_dict: dict) -> float:
    score = 0.0

    top_text = _top_region_text(page_dict)
    full_text = _full_region_text(page_dict)

    # ---- naming / title first ----
    if _has_panelboard_title(top_text):
        score += 30.0
    elif _has_panelboard_title(full_text):
        score += 14.0

    if "panelboards breaker type" in top_text:
        score += 16.0
    if "distribution panelboards" in top_text:
        score += 16.0
    if "power and lighting panelboards" in top_text:
        score += 16.0
    if "lighting and appliance panelboards" in top_text:
        score += 14.0
    if "low voltage panelboards" in top_text:
        score += 14.0

    # ---- numbers only help, never decide ----
    if re.search(r"\bsection\s+26\s+24\s+\d{2}(?:\s+\d+)?\b", top_text):
        score += 8.0
    elif re.search(r"\b26\s+24\s+\d{2}(?:\s+\d+)?\b", top_text):
        score += 4.0

    # ---- content support ----
    score += float(_panelboard_support_score(full_text))

    # ---- penalties ----
    score -= float(_panelboard_reference_penalty(full_text)) * 8.0
    score -= float(_panelboard_stop_score(top_text)) * 10.0
    score -= float(_panelboard_stop_score(full_text)) * 2.0
    score += _score_section_code_proximity(page_dict)

    return score


def score_section_header_candidate(page_dict: dict) -> int:
    score = 0

    top_text = _top_region_text(page_dict)
    full_text = _full_region_text(page_dict)

    if _has_panelboard_title(top_text):
        score += 40
    elif _has_panelboard_title(full_text):
        score += 20

    if "panelboards breaker type" in top_text:
        score += 20
    if "distribution panelboards" in top_text:
        score += 20
    if "power and lighting panelboards" in top_text:
        score += 20
    if "lighting and appliance panelboards" in top_text:
        score += 18
    if "low voltage panelboards" in top_text:
        score += 18

    if re.search(r"\bsection\s+26\s+24\s+\d{2}(?:\s+\d+)?\b", top_text):
        score += 10

    score += _panelboard_support_score(full_text) * 2
    score -= _panelboard_reference_penalty(full_text) * 10
    score -= _panelboard_stop_score(top_text) * 12
    score += int(_score_section_code_proximity(page_dict))

    return score


def get_section_header_line(page_dict: dict) -> Optional[str]:
    candidate_regions = [
        page_dict["regions"]["top_center"],
        page_dict["regions"]["top_left"],
        page_dict["regions"]["top_right"],
        page_dict["regions"]["top_band"],
        page_dict["text"],
    ]

    header_patterns = [
        r"section\s+26\s+24\s+\d{2}(?:\s+\d+)?\s+panelboards?\s+breaker\s+type",
        r"section\s+26\s+24\s+\d{2}(?:\s+\d+)?\s+distribution\s+panelboards?",
        r"section\s+26\s+24\s+\d{2}(?:\s+\d+)?\s+power\s+and\s+lighting\s+panelboards?",
        r"section\s+26\s+24\s+\d{2}(?:\s+\d+)?\s+lighting\s+and\s+appliance\s+panelboards?",
        r"section\s+26\s+24\s+\d{2}(?:\s+\d+)?\s+low\s+voltage\s+panelboards?",
        r"section\s+26\s+24\s+\d{2}(?:\s+\d+)?\s+panelboards?",
        r"panelboards?\s+breaker\s+type",
        r"distribution\s+panelboards?",
        r"power\s+and\s+lighting\s+panelboards?",
        r"lighting\s+and\s+appliance\s+panelboards?",
        r"low\s+voltage\s+panelboards?",
        r"panelboards?",
    ]

    for region_text in candidate_regions:
        norm = normalize_for_matching(region_text)
        for pat in header_patterns:
            m = re.search(pat, norm, re.IGNORECASE)
            if m:
                return m.group(0)

    return None


def find_panelboard_section_candidates(pages: List[dict]) -> List[dict]:
    candidates = []

    for idx, page in enumerate(pages):
        if not _looks_like_panelboard_section_start(page):
            continue

        score = score_section_header_candidate(page)
        candidates.append({
            "idx": idx,
            "page_num": page["page_num"],
            "section_id": None,
            "score": score,
            "title": get_section_header_line(page),
        })

    return candidates


def choose_best_panelboard_candidate(candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None

    # Sort by page order first
    candidates = sorted(candidates, key=lambda c: c["idx"])

    # Group nearby candidates into local runs (same section area)
    runs = []
    current_run = [candidates[0]]

    for cand in candidates[1:]:
        prev = current_run[-1]

        # pages within 2 pages of each other are treated as one section run
        if cand["idx"] - prev["idx"] <= 2:
            current_run.append(cand)
        else:
            runs.append(current_run)
            current_run = [cand]

    runs.append(current_run)

    # Score each run by its strongest page, but return the earliest page in that run
    best_run = max(
        runs,
        key=lambda run: max(c["score"] for c in run)
    )

    return best_run[0]


def find_best_start_page(pages: List[dict]) -> Tuple[Optional[int], float, str]:
    best_idx = None
    best_score = -999.0
    best_mode = "unknown"

    for idx, page in enumerate(pages):
        score = score_page_for_panelboard_relevance(page)

        top_text = _top_region_text(page)
        full_text = _full_region_text(page)

        has_header_like_title = _has_panelboard_title(top_text)
        has_panel_title_anywhere = _has_panelboard_title(full_text)

        if has_header_like_title:
            mode = "dedicated_panelboard_section"
            score += 6.0
        elif has_panel_title_anywhere:
            mode = "panelboard_like_section"
        else:
            mode = "unknown"

        if score > best_score:
            best_idx = idx
            best_score = score
            best_mode = mode

    if best_score < 6.0:
        return None, best_score, "unknown"

    return best_idx, best_score, best_mode


def refine_to_section_start(pages: List[dict], best_idx: int) -> int:
    lookback = max(0, best_idx - 5)
    best_start = best_idx

    best_page_code = _extract_section_code(
        " ".join([
            pages[best_idx]["regions"]["top_left"],
            pages[best_idx]["regions"]["top_center"],
            pages[best_idx]["regions"]["top_right"],
            pages[best_idx]["regions"]["top_band"],
            pages[best_idx]["text"],
        ])
    )

    if best_page_code and best_page_code[0] == 26 and best_page_code[1] == 24:
        for idx in range(lookback, best_idx + 1):
            page_code = _extract_section_code(
                " ".join([
                    pages[idx]["regions"]["top_left"],
                    pages[idx]["regions"]["top_center"],
                    pages[idx]["regions"]["top_right"],
                    pages[idx]["regions"]["top_band"],
                    pages[idx]["text"],
                ])
            )

            if page_code and page_code[0] == 26 and page_code[1] == 24:
                best_start = idx
                break

        return best_start

    candidates = []
    for idx in range(lookback, best_idx + 1):
        if _looks_like_panelboard_section_start(pages[idx]):
            score = score_section_header_candidate(pages[idx])
            candidates.append({
                "idx": idx,
                "score": score,
            })

    if not candidates:
        return best_idx

    candidates.sort(key=lambda x: (x["idx"], -x["score"]))
    return candidates[0]["idx"]


def collect_multi_page_section(
    pages: List[dict],
    start_idx: int,
    mode: str,
    target_section_id: Optional[str] = None
) -> Tuple[str, int]:
    collected = []
    end_idx = start_idx

    for idx in range(start_idx, len(pages)):
        page_text = normalize_preserve_lines(pages[idx]["text"])
        top_text = _top_region_text(pages[idx])
        full_text = _full_region_text(pages[idx])

        if idx > start_idx:
            # Compact spec mode:
            # stop when the next compact 26 xx xx section starts and it is not panelboards.
            if mode == "inline_compact_panelboard_section":
                page_lines = split_nonempty_lines(page_text)

                for line in page_lines[:20]:
                    line_norm = normalize_for_matching(line)

                    looks_like_section_line = bool(
                        re.match(r"^\s*26\s*\d{2}\s*\d{2}(?:\.\d{2})?\s*[-–—]?\s*", line, re.IGNORECASE)
                    )

                    if looks_like_section_line:
                        if "panelboard" not in line_norm and "switchboards and panelboards" not in line_norm:
                            return "\n".join(collected), end_idx

            # Stop when a new non-panelboard section clearly begins
            if "section" in top_text and not _has_panelboard_title(top_text):
                if _panelboard_stop_score(top_text) > 0:
                    break

            # Stop if the page has drifted away strongly
            support_score = _panelboard_support_score(full_text)
            stop_score = _panelboard_stop_score(full_text)

            if stop_score >= 2 and stop_score > support_score:
                break

        collected.append(page_text)
        end_idx = idx

        if (idx - start_idx) >= 16:
            break

    return "\n".join(collected), end_idx


# ============================================================
# Line Window Helpers
# ============================================================

def split_nonempty_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in normalize_preserve_lines(text).splitlines()]
    return [ln for ln in lines if ln]


def get_line_window(lines: List[str], idx: int) -> str:
    start = max(0, idx - 1)
    end = min(len(lines), idx + 2)
    return "\n".join(lines[start:end]).strip()


def get_expanded_line_window(lines: List[str], idx: int) -> str:
    start = max(0, idx - 2)
    end = min(len(lines), idx + 3)
    return "\n".join(lines[start:end]).strip()

def resolve_requirement_strength(line_text: str, small_window: str, big_window: str, target_phrases: List[str]) -> Optional[str]:
    """
    Prefer the current line first.
    Only fall back to larger windows if line-level text is inconclusive.
    """
    line_pol = classify_requirement_strength(line_text, target_phrases)
    if line_pol in ("required", "allowed", "rejected", "mentioned"):
        return line_pol

    small_pol = classify_requirement_strength(small_window, target_phrases)
    if small_pol in ("required", "allowed", "rejected", "mentioned"):
        return small_pol

    big_pol = classify_requirement_strength(big_window, target_phrases)
    if big_pol in ("required", "allowed", "rejected", "mentioned"):
        return big_pol

    return None

def window_has_any_phrase(window_text: str, phrases: List[str]) -> bool:
    norm = normalize_for_matching(window_text)
    return any(normalize_for_matching(p) in norm for p in phrases)


def classify_requirement_strength(window_text: str, target_phrases: List[str]) -> Optional[str]:
    """
    Return one of:
    - 'required'
    - 'allowed'
    - 'rejected'
    - 'mentioned'
    - None

    Only evaluates polarity if one of the target phrases is actually present.
    """
    norm = normalize_for_matching(window_text)

    if not window_has_any_phrase(window_text, target_phrases):
        return None

    reject_patterns = [
        "not acceptable",
        "not allowed",
        "not permitted",
        "are prohibited",
        "is prohibited",
        "prohibited",
        "shall not",
        "shall not be used",
        "unacceptable",
    ]

    require_patterns = [
        "shall be",
        "must be",
        "is required",
        "are required",
        "provide",
        "only",
        "type",
    ]

    allow_patterns = [
        "allowed",
        "permitted",
        "acceptable",
        "may be used",
        "may be provided",
        "can be used",
    ]

    if any(p in norm for p in reject_patterns):
        return "rejected"

    if any(p in norm for p in require_patterns):
        return "required"

    if any(p in norm for p in allow_patterns):
        return "allowed"

    return "mentioned"

def find_line_hits(lines: List[str], patterns: List[re.Pattern]) -> List[dict]:
    hits = []

    for idx, line in enumerate(lines):
        for pat in patterns:
            m = pat.search(line)
            if m:
                hits.append({
                    "line_index": idx,
                    "line": line.strip(),
                    "window": get_line_window(lines, idx),
                    "matched_text": m.group(0),
                    "match_start": m.start(),
                    "match_end": m.end(),
                })
                break

    return hits

def _make_evidence(
    page: Optional[int],
    source: str,
    snippet: str,
    matched_phrase: Optional[str] = None,
    matched_line: Optional[str] = None,
    page_index: Optional[int] = None,
    line_index: Optional[int] = None,
    image_path: Optional[str] = None,
    full_page_image_path: Optional[str] = None,
) -> EvidenceItem:
    return EvidenceItem(
        page=page,
        source=source,
        snippet=snippet.strip(),
        matched_phrase=matched_phrase,
        matched_line=(matched_line or "").strip() or None,
        image_path=image_path,
        full_page_image_path=full_page_image_path,
        page_index=page_index,
        line_index=line_index,
    )

# ============================================================
# Concept Scanners
# ============================================================

def scan_square_d_allowed(lines: List[str], page: Optional[int], source: str) -> DetectedValue:
    sqd_patterns = [
        re.compile(r"\bsquare[\s-]?d\b", re.IGNORECASE),
        re.compile(r"\bsqd\b", re.IGNORECASE),
        re.compile(r"\bschneider\b", re.IGNORECASE),
        re.compile(r"\bschneider\s+electric\b", re.IGNORECASE),
        re.compile(r"\bschneidier\s+elec\b", re.IGNORECASE),
    ]

    competitor_patterns = [
        re.compile(r"\beaton\b", re.IGNORECASE),
        re.compile(r"\bcutler[\s-]?hammer\b", re.IGNORECASE),
        re.compile(r"\babb\b", re.IGNORECASE),
        re.compile(r"\bsiemens\b", re.IGNORECASE),
        re.compile(r"\bgeneral\s+electric\b", re.IGNORECASE),
        re.compile(r"\bwestinghouse\b", re.IGNORECASE),
    ]

    sqd_hits = []
    competitor_only_hits = []

    for idx, line in enumerate(lines):
        line_text = line.strip()
        window = get_line_window(lines, idx)

        sqd_match = None
        for pat in sqd_patterns:
            m = pat.search(line_text)
            if m:
                sqd_match = m
                break

        competitor_match = None
        for pat in competitor_patterns:
            m = pat.search(line_text)
            if m:
                competitor_match = m
                break

        # If Square D appears anywhere on the same line, that line counts as allowed
        if sqd_match:
            sqd_hits.append({
                "line_index": idx,
                "line": line_text,
                "window": window,
                "matched_text": sqd_match.group(0),
            })
            continue

        # Only count competitor evidence if Square D is NOT on the same line
        if competitor_match:
            competitor_only_hits.append({
                "line_index": idx,
                "line": line_text,
                "window": window,
                "matched_text": competitor_match.group(0),
            })

    if sqd_hits:
        return DetectedValue(
            value=True,
            matched_phrase="square d found in narrowed section",
            evidence=[
                _make_evidence(
                    page,
                    source,
                    h["window"],
                    matched_phrase=h["matched_text"],
                    matched_line=h["line"],
                    line_index=h["line_index"],
                )
                for h in sqd_hits[:5]
            ]
        )

    if competitor_only_hits:
        return DetectedValue(
            value=False,
            matched_phrase="competitor found without square d",
            evidence=[
                _make_evidence(
                    page,
                    source,
                    h["window"],
                    matched_phrase=h["matched_text"],
                    matched_line=h["line"],
                    line_index=h["line_index"],
                )
                for h in competitor_only_hits[:5]
            ]
        )

    return DetectedValue(
        value=True,
        matched_phrase="no manufacturers mentioned; defaulting to allowed",
        evidence=[]
    )

def _get_nearest_preceding_equipment_context(lines: List[str], idx: int, lookback: int = 8) -> Optional[str]:
    """
    Look backward from the matched line and return the nearest relevant equipment context:
    - 'panelboards'
    - 'switchboards'
    - None
    """
    start = max(0, idx - lookback)

    for j in range(idx, start - 1, -1):
        norm = normalize_for_matching(lines[j])

        if "panelboards" in norm or "panelboard" in norm:
            return "panelboards"

        if "switchboards" in norm or "switchboard" in norm:
            return "switchboards"

    return None

def scan_bussing_material(lines: List[str], page: Optional[int], source: str) -> DetectedValue:
    copper_patterns = [
        re.compile(r"\bcopper\b", re.IGNORECASE),
    ]

    copper_phrases = [
        "copper",
    ]

    bus_relevance_terms = [
        "bus",
        "buses",
        "buss",
        "bussing",
        "bus bars",
        "busbar",
        "panelboard buses",
        "main bus",
        "phase bus",
    ]
    bus_relevance_terms_norm = [normalize_for_matching(t) for t in bus_relevance_terms]

    copper_required_hits = []

    for idx, line in enumerate(lines):
        small_window = get_line_window(lines, idx)
        big_window = get_expanded_line_window(lines, idx)

        line_norm = normalize_for_matching(line)
        small_norm = normalize_for_matching(small_window)
        big_norm = normalize_for_matching(big_window)

        copper_match = None
        for p in copper_patterns:
            m = p.search(line)
            if m:
                copper_match = m
                break

        if not copper_match:
            continue

        line_has_bus_context = any(term in line_norm for term in bus_relevance_terms_norm)
        small_has_bus_context = any(term in small_norm for term in bus_relevance_terms_norm)
        big_has_bus_context = any(term in big_norm for term in bus_relevance_terms_norm)

        if not (line_has_bus_context or small_has_bus_context or big_has_bus_context):
            continue

        # Prefer the nearest PRECEDING equipment context before the copper line.
        # This prevents a switchboard copper line from being accepted just because
        # a later PANELBOARDS header appears below it in the same snippet/window.
        preceding_context = _get_nearest_preceding_equipment_context(lines, idx, lookback=8)

        if preceding_context == "switchboards":
            continue

        pol = resolve_requirement_strength(line, small_window, big_window, copper_phrases)

        if pol in ("required", "allowed", "mentioned"):
            copper_required_hits.append({
                "line_index": idx,
                "line": line.strip(),
                "window": big_window,
                "matched_text": copper_match.group(0),
            })

    if copper_required_hits:
        return DetectedValue(
            value="COPPER",
            matched_phrase="copper explicitly required",
            evidence=[
                _make_evidence(
                    page,
                    source,
                    h["window"],
                    matched_phrase="copper",
                    matched_line=h["line"],
                    line_index=h["line_index"],
                )
                for h in copper_required_hits[:5]
            ]
        )

    return DetectedValue(value=None, matched_phrase=None, evidence=[])


def scan_breaker_mounting(lines: List[str], page: Optional[int], source: str) -> DetectedValue:
    bolt_patterns = [
        re.compile(r"\bbolt[\s-]?on\b", re.IGNORECASE),
        re.compile(r"\bbolt[\s-]?in\b", re.IGNORECASE),
    ]

    plug_patterns = [
        re.compile(r"\bplug[\s-]?on\b", re.IGNORECASE),
        re.compile(r"\bplug[\s-]?in\b", re.IGNORECASE),
    ]

    breaker_terms = [
        "breaker",
        "breakers",
        "circuit breaker",
        "circuit breakers",
        "molded case",
        "molded-case",
    ]
    breaker_terms_norm = [normalize_for_matching(t) for t in breaker_terms]

    bolt_phrases = [
        "bolt on",
        "bolt in",
    ]

    plug_phrases = [
        "plug on",
        "plug in",
    ]

    bolt_required_hits = []
    plug_required_hits = []

    for idx, line in enumerate(lines):
        small_window = get_line_window(lines, idx)
        big_window = get_expanded_line_window(lines, idx)
        big_window_norm = normalize_for_matching(big_window)

        if not any(term in big_window_norm for term in breaker_terms_norm):
            continue

        bolt_match = None
        for p in bolt_patterns:
            m = p.search(line)
            if m:
                bolt_match = m
                break

        if bolt_match:
            pol = resolve_requirement_strength(line, small_window, big_window, bolt_phrases)
            if pol in ("required", "allowed", "mentioned"):
                bolt_required_hits.append({
                    "line_index": idx,
                    "line": line.strip(),
                    "window": big_window,
                    "matched_text": bolt_match.group(0),
                })

        plug_match = None
        for p in plug_patterns:
            m = p.search(line)
            if m:
                plug_match = m
                break

        if plug_match:
            pol = resolve_requirement_strength(line, small_window, big_window, plug_phrases)
            if pol in ("required", "allowed", "mentioned"):
                plug_required_hits.append({
                    "line_index": idx,
                    "line": line.strip(),
                    "window": big_window,
                    "matched_text": plug_match.group(0),
                })

    if bolt_required_hits:
        return DetectedValue(
            value=False,
            matched_phrase="bolt-on explicitly required",
            evidence=[
                _make_evidence(
                    page,
                    source,
                    h["window"],
                    matched_phrase="bolt on",
                    matched_line=h["line"],
                    line_index=h["line_index"],
                )
                for h in bolt_required_hits[:5]
            ]
        )

    if plug_required_hits:
        return DetectedValue(
            value=True,
            matched_phrase="plug-on explicitly required",
            evidence=[
                _make_evidence(
                    page,
                    source,
                    h["window"],
                    matched_phrase="plug on",
                    matched_line=h["line"],
                    line_index=h["line_index"],
                )
                for h in plug_required_hits[:5]
            ]
        )

    return DetectedValue(value=None, matched_phrase=None, evidence=[])

def scan_rating_type(lines: List[str], page: Optional[int], source: str) -> DetectedValue:
    fully_patterns = [
        re.compile(r"\bfully[\s-]?rated\b", re.IGNORECASE),
    ]

    series_patterns = [
        re.compile(r"\bseries[\s-]?rated\b", re.IGNORECASE),
        re.compile(r"\bseries[\s-]?rating\b", re.IGNORECASE),
    ]

    fully_phrases = [
        "fully rated",
    ]

    series_phrases = [
        "series rated",
        "series rating",
    ]

    rating_relevance_terms = [
        "panelboard",
        "panelboards",
        "device",
        "devices",
        "breaker",
        "breakers",
        "interrupting rating",
        "short circuit rating",
        "short-circuit rating",
        "rating",
        "ratings",
        "aic",
        "sccr",
    ]
    rating_relevance_terms_norm = [normalize_for_matching(t) for t in rating_relevance_terms]

    fully_required_hits = []
    series_required_hits = []
    series_rejected_hits = []

    for idx, line in enumerate(lines):
        small_window = get_line_window(lines, idx)
        big_window = get_expanded_line_window(lines, idx)

        line_norm = normalize_for_matching(line)
        small_norm = normalize_for_matching(small_window)
        big_norm = normalize_for_matching(big_window)

        line_has_rating_context = any(term in line_norm for term in rating_relevance_terms_norm)
        small_has_rating_context = any(term in small_norm for term in rating_relevance_terms_norm)
        big_has_rating_context = any(term in big_norm for term in rating_relevance_terms_norm)

        fully_match = None
        for p in fully_patterns:
            m = p.search(line)
            if m:
                fully_match = m
                break

        if fully_match:
            if not (line_has_rating_context or small_has_rating_context or big_has_rating_context):
                continue

            pol = resolve_requirement_strength(line, small_window, big_window, fully_phrases)

            if pol in ("required", "allowed", "mentioned"):
                fully_required_hits.append({
                    "line_index": idx,
                    "line": line.strip(),
                    "window": big_window,
                    "matched_text": fully_match.group(0),
                })

        series_match = None
        for p in series_patterns:
            m = p.search(line)
            if m:
                series_match = m
                break

        if series_match:
            if not (line_has_rating_context or small_has_rating_context or big_has_rating_context):
                continue

            pol = resolve_requirement_strength(line, small_window, big_window, series_phrases)

            if pol == "rejected":
                series_rejected_hits.append({
                    "line_index": idx,
                    "line": line.strip(),
                    "window": big_window,
                    "matched_text": series_match.group(0),
                })
            elif pol in ("required", "allowed", "mentioned"):
                series_required_hits.append({
                    "line_index": idx,
                    "line": line.strip(),
                    "window": big_window,
                    "matched_text": series_match.group(0),
                })

    if fully_required_hits:
        return DetectedValue(
            value="FULLY_RATED",
            matched_phrase="fully rated explicitly required",
            evidence=[
                _make_evidence(
                    page,
                    source,
                    h["window"],
                    matched_phrase="fully rated",
                    matched_line=h["line"],
                    line_index=h["line_index"],
                )
                for h in fully_required_hits[:5]
            ]
        )

    if series_rejected_hits:
        return DetectedValue(
            value="FULLY_RATED",
            matched_phrase="series rated explicitly rejected",
            evidence=[
                _make_evidence(
                    page,
                    source,
                    h["window"],
                    matched_phrase="series rated",
                    matched_line=h["line"],
                    line_index=h["line_index"],
                )
                for h in series_rejected_hits[:5]
            ]
        )

    if series_required_hits:
        return DetectedValue(
            value="SERIES_RATED",
            matched_phrase="series rated explicitly required",
            evidence=[
                _make_evidence(
                    page,
                    source,
                    h["window"],
                    matched_phrase="series rated",
                    matched_line=h["line"],
                    line_index=h["line_index"],
                )
                for h in series_required_hits[:5]
            ]
        )

    return DetectedValue(
        value="SERIES_RATED",
        matched_phrase="no fully rated requirement found; defaulting to series rated",
        evidence=[]
    )

# ============================================================
# Main Analysis
# ============================================================

def _score_evidence_item(category: str, ev: EvidenceItem) -> int:
    line = normalize_for_matching(ev.matched_line or "")
    snippet = normalize_for_matching(ev.snippet or "")
    text = f"{line} {snippet}".strip()

    score = 0

    # Prefer evidence that already rendered successfully
    if ev.image_path:
        score += 10
    if ev.full_page_image_path:
        score += 5

    if category == "bussing_material":
        if "panelboard bus" in text:
            score += 60
        if "bus material" in text or "material" in line:
            score += 50
        if "main bus" in text:
            score += 40
        if "equipment ground bus" in text or "ground bus" in text:
            score += 20
        if "neutral terminal bus" in text or "neutral bus" in text:
            score += 15
        if "copper" in text:
            score += 10
        if "shall" in text or "only" in text:
            score += 10

    elif category == "allow_plug_on_breakers":
        if "shall be bolt on breakers" in text:
            score += 110
        if "shall be bolt on" in text:
            score += 95
        if "bolt on breakers" in text:
            score += 80
        if "panelboards with bolt on type circuit breakers" in text:
            score += 60
        if "bolt on type circuit breakers" in text:
            score += 50
        if "branch overcurrent protective devices" in text:
            score += 35
        if "bus connection" in text:
            score += 10
        if "bolt on" in text:
            score += 10
        if "shall" in text:
            score += 15
        if "type" in text:
            score += 5

        # Penalize manufacturer-list style wrapped lines
        if "manufactured by" in text:
            score -= 40
        if "approved equal" in text or "or equal" in text:
            score -= 25

    elif category == "rating_type":
        if "shall be fully rated" in text:
            score += 80
        if "fully rated" in text:
            score += 50
        if "series rated" in text and ("not acceptable" in text or "not allowed" in text or "not permitted" in text):
            score += 70
        if "interrupting ratings" in text or "sccr" in text or "short circuit current rating" in text:
            score += 25

    elif category == "square_d_allowed":
        if "schneider electric" in text and "square d" in text:
            score += 80
        elif "square d" in text:
            score += 50
        elif "schneider" in text:
            score += 25

        if "manufacturers" in snippet:
            score += 25
        if "manufactured by" in text:
            score += 25
        if "approved equal" in text or "or equal" in text:
            score += 10

        # Prefer panelboard-related context
        if "panelboard" in text or "panelboards" in text:
            score += 25

        # Penalize unrelated equipment context
        if "controller" in text or "controllers" in text:
            score -= 70
        if "comparable to" in text:
            score -= 60
        if "meter" in text:
            score -= 30
        if "photocell" in text:
            score -= 20

    # Slight preference for shorter, more direct lines
    line_len = len(ev.matched_line or "")
    if 20 <= line_len <= 140:
        score += 5
    elif line_len < 12:
        score -= 10
    elif line_len > 180:
        score -= 5

    # Penalize obviously wrapped or incomplete line fragments
    raw_line = (ev.matched_line or "").strip()
    if raw_line.endswith(("type", "with", "by", "of", "and", "or", "(", "-", "be", "shall")):
        score -= 25

    return score


def _pick_best_evidence(category: str, evidence_list: List[EvidenceItem]) -> List[EvidenceItem]:
    if not evidence_list:
        return []

    ranked = sorted(
        evidence_list,
        key=lambda ev: (
            _score_evidence_item(category, ev),
            -(ev.page or 999999),
            -(ev.line_index or 999999),
        ),
        reverse=True
    )

    return [ranked[0]]

def analyze_panelboard_section_text(
    section_text: str,
    section_pages: List[dict],
    source_label: str
) -> Dict[str, DetectedValue]:
    """
    Use whole-section text for final decision values,
    but rebuild evidence page-by-page so each evidence item has the correct page
    and a page-local matched line.
    """
    whole_lines = split_nonempty_lines(section_text)

    # Final decision values still come from whole-section analysis
    whole_bussing = scan_bussing_material(whole_lines, None, source_label)
    whole_breaker = scan_breaker_mounting(whole_lines, None, source_label)
    whole_rating = scan_rating_type(whole_lines, None, source_label)

    # Rebuild evidence page-by-page so image generation has real page anchors
    bussing_evidence = []
    breaker_evidence = []
    rating_fully_evidence = []
    rating_series_rejected_evidence = []
    rating_series_required_evidence = []

    for page_dict in section_pages:
        page_num = page_dict["page_num"]
        page_lines = split_nonempty_lines(page_dict["text"])

        b = scan_bussing_material(page_lines, page_num, source_label)
        if b.evidence:
            bussing_evidence.extend(b.evidence)

        br = scan_breaker_mounting(page_lines, page_num, source_label)
        if br.evidence:
            breaker_evidence.extend(br.evidence)

        r = scan_rating_type(page_lines, page_num, source_label)
        if r.evidence:
            if r.matched_phrase == "fully rated explicitly required":
                rating_fully_evidence.extend(r.evidence)
            elif r.matched_phrase == "series rated explicitly rejected":
                rating_series_rejected_evidence.extend(r.evidence)
            elif r.matched_phrase == "series rated explicitly required":
                rating_series_required_evidence.extend(r.evidence)

    # Apply best page-level evidence back onto the whole-section decisions
    whole_bussing.evidence = _pick_best_evidence("bussing_material", bussing_evidence)
    whole_breaker.evidence = _pick_best_evidence("allow_plug_on_breakers", breaker_evidence)

    if whole_rating.value == "FULLY_RATED":
        if whole_rating.matched_phrase == "fully rated explicitly required":
            whole_rating.evidence = _pick_best_evidence("rating_type", rating_fully_evidence)
        elif whole_rating.matched_phrase == "series rated explicitly rejected":
            whole_rating.evidence = _pick_best_evidence("rating_type", rating_series_rejected_evidence)
    elif whole_rating.value == "SERIES_RATED":
        if whole_rating.matched_phrase == "series rated explicitly required":
            whole_rating.evidence = _pick_best_evidence("rating_type", rating_series_required_evidence)

    return {
        "bussing_material": whole_bussing,
        "allow_plug_on_breakers": whole_breaker,
        "rating_type": whole_rating,
    }


def analyze_pdf_panelboard_specs(pdf_path: str) -> SpecAnalysisResult:
    pages = extract_pdf_pages_text(pdf_path)

    header_candidates = find_panelboard_section_candidates(pages)
    chosen_candidate = choose_best_panelboard_candidate(header_candidates)

    broad_best_idx, broad_best_score, broad_mode_guess = find_best_start_page(pages)

    chosen_layout = None
    if chosen_candidate:
        chosen_layout = classify_panelboard_page_layout(pages[chosen_candidate["idx"]])

    if chosen_candidate and chosen_layout == "strong_top_header":
        # Normal doc path. Keep existing behavior.
        start_idx = chosen_candidate["idx"]
        mode_guess = "dedicated_panelboard_section"
        chosen_section_id = None
        chosen_score = float(chosen_candidate["score"])
        chosen_title = chosen_candidate["title"]
        selection_path = "exact_candidate_strong_top_header"

    elif chosen_candidate and chosen_layout == "inline_compact":
        # Weird compact doc path. Only now do the safe one-page backtrack.
        original_idx = chosen_candidate["idx"]
        start_idx = backtrack_to_inline_panelboard_start(pages, original_idx)
        mode_guess = "inline_compact_panelboard_section"
        chosen_section_id = None
        chosen_score = float(chosen_candidate["score"])
        chosen_title = (
            get_inline_panelboard_title_from_page(pages[start_idx])
            or chosen_candidate["title"]
        )
        selection_path = "inline_compact_backtrack"

    else:
        # Existing fallback behavior
        use_exact_candidate = False
        if chosen_candidate:
            candidate_score = float(chosen_candidate["score"])
            if broad_best_idx is None:
                use_exact_candidate = True
            elif candidate_score >= 18:
                use_exact_candidate = True

        if use_exact_candidate and chosen_candidate:
            start_idx = chosen_candidate["idx"]
            mode_guess = "dedicated_panelboard_section"
            chosen_section_id = None
            chosen_score = float(chosen_candidate["score"])
            chosen_title = chosen_candidate["title"]
            selection_path = "exact_candidate"
        else:
            if broad_best_idx is None:
                return SpecAnalysisResult(
                    section_match=None,
                    panelboards={
                        "bussing_material": DetectedValue(value=None),
                        "allow_plug_on_breakers": DetectedValue(value=None),
                        "rating_type": DetectedValue(value=None),
                    },
                    job_flags={
                        "square_d_allowed": DetectedValue(
                            value=True,
                            matched_phrase="no panelboard section found; defaulting to allowed",
                            evidence=[]
                        ),
                    },
                    debug={
                        "reason": "No likely panelboard section found",
                        "default_behavior": "square_d_allowed=True when no panelboard section is found",
                        "page_scores": [
                            {
                                "page_num": p["page_num"],
                                "score": score_page_for_panelboard_relevance(p)
                            }
                            for p in pages
                        ],
                        "header_candidates": header_candidates,
                    }
                )

            start_idx = refine_to_section_start(pages, broad_best_idx)
            mode_guess = broad_mode_guess
            chosen_section_id = None
            chosen_score = float(broad_best_score)
            chosen_title = get_section_header_line(pages[start_idx])
            selection_path = "broad_fallback"

    section_text, end_idx = collect_multi_page_section(
        pages,
        start_idx,
        mode_guess,
        target_section_id=chosen_section_id
    )

    section_pages = pages[start_idx:end_idx + 1]

    panelboard_values = analyze_panelboard_section_text(
        section_text,
        section_pages,
        "whole_section_line_scan"
    )

    whole_square_d_allowed = scan_square_d_allowed(
        split_nonempty_lines(section_text),
        None,
        "whole_section_line_scan"
    )

    square_d_evidence = []
    competitor_evidence = []

    for page_dict in section_pages:
        page_num = page_dict["page_num"]
        page_lines = split_nonempty_lines(page_dict["text"])
        sqd_page = scan_square_d_allowed(page_lines, page_num, "whole_section_line_scan")

        if sqd_page.evidence:
            if sqd_page.value is True:
                square_d_evidence.extend(sqd_page.evidence)
            elif sqd_page.value is False:
                competitor_evidence.extend(sqd_page.evidence)

    if whole_square_d_allowed.value is True:
        whole_square_d_allowed.evidence = _pick_best_evidence("square_d_allowed", square_d_evidence)
    elif whole_square_d_allowed.value is False:
        whole_square_d_allowed.evidence = _pick_best_evidence("square_d_allowed", competitor_evidence)

    square_d_allowed = whole_square_d_allowed

    section_match = SectionMatch(
        mode=mode_guess,
        start_page=pages[start_idx]["page_num"],
        end_page=pages[end_idx]["page_num"],
        score=chosen_score,
        title=chosen_title
    )

    return SpecAnalysisResult(
        section_match=section_match,
        panelboards=panelboard_values,
        job_flags={"square_d_allowed": square_d_allowed},
        debug={
            "selection_path": selection_path,
            "selected_section_id": chosen_section_id,
            "all_header_candidates": header_candidates,
            "chosen_layout": chosen_layout,
            "broad_best_idx": broad_best_idx,
            "broad_best_page_num": pages[broad_best_idx]["page_num"] if broad_best_idx is not None else None,
            "refined_start_page_index": start_idx,
            "refined_start_page_num": pages[start_idx]["page_num"],
            "mode_guess": mode_guess,
            "section_preview": section_text[:1200],
            "page_scores": [
                {
                    "page_num": p["page_num"],
                    "score": score_page_for_panelboard_relevance(p)
                }
                for p in pages
            ]
        }
    )


def apply_spec_analysis_to_defaults(default_overrides: dict, spec_result: SpecAnalysisResult) -> dict:
    merged = {**default_overrides}
    panelboards = dict(merged.get("panelboards", {}))
    detected = spec_result.panelboards

    if detected["bussing_material"].value is not None:
        panelboards["bussing_material"] = detected["bussing_material"].value

    if detected["allow_plug_on_breakers"].value is not None:
        panelboards["allow_plug_on_breakers"] = detected["allow_plug_on_breakers"].value

    if detected["rating_type"].value is not None:
        panelboards["rating_type"] = detected["rating_type"].value

    merged["panelboards"] = panelboards
    return merged

def build_detected_overrides_payload(spec_result: SpecAnalysisResult) -> dict:
    return {
        "panelboards": {
            "bussing_material": (
                spec_result.panelboards.get("bussing_material").value
                if spec_result.panelboards.get("bussing_material") else None
            ),
            "allow_plug_on_breakers": (
                spec_result.panelboards.get("allow_plug_on_breakers").value
                if spec_result.panelboards.get("allow_plug_on_breakers") else None
            ),
            "rating_type": (
                spec_result.panelboards.get("rating_type").value
                if spec_result.panelboards.get("rating_type") else None
            ),
        },
        "job_flags": {
            "square_d_allowed": (
                spec_result.job_flags.get("square_d_allowed").value
                if spec_result.job_flags.get("square_d_allowed") else None
            ),
        }
    }


def _copy_specs_pdf_to_data(pdf_path: Path) -> Path:
    """
    Save a copy of the uploaded specs PDF into:
    ~/Spec_Sheet_Analysis/data/
    """
    import shutil

    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"Specs PDF not found: {pdf_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.name)
    saved_name = f"{stamp}__{safe_name}"
    saved_path = SPEC_DATA_ROOT / saved_name

    shutil.copy2(str(pdf_path), str(saved_path))
    return saved_path

# ============================================================
# Output Helpers
# ============================================================

def build_human_summary(spec_result_dict: dict) -> dict:
    panelboards = spec_result_dict.get("panelboards") or {}
    job_flags = spec_result_dict.get("job_flags") or {}
    section_match = spec_result_dict.get("section_match") or {}

    return {
        "section_match": {
            "mode": section_match.get("mode"),
            "start_page": section_match.get("start_page"),
            "end_page": section_match.get("end_page"),
            "score": section_match.get("score"),
            "title": section_match.get("title"),
        },
        "panelboard_defaults_detected": {
            "bussing_material": (panelboards.get("bussing_material") or {}).get("value"),
            "allow_plug_on_breakers": (panelboards.get("allow_plug_on_breakers") or {}).get("value"),
            "rating_type": (panelboards.get("rating_type") or {}).get("value"),
        },
        "job_flags": {
            "square_d_allowed": (job_flags.get("square_d_allowed") or {}).get("value"),
        },
    }


def print_detection_summary(spec_result_dict: dict, merged_defaults: dict):
    section_match = spec_result_dict.get("section_match") or {}
    panelboards = spec_result_dict.get("panelboards") or {}
    job_flags = spec_result_dict.get("job_flags") or {}

    print("\n=== SPEC SECTION MATCH ===")
    print("mode       :", section_match.get("mode"))
    print("start_page :", section_match.get("start_page"))
    print("end_page   :", section_match.get("end_page"))
    print("score      :", section_match.get("score"))
    print("title      :", section_match.get("title"))

    print("\n=== DETECTED PANELBOARD DEFAULTS ===")
    print("bussing_material        :", (panelboards.get("bussing_material") or {}).get("value"))
    print("allow_plug_on_breakers  :", (panelboards.get("allow_plug_on_breakers") or {}).get("value"))
    print("rating_type             :", (panelboards.get("rating_type") or {}).get("value"))

    print("\n=== JOB FLAGS ===")
    print("square_d_allowed        :", (job_flags.get("square_d_allowed") or {}).get("value"))

    print("\n=== MERGED PANELBOARD DEFAULTS ===")
    merged_panelboards = merged_defaults.get("panelboards", {})
    print("bussing_material        :", merged_panelboards.get("bussing_material"))
    print("allow_plug_on_breakers  :", merged_panelboards.get("allow_plug_on_breakers"))
    print("rating_type             :", merged_panelboards.get("rating_type"))
    print("allow_feed_thru_lugs    :", merged_panelboards.get("allow_feed_thru_lugs"))
    print("default_trim_style      :", merged_panelboards.get("default_trim_style"))
    print("enclosure               :", merged_panelboards.get("enclosure"))
    print("allow_square_d_spd      :", merged_panelboards.get("allow_square_d_spd"))

def analyze_specs_pdf_for_ui(pdf_path: str, job_dir: str) -> dict:
    """
    App-callable specs analyzer entrypoint.

    Inputs:
      - pdf_path: actual uploaded specs PDF path from uplink
      - job_dir: job folder created by uplink

    Returns:
      UI-friendly dict containing:
      - summary
      - detected_overrides
      - merged_ui_overrides
      - evidence_images
    """
    run_start_ts_ms = now_ts_ms()
    run_start_perf = time.perf_counter()

    pdf_path = Path(pdf_path).expanduser().resolve()
    job_dir = Path(job_dir).expanduser().resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    # Save permanent copy of uploaded specs PDF
    data_pdf_path = _copy_specs_pdf_to_data(pdf_path)

    default_ui_overrides = build_default_ui_overrides()

    spec_result = analyze_pdf_panelboard_specs(str(pdf_path))

    # Save evidence images into the job folder
    evidence_images = []

    for key in ["bussing_material", "allow_plug_on_breakers", "rating_type"]:
        item = spec_result.panelboards.get(key)
        if not item:
            continue

        for i, ev in enumerate(item.evidence):
            if not ev.page or not ev.matched_phrase or not ev.matched_line:
                continue

            image_name = f"{key}_evidence_{i+1}_page_{ev.page}.png"
            full_page_image_name = f"{key}_evidence_{i+1}_page_{ev.page}_full.png"

            image_path = job_dir / image_name
            full_page_image_path = job_dir / full_page_image_name

            saved = generate_evidence_image(
                pdf_path=str(pdf_path),
                page_number=ev.page,
                snippet=ev.snippet,
                matched_phrase=ev.matched_phrase,
                matched_line=ev.matched_line,
                output_path=str(image_path),
                full_page_output_path=str(full_page_image_path),
            )

            if saved:
                ev.image_path = image_name
                ev.full_page_image_path = full_page_image_name

                evidence_images.append({
                    "category": key,
                    "label": {
                        "bussing_material": "Bussing Material",
                        "allow_plug_on_breakers": "Allow Plug-On Breakers",
                        "rating_type": "Rating Type",
                    }.get(key, key),
                    "value": item.value,
                    "page": ev.page,
                    "matched_phrase": ev.matched_phrase,
                    "matched_line": ev.matched_line,
                    "snippet": ev.snippet,
                    "image_path": image_name,
                    "full_page_image_path": full_page_image_name,
                })

    for key in ["square_d_allowed"]:
        item = spec_result.job_flags.get(key)
        if not item:
            continue

        for i, ev in enumerate(item.evidence):
            if not ev.page or not ev.matched_phrase or not ev.matched_line:
                continue

            image_name = f"{key}_evidence_{i+1}_page_{ev.page}.png"
            full_page_image_name = f"{key}_evidence_{i+1}_page_{ev.page}_full.png"

            image_path = job_dir / image_name
            full_page_image_path = job_dir / full_page_image_name

            saved = generate_evidence_image(
                pdf_path=str(pdf_path),
                page_number=ev.page,
                snippet=ev.snippet,
                matched_phrase=ev.matched_phrase,
                matched_line=ev.matched_line,
                output_path=str(image_path),
                full_page_output_path=str(full_page_image_path),
            )

            if saved:
                ev.image_path = image_name
                ev.full_page_image_path = full_page_image_name

                evidence_images.append({
                    "category": key,
                    "label": "Square D Allowed",
                    "value": item.value,
                    "page": ev.page,
                    "matched_phrase": ev.matched_phrase,
                    "matched_line": ev.matched_line,
                    "snippet": ev.snippet,
                    "image_path": image_name,
                    "full_page_image_path": full_page_image_name,
                })

    spec_result_dict = ensure_json_safe(spec_result.to_dict())
    merged_defaults = apply_spec_analysis_to_defaults(default_ui_overrides, spec_result)

    cycle_time_ms = int((time.perf_counter() - run_start_perf) * 1000)
    parse_done_ts_ms = now_ts_ms()

    summary = build_human_summary(spec_result_dict)
    detected_overrides = build_detected_overrides_payload(spec_result)

    # optional debug/json dumps in job dir
    summary_json_path = job_dir / "summary.json"
    full_dump_path = job_dir / "full_spec_analysis_dump.json"

    ui_result = {
        "ok": True,
        "saved_pdf": str(pdf_path),
        "data_saved_pdf": str(data_pdf_path),
        "job_dir": str(job_dir),
        "cycle_time_ms": cycle_time_ms,
        "cycle_time_str": ms_to_readable(cycle_time_ms),
        "noticed_ts_ms": run_start_ts_ms,
        "parse_done_ts_ms": parse_done_ts_ms,
        "summary": summary,
        "detected_overrides": ensure_json_safe(detected_overrides),
        "merged_ui_overrides": ensure_json_safe(merged_defaults),
        "evidence_images": ensure_json_safe(evidence_images),
        "spec_analysis_result": ensure_json_safe(spec_result_dict),
    }

    try:
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "ok": True,
                "summary": ui_result["summary"],
                "detected_overrides": ui_result["detected_overrides"],
                "merged_ui_overrides": ui_result["merged_ui_overrides"],
                "evidence_images": ui_result["evidence_images"],
                "saved_pdf": ui_result["saved_pdf"],
                "data_saved_pdf": ui_result["data_saved_pdf"],
                "cycle_time_ms": ui_result["cycle_time_ms"],
                "cycle_time_str": ui_result["cycle_time_str"],
            }, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[WARN] Could not write summary json: {e}")

    try:
        with open(full_dump_path, "w", encoding="utf-8") as f:
            json.dump(ui_result, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[WARN] Could not write full dump json: {e}")

    return ui_result

# ============================================================
# Main
# ============================================================

def main():
    run_start_ts_ms = now_ts_ms()
    run_start_perf = time.perf_counter()

    if not INPUT_PDF.exists():
        raise FileNotFoundError(f"Input PDF not found: {INPUT_PDF}")

    job_id = build_job_id(INPUT_PDF)
    job_dir = SPEC_OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = job_dir / "summary.json"
    raw_dump_path = job_dir / "full_spec_analysis_dump.json"

    print(f"\n[JOB] job_id={job_id}")
    print(f"[JOB] job_dir={job_dir}")
    print(f"[JOB] input_pdf={INPUT_PDF}")

    default_ui_overrides = build_default_ui_overrides()

    print("\n[SpecAnalysis] starting...")
    spec_result = analyze_pdf_panelboard_specs(str(INPUT_PDF))
    spec_result_dict = ensure_json_safe(spec_result.to_dict())
    # Generate proof images for detected panelboard evidence
    for key in ["bussing_material", "allow_plug_on_breakers", "rating_type"]:
        item = spec_result.panelboards.get(key)
        if not item:
            continue

        for i, ev in enumerate(item.evidence):
            if not ev.page or not ev.matched_phrase or not ev.matched_line:
                continue

            image_name = f"{key}_evidence_{i+1}_page_{ev.page}.png"
            full_page_image_name = f"{key}_evidence_{i+1}_page_{ev.page}_full.png"

            image_path = job_dir / image_name
            full_page_image_path = job_dir / full_page_image_name

            saved = generate_evidence_image(
                pdf_path=str(INPUT_PDF),
                page_number=ev.page,
                snippet=ev.snippet,
                matched_phrase=ev.matched_phrase,
                matched_line=ev.matched_line,
                output_path=str(image_path),
                full_page_output_path=str(full_page_image_path),
            )

            if saved:
                ev.image_path = saved["zoom_image_path"]
                ev.full_page_image_path = saved["full_page_image_path"]
            else:
                print(f"[IMG FAIL] key={key} page={ev.page} phrase={ev.matched_phrase!r} line={ev.matched_line!r}")
                if key == "allow_plug_on_breakers":
                    print(f"[BOLT FAIL] snippet={ev.snippet!r}")

    # Generate proof images for job flags
    for key in ["square_d_allowed"]:
        item = spec_result.job_flags.get(key)
        if not item:
            continue

        for i, ev in enumerate(item.evidence):
            if not ev.page or not ev.matched_phrase or not ev.matched_line:
                continue

            image_name = f"{key}_evidence_{i+1}_page_{ev.page}.png"
            full_page_image_name = f"{key}_evidence_{i+1}_page_{ev.page}_full.png"

            image_path = job_dir / image_name
            full_page_image_path = job_dir / full_page_image_name

            saved = generate_evidence_image(
                pdf_path=str(INPUT_PDF),
                page_number=ev.page,
                snippet=ev.snippet,
                matched_phrase=ev.matched_phrase,
                matched_line=ev.matched_line,
                output_path=str(image_path),
                full_page_output_path=str(full_page_image_path),
            )

            if saved:
                ev.image_path = saved["zoom_image_path"]
                ev.full_page_image_path = saved["full_page_image_path"]
            else:
                print(f"[IMG FAIL] key={key} page={ev.page} phrase={ev.matched_phrase!r} line={ev.matched_line!r}")

    # refresh dict after image paths are added
    spec_result_dict = ensure_json_safe(spec_result.to_dict())

    merged_defaults = apply_spec_analysis_to_defaults(default_ui_overrides, spec_result)

    cycle_time_ms = int((time.perf_counter() - run_start_perf) * 1000)
    parse_done_ts_ms = now_ts_ms()

    human_summary = build_human_summary(spec_result_dict)

    summary = {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "saved_pdf": str(INPUT_PDF),
        "output_dir": str(job_dir),
        "cycle_time_ms": cycle_time_ms,
        "cycle_time_str": ms_to_readable(cycle_time_ms),
        "noticed_ts_ms": run_start_ts_ms,
        "parse_done_ts_ms": parse_done_ts_ms,
        "summary": human_summary,
        "merged_ui_overrides": ensure_json_safe(merged_defaults),
        "square_d_allowed": (
            ((spec_result_dict.get("job_flags") or {}).get("square_d_allowed") or {}).get("value")
        ),
    }

    full_dump = {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "saved_pdf": str(INPUT_PDF),
        "output_dir": str(job_dir),
        "cycle_time_ms": cycle_time_ms,
        "cycle_time_str": ms_to_readable(cycle_time_ms),
        "noticed_ts_ms": run_start_ts_ms,
        "parse_done_ts_ms": parse_done_ts_ms,
        "spec_analysis_result": spec_result_dict,
        "original_ui_overrides": ensure_json_safe(default_ui_overrides),
        "merged_ui_overrides": ensure_json_safe(merged_defaults),
    }

    print_detection_summary(spec_result_dict, merged_defaults)

    square_d_allowed = (
        ((spec_result_dict.get("job_flags") or {}).get("square_d_allowed") or {}).get("value")
    )
    if spec_result.section_match is None:
        print("\n[SpecAnalysis] No panelboard section found. Defaulting Square D to allowed.")
    elif square_d_allowed is False:
        print("\n[SpecAnalysis] BAIL CONDITION: Square D / SQD / Schneider token not found in panelboard section.")
    else:
        print("\n[SpecAnalysis] Square D allowed. Defaults can be used.")

    try:
        with open(raw_dump_path, "w", encoding="utf-8") as f:
            json.dump(full_dump, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[WROTE RAW DUMP] {raw_dump_path}")
    except Exception as e:
        print(f"[WARN] Could not write raw dump: {e}")

    try:
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"[WROTE SUMMARY] {summary_json_path}")
    except Exception as e:
        print(f"[ERROR] Could not write summary: {e}")
        raise

    print(f"\n[SpecAnalysis] Done.")
    print(f"[SpecAnalysis] cycle_time={summary['cycle_time_str']}")


if __name__ == "__main__":
    main()