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
INPUT_PDF = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/S1.pdf").expanduser()
SPEC_OUTPUT_ROOT = Path("~/Spec_Sheet_Analysis/Results").expanduser()
SPEC_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


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

PANELBOARD_SECTION_PRIORITY = {
    "26 24 16.16": 0,   # highest priority
    "26 24 16.13": 1,
    "26 24 16": 2,
}

PANELBOARD_SECTION_HEADER_PATTERNS = [
    (re.compile(r"\bsection\s+26\s+24\s+16\s+16\b", re.IGNORECASE), "26 24 16.16"),
    (re.compile(r"\bsection\s+26\s+24\s+16\s+13\b", re.IGNORECASE), "26 24 16.13"),
    (re.compile(r"\bsection\s+26\s+24\s+16\b", re.IGNORECASE), "26 24 16"),
    (re.compile(r"\b26\s+24\s+16\s+16\b", re.IGNORECASE), "26 24 16.16"),
    (re.compile(r"\b26\s+24\s+16\s+13\b", re.IGNORECASE), "26 24 16.13"),
    (re.compile(r"\b26\s+24\s+16\b", re.IGNORECASE), "26 24 16"),
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

def _find_best_page_line_index(page_text: str, matched_line: str) -> Optional[int]:
    page_lines = [ln.strip() for ln in normalize_preserve_lines(page_text).splitlines() if ln.strip()]
    target = normalize_for_matching(matched_line)

    if not target:
        return None

    # Exact normalized match first
    for i, line in enumerate(page_lines):
        if normalize_for_matching(line) == target:
            return i

    # Fallback: containment
    for i, line in enumerate(page_lines):
        norm_line = normalize_for_matching(line)
        if target in norm_line or norm_line in target:
            return i

    return None

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
    Reliable evidence-image generator:
    - uses page extracted text lines to find the matched line's vertical position
    - uses pdfium phrase search to get actual phrase bbox
    - chooses the phrase hit closest to the matched line's vertical band
    - renders a full-page highlight and a full-width zoom band
    """
    pdf = None
    page = None
    textpage = None

    try:
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[page_number - 1]
        textpage = page.get_textpage()

        page_height_pdf = float(page.get_height())

        page_text = textpage.get_text_bounded() or ""
        page_lines = [ln.strip() for ln in normalize_preserve_lines(page_text).splitlines() if ln.strip()]
        total_page_lines = max(len(page_lines), 1)

        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # -------------------------------
        # 1. Find matched line vertical position from extracted page text
        # -------------------------------
        page_line_index = _find_best_page_line_index(page_text, matched_line)
        if page_line_index is None:
            return None

        line_ratio = (page_line_index + 0.5) / total_page_lines
        estimated_line_center_y_px = int(pil_image.size[1] * line_ratio)

        # -------------------------------
        # 2. Find all phrase hits on page and choose closest one vertically
        # -------------------------------
        phrase_hits = []
        phrase_search = None

        try:
            phrase_search = textpage.search(matched_phrase, match_case=False, match_whole_word=False)

            while True:
                hit = phrase_search.get_next()
                if not hit:
                    break

                start_index, count = hit
                bbox = _get_span_bbox(textpage, start_index, count)
                if not bbox:
                    continue

                px1, py1, px2, py2 = _pdf_rect_to_pixel_rect(bbox, page_height_pdf, scale)
                phrase_center_y = (py1 + py2) // 2

                phrase_hits.append({
                    "bbox_pdf": bbox,
                    "bbox_px": (px1, py1, px2, py2),
                    "center_y": phrase_center_y,
                    "distance": abs(phrase_center_y - estimated_line_center_y_px),
                })

        finally:
            try:
                if phrase_search is not None:
                    phrase_search.close()
            except Exception:
                pass

        if not phrase_hits:
            return None

        best_phrase_hit = sorted(phrase_hits, key=lambda h: h["distance"])[0]
        phrase_bbox = best_phrase_hit["bbox_pdf"]
        fx1, fy1, fx2, fy2 = best_phrase_hit["bbox_px"]
        phrase_center_y = best_phrase_hit["center_y"]

        # -------------------------------
        # 3. Full page annotated image
        # -------------------------------
        full_page_img = pil_image.copy()
        full_draw = ImageDraw.Draw(full_page_img, "RGBA")

        full_draw.rectangle(
            [(fx1 - 4, fy1 - 2), (fx2 + 4, fy2 + 2)],
            fill=(255, 235, 59, 120),
            outline=(255, 193, 7, 220),
            width=2,
        )

        full_page_img.save(full_page_output_path)

        # -------------------------------
        # 4. Full-width context band around phrase/line area
        # -------------------------------
        band_height_px = max(220, int(full_page_img.size[1] * context_band_ratio))
        band_top = max(0, phrase_center_y - band_height_px // 2)
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
            fill=(255, 235, 59, 120),
            outline=(255, 193, 7, 220),
            width=2,
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

def score_page_for_panelboard_relevance(page_dict: dict) -> float:
    score = 0.0

    full_text = page_dict["text"] or ""
    norm_full = normalize_for_matching(full_text)

    top_band = normalize_for_matching(page_dict["regions"]["top_band"])
    top_left = normalize_for_matching(page_dict["regions"]["top_left"])
    top_center = normalize_for_matching(page_dict["regions"]["top_center"])
    top_right = normalize_for_matching(page_dict["regions"]["top_right"])
    top_regions = [top_center, top_left, top_right, top_band]

    if re.search(r"\bsection\s+26\s+24\s+16\b", norm_full):
        score += 20.0

    if any(re.search(r"\bsection\s+26\s+24\s+16\b", r) for r in top_regions):
        score += 18.0

    if any(re.search(r"\b26\s+24\s+16\b", r) for r in top_regions):
        score += 10.0

    if any("distribution panelboards" in r for r in top_regions):
        score += 16.0
    elif any("lighting and appliance panelboards" in r for r in top_regions):
        score += 14.0
    elif any("low voltage panelboards" in r for r in top_regions):
        score += 14.0
    elif any("panelboards" in r for r in top_regions):
        score += 6.0

    if re.search(r"\b26\s+24\s+16(?:\s+\d+)?\s*1\b", norm_full):
        score += 18.0

    support_terms = [
        "bolt on",
        "bolt in",
        "fully rated",
        "copper",
        "manufacturers",
        "breaker",
        "breakers",
        "bus",
        "buses",
        "panelboard buses",
    ]
    for term in support_terms:
        if term in norm_full:
            score += 1.0

    reference_terms = [
        "related work",
        "references",
        "section includes the following",
        "this section includes the following",
        "table of contents",
        "contents",
    ]
    for term in reference_terms:
        if term in norm_full:
            score -= 8.0

    all_section_refs = re.findall(r"\b26\s+\d{2}\s+\d{2}(?:\.\d+)?\b", norm_full)
    unique_refs = set(all_section_refs)
    if len(unique_refs) >= 4:
        score -= 18.0
    elif len(unique_refs) >= 2:
        score -= 6.0

    if re.search(r"\b26\s+24\s+16\.\d+\b", norm_full) and "section 26 24 16" not in norm_full:
        score -= 8.0

    return score


def score_section_header_candidate(page_dict: dict) -> int:
    full_text = normalize_for_matching(page_dict["text"])

    top_band = normalize_for_matching(page_dict["regions"]["top_band"])
    middle_band = normalize_for_matching(page_dict["regions"]["middle_band"])
    bottom_band = normalize_for_matching(page_dict["regions"]["bottom_band"])

    top_left = normalize_for_matching(page_dict["regions"]["top_left"])
    top_center = normalize_for_matching(page_dict["regions"]["top_center"])
    top_right = normalize_for_matching(page_dict["regions"]["top_right"])

    bottom_left = normalize_for_matching(page_dict["regions"]["bottom_left"])
    bottom_center = normalize_for_matching(page_dict["regions"]["bottom_center"])
    bottom_right = normalize_for_matching(page_dict["regions"]["bottom_right"])

    score = 0
    top_regions = [top_center, top_left, top_right, top_band]
    bottom_regions = [bottom_left, bottom_center, bottom_right, bottom_band]

    if any("section 26 24 16" in r or "section 262416" in r for r in top_regions):
        score += 30

    if any("distribution panelboards" in r for r in top_regions):
        score += 20
    elif any("lighting and appliance panelboards" in r for r in top_regions):
        score += 18
    elif any("low voltage panelboards" in r for r in top_regions):
        score += 18
    elif any("panelboards" in r for r in top_regions):
        score += 10

    if any("26 24 16" in r or "262416" in r for r in top_regions):
        score += 16

    if any("panelboards" in r for r in top_regions):
        score += 8

    if "26 24 16" in full_text and "panelboards" in full_text:
        score += 12

    if re.search(r"\b26\s+24\s+16(?:\s+\d+)?\s*1\b", full_text):
        score += 28

    opening_terms = [
        "part 1 general",
        "1 1 related documents",
        "1 1 description",
        "1 1 summary",
        "1 1 references",
        "1 1 related work",
    ]
    opening_hits = sum(1 for term in opening_terms if term in full_text)
    score += min(opening_hits * 8, 24)

    if any("section 26 24 16" in r or "section 262416" in r for r in bottom_regions):
        score -= 12

    if any("low voltage panelboards" in r or "distribution panelboards" in r for r in bottom_regions):
        score -= 8

    all_section_refs = re.findall(r"\b26\s+\d{2}\s+\d{2}(?:\.\d+)?\b", full_text)
    unique_section_refs = set(all_section_refs)
    if len(unique_section_refs) >= 3:
        score -= 18
    elif len(unique_section_refs) == 2:
        score -= 8

    toc_terms = [
        "table of contents",
        "contents",
        "division 26",
        "building standards",
        "technical standards",
    ]
    if any(term in full_text for term in toc_terms):
        score -= 18

    deep_body_terms = [
        "surge protective device",
        "ground fault circuit interrupter",
        "fabrication and features",
    ]
    deep_body_hits = sum(1 for term in deep_body_terms if term in middle_band)
    score -= min(deep_body_hits * 2, 10)

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
        r"section\s+26\s+24\s+16\s+16\s+distribution\s+panelboards",
        r"section\s+26\s+24\s+16\s+13\s+lighting\s+and\s+appliance\s+panelboards",
        r"section\s+26\s+24\s+16\s+low\s+voltage\s+panelboards",
        r"section\s+26\s+24\s+16\s+panelboards",
        r"26\s+24\s+16\s+16\s+distribution\s+panelboards",
        r"26\s+24\s+16\s+13\s+lighting\s+and\s+appliance\s+panelboards",
        r"26\s+24\s+16\s+low\s+voltage\s+panelboards",
        r"26\s+24\s+16\s+panelboards",
        r"26\s+24\s+16\s+panelboards",
        r"26\s+24\s+00\s+switchboards\s+and\s+panelboards",
        r"panelboards",
    ]

    for region_text in candidate_regions:
        norm = normalize_for_matching(region_text)
        for pat in header_patterns:
            m = re.search(pat, norm, re.IGNORECASE)
            if m:
                return m.group(0)

    return None


def get_panelboard_section_identity(page_dict: dict) -> Optional[str]:
    candidate_regions = [
        page_dict["regions"]["top_center"],
        page_dict["regions"]["top_left"],
        page_dict["regions"]["top_right"],
        page_dict["regions"]["top_band"],
        page_dict["text"],
    ]

    for region_text in candidate_regions:
        norm = normalize_for_matching(region_text)
        for pattern, section_id in PANELBOARD_SECTION_HEADER_PATTERNS:
            if pattern.search(norm):
                return section_id

    return None


def find_panelboard_section_candidates(pages: List[dict]) -> List[dict]:
    candidates = []

    for idx, page in enumerate(pages):
        section_id = get_panelboard_section_identity(page)
        if not section_id:
            continue

        score = score_section_header_candidate(page)
        candidates.append({
            "idx": idx,
            "page_num": page["page_num"],
            "section_id": section_id,
            "score": score,
            "title": get_section_header_line(page),
        })

    return candidates


def choose_best_panelboard_candidate(candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda c: (
            PANELBOARD_SECTION_PRIORITY.get(c["section_id"], 999),
            -c["score"],
            c["idx"],
        )
    )
    return candidates[0]


def find_best_start_page(pages: List[dict]) -> Tuple[Optional[int], float, str]:
    best_idx = None
    best_score = -999.0
    best_mode = "unknown"

    for idx, page in enumerate(pages):
        score = score_page_for_panelboard_relevance(page)

        full_text = normalize_for_matching(page["text"])
        top_band = normalize_for_matching(page["regions"]["top_band"])
        top_left = normalize_for_matching(page["regions"]["top_left"])
        top_center = normalize_for_matching(page["regions"]["top_center"])
        top_right = normalize_for_matching(page["regions"]["top_right"])
        top_regions = [top_center, top_left, top_right, top_band]

        has_true_header = (
            re.search(r"\bsection\s+26\s+24\s+16\b", full_text) is not None
            or any(re.search(r"\bsection\s+26\s+24\s+16\b", r) for r in top_regions)
        )

        has_panel_title = any(
            term in " ".join(top_regions)
            for term in [
                "panelboards",
                "low voltage panelboards",
                "distribution panelboards",
                "lighting and appliance panelboards",
            ]
        ) or any(
            term in full_text
            for term in [
                "panelboards",
                "low voltage panelboards",
                "distribution panelboards",
                "lighting and appliance panelboards",
            ]
        )

        if has_true_header and has_panel_title:
            score += 10.0
            mode = "dedicated_panelboard_section"
        elif has_panel_title:
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
    lookback = max(0, best_idx - 20)
    candidates = []

    for idx in range(lookback, best_idx + 1):
        full_text = normalize_for_matching(pages[idx]["text"])
        top_band = normalize_for_matching(pages[idx]["regions"]["top_band"])
        top_left = normalize_for_matching(pages[idx]["regions"]["top_left"])
        top_center = normalize_for_matching(pages[idx]["regions"]["top_center"])
        top_right = normalize_for_matching(pages[idx]["regions"]["top_right"])
        top_regions = [top_center, top_left, top_right, top_band]

        has_true_section_header = (
            re.search(r"\bsection\s+26\s+24\s+16\b", full_text) is not None
            or any(re.search(r"\bsection\s+26\s+24\s+16\b", r) for r in top_regions)
        )

        has_panelboard_title = any("panelboards" in r for r in top_regions) or "panelboards" in full_text

        if has_true_section_header and has_panelboard_title:
            score = score_section_header_candidate(pages[idx])
            candidates.append({
                "idx": idx,
                "page_num": pages[idx]["page_num"],
                "score": score,
            })

    if not candidates:
        return best_idx

    candidates.sort(key=lambda x: (x["score"], x["idx"]))
    return candidates[-1]["idx"]


def collect_multi_page_section(
    pages: List[dict],
    start_idx: int,
    mode: str,
    target_section_id: Optional[str] = None
) -> Tuple[str, int]:
    collected = []
    end_idx = start_idx
    target_norm = normalize_for_matching(target_section_id or "")

    for idx in range(start_idx, len(pages)):
        page_text = normalize_preserve_lines(pages[idx]["text"])
        page_match = normalize_for_matching(page_text)

        if idx > start_idx:
            if target_section_id in ("26 24 16.13", "26 24 16.16"):
                sibling_headers = [
                    "section 26 24 16 13",
                    "section 26 24 16 16",
                ]

                page_has_sibling_header = any(h in page_match for h in sibling_headers)
                page_is_target = f"section {target_norm}" in page_match

                if page_has_sibling_header and not page_is_target:
                    break

            if (
                "section 26 24 19" in page_match or
                "section 262419" in page_match or
                re.search(r"\b26\s+24\s+19\b", page_match)
            ):
                break

            top_header_text = normalize_for_matching(
                " ".join([
                    pages[idx]["regions"]["top_left"],
                    pages[idx]["regions"]["top_center"],
                    pages[idx]["regions"]["top_right"],
                    pages[idx]["regions"]["top_band"],
                ])
            )

            real_new_section_header = re.search(
                r"\bsection\s+26\s+\d{2}\s+\d{2}(?:\s+\d+)?\b",
                top_header_text
            )

            if real_new_section_header:
                if target_section_id:
                    exact_target_phrase = f"section {target_norm}"
                    if exact_target_phrase not in top_header_text:
                        break
                else:
                    if not ("section 26 24 16" in top_header_text or "section 262416" in top_header_text):
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
        re.compile(r"\bsquare\s*d\b", re.IGNORECASE),
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

    sqd_hits = find_line_hits(lines, sqd_patterns)
    competitor_hits = find_line_hits(lines, competitor_patterns)

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

    if competitor_hits:
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
                for h in competitor_hits[:5]
            ]
        )

    return DetectedValue(
        value=True,
        matched_phrase="no manufacturers mentioned; defaulting to allowed",
        evidence=[]
    )


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

def analyze_panelboard_section_text(section_text: str, page: Optional[int], source_label: str) -> Dict[str, DetectedValue]:
    lines = split_nonempty_lines(section_text)

    return {
        "bussing_material": scan_bussing_material(lines, page, source_label),
        "allow_plug_on_breakers": scan_breaker_mounting(lines, page, source_label),
        "rating_type": scan_rating_type(lines, page, source_label),
    }


def analyze_pdf_panelboard_specs(pdf_path: str) -> SpecAnalysisResult:
    pages = extract_pdf_pages_text(pdf_path)

    header_candidates = find_panelboard_section_candidates(pages)
    chosen_candidate = choose_best_panelboard_candidate(header_candidates)

    broad_best_idx, broad_best_score, broad_mode_guess = find_best_start_page(pages)

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
        chosen_section_id = chosen_candidate["section_id"]
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
                    "square_d_allowed": DetectedValue(value=False),
                },
                debug={
                    "reason": "No likely panelboard section found",
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
        chosen_section_id = get_panelboard_section_identity(pages[start_idx])
        chosen_score = float(broad_best_score)
        chosen_title = get_section_header_line(pages[start_idx])
        selection_path = "broad_fallback"

    section_text, end_idx = collect_multi_page_section(
        pages,
        start_idx,
        mode_guess,
        target_section_id=chosen_section_id
    )

    panelboard_values = analyze_panelboard_section_text(
        section_text,
        pages[start_idx]["page_num"],
        "whole_section_line_scan"
    )

    square_d_allowed = scan_square_d_allowed(
        split_nonempty_lines(section_text),
        pages[start_idx]["page_num"],
        "whole_section_line_scan"
    )

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
    if square_d_allowed is False:
        print("\n[SpecAnalysis] BAIL CONDITION: Square D / SQD / Schneider token not found.")
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