#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#V1

import os
import re
import sys
import json
import time
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
INPUT_PDF = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/S6.pdf").expanduser()
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
# Spec Analyzer Models
# ============================================================

@dataclass
class EvidenceItem:
    page: Optional[int]
    source: str
    snippet: str


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
# Spec Analyzer Config
# ============================================================

PANELBOARD_SECTION_NUMBER_PATTERNS = [
    re.compile(r"\b26\s*[-.]?\s*24\s*[-.]?\s*16(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\b262416(?:\.\d+)?\b", re.IGNORECASE),
]

BROADER_DISTRIBUTION_SECTION_PATTERNS = [
    re.compile(r"\b26\s*[-.]?\s*24\s*[-.]?\s*00\b", re.IGNORECASE),
    re.compile(r"\b262400\b", re.IGNORECASE),
    re.compile(r"\belectrical\s+distribution\s+equipment\b", re.IGNORECASE),
]

PANELBOARD_TITLE_PATTERNS = [
    re.compile(r"\bpanelboards?\b", re.IGNORECASE),
    re.compile(r"\blow[\s-]?voltage\s+panelboards?\b", re.IGNORECASE),
]

PREFERRED_SUBSECTION_PATTERNS = [
    re.compile(r"\bbranch\s+panelboards?\b", re.IGNORECASE),
    re.compile(r"\blighting\s+and\s+appliance\s+panelboards?\b", re.IGNORECASE),
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

SECTION_HEADER_LINE_PATTERNS = [
    re.compile(
        r"^\s*(section\s+)?26\s*[-.]?\s*24\s*[-.]?\s*16(?:\.\d+)?\b.*(?:low\s*[- ]?\s*voltage\s+)?panelboards?\b",
        re.IGNORECASE
    ),
    re.compile(
        r"^\s*(section\s+)?262416(?:\.\d+)?\b.*(?:low\s*[- ]?\s*voltage\s+)?panelboards?\b",
        re.IGNORECASE
    ),
    re.compile(
        r"^\s*(section\s+)?26\s*[-.]?\s*24\s*[-.]?\s*00\b.*electrical\s+distribution\s+equipment",
        re.IGNORECASE
    ),
    re.compile(
        r"^\s*(section\s+)?262400\b.*electrical\s+distribution\s+equipment",
        re.IGNORECASE
    ),
]

NEXT_MAJOR_SECTION_PATTERN = re.compile(
    r"^\s*(section\s+)?26\s*[-.]?\s*\d{2}\s*[-.]?\s*\d{2}(?:\.\d+)?\b",
    re.IGNORECASE
)

SUBSECTION_HEADER_PATTERNS = [
    re.compile(r"^\s*\d+[\.\)]?\s*branch\s+panelboards?\b", re.IGNORECASE),
    re.compile(r"^\s*branch\s+panelboards?\b", re.IGNORECASE),
    re.compile(r"^\s*\d+[\.\)]?\s*lighting\s+and\s+appliance\s+panelboards?\b", re.IGNORECASE),
    re.compile(r"^\s*lighting\s+and\s+appliance\s+panelboards?\b", re.IGNORECASE),
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
# PDF Text Extraction (pypdfium2)
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

            # Whole page
            full_text = textpage.get_text_bounded() or ""

            # Regions
            top_cut = height * 0.78
            mid_top = height * 0.40
            bottom_cut = height * 0.22

            left_1 = 0.0
            left_2 = width / 3.0
            left_3 = 2.0 * width / 3.0
            right = width

            top_band = _extract_region_text(
                textpage, left=0, bottom=top_cut, right=width, top=height
            )
            middle_band = _extract_region_text(
                textpage, left=0, bottom=bottom_cut, right=width, top=top_cut
            )
            bottom_band = _extract_region_text(
                textpage, left=0, bottom=0, right=width, top=bottom_cut
            )

            top_left = _extract_region_text(
                textpage, left=left_1, bottom=top_cut, right=left_2, top=height
            )
            top_center = _extract_region_text(
                textpage, left=left_2, bottom=top_cut, right=left_3, top=height
            )
            top_right = _extract_region_text(
                textpage, left=left_3, bottom=top_cut, right=right, top=height
            )

            bottom_left = _extract_region_text(
                textpage, left=left_1, bottom=0, right=left_2, top=bottom_cut
            )
            bottom_center = _extract_region_text(
                textpage, left=left_2, bottom=0, right=left_3, top=bottom_cut
            )
            bottom_right = _extract_region_text(
                textpage, left=left_3, bottom=0, right=right, top=bottom_cut
            )

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

# ============================================================
# Text Normalization Helpers
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


def tokenize(text: str) -> List[str]:
    return normalize_for_matching(text).split()


def find_keyword_windows(original_text: str, terms: List[str], window_chars: int = 180) -> List[str]:
    out = []
    text_lower = (original_text or "").lower()

    for term in terms:
        start = 0
        term_lower = term.lower()

        while True:
            idx = text_lower.find(term_lower, start)
            if idx == -1:
                break

            s = max(0, idx - window_chars)
            e = min(len(original_text), idx + len(term) + window_chars)
            out.append(original_text[s:e].strip())
            start = idx + len(term_lower)

    return out


def sentenceish_chunks(text: str) -> List[str]:
    text = normalize_preserve_lines(text)
    raw_parts = re.split(r"[\n\r]+", text)
    parts = []

    for part in raw_parts:
        p = part.strip()
        if not p:
            continue

        subparts = re.split(r"(?<=[.;:])\s{1,}", p)
        for sp in subparts:
            s = sp.strip()
            if s:
                parts.append(s)

    return parts


# ============================================================
# Section Finding
# ============================================================

def score_page_for_panelboard_relevance(text: str) -> float:
    score = 0.0
    t = text or ""

    if any(p.search(t) for p in PANELBOARD_SECTION_NUMBER_PATTERNS):
        score += 10.0

    if any(p.search(t) for p in PANELBOARD_TITLE_PATTERNS):
        score += 8.0

    if any(p.search(t) for p in PREFERRED_SUBSECTION_PATTERNS):
        score += 7.0

    if any(p.search(t) for p in BROADER_DISTRIBUTION_SECTION_PATTERNS):
        score += 5.0

    support_terms = [
        "manufacturers",
        "materials",
        "construction",
        "products",
        "bus",
        "buss",
        "bolt-on",
        "bolt on",
        "bolt-in",
        "bolt in",
        "plug-in",
        "plug in",
        "plug-on",
        "plug on",
        "series rated",
        "fully rated",
    ]

    norm = normalize_for_matching(t)
    for term in support_terms:
        if normalize_for_matching(term) in norm:
            score += 1.0

    return score


def get_section_header_line(page_dict: dict) -> Optional[str]:
    """
    Prefer top-band / top-corner matches over whole-page text.
    """
    candidate_regions = [
        page_dict["regions"]["top_center"],
        page_dict["regions"]["top_left"],
        page_dict["regions"]["top_right"],
        page_dict["regions"]["top_band"],
        page_dict["text"],
    ]

    header_patterns = [
        r"section\s+26\s+24\s+16(?:\.\d+)?\s+low\s+voltage\s+panelboards",
        r"section\s+26\s+24\s+16(?:\.\d+)?\s+panelboards",
        r"26\s+24\s+16(?:\.\d+)?\s+low\s+voltage\s+panelboards",
        r"26\s+24\s+16(?:\.\d+)?\s+panelboards",
        r"262416(?:\.\d+)?\s+low\s+voltage\s+panelboards",
        r"262416(?:\.\d+)?\s+panelboards",
    ]

    for region_text in candidate_regions:
        norm = normalize_for_matching(region_text)
        for pat in header_patterns:
            m = re.search(pat, norm, re.IGNORECASE)
            if m:
                return m.group(0)

    return None

def find_best_start_page(pages: List[dict]) -> Tuple[Optional[int], float, str]:
    best_idx = None
    best_score = -1.0
    best_mode = "unknown"

    for idx, page in enumerate(pages):
        text = page["text"]
        score = score_page_for_panelboard_relevance(text)

        has_dedicated = (
            any(p.search(text) for p in PANELBOARD_SECTION_NUMBER_PATTERNS)
            and any(p.search(text) for p in PANELBOARD_TITLE_PATTERNS)
        )
        has_broader = any(p.search(text) for p in BROADER_DISTRIBUTION_SECTION_PATTERNS)

        if has_dedicated:
            score += 8.0
            mode = "dedicated_panelboard_section"
        elif has_broader:
            mode = "distribution_equipment_section"
        else:
            mode = "unknown"

        if score > best_score:
            best_idx = idx
            best_score = score
            best_mode = mode

    if best_score < 8.0:
        return None, best_score, "unknown"

    return best_idx, best_score, best_mode

def score_section_header_candidate(page_dict: dict) -> int:
    """
    Layout-aware header scoring.
    Strongly rewards actual section-opening pages and penalizes
    nearby reference pages or previous-section spillover pages.
    """
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

    # ------------------------------------------------------------
    # Strong positives: actual section header in top regions
    # ------------------------------------------------------------

    if any("section 26 24 16" in r or "section 262416" in r for r in top_regions):
        score += 30

    if any("low voltage panelboards" in r for r in top_regions):
        score += 22
    elif any("panelboards" in r for r in top_regions):
        score += 10

    if any("26 24 16" in r or "262416" in r for r in top_regions):
        score += 16

    # Exact section-title pattern gets extra credit
    exact_header_patterns = [
        r"section\s+26\s+24\s+16(?:\.\d+)?\s+low\s+voltage\s+panelboards",
        r"section\s+26\s+24\s+16(?:\.\d+)?\s+panelboards",
        r"26\s+24\s+16(?:\.\d+)?\s+low\s+voltage\s+panelboards",
        r"26\s+24\s+16(?:\.\d+)?\s+panelboards",
    ]
    if any(re.search(pat, top_band, re.IGNORECASE) for pat in exact_header_patterns):
        score += 24

    # Very strong "first page of section" marker
    if re.search(r"\b26\s+24\s+16(?:\.\d+)?\s*[-–—]\s*1\b", full_text):
        score += 35

    # True opener pages usually immediately contain these
    opening_terms = [
        "part 1 general",
        "part 1  general",
        "1 1 related documents",
        "1 1 summary",
        "1 1 description",
        "1 1 references",
    ]
    opening_hits = sum(1 for term in opening_terms if term in full_text)
    score += min(opening_hits * 10, 25)

    # ------------------------------------------------------------
    # Negatives: footer/reference/header carryover
    # ------------------------------------------------------------

    # Footer mention of section title/number is weaker
    if any("section 26 24 16" in r or "section 262416" in r for r in bottom_regions):
        score -= 12

    if any("low voltage panelboards" in r for r in bottom_regions):
        score -= 10

    # Many different section refs = probably TOC/reference-like
    all_section_refs = re.findall(r"\b26\s+\d{2}\s+\d{2}(?:\.\d+)?\b", full_text)
    unique_section_refs = set(all_section_refs)
    if len(unique_section_refs) >= 3:
        score -= 24
    elif len(unique_section_refs) == 2:
        score -= 10

    toc_terms = [
        "table of contents",
        "contents",
        "division 26",
        "building standards",
        "technical standards",
    ]
    if any(term in full_text for term in toc_terms):
        score -= 18

    # ------------------------------------------------------------
    # Negatives: looks like previous/deep body section instead of opener
    # ------------------------------------------------------------

    deep_body_terms = [
        "circuit breakers",
        "bussing shall",
        "fully rated",
        "series rated",
        "manufacturers",
        "products",
        "construction",
        "fabrication and features",
        "surge protective device",
        "ground fault circuit interrupter",
    ]
    deep_body_hits = sum(1 for term in deep_body_terms if term in middle_band)
    score -= min(deep_body_hits * 2, 14)

    # Specific penalty for previous-section spillover language that fooled page 213
    previous_section_terms = [
        "switchboards",
        "switchboard",
        "26 24 13",
        "26 24 13 21",
    ]
    prev_hits = sum(1 for term in previous_section_terms if term in full_text)
    score -= min(prev_hits * 8, 24)

    return score

def refine_to_section_start(pages: List[dict], best_idx: int) -> int:
    """
    Find the best actual header page in the lookback window using layout-aware scoring.
    """
    lookback = max(0, best_idx - 12)
    candidates = []

    for idx in range(lookback, best_idx + 1):
        full_text = normalize_for_matching(pages[idx]["text"])
        top_band = normalize_for_matching(pages[idx]["regions"]["top_band"])
        bottom_band = normalize_for_matching(pages[idx]["regions"]["bottom_band"])

        has_section_num = ("26 24 16" in full_text or "262416" in full_text)
        has_panelboards = ("panelboards" in full_text)

        if not (has_section_num and has_panelboards):
            continue

        score = score_section_header_candidate(pages[idx])

        candidates.append({
            "idx": idx,
            "page_num": pages[idx]["page_num"],
            "score": score,
            "top_band": top_band[:250],
            "bottom_band": bottom_band[:250],
        })

    if not candidates:
        return best_idx

    candidates.sort(key=lambda x: (x["score"], x["idx"]))
    return candidates[-1]["idx"]

def collect_multi_page_section(pages: List[dict], start_idx: int, mode: str) -> Tuple[str, int]:
    collected = []
    end_idx = start_idx

    for idx in range(start_idx, len(pages)):
        page_text = normalize_preserve_lines(pages[idx]["text"])
        page_match = normalize_for_matching(page_text)

        if idx > start_idx:
            # Hard stop for the next known section in this family
            if (
                "section 26 24 19" in page_match or
                "section 262419" in page_match or
                re.search(r"\b26\s+24\s+19\b", page_match)
            ):
                break

            # Generic stop at next Division 26 section,
            # but only if it is clearly NOT still our own section.
            next_section_match = re.search(r"\bsection\s+26\s+\d{2}\s+\d{2}(?:\.\d+)?\b", page_match)
            if next_section_match:
                if not ("26 24 16" in page_match or "262416" in page_match):
                    break

        collected.append(page_text)
        end_idx = idx

        # generous cap for now
        if (idx - start_idx) >= 14:
            break

    return "\n".join(collected), end_idx

def isolate_preferred_panelboard_subsection(section_text: str) -> Tuple[str, str]:
    lines = normalize_preserve_lines(section_text).splitlines()

    headers = []
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        for pat in SUBSECTION_HEADER_PATTERNS:
            if pat.search(line_stripped):
                headers.append((i, line_stripped))
                break

    if not headers:
        return section_text, "whole_section"

    preferred_order = [
        re.compile(r"\bbranch\s+panelboards?\b", re.IGNORECASE),
        re.compile(r"\blighting\s+and\s+appliance\s+panelboards?\b", re.IGNORECASE),
    ]

    target_idx = None
    target_label = None

    for pref_pat in preferred_order:
        for i, label in headers:
            if pref_pat.search(label):
                target_idx = i
                target_label = label
                break
        if target_idx is not None:
            break

    if target_idx is None:
        return section_text, "whole_section"

    next_idx = len(lines)
    for i, label in headers:
        if i > target_idx:
            next_idx = i
            break

    focused = "\n".join(lines[target_idx:next_idx]).strip()
    return focused if focused else section_text, f"subsection:{target_label}"


# ============================================================
# Detection Helpers
# ============================================================

def _make_evidence(page: Optional[int], source: str, snippet: str) -> EvidenceItem:
    return EvidenceItem(page=page, source=source, snippet=snippet.strip())


def _contains_any_normalized(text: str, phrases: List[str]) -> Optional[str]:
    norm = normalize_for_matching(text)
    for phrase in phrases:
        p = normalize_for_matching(phrase)
        if p in norm:
            return phrase
    return None


def _token_near_token(text: str, group_a: List[str], group_b: List[str], max_gap: int = 4) -> Optional[str]:
    toks = tokenize(text)
    if not toks:
        return None

    a_norm = {normalize_for_matching(x) for x in group_a}
    b_norm = {normalize_for_matching(x) for x in group_b}

    for i, tok in enumerate(toks):
        if tok in a_norm:
            start = max(0, i - max_gap)
            end = min(len(toks), i + max_gap + 1)

            for j in range(start, end):
                if toks[j] in b_norm:
                    return " ".join(toks[max(0, i - 3):min(len(toks), i + 4)])

    return None


# ============================================================
# Detectors
# ============================================================

def detect_square_d_allowed(text: str, page: Optional[int], source: str) -> DetectedValue:
    lines = [ln.strip() for ln in normalize_preserve_lines(text).splitlines()]
    lines = [ln for ln in lines if ln]

    sqd_tokens = [normalize_for_matching(t) for t in MANUFACTURER_TOKENS]
    competitor_tokens = [
        normalize_for_matching(t)
        for t in COMPETITOR_MANUFACTURER_TOKENS
        if normalize_for_matching(t) != "ge"
    ]

    # ------------------------------------------------------------
    # Strong start patterns for actual manufacturer lists
    # ------------------------------------------------------------
    strong_start_patterns = [
        re.compile(r"^\d+(?:\.\d+)?\s+manufacturers\s*$", re.IGNORECASE),   # 2.1 MANUFACTURERS
        re.compile(r"^manufacturers\s*$", re.IGNORECASE),                   # MANUFACTURERS
        re.compile(r"^manufacturers\s*:", re.IGNORECASE),                   # Manufacturers:
        re.compile(r"^[A-Z]\.\s*manufacturers\s*:", re.IGNORECASE),         # A. Manufacturers:
        re.compile(r"^[A-Z]\.\s*acceptable manufacturers\s*:", re.IGNORECASE),
        re.compile(r"^[A-Z]\.\s*approved manufacturers\s*:", re.IGNORECASE),
        re.compile(r"^[A-Z]\.\s*basis of design manufacturers\s*:", re.IGNORECASE),
        re.compile(r"^acceptable manufacturers\s*:?", re.IGNORECASE),
        re.compile(r"^approved manufacturers\s*:?", re.IGNORECASE),
        re.compile(r"^basis of design manufacturers\s*:?", re.IGNORECASE),
        re.compile(r"^basis-of-design manufacturers\s*:?", re.IGNORECASE),
    ]

    context_follow_patterns = [
        re.compile(r"subject to compliance", re.IGNORECASE),
        re.compile(r"provide products by one of the following", re.IGNORECASE),
        re.compile(r"one of the following", re.IGNORECASE),
        re.compile(r"or equal", re.IGNORECASE),
    ]

    # ------------------------------------------------------------
    # Find the real start of a manufacturer list
    # ------------------------------------------------------------
    start_idx = None

    for i, line in enumerate(lines):
        line_clean = line.strip()
        norm_line = normalize_for_matching(line_clean)

        # Reject obvious false positives like NEMA references
        if "national electrical manufacturers association" in norm_line:
            continue

        # Strong direct match
        if any(p.search(line_clean) for p in strong_start_patterns):
            start_idx = i
            break

        # Also allow a manufacturer line if the next few lines clearly look like a vendor list block
        if "manufacturers" in norm_line and "national electrical manufacturers association" not in norm_line:
            lookahead = "\n".join(lines[i:i+6])
            norm_lookahead = normalize_for_matching(lookahead)

            if any(p.search(lookahead) for p in context_follow_patterns):
                start_idx = i
                break

    # No actual manufacturer list found -> allow by default
    if start_idx is None:
        return DetectedValue(
            value=True,
            matched_phrase="no manufacturer list found; defaulting to allowed",
            evidence=[]
        )

    # ------------------------------------------------------------
    # Take the next X raw lines starting from the actual manufacturer list
    # ------------------------------------------------------------
    max_forward_lines = 22
    raw_block_lines = lines[start_idx:start_idx + max_forward_lines]

    trimmed_block = []
    for idx, line in enumerate(raw_block_lines):
        norm_line = normalize_for_matching(line)
        trimmed_block.append(line)

        # After we have captured enough of the list, stop on a clear new subsection
        if idx >= 5:
            if re.match(r"^\d+\.\d+\s+[A-Z]", line):   # e.g. 2.2 PANELBOARDS
                trimmed_block.pop()  # remove the new subsection line
                break

            if any(term in norm_line for term in [
                "execution",
                "installation",
                "fabrication and features",
                "short circuit current rating",
                "panelboard short circuit current rating",
            ]):
                trimmed_block.pop()
                break

    block_text = "\n".join(trimmed_block)
    norm_block = normalize_for_matching(block_text)

    sqd_found = any(token in norm_block for token in sqd_tokens)
    competitor_found = [token for token in competitor_tokens if token in norm_block]

    if sqd_found:
        return DetectedValue(
            value=True,
            matched_phrase="square d found in manufacturer block",
            evidence=[_make_evidence(page, source, block_text[:1200])]
        )

    if competitor_found:
        return DetectedValue(
            value=False,
            matched_phrase="competitors listed without square d",
            evidence=[_make_evidence(page, source, block_text[:1200])]
        )

    return DetectedValue(
        value=True,
        matched_phrase="manufacturer context found but no explicit manufacturer list; defaulting to allowed",
        evidence=[_make_evidence(page, source, block_text[:1200])]
    )

def detect_bussing_material(text: str, page: Optional[int], source: str) -> DetectedValue:
    chunks = sentenceish_chunks(text)

    copper_hits = []
    aluminum_hits = []
    bus_terms = ["bus", "buss", "bussing"]

    for chunk in chunks:
        if _token_near_token(chunk, ["copper", "cu"], bus_terms, max_gap=6):
            copper_hits.append(chunk)

        if _token_near_token(chunk, ["aluminum", "aluminium", "al"], bus_terms, max_gap=6):
            aluminum_hits.append(chunk)

        if _contains_any_normalized(chunk, ["copper bus", "copper buss", "plated copper", "copper bussing"]):
            copper_hits.append(chunk)

        if _contains_any_normalized(chunk, ["aluminum bus", "aluminum buss", "aluminum bussing"]):
            aluminum_hits.append(chunk)

    if copper_hits and not aluminum_hits:
        return DetectedValue(
            value="COPPER",
            matched_phrase="copper near bus/buss",
            evidence=[_make_evidence(page, source, s) for s in copper_hits[:5]]
        )

    if aluminum_hits and not copper_hits:
        return DetectedValue(
            value="ALUMINUM",
            matched_phrase="aluminum near bus/buss",
            evidence=[_make_evidence(page, source, s) for s in aluminum_hits[:5]]
        )

    if copper_hits and aluminum_hits:
        if len(copper_hits) >= len(aluminum_hits):
            return DetectedValue(
                value="COPPER",
                matched_phrase="conflict resolved by evidence count",
                evidence=[_make_evidence(page, source, s) for s in copper_hits[:5]]
            )

        return DetectedValue(
            value="ALUMINUM",
            matched_phrase="conflict resolved by evidence count",
            evidence=[_make_evidence(page, source, s) for s in aluminum_hits[:5]]
        )

    return DetectedValue(value=None, matched_phrase=None, evidence=[])

def detect_breaker_mounting(text: str, page: Optional[int], source: str) -> DetectedValue:
    chunks = sentenceish_chunks(text)

    plug_phrases = [
        "plug on",
        "plug-on",
        "plug in",
        "plug-in",
        "plugin",
        "plug in circuit breaker type",
        "plug on type",
    ]

    bolt_phrases = [
        "bolt on",
        "bolt-on",
        "bolt in",
        "bolt-in",
        "boltin",
        "bolt on type",
        "bolt on breakers",
        "bolt-on breakers",
        "limited to bolt-on",
        "limited to bolt on",
    ]

    plug_hits = []
    bolt_hits = []

    for chunk in chunks:
        if _contains_any_normalized(chunk, plug_phrases):
            plug_hits.append(chunk)
        if _contains_any_normalized(chunk, bolt_phrases):
            bolt_hits.append(chunk)

    # Per your rule: if both appear, default to bolt-on
    if bolt_hits:
        return DetectedValue(
            value=False,
            matched_phrase="bolt-on/bolt-in language found",
            evidence=[_make_evidence(page, source, s) for s in bolt_hits[:5]]
        )

    if plug_hits:
        return DetectedValue(
            value=True,
            matched_phrase="plug-on/plug-in language found",
            evidence=[_make_evidence(page, source, s) for s in plug_hits[:5]]
        )

    return DetectedValue(value=None, matched_phrase=None, evidence=[])

def detect_rating_type(text: str, page: Optional[int], source: str) -> DetectedValue:
    chunks = sentenceish_chunks(text)

    # Strong negatives against series rating -> FULLY_RATED
    fully_priority_patterns = [
        "series rated prohibited",
        "series-rated prohibited",
        "series rated panelboards are prohibited",
        "series-rated panelboards are prohibited",
        "series rating shall not be used",
        "series ratings shall not be used",
        "series rated not allowed",
        "series rating not allowed",
        "series-rated not allowed",
        "series rated are not permitted",
        "series-rated are not permitted",
        "series rated is not permitted",
        "series-rated is not permitted",
        "series rated not permitted",
        "series-rated not permitted",
        "series rated panelboards are not permitted",
        "series-rated panelboards are not permitted",
        "fully rated",
        "bussing shall be fully rated",
        "bussing is fully rated",
    ]

    # True positive permission language -> SERIES_RATED
    series_allowed_patterns = [
        "series rated allowed",
        "series-rated allowed",
        "series rated permitted",
        "series-rated permitted",
        "series ratings permitted",
        "series rating permitted",
        "series rated acceptable",
        "series rating acceptable",
        "series-rated acceptable",
        "series combination ratings are permitted",
        "series combination rating permitted",
    ]

    norm_text = normalize_for_matching(text)

    # Pass 1: exact high-priority negative matches
    for phrase in fully_priority_patterns:
        if normalize_for_matching(phrase) in norm_text:
            snippets = find_keyword_windows(text, [phrase], window_chars=150)
            ev = [_make_evidence(page, source, s) for s in snippets[:5]]
            return DetectedValue(
                value="FULLY_RATED",
                matched_phrase=phrase,
                evidence=ev
            )

    # Pass 2: exact explicit allowance matches
    for phrase in series_allowed_patterns:
        if normalize_for_matching(phrase) in norm_text:
            snippets = find_keyword_windows(text, [phrase], window_chars=150)
            ev = [_make_evidence(page, source, s) for s in snippets[:5]]
            return DetectedValue(
                value="SERIES_RATED",
                matched_phrase=phrase,
                evidence=ev
            )

    # Pass 3: chunk-level logic
    fully_hits = []
    series_hits = []

    negative_words = [
        "prohibited",
        "not allowed",
        "shall not",
        "not be used",
        "not permitted",
        "are not permitted",
        "is not permitted",
    ]

    positive_words = [
        "allowed",
        "permitted",
        "acceptable",
    ]

    for chunk in chunks:
        n = normalize_for_matching(chunk)

        if "fully rated" in n:
            fully_hits.append(chunk)
            continue

        has_series_phrase = ("series rated" in n or "series rating" in n or "series rated panelboards" in n)

        if has_series_phrase and any(word in n for word in negative_words):
            fully_hits.append(chunk)
            continue

        if has_series_phrase and any(word in n for word in positive_words):
            # protect against "not permitted" containing "permitted"
            if not any(word in n for word in negative_words):
                series_hits.append(chunk)

    if fully_hits:
        return DetectedValue(
            value="FULLY_RATED",
            matched_phrase="fully rated / series prohibited language found",
            evidence=[_make_evidence(page, source, s) for s in fully_hits[:5]]
        )

    if series_hits:
        return DetectedValue(
            value="SERIES_RATED",
            matched_phrase="series rated allowed language found",
            evidence=[_make_evidence(page, source, s) for s in series_hits[:5]]
        )

    return DetectedValue(value=None, matched_phrase=None, evidence=[])

# ============================================================
# Main Spec Analysis Functions
# ============================================================

def analyze_panelboard_section_text(section_text: str, page: Optional[int], source_label: str) -> Dict[str, DetectedValue]:
    return {
        "bussing_material": detect_bussing_material(section_text, page, source_label),
        "allow_plug_on_breakers": detect_breaker_mounting(section_text, page, source_label),
        "rating_type": detect_rating_type(section_text, page, source_label),
    }


def analyze_pdf_panelboard_specs(pdf_path: str) -> SpecAnalysisResult:
    pages = extract_pdf_pages_text(pdf_path)

    best_idx, best_score, mode_guess = find_best_start_page(pages)
    if best_idx is None:
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
                        "score": score_page_for_panelboard_relevance(p["text"])
                    }
                    for p in pages
                ]
            }
        )

    start_idx = refine_to_section_start(pages, best_idx)

    debug_candidates = []
    lookback = max(0, best_idx - 12)
    for idx in range(lookback, best_idx + 1):
        text_match = normalize_for_matching(pages[idx]["text"])
        if ("26 24 16" in text_match or "262416" in text_match) and ("panelboards" in text_match):
            debug_candidates.append({
                "page_num": pages[idx]["page_num"],
                "header_score": score_section_header_candidate(pages[idx]),
                "top_band_preview": normalize_for_matching(pages[idx]["regions"]["top_band"])[:220],
                "bottom_band_preview": normalize_for_matching(pages[idx]["regions"]["bottom_band"])[:220],
            })

    section_text, end_idx = collect_multi_page_section(pages, start_idx, mode_guess)
    header_line = get_section_header_line(pages[start_idx])
    focused_text, focused_source = isolate_preferred_panelboard_subsection(section_text)

    panelboard_values = analyze_panelboard_section_text(
        focused_text,
        pages[start_idx]["page_num"],
        focused_source
    )

    square_d_allowed = detect_square_d_allowed(
        section_text,
        pages[start_idx]["page_num"],
        "section_manufacturer_scan"
    )

    section_match = SectionMatch(
        mode=mode_guess,
        start_page=pages[start_idx]["page_num"],
        end_page=pages[end_idx]["page_num"],
        score=best_score,
        title=header_line
    )

    return SpecAnalysisResult(
        section_match=section_match,
        panelboards=panelboard_values,
        job_flags={"square_d_allowed": square_d_allowed},
        debug={
            "best_start_page_index": best_idx,
            "header_candidates": debug_candidates,
            "refined_start_page_index": start_idx,
            "best_start_page_num": pages[best_idx]["page_num"],
            "refined_start_page_num": pages[start_idx]["page_num"],
            "mode_guess": mode_guess,
            "focused_source": focused_source,
            "section_preview": section_text[:1200],
            "focused_preview": focused_text[:1200],
            "page_scores": [
                {
                    "page_num": p["page_num"],
                    "score": score_page_for_panelboard_relevance(p["text"])
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
    panelboards = spec_result_dict.get("panelboards", {})
    job_flags = spec_result_dict.get("job_flags", {})
    section_match = spec_result_dict.get("section_match", {})

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

    # ---- 1) Run spec analysis ----
    print("\n[SpecAnalysis] starting...")
    spec_result = analyze_pdf_panelboard_specs(str(INPUT_PDF))
    spec_result_dict = ensure_json_safe(spec_result.to_dict())

    # ---- 2) Merge into defaults ----
    merged_defaults = apply_spec_analysis_to_defaults(default_ui_overrides, spec_result)

    # ---- 3) Build output payloads ----
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

    # ---- 4) Console printout ----
    print_detection_summary(spec_result_dict, merged_defaults)

    square_d_allowed = (
        ((spec_result_dict.get("job_flags") or {}).get("square_d_allowed") or {}).get("value")
    )
    if square_d_allowed is False:
        print("\n[SpecAnalysis] BAIL CONDITION: Square D / SQD / Schneider token not found.")
    else:
        print("\n[SpecAnalysis] Square D allowed. Defaults can be used.")

    # ---- 5) Write files ----
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