# AnchoringClasses/BreakerFooterFinder.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import os
import cv2
import numpy as np
from difflib import SequenceMatcher
from contextlib import contextmanager
import re
import time
import fcntl

# Same values you use in HeaderBandScanner
_HDR_OCR_SCALE = 2.0
_HDR_OCR_ALLOWLIST = None
_HDR_MIN_CONF = 0.35

@dataclass
class FooterResult:
    footer_y: Optional[int]
    token_y: Optional[int]
    token_val: Optional[int]
    panel_size: Optional[int]
    dbg_marks: List[Tuple[int, str]]  # (y, label) for overlays
    vlines_x: List[int] = field(default_factory=list)              # all vertical-line centers (gray coords)
    cct_cols: List[Tuple[int, int]] = field(default_factory=list)  # [(xl, xr), ...] in gray coords


class BreakerFooterFinder:
    """
    Footer finder (work in progress).

    Step 1: mirror the parser logic to crop the HEADER BAND between
    header_y and header_bottom_y.
    Step 2: run the exact same vertical-line detector used by the parser
    inside this band to get column separators.
    Step 3: run the same OCR + header scoring as the parser, but only
            return the CKT/CCT column(s) as cct_cols.
    """

    # panel size 
    PANEL_FOOTER_MAP: Dict[int, set[int]] = {
    84: set(range(73, 85)),
    72: set(range(67, 73)),
    66: set(range(55, 67)),
    54: set(range(43, 55)),
    42: set(range(31, 43)),
    30: set(range(19, 31)),
    18: set(range(1, 19)),
}

    # search largest size first
    PANEL_SIZE_ORDER: List[int] = [84, 72, 66, 54, 42, 30, 18]
    MAX_CONTINUED_SECTION_CKT = 168

    # flat set of all token values we care about
    FOOTER_TOKEN_VALUES = set().union(*PANEL_FOOTER_MAP.values())

    def __init__(
        self,
        reader,
        bottom_trim_frac: float = 0.15,
        top_trim_frac: float = 0.50,
        upscale_factor: float = 1.0,
        debug: bool = False,
    ):
        self.reader = reader
        self.bottom_trim_frac = float(bottom_trim_frac)
        self.top_trim_frac = float(top_trim_frac)
        self.upscale_factor = float(upscale_factor)
        self.debug = debug
        # optional: where to dump vertical mask + column crops / debug images
        self.debug_dir: Optional[str] = None

        # Inter-process OCR lock for footer OCR only.
        # This prevents concurrent EasyOCR footer calls from different worker
        # processes from crashing under load.
        self.ocr_lock_path = "/tmp/argus_footer_ocr.lock"
        self.ocr_lock_timeout_sec = 180

    def _extract_last_plausible_circuit_number(self, text: str) -> List[int]:
        """
        Extract circuit numbers from one OCR token.

        Normal numeric OCR keeps all plausible numbers:
            "42"       -> [42]
            "41 42"    -> [41, 42]

        Mixed identifiers or separated codes keep only the last plausible number:
            "123-45-42"     -> [42]
            "123-42-30"     -> [30]
            "123-42-1124"   -> [42]
            "BGBOI-112-42"  -> [42]

        Large values such as 1124 are rejected as a whole.
        They are never shortened into 24.
        """
        raw = str(text or "").strip()

        if not raw:
            return []

        all_numbers = [
            int(match.group())
            for match in re.finditer(r"\d+", raw)
        ]

        plausible_numbers = [
            value
            for value in all_numbers
            if 1 <= value <= self.MAX_CONTINUED_SECTION_CKT
        ]

        if not plausible_numbers:
            return []

        has_letters = bool(re.search(r"[A-Za-z]", raw))
        has_identifier_separator = bool(re.search(r"[-_/]", raw))

        # Mixed identifier/code: favor the final plausible numeric group.
        if has_letters or has_identifier_separator:
            return [plausible_numbers[-1]]

        # Clean numeric OCR: preserve the existing behavior.
        return plausible_numbers

    def _round_up_to_standard_panel_size(self, raw_size: int) -> Optional[int]:
        """
        Round a raw circuit count up to the next supported standard panel size.
        Examples:
          24 -> 30
          60 -> 66
          84 -> 84
        """
        if raw_size is None or raw_size <= 0:
            return None

        for size in sorted(self.PANEL_SIZE_ORDER):
            if raw_size <= size:
                return size

        return None

    def _candidate_has_numeric_footer_value(self, cand: Dict) -> bool:
        """
        True when a candidate contains at least one value that made it into footer_token_candidates.

        We are intentionally NOT requiring the raw OCR token to be digits-only because some
        drawings/OCR may produce mixed text around legitimate circuit numbers.
        """
        if not cand:
            return False

        try:
            val = int(cand.get("val"))
        except Exception:
            return False

        return val in self.FOOTER_TOKEN_VALUES


    def _candidate_is_words_only(self, cand: Dict) -> bool:
        """
        True when OCR text contains letters but no usable number.

        Examples:
        'NOTES'
        'TOTAL LOAD'
        'PANEL-L1' may contain a digit, so this is not words-only.
        """
        raw = str(cand.get("raw_text", "") or "").strip()

        if not raw:
            return False

        has_letter = bool(re.search(r"[A-Za-z]", raw))
        has_digit = bool(re.search(r"\d", raw))

        return has_letter and not has_digit


    def _row_group_footer_candidates(
        self,
        candidates: List[Dict],
        row_merge_px: float = 18.0,
    ) -> List[Dict]:
        """
        Group footer candidates into approximate visual rows based on y_page.

        A row may contain:
        - left circuit number
        - right circuit number
        - OCR junk on the same row
        """
        if not candidates:
            return []

        ordered = sorted(
            candidates,
            key=lambda c: (
                float(c.get("y_page", 0.0)),
                float(c.get("x_page", 0.0)),
                -float(c.get("conf", 0.0)),
            )
        )

        rows: List[Dict] = []

        for cand in ordered:
            y = float(cand.get("y_page", 0.0))

            if not rows:
                rows.append({
                    "y": y,
                    "candidates": [cand],
                    "values": {int(cand["val"])} if self._candidate_has_numeric_footer_value(cand) else set(),
                    "has_words_only": self._candidate_is_words_only(cand),
                })
                continue

            last = rows[-1]

            if abs(y - float(last["y"])) <= row_merge_px:
                last["candidates"].append(cand)

                if self._candidate_has_numeric_footer_value(cand):
                    last["values"].add(int(cand["val"]))

                if self._candidate_is_words_only(cand):
                    last["has_words_only"] = True

                # Average y keeps same-row left/right numbers together.
                last["y"] = sum(float(x.get("y_page", 0.0)) for x in last["candidates"]) / len(last["candidates"])

            else:
                rows.append({
                    "y": y,
                    "candidates": [cand],
                    "values": {int(cand["val"])} if self._candidate_has_numeric_footer_value(cand) else set(),
                    "has_words_only": self._candidate_is_words_only(cand),
                })

        return rows

    def _filter_normal_candidates_by_expected_row(
        self,
        candidates: List[Dict],
        header_bottom_y: int,
        max_value_delta: int = 4,
    ) -> List[Dict]:
        """
        Sanity-check normal panel circuit OCR against the candidate's estimated row.

        OCR remains authoritative:
        - The detected OCR value is never replaced.
        - The expected value is used only to reject extreme outliers.
        - A difference of up to 4 circuit numbers is allowed, which equals
          approximately two physical panel rows.

        Examples:
          expected=9, detected=9   -> keep
          expected=9, detected=13  -> keep (two-row tolerance)
          expected=9, detected=79  -> reject
        """
        original_candidates = list(candidates or [])

        if len(original_candidates) < 4:
            return original_candidates

        if not isinstance(header_bottom_y, (int, float)):
            return original_candidates

        # ------------------------------------------------------------
        # Estimate physical row pitch from reasonable OCR progression.
        #
        # Do this separately within each side. We do NOT require matching
        # left/right pairs.
        # ------------------------------------------------------------
        pitch_samples: List[float] = []

        for side in ("left", "right"):
            side_candidates = sorted(
                [
                    c for c in original_candidates
                    if str(c.get("side", "")).lower() == side
                ],
                key=lambda c: float(c.get("y_page", 0.0)),
            )

            for current, next_candidate in zip(
                side_candidates,
                side_candidates[1:],
            ):
                try:
                    current_val = int(current["val"])
                    next_val = int(next_candidate["val"])
                    current_y = float(current["y_page"])
                    next_y = float(next_candidate["y_page"])
                except (KeyError, TypeError, ValueError):
                    continue

                value_delta = next_val - current_val
                y_delta = next_y - current_y

                # Same-side circuit numbers normally increase by two per row.
                # Allow up to four missing physical rows when estimating pitch.
                if value_delta not in {2, 4, 6, 8}:
                    continue

                if y_delta <= 0:
                    continue

                rows_advanced = value_delta / 2.0
                pitch = y_delta / rows_advanced

                # Broad limits only prevent obviously invalid pitch samples.
                if 8.0 <= pitch <= 250.0:
                    pitch_samples.append(float(pitch))

        # If OCR is too sparse to establish row spacing, preserve the
        # existing behavior instead of guessing.
        if len(pitch_samples) < 2:
            if self.debug:
                print(
                    "[BreakerFooterFinder] Expected-row validation skipped: "
                    f"only {len(pitch_samples)} usable pitch samples."
                )
            return original_candidates

        row_pitch = float(np.median(pitch_samples))
        first_row_center_y = float(header_bottom_y) + (row_pitch * 0.5)

        validated_candidates: List[Dict] = []

        if self.debug:
            print(
                "[BreakerFooterFinder] Expected-row validation: "
                f"row_pitch={row_pitch:.1f}, "
                f"first_row_center_y={first_row_center_y:.1f}, "
                f"max_value_delta={max_value_delta}"
            )

        for candidate in original_candidates:
            try:
                detected_value = int(candidate["val"])
                candidate_y = float(candidate["y_page"])
                side = str(candidate.get("side", "")).lower()
            except (KeyError, TypeError, ValueError):
                continue

            if side not in {"left", "right"}:
                validated_candidates.append(candidate)
                continue

            estimated_row_index = int(
                round(
                    (candidate_y - first_row_center_y)
                    / row_pitch
                )
            )

            if estimated_row_index < 0:
                if self.debug:
                    print(
                        "[BreakerFooterFinder] Expected-row REJECT: "
                        f"val={detected_value}, y={candidate_y:.1f}, "
                        f"side={side}, reason=above_first_body_row"
                    )
                continue

            if side == "left":
                expected_value = 1 + (estimated_row_index * 2)
            else:
                expected_value = 2 + (estimated_row_index * 2)

            value_delta = abs(detected_value - expected_value)
            is_valid = value_delta <= max_value_delta

            if self.debug:
                decision = "KEEP" if is_valid else "REJECT"

                print(
                    "[BreakerFooterFinder] ROW CHECK: "
                    f"row={estimated_row_index + 1:02d}, "
                    f"side={side:<5}, "
                    f"expected={expected_value:>3}, "
                    f"detected={detected_value:>3}, "
                    f"delta={value_delta:>3} "
                    f"-> {decision}"
                )

            if not is_valid:
                continue

            # Preserve the detected OCR value. These extra fields are only
            # diagnostic and do not alter the existing bucket logic.
            validated = dict(candidate)
            validated["expected_value"] = int(expected_value)
            validated["expected_delta"] = int(value_delta)
            validated["estimated_row_index"] = int(estimated_row_index)

            validated_candidates.append(validated)

        if self.debug:
            print(
                "[BreakerFooterFinder] Expected-row validation complete: "
                f"kept={len(validated_candidates)}, "
                f"rejected={len(original_candidates) - len(validated_candidates)}"
            )

        return validated_candidates

    def _find_footer_candidate_cutoff_y(
        self,
        footer_token_candidates: List[Dict],
        all_numeric_candidates: List[Dict],
    ) -> Optional[Dict]:
        """
        Finds the point where normal circuit-number spacing appears to end.

        Rules:
        1. Use rows 3-7 as a spacing sample when possible.
        2. If a later gap is clearly larger than normal row spacing, cut off below that gap.
        3. If a row becomes words-only after numeric rows have started, cut off at that row.

        Returns:
        {
            "cutoff_y": float,
            "reason": "large_gap" | "words_only",
            ...
        }
        """
        # Use all_numeric_candidates for row detection because it includes 1..168,
        # not just the footer bucket values. This keeps spacing smarter on continued sections.
        candidate_pool = list(all_numeric_candidates or footer_token_candidates or [])

        if len(candidate_pool) < 6:
            return None

        rows = self._row_group_footer_candidates(candidate_pool)

        # Keep rows that actually have numeric values.
        numeric_rows = [r for r in rows if r.get("values")]

        if len(numeric_rows) < 6:
            return None

        # ------------------------------------------------------------
        # Rule A: words-only row after numeric sequence means junk starts.
        # ------------------------------------------------------------
        numeric_started = False

        for row in rows:
            if row.get("values"):
                numeric_started = True
                continue

            if numeric_started and row.get("has_words_only"):
                return {
                    "cutoff_y": float(row["y"]) - 1.0,
                    "reason": "words_only",
                    "row_y": float(row["y"]),
                }

        # ------------------------------------------------------------
        # Rule B: large whitespace gap compared to rows 3-7.
        # ------------------------------------------------------------
        # Use numeric rows only for spacing, because junk words can distort row grouping.
        numeric_rows = sorted(numeric_rows, key=lambda r: float(r["y"]))

        if len(numeric_rows) < 7:
            sample_rows = numeric_rows[2:]
        else:
            sample_rows = numeric_rows[2:7]

        if len(sample_rows) < 3:
            return None

        sample_gaps = [
            float(sample_rows[i + 1]["y"]) - float(sample_rows[i]["y"])
            for i in range(len(sample_rows) - 1)
        ]

        sample_gaps = [g for g in sample_gaps if 8.0 <= g <= 250.0]

        if len(sample_gaps) < 2:
            return None

        avg_gap = float(np.mean(sample_gaps))
        median_gap = float(np.median(sample_gaps))

        # Conservative threshold:
        # - at least 2.25x normal spacing
        # - and at least normal + 90px
        # This avoids triggering on slightly taller rows.
        large_gap_threshold = max(avg_gap * 2.25, median_gap + 90.0)

        for i in range(len(numeric_rows) - 1):
            current_row = numeric_rows[i]
            next_row = numeric_rows[i + 1]

            gap = float(next_row["y"]) - float(current_row["y"])

            if gap >= large_gap_threshold:
                return {
                    "cutoff_y": float(current_row["y"]) + max(10.0, median_gap * 0.50),
                    "reason": "large_gap",
                    "gap": float(gap),
                    "avg_gap": float(avg_gap),
                    "median_gap": float(median_gap),
                    "current_row_y": float(current_row["y"]),
                    "next_row_y": float(next_row["y"]),
                    "current_values": sorted(current_row.get("values", set())),
                    "next_values": sorted(next_row.get("values", set())),
                }

        return None

    def _infer_continued_section_start(
        self,
        all_numeric_candidates: List[Dict],
        page_height: int,
    ) -> Optional[int]:
        """
        Use only the first few numeric hits directly under the top of the analyzed
        crop to decide whether this is a continued section.

        Logic:
          - If we see at least 2 low-start values (<= 31), treat as normal mode.
          - Otherwise, if the first few visible values are high, use the smallest
            of those first visible high values as the continued-section start.

        This intentionally uses only the first few numbers under the top line,
        not numbers farther down the crop.
        """
        if not all_numeric_candidates:
            return None

        # Only inspect the very top portion of the analyzed body
        top_cutoff = page_height * 0.30
        top_candidates = [
            c for c in all_numeric_candidates
            if c["y_page"] <= top_cutoff
        ]

        if not top_candidates:
            top_candidates = sorted(
                all_numeric_candidates,
                key=lambda c: (c["y_page"], c["x_page"], -c["conf"])
            )[:16]

        if not top_candidates:
            return None

        # Reading order from the top of the crop
        top_candidates = sorted(
            top_candidates,
            key=lambda c: (c["y_page"], c["x_page"], -c["conf"])
        )

        # Keep the first few UNIQUE values only
        seen = set()
        first_vals = []
        for c in top_candidates:
            v = int(c["val"])
            if v not in seen:
                seen.add(v)
                first_vals.append(v)
            if len(first_vals) >= 8:
                break

        if not first_vals:
            return None

        low_vals = [v for v in first_vals if v <= 31]
        high_vals = [v for v in first_vals if v > 31]

        # Normal panel: we have clear low-start evidence near the top
        if len(low_vals) >= 2:
            return None

        # Continued section: use the first few visible high numbers near the top
        if high_vals:
            return min(high_vals)

        return None

    def _choose_continued_section_bottom_candidate(
        self,
        all_numeric_candidates: List[Dict],
        start_num: int,
        page_height: int,
    ) -> Optional[Dict]:
        """
        For continued sections, choose the real bottom anchor candidate using the
        lower part of the analyzed crop.

        We want the largest plausible number near the bottom, since that should
        be the footer number for the continued section.
        """
        if not all_numeric_candidates or start_num is None:
            return None

        bottom_cutoff = page_height * 0.55
        bottom_candidates = [
            c for c in all_numeric_candidates
            if c["y_page"] >= bottom_cutoff and c["val"] >= start_num
        ]

        if not bottom_candidates:
            bottom_candidates = [
                c for c in all_numeric_candidates
                if c["val"] >= start_num
            ]

        if not bottom_candidates:
            return None

        return max(
            bottom_candidates,
            key=lambda c: (c["val"], c["y_page"], c["conf"])
        )

    def _ensure_debug_dir(self, analyzer_result: Dict) -> str:
        """
        Resolve a debug directory path based on analyzer_result, similar
        to HeaderBandScanner._ensure_debug_dir().
        """
        src_dir = analyzer_result.get("src_dir") or os.path.dirname(
            analyzer_result.get("src_path", "") or "."
        )
        debug_dir = analyzer_result.get("debug_dir") or os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)
        return debug_dir

    @contextmanager
    def _ocr_lock(self):
        """
        Inter-process lock used only for footer OCR calls.
        This serializes EasyOCR footer calls across worker processes so
        GPU/native EasyOCR code is not hit concurrently.
        """
        lock_path = self.ocr_lock_path
        timeout_sec = max(1, int(self.ocr_lock_timeout_sec))

        lock_dir = os.path.dirname(lock_path) or "/tmp"
        os.makedirs(lock_dir, exist_ok=True)

        f = open(lock_path, "w")
        acquired = False
        start = time.time()

        try:
            while (time.time() - start) < timeout_sec:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    time.sleep(0.05)

            if not acquired:
                raise TimeoutError(
                    f"Timed out waiting for footer OCR lock after {timeout_sec}s"
                )

            yield

        finally:
            try:
                if acquired:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass


    def _safe_footer_readtext(
        self,
        img: np.ndarray,
        *,
        detail: int = 1,
        paragraph: bool = False,
        allowlist=None,
        mag_ratio: float = 1.0,
        contrast_ths: float = 0.05,
        adjust_contrast: float = 0.7,
        text_threshold: float = 0.4,
        low_text: float = 0.25,
        dbg_label: str = "",
    ):
        """
        Wrapper for footer OCR only.
        Serializes OCR across processes to avoid EasyOCR/PyTorch segfaults.
        """
        if self.reader is None:
            return []

        with self._ocr_lock():
            if self.debug:
                print(f"[BreakerFooterFinder] OCR LOCK ACQUIRED for {dbg_label}")

            out = self.reader.readtext(
                img,
                detail=detail,
                paragraph=paragraph,
                allowlist=allowlist,
                mag_ratio=mag_ratio,
                contrast_ths=contrast_ths,
                adjust_contrast=adjust_contrast,
                text_threshold=text_threshold,
                low_text=low_text,
            )

            if self.debug:
                print(
                    f"[BreakerFooterFinder] OCR LOCK RELEASE for {dbg_label} "
                    f"(detections={len(out)})"
                )

            return out
    
    def _find_header_verticals(self, band: np.ndarray) -> List[int]:
        """
        Detect vertical grid lines inside the *header band only*.

        This is a direct copy of HeaderBandScanner._find_header_verticals.
        """
        if band is None or band.size == 0:
            return []

        H_band, W_band = band.shape

        # 1) binarize
        blur = cv2.GaussianBlur(band, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        # 2) emphasize vertical strokes with a tall, skinny kernel
        Kv = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(15, int(0.40 * H_band)),  # 40% of band height – enough to glue segments
            ),
        )
        v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)

        # 3) connected components -> filter long, skinny, full-height-ish components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            v_candidates,
            connectivity=8,
        )

        if num_labels <= 1:
            return []

        min_full_len = int(0.85 * H_band)             # must span at least 85% of header height
        max_thick    = max(2, int(0.02 * W_band))     # must be thin
        top_margin   = 3                              # must touch near top
        bot_margin   = 3                              # and near bottom

        xs_raw: List[int] = []

        for i in range(1, num_labels):  # label 0 is background
            x, y, w, h, area = stats[i]
            if h < min_full_len:
                continue
            if w > max_thick:
                continue

            # Require line to effectively touch both top and bottom of header band
            if y > top_margin:
                continue
            if (y + h) < (H_band - bot_margin):
                continue

            x_center = x + w // 2
            xs_raw.append(int(x_center))

        if not xs_raw:
            return []

        xs_raw.sort()

        # collapse near-duplicates into one X per visual line
        collapsed: List[int] = []
        MERGE_PX = 4
        for x in xs_raw:
            if not collapsed or abs(x - collapsed[-1]) > MERGE_PX:
                collapsed.append(x)

        return collapsed

    def _find_footer_line_from_anchor(
        self,
        gray_lines: np.ndarray,
        anchor_y: float,
        search_down_px: int = 60,
    ) -> Optional[int]:
        """
        Starting at anchor_y (footer token y), look downward up to search_down_px
        for a strong horizontal line spanning most of the width.
        Returns the page Y of that line or None if not found.
        """
        if gray_lines is None or anchor_y is None:
            return None

        H, W = gray_lines.shape[:2]
        y_start = int(max(0, min(anchor_y, H - 1)))
        y_end = int(min(H, y_start + max(10, search_down_px)))
        if y_end <= y_start + 2:
            return None

        band = gray_lines[y_start:y_end, :]
        if band.size == 0:
            return None

        # binarize
        blur = cv2.GaussianBlur(band, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        band_h = band.shape[0]

        # emphasize horizontal strokes
        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(40, int(0.60 * W)), 1),   # wide kernel: 60% of page width
        )
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            h_candidates, connectivity=8
        )
        if num_labels <= 1:
            return None

        min_len = int(0.60 * W)                 # at least 60% width
        max_thick = max(3, int(0.03 * band_h))  # thin-ish line

        line_ys: List[int] = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w < min_len:
                continue
            if h > max_thick:
                continue
            if area <= 0:
                continue

            y_center = y + h // 2
            line_ys.append(int(y_center))

        if not line_ys:
            return None

        # choose the first strong line below the anchor
        rel_y = min(line_ys)
        return int(y_start + rel_y)

    def _snap_footer_line_from_token(
        self,
        gray_lines: np.ndarray,
        token_y: float,
        max_down: int = 60,
    ) -> Optional[int]:
        """
        Given the PAGE y of the footer token text (panel size, e.g. 84 / 41),
        look downward a short distance for a strong horizontal line that spans
        across most of the panel.

        This mirrors the header snapping approach:
          - build a horizontal-line mask in a band below token_y
          - search downward from token_y for the first horizontal run
          - return its absolute y in page coords
        """
        if gray_lines is None:
            return None

        H, W = gray_lines.shape[:2]

        # band from just above token_y downwards a bit
        y1_band = max(0, int(token_y) - 2)
        y2_band = min(H, int(token_y) + max_down + 10)  # small cushion

        if y2_band <= y1_band + 1:
            return None

        band = gray_lines[y1_band:y2_band, :]
        if band is None or band.size == 0:
            return None

        # --- build horizontal-line mask (similar to header snapping) ---
        blur = cv2.GaussianBlur(band, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        # emphasize HORIZONTAL structures
        klen_target = int(W * 0.70)                  # ~70% of page width
        klen        = int(min(W - 2, max(70, klen_target)))
        K = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, K, iterations=1)

        band_h = horiz.shape[0]

        # token baseline in band-relative coords
        y_rel_start = int(token_y) - y1_band
        if y_rel_start < 0 or y_rel_start >= band_h:
            return None

        def has_white_at(y_idx: int) -> bool:
            y0 = max(0, y_idx - 2)
            y1 = min(band_h, y_idx + 3)
            slice_ = horiz[y0:y1, :]
            return bool(slice_.any())

        best_down_rel = None
        steps_down = 0
        cur = int(y_rel_start)

        while cur < band_h and steps_down <= max_down:
            if has_white_at(cur):
                best_down_rel = cur
                break
            cur += 1
            steps_down += 1

        if best_down_rel is None:
            return None

        footer_line_y = y1_band + best_down_rel

        return int(footer_line_y)

    def _score_header_columns(self, column_groups: List[Dict]) -> Dict:
        """
        Given column_groups (each with .items = OCR tokens), assign a semantic
        role per column: 'ckt', 'description', 'trip', 'poles', 'combo', or ignored.
        Also returns a panel-level layout:

          layout: 'combined'   -> trip + poles live in same "hero" column (per side)
                  'separated'  -> trip and poles live in different hero columns (per side)
                  'unknown'    -> anything else / insufficient signal

        Hero-word priorities (for breaking ties between hero columns):

          Trip heroes (highest → lowest):
            AMP/AMPS  >  TRIP/TRIPPING  >  LOAD  >  SIZE  >  BREAKER/BKR/BRKR/CB

          Poles heroes:
            POLE/POLES/PO/PO. (strong)  >  bare 'P' (weak)

        IMPORTANT:
          - CKT/CCT and DESCRIPTION/NAME always outrank trip/poles for a column.
            If a column has any ckt/description signal, it will NEVER be
            assigned as trip/poles/combo.
        """
        def norm_word(s: str) -> str:
            """
            Normalize a single OCR token to reduce typical OCR mistakes:

              - 0 / O / Q -> O
              - 1 / I / L / J / | / ! -> I
              - 5 / S / $ -> S
              - 2 / Z -> Z
              - 8 / B -> B

            Then keep only A-Z0-9.
            """
            if not s:
                return ""

            s = str(s).upper()
            out = []

            for ch in s:
                if ch in "0OQ":
                    out.append("O")
                elif ch in "1ILJ!|":
                    out.append("I")
                elif ch in "5S$":
                    out.append("S")
                elif ch in "2Z":
                    out.append("Z")
                elif ch in "8B":
                    out.append("B")
                elif ch.isalnum():
                    out.append(ch)
                # everything else is dropped

            return "".join(out)

        def _is_ckt_label(word: str) -> bool:
            """
            Strictly recognize CKT/CCT and common OCR artifacts caused
            by an adjacent vertical grid line.

            norm_word() collapses I, L, 1, !, and | into I, so:
                CKTI, CKT1, CKT!, CKTL, CKT| -> CKTI
                ICKT, 1CKT, !CKT, LCKT, |CKT -> ICKT
            """
            return norm_word(word) in {
                "CKT",
                "CCT",
                "CKTI",
                "ICKT",
            }

        def _is_like(word: str, targets: List[str], base_threshold: float = 0.78) -> bool:
            """
            General fuzzy match `word` against a list of canonical `targets`.

            - Uses norm_word() on both sides.
            - Uses SequenceMatcher ratio.
            - Slightly stricter threshold for very short targets.
            """
            w = norm_word(word)
            if not w:
                return False

            for t in targets:
                t_norm = norm_word(t)
                if not t_norm:
                    continue

                # Bump threshold for very short targets (to avoid random matches)
                if len(t_norm) <= 3:
                    threshold = max(base_threshold, 0.88)
                else:
                    threshold = base_threshold

                if SequenceMatcher(a=w, b=t_norm).ratio() >= threshold:
                    return True

            return False

        def _hero_match(word: str, targets: List[str]) -> bool:
            """
            Stricter fuzzy match specifically for hero words (trip/poles).

              - After the standard fuzzy match on the whole token, we also allow
                a hero word that is "buried" in the OCR token with a tiny bit of
                junk on the edge, e.g.:
                  AMPS!  -> AMPS
                  POLET  -> POLE
                by checking letter-only prefix/suffix matches with <=1 extra char.
            """
            w = norm_word(word)
            if not w:
                return False

            # Letters-only version of the OCR token
            w_letters = "".join(ch for ch in w if ch.isalpha())

            for t in targets:
                t_norm = norm_word(t)
                if not t_norm:
                    continue

                # Letters-only version of the target hero word
                t_letters = "".join(ch for ch in t_norm if ch.isalpha())

                # --- primary: whole-token fuzzy ratio (existing behavior) ---
                if len(t_norm) <= 4:
                    threshold = 0.90
                else:
                    threshold = 0.75

                if SequenceMatcher(a=w, b=t_norm).ratio() >= threshold:
                    return True

                # --- NEW: "hero buried at edge" fallback ---
                # Allow cases like "AMPS!" or "POLET" where the core hero word
                # is at the start or end with at most one extra letter.
                if w_letters and t_letters:
                    if (
                        (w_letters.startswith(t_letters) or w_letters.endswith(t_letters))
                        and abs(len(w_letters) - len(t_letters)) <= 1
                    ):
                        return True

            return False

        # ---------- First pass: compute scores + hero ranks per column ----------
        col_infos: List[Dict] = []

        for col in column_groups:
            items = col.get("items", []) or []

            # texts will now be a flat list of individual words,
            # e.g. ['Circuit', 'Description'] instead of ['Circuit Description']
            texts: List[str] = []
            # (word, is_sentence_like)
            word_entries: List[Tuple[str, bool]] = []

            # Build text + word list, but only from tokens with sufficient OCR confidence
            for tok in items:
                txt = str(tok.get("text", "")).strip()
                if not txt:
                    continue

                conf = float(tok.get("conf", 0.0))
                if conf < _HDR_MIN_CONF:
                    continue  # too noisy for scoring

                # Split into word-ish chunks
                pieces = [w for w in re.split(r"[\s/,;-]+", txt) if w.strip()]
                if not pieces:
                    continue

                # For debug: store *separated* words
                for raw in pieces:
                    raw_clean = raw.strip()
                    if raw_clean:
                        texts.append(raw_clean)

                # Treat long, multi-word chunks as sentence-like notes
                # Example: "(d) PROVIDE WITH SHUNT TRIP BREAKER." -> many words -> sentence-like
                is_sentence_like = len(pieces) >= 3

                for raw in pieces:
                    raw = raw.strip()
                    if not raw:
                        continue

                    w_norm = norm_word(raw)
                    if not w_norm:
                        continue

                    # Skip pure-numeric tokens (no letters) so "3.3", "20", "202" etc
                    # do not drive header role scoring. Mixed tokens like "tr1p" or "20A"
                    # still participate because they have letters.
                    if not any(ch.isalpha() for ch in w_norm):
                        continue

                    word_entries.append((raw, is_sentence_like))

            # --- Detect "strong" CKT label (e.g. 'CKT #', 'CKT NO.', 'CKT NUMBER') ---
            has_ckt_core = any(
                _is_ckt_label(t) for t in texts
            )
            has_ckt_number_assoc = False
            for t in texts:
                tu = t.strip().upper()
                if tu in ("#", "NO", "NO.", "NUMBER", "NUM", "NUM.", "NBR", "NBR."):
                    has_ckt_number_assoc = True
                    break

            # Base structure for this column
            has_notes = False
            score = {
                "ckt": 0,
                "description": 0,
                "trip": 0,   # kept for debugging / future use
                "poles": 0,
            }
            hero_trip_rank = 0  # 0 = no trip hero; 1..5 = increasing strength
            hero_poles_rank = 0  # 0 = no poles hero; 1..4 = increasing strength

            if not word_entries:
                col_infos.append(
                    {
                        "index": col["index"],
                        "x_left": col["x_left"],
                        "x_right": col["x_right"],
                        "texts": texts,
                        "has_notes": has_notes,
                        "ignoredReason": None,
                        "score": score,
                        "hero_trip_rank": hero_trip_rank,
                        "hero_poles_rank": hero_poles_rank,
                        "has_ckt_signal": False,
                        "has_desc_signal": False,
                        "role": None,  # final role filled later
                    }
                )
                continue

            # Score words + hero ranks
            for w_raw, is_sentence_like in word_entries:
                w_norm = norm_word(w_raw)
                if not w_norm:
                    continue

                # --- notes / remarks (only ignore on NOTES-like; ABC/phase is allowed) ---
                # We *do* want sentence-like chunks to still trip the "notes" flag.
                if _is_like(
                    w_raw,
                    ["NOTE", "NOTES", "REMARK", "REMARKS", "COMMENT", "COMMENTS"],
                    base_threshold=0.80,
                ):
                    has_notes = True

                # If this word came from a sentence-like chunk, don't let it
                # drive header-role scoring (treat as inline note only).
                if is_sentence_like:
                    continue

                # --- CKT / CCT / grid-line OCR variants / NO. ---
                if (
                    _is_ckt_label(w_raw)
                    or _is_like(
                        w_raw,
                        ["NO", "NO."],
                        base_threshold=0.95,
                    )
                ):
                    score["ckt"] += 3

                # --- DESCRIPTION / DESIGNATION / NAME (looser; allow ~2–3 errors) ---
                if _is_like(
                    w_raw,
                    [
                        "DESCRIPTION",
                        "CIRCUIT DESCRIPTION",
                        "LOAD DESCRIPTION",
                        "DESIGNATION",
                        "LOAD DESIGNATION",
                        "NAME",
                    ],
                    base_threshold=0.70,  # long words; 0.70 ~ up to ~3 mismatches
                ):
                    score["description"] += 3

                # ---------- Trip hero ranks ----------
                # Priority:
                #   AMP/AMPS (5) > TRIP/TRIPPING (4) > LOAD (3) > SIZE (2) > BREAKER/BKR/BRKR/CB (1)
                if _hero_match(w_raw, ["AMP", "AMPS"]):
                    hero_trip_rank = max(hero_trip_rank, 5)
                elif _hero_match(w_raw, ["TRIP", "TRIPPING"]):
                    hero_trip_rank = max(hero_trip_rank, 4)
                elif _hero_match(w_raw, ["LOAD"]):
                    hero_trip_rank = max(hero_trip_rank, 3)
                elif _hero_match(w_raw, ["SIZE"]):
                    hero_trip_rank = max(hero_trip_rank, 2)
                elif _hero_match(w_raw, ["BREAKER", "BKR", "BRKR", "CB"]):
                    hero_trip_rank = max(hero_trip_rank, 1)

                # ---------- Poles hero ranks ----------
                #   POLE/POLES/PO/PO. (4) > bare 'P' (1)
                if _hero_match(w_raw, ["POLE", "POLES", "PO.", "PO"]):
                    hero_poles_rank = max(hero_poles_rank, 4)
                elif w_norm == "P":
                    hero_poles_rank = max(hero_poles_rank, 1)

            # Extra boost for "CKT + number-ish" labels ---
            # Example headers:
            #   "CKT #"
            #   "CKT NO."
            #   "CKT NUMBER"
            # This makes that column win over a plain "CKT" / "CKT TAG" column.
            if has_ckt_core and has_ckt_number_assoc:
                score["ckt"] += 2
            has_ckt_signal = score["ckt"] > 0
            has_desc_signal = score["description"] > 0

            col_infos.append(
                {
                    "index": col["index"],
                    "x_left": col["x_left"],
                    "x_right": col["x_right"],
                    "texts": texts,
                    "has_notes": has_notes,
                    "ignoredReason": "notes" if has_notes else None,
                    "score": score,
                    "hero_trip_rank": hero_trip_rank,
                    "hero_poles_rank": hero_poles_rank,
                    "has_ckt_signal": has_ckt_signal,
                    "has_desc_signal": has_desc_signal,
                    "role": None,  # filled later
                }
            )

        if not col_infos:
            return {
                "roles": {},
                "columns": [],
                "layout": "unknown",
            }

        # ---------- First: assign ckt / description from fuzzy scores ----------
        for info in col_infos:
            if info["has_notes"]:
                info["role"] = None
                continue

            s = info["score"]
            best_role = None
            best_score_val = 0

            # Only compete ckt vs description here; trip/poles/combo handled via hero logic
            for r in ("ckt", "description"):
                if s[r] > best_score_val:
                    best_score_val = s[r]
                    best_role = r

            # Threshold of 2 keeps us from random noise, but:
            # even if we don't cross 2, the presence of ANY ckt/desc
            # will still block trip/poles later via has_ckt_signal/has_desc_signal.
            if best_role is not None and best_score_val >= 2:
                info["role"] = best_role
            else:
                info["role"] = None

        # ---------- Normalize multiple CKT columns (one primary per side) ----------
        # We can have up to TWO real CKT columns: one on the left, one on the right.
        ckt_candidates = [
            info
            for info in col_infos
            if (not info["has_notes"])
               and info["score"]["ckt"] > 0
               and (info.get("ignoredReason") is None)
        ]

        if ckt_candidates:
            # Use geometry to split into left/right
            global_left = min(c["x_left"] for c in ckt_candidates)
            global_right = max(c["x_right"] for c in ckt_candidates)
            center_x = 0.5 * (global_left + global_right)

            for info in ckt_candidates:
                col_center = 0.5 * (info["x_left"] + info["x_right"])
                info["ckt_side"] = "left" if col_center < center_x else "right"

            def _has_number_assoc(texts: List[str]) -> bool:
                # Same idea as has_ckt_number_assoc: '#', NO, NUMBER, etc.
                for t in texts or []:
                    tu = t.strip().upper()
                    if tu in ("#", "NO", "NO.", "NUMBER", "NUM", "NUM.", "NBR", "NBR."):
                        return True
                return False

            def _rank_ckt(info: Dict) -> tuple:
                # Higher CKT score first, then prefer the one that has "#/NO/NUMBER",
                # then slightly favor the left-most within that side.
                has_num = 1 if _has_number_assoc(info["texts"]) else 0
                return (
                    info["score"]["ckt"],
                    has_num,
                    -info["x_left"],
                )

            primary_ckt_indices = set()

            # Pick best per side (left + right)
            for side in ("left", "right"):
                side_list = [c for c in ckt_candidates if c["ckt_side"] == side]
                if not side_list:
                    continue
                primary = max(side_list, key=_rank_ckt)
                primary_ckt_indices.add(primary["index"])

            # Winners keep/receive role='ckt'; others are demoted to secondaryCkt
            for info in ckt_candidates:
                if info["index"] in primary_ckt_indices:
                    if info["role"] is None:
                        info["role"] = "ckt"
                else:
                    if info["role"] == "ckt":
                        info["role"] = None
                    if not info.get("ignoredReason"):
                        info["ignoredReason"] = "secondaryCkt"

        # ---------- Hero candidates: only pure trip/poles columns ----------
        # We NEVER allow a column that has any ckt/description signal to become
        # trip/poles/combo. That enforces "NAME beats BKR", "CKT beats everything",
        hero_candidates = [
            info
            for info in col_infos
            if (not info["has_notes"])
               and info["role"] is None
               and not info.get("has_ckt_signal")
               and not info.get("has_desc_signal")
        ]

        if not hero_candidates:
            # nothing to do for trip/poles
            role_to_index: Dict[str, int] = {}
            summaries: List[Dict] = []
            for info in col_infos:
                role = info.get("role")
                ignored_reason = info.get("ignoredReason")
                if role and role not in role_to_index:
                    role_to_index[role] = info["index"]
                summaries.append(
                    {
                        "index": info["index"],
                        "x_left": info["x_left"],
                        "x_right": info["x_right"],
                        "texts": info["texts"],
                        "role": role,
                        "ignoredReason": ignored_reason,
                    }
                )
            return {
                "roles": role_to_index,
                "columns": summaries,
                "layout": "unknown",
            }

        # ---------- Determine left/right side using geometric center ----------
        global_left = min(info["x_left"] for info in hero_candidates)
        global_right = max(info["x_right"] for info in hero_candidates)
        center_x = 0.5 * (global_left + global_right)

        for info in col_infos:
            col_center = 0.5 * (info["x_left"] + info["x_right"])
            info["side"] = "left" if col_center < center_x else "right"

        # ---------- Pick best hero trip/poles per side ----------
        best_trip_col = {"left": None, "right": None}
        best_trip_rank = {"left": 0, "right": 0}
        best_poles_col = {"left": None, "right": None}
        best_poles_rank = {"left": 0, "right": 0}

        for info in hero_candidates:
            side = info["side"]
            ht = info["hero_trip_rank"]
            hp = info["hero_poles_rank"]

            if ht > best_trip_rank[side]:
                best_trip_rank[side] = ht
                best_trip_col[side] = info

            if hp > best_poles_rank[side]:
                best_poles_rank[side] = hp
                best_poles_col[side] = info

        # ---------- Panel-level layout decision ----------
        combo_side_exists = False
        for side in ("left", "right"):
            tcol = best_trip_col[side]
            pcol = best_poles_col[side]
            if (
                tcol is not None
                and pcol is not None
                and tcol["index"] == pcol["index"]
                and best_trip_rank[side] > 0
                and best_poles_rank[side] > 0
            ):
                combo_side_exists = True
                break

        any_trip_hero = any(best_trip_rank[side] > 0 for side in ("left", "right"))
        any_poles_hero = any(best_poles_rank[side] > 0 for side in ("left", "right"))

        if combo_side_exists:
            layout = "combined"
        elif any_trip_hero and any_poles_hero:
            layout = "separated"
        else:
            layout = "unknown"

        # ---------- Hero-based assignment for trip/poles/combo (per side) ----------
        # We never override an existing 'ckt' or 'description' role.

        if layout == "combined":
            # Per side: if best trip and poles are same col, mark as combo
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                pcol = best_poles_col[side]
                if (
                    tcol is not None
                    and pcol is not None
                    and tcol["index"] == pcol["index"]
                    and best_trip_rank[side] > 0
                    and best_poles_rank[side] > 0
                ):
                    if tcol["role"] is None:
                        tcol["role"] = "combo"

        elif layout == "separated":
            # One trip + one poles per side (up to 2 of each overall)
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                if tcol is not None and best_trip_rank[side] > 0:
                    if tcol["role"] is None:
                        tcol["role"] = "trip"

                pcol = best_poles_col[side]
                if pcol is not None and best_poles_rank[side] > 0:
                    # Don't override a trip column with poles on the same side
                    if pcol["role"] is None and (tcol is None or pcol["index"] != tcol["index"]):
                        pcol["role"] = "poles"

        else:  # layout == "unknown"
            # Still pick best trip + best poles per side if present
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                if tcol is not None and best_trip_rank[side] > 0:
                    if tcol["role"] is None:
                        tcol["role"] = "trip"

                pcol = best_poles_col[side]
                if pcol is not None and best_poles_rank[side] > 0:
                    if pcol["role"] is None and (tcol is None or pcol["index"] != tcol["index"]):
                        pcol["role"] = "poles"

        # ---------- Build roles map + summaries ----------
        role_to_index: Dict[str, int] = {}
        summaries: List[Dict] = []

        for info in col_infos:
            role = info.get("role")
            ignored_reason = info.get("ignoredReason")

            # First column seen for a role wins in the roles map
            if role and role not in role_to_index:
                role_to_index[role] = info["index"]

            summaries.append(
                {
                    "index": info["index"],
                    "x_left": info["x_left"],
                    "x_right": info["x_right"],
                    "texts": info["texts"],  # per-word
                    "role": role,
                    "ignoredReason": ignored_reason,
                }
            )

        return {
            "roles": role_to_index,
            "columns": summaries,
            "layout": layout,
        }

    def find_footer(
        self,
        analyzer_result: Optional[Dict] = None,
        **kwargs,
    ) -> FooterResult:
        """
        Current behavior:
          - Crop the header band [header_y, header_bottom_y) exactly like the parser.
          - Find vertical grid lines inside that band (column separators).
          - OCR each column band with the same settings as the parser.
          - Run the same header scoring and extract only CKT/CCT columns.

        Supports two calling styles:
          - find_footer(analyzer_result={...})
          - find_footer(gray=..., gridless_gray=..., header_y=..., header_bottom_y=..., src_path=..., src_dir=..., debug_dir=...)
        """
        # --- 0) normalize inputs into a single analyzer_result dict ---
        if analyzer_result is None:
            # Called with keyword args like gray=..., header_y=..., etc.
            analyzer_result = dict(kwargs)
        else:
            # Merge any extra kwargs on top (kwargs wins)
            tmp = dict(analyzer_result)
            tmp.update(kwargs)
            analyzer_result = tmp

        # --- 0b) pull basic images + header geometry from analyzer_result ---
        raw_gray      = analyzer_result.get("gray", None)          # with grid lines
        gridless_gray = analyzer_result.get("gridless_gray", None) # after degridding

        # use gridless for OCR/inspection if available, else raw
        gray_ocr   = gridless_gray if gridless_gray is not None else raw_gray
        gray_lines = raw_gray if raw_gray is not None else gray_ocr

        header_y        = analyzer_result.get("header_y")
        header_bottom_y = analyzer_result.get("header_bottom_y")
        src_path        = analyzer_result.get("src_path")
        footer_struct_y = analyzer_result.get("footer_struct_y")

        dbg_marks: List[Tuple[int, str]] = []

        # --- 1) sanity check ---
        if gray_ocr is None or gray_lines is None or header_y is None:
            if self.debug:
                print(
                    "[BreakerFooterFinder] Missing gray/header_y; "
                    "cannot crop header band."
                )
            return FooterResult(
                footer_y=None,
                token_y=None,
                token_val=None,
                panel_size=None,
                dbg_marks=dbg_marks,
                vlines_x=[],
                cct_cols=[],
            )

        H, W = gray_ocr.shape[:2]

        if not isinstance(header_bottom_y, (int, float)) or not isinstance(header_y, (int, float)):
            if self.debug:
                print(
                    "[BreakerFooterFinder] Missing header_bottom_y/header_y; "
                    "cannot determine header band."
                )
                print(f"  header_y={header_y!r}, header_bottom_y={header_bottom_y!r}")
            return FooterResult(
                footer_y=None,
                token_y=None,
                token_val=None,
                panel_size=None,
                dbg_marks=dbg_marks,
                vlines_x=[],
                cct_cols=[],
            )

        # --- 2) define the header band in page coordinates ---
        y1 = max(0, int(header_y))
        y2 = min(H, int(header_bottom_y))

        dbg_marks.append((y1, "HEADER_TOP"))
        dbg_marks.append((y2, "HEADER_BOTTOM"))

        # Require a non-trivial band height
        if y2 <= y1 + 4:
            if self.debug:
                print(
                    "[BreakerFooterFinder] Header band too small or invalid: "
                    f"y1={y1}, y2={y2}, H={H}. Bailing."
                )
            return FooterResult(
                footer_y=None,
                token_y=None,
                token_val=None,
                panel_size=None,
                dbg_marks=dbg_marks,
                vlines_x=[],
                cct_cols=[],
            )

        band_ocr   = gray_ocr[y1:y2, :]
        band_lines = gray_lines[y1:y2, :]

        debug_dir = self.debug_dir or self._ensure_debug_dir(analyzer_result)

        # --- 3) optional debug: save RAW cropped header band used by footer finder ---
        if self.debug:
            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]
            debug_img_raw_path = os.path.join(
                debug_dir,
                f"{base}_footer_header_band_raw.png",
            )
            try:
                cv2.imwrite(debug_img_raw_path, band_ocr)
            except Exception as e:
                print(
                    f"[BreakerFooterFinder] Failed to write footer header band image: {e}"
                )

        # --- 4) find header-local vertical lines (column dividers), same as parser ---
        header_v_cols = self._find_header_verticals(band_lines)
        v_cols = sorted(int(x) for x in header_v_cols)

        if self.debug:
            print(f"[BreakerFooterFinder] header_v_cols (lines band) = {v_cols}")

        # --- 5) OCR columns exactly like parser, then score + pick CKT columns ---
        tokens: List[Dict] = []
        column_groups: List[Dict] = []

        if self.reader is not None and v_cols and len(v_cols) >= 2:
            H_band, W_band = band_ocr.shape[:2]

            for i in range(len(v_cols) - 1):
                x_left = v_cols[i]
                x_right = v_cols[i + 1]

                # guard against degenerate / reversed intervals
                if x_right <= x_left or x_left < 0 or x_right > W_band:
                    continue

                col_group = {
                    "index": len(column_groups) + 1,
                    "x_left": int(x_left),
                    "x_right": int(x_right),
                    "items": [],
                }
                column_groups.append(col_group)

                col_band = band_ocr[:, x_left:x_right]
                if col_band.size == 0:
                    continue

                col_band_up = cv2.resize(
                    col_band,
                    None,
                    fx=_HDR_OCR_SCALE,
                    fy=_HDR_OCR_SCALE,
                    interpolation=cv2.INTER_CUBIC,
                )

                try:
                    dets = self._safe_footer_readtext(
                        col_band_up,
                        detail=1,
                        paragraph=False,
                        allowlist=_HDR_OCR_ALLOWLIST,
                        mag_ratio=1.0,
                        contrast_ths=0.05,
                        adjust_contrast=0.7,
                        text_threshold=0.4,
                        low_text=0.25,
                        dbg_label=f"footer_header_col_{i}",
                    )
                except Exception as e:
                    if self.debug:
                        print(f"[BreakerFooterFinder] OCR failed on header column {i}: {e}")
                    dets = []

                for box, txt, conf in dets:
                    try:
                        conf_f = float(conf or 0.0)
                    except Exception:
                        conf_f = 0.0

                    pts_band_local = [
                        (
                            int(p[0] / _HDR_OCR_SCALE),
                            int(p[1] / _HDR_OCR_SCALE),
                        )
                        for p in box
                    ]
                    xs = [p[0] for p in pts_band_local]
                    ys = [p[1] for p in pts_band_local]

                    x1_local = max(0, min(xs))
                    x2_local = min(col_band.shape[1] - 1, max(xs))
                    y1b = max(0, min(ys))
                    y2b = min(H_band - 1, max(ys))

                    x1b = x_left + x1_local
                    x2b = x_left + x2_local

                    y1_abs = y1 + y1b
                    y2_abs = y1 + y2b

                    tok = {
                        "text": str(txt or "").strip(),
                        "conf": conf_f,
                        "box_band": [int(x1b), int(y1b), int(x2b), int(y2b)],
                        "box_page": [int(x1b), int(y1_abs), int(x2b), int(y2_abs)],
                    }
                    tokens.append(tok)
                    col_group["items"].append(tok)
        else:
            if self.debug:
                print("[BreakerFooterFinder] No full-height verticals; skipping header OCR.")

        # score columns using the exact same logic as the parser
        normalized_columns = self._score_header_columns(column_groups)

        # keep only CKT/CCT columns
        cct_cols: List[Tuple[int, int]] = []
        for col in normalized_columns.get("columns", []):
            if col.get("role") == "ckt":
                cct_cols.append((int(col["x_left"]), int(col["x_right"])))

        # --- 5b) For each CKT/CCT column, crop body (header_bottom_y -> page bottom),
        #          up-res, trim top/bottom 10%, OCR everything, and save ONLY an overlay.
        footer_token_candidates: List[Dict] = []
        all_numeric_candidates: List[Dict] = []
        if cct_cols:
            y_body_start = int(header_bottom_y)
            # clamp start in range
            y_body_start = max(0, min(y_body_start, H - 1))
            y_body_end = H

            if self.debug:
                print(
                    f"[BreakerFooterFinder] Cropping CKT body columns from "
                    f"y={y_body_start} to y={y_body_end} (H={H})."
                )

            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]

            side_overlay_written = {"left": False, "right": False}

            for idx, (xl, xr) in enumerate(cct_cols, start=1):
                xl_clamped = max(0, min(int(xl), W - 1))
                xr_clamped = max(xl_clamped + 1, min(int(xr), W))

                # decide side for this column based on its center
                col_center_page = 0.5 * (xl_clamped + xr_clamped)
                side_for_col = "left" if col_center_page < (W * 0.5) else "right"

                col_body = gray_ocr[y_body_start:y_body_end, xl_clamped:xr_clamped]
                if col_body.size == 0:
                    if self.debug:
                        print(
                            f"[BreakerFooterFinder] Empty CKT body crop for col {idx}: "
                            f"xl={xl_clamped}, xr={xr_clamped}"
                        )
                    continue

                # Up-res the full body crop
                col_body_up = cv2.resize(
                    col_body,
                    None,
                    fx=_HDR_OCR_SCALE,
                    fy=_HDR_OCR_SCALE,
                    interpolation=cv2.INTER_CUBIC,
                )

                # --- keep the top exactly at header_bottom_y; only trim the bottom ---
                H_body_up, W_body_up = col_body_up.shape[:2]
                bottom_trim = int(0.10 * H_body_up)

                y_top = 0
                y_bot = max(y_top + 1, H_body_up - bottom_trim)

                col_body_mid = col_body_up[y_top:y_bot, :]
                if col_body_mid.size == 0:
                    if self.debug:
                        print(
                            f"[BreakerFooterFinder] Trimmed CKT body crop empty for col {idx}: "
                            f"H_body_up={H_body_up}, bottom_trim={bottom_trim}"
                        )
                    continue

                # File name for mid-band overlay ONLY (per side)
                mid_name_overlay = (
                    f"{base}_footer_ckt_body_{side_for_col}_mid80_overlay.png"
                )
                mid_path_overlay = os.path.join(debug_dir, mid_name_overlay)

                # --- OCR on the trimmed mid-band ---
                dets = []
                if self.reader is not None:
                    try:
                        dets = self._safe_footer_readtext(
                            col_body_mid,
                            detail=1,
                            paragraph=False,
                            allowlist=_HDR_OCR_ALLOWLIST,
                            mag_ratio=1.0,
                            contrast_ths=0.05,
                            adjust_contrast=0.7,
                            text_threshold=0.4,
                            low_text=0.25,
                            dbg_label=f"footer_body_col_{idx}_{side_for_col}",
                        )
                    except Exception as e:
                        if self.debug:
                            print(
                                f"[BreakerFooterFinder] OCR failed on CKT body col {idx} mid-band: {e}"
                            )
                        dets = []

                if self.debug:
                    print(f"[BreakerFooterFinder] OCR results for CKT body col {idx} (mid 80%):")
                    for j, (box, txt, conf) in enumerate(dets, start=1):
                        try:
                            conf_f = float(conf or 0.0)
                        except Exception:
                            conf_f = 0.0
                        print(f"  [{idx}.{j}] '{txt}' conf={conf_f:.2f} box={box}")

                # --- collect numeric footer token candidates (map to PAGE coords) ---
                for (box, txt, conf) in dets:
                    # pull all integer substrings from the OCR text
                    nums = self._extract_last_plausible_circuit_number(txt)
                    if not nums:
                        continue

                    try:
                        conf_f = float(conf or 0.0)
                    except Exception:
                        conf_f = 0.0

                    # ignore low-confidence numbers
                    if conf_f < 0.50:
                        continue

                    # box is in col_body_mid coords (upscaled)
                    pts = [(int(p[0]), int(p[1])) for p in box]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if not xs or not ys:
                        continue

                    x_mid_c = 0.5 * (min(xs) + max(xs))
                    y_mid_c = 0.5 * (min(ys) + max(ys))

                    y_up_c = y_top + y_mid_c
                    x_up_c = x_mid_c

                    y_body_rel = y_up_c / _HDR_OCR_SCALE
                    x_body_rel = x_up_c / _HDR_OCR_SCALE

                    y_page_c = y_body_start + y_body_rel
                    x_page_c = xl_clamped + x_body_rel

                    side = "left" if x_page_c < (W * 0.5) else "right"

                    for val in nums:
                        cand = {
                            "val": int(val),
                            "conf": conf_f,
                            "y_page": float(y_page_c),
                            "x_page": float(x_page_c),
                            "side": side,
                            "col_idx": idx,
                            "raw_text": str(txt),
                        }

                        # keep all plausible circuit numbers so we can detect continued sections
                        if 1 <= val <= self.MAX_CONTINUED_SECTION_CKT:
                            all_numeric_candidates.append(cand)

                        # keep the original footer candidates EXACTLY as before for normal mode
                        if val in self.FOOTER_TOKEN_VALUES:
                            footer_token_candidates.append(cand)

                # --- Build overlay image with OCR boxes + text (with confidence) ---
                if self.debug and col_body_mid.size > 0 and not side_overlay_written[side_for_col]:
                    # make color copy for drawing
                    if len(col_body_mid.shape) == 2:
                        vis_mid = cv2.cvtColor(col_body_mid, cv2.COLOR_GRAY2BGR)
                    else:
                        vis_mid = col_body_mid.copy()

                    for box, txt, conf in dets:
                        try:
                            conf_f = float(conf or 0.0)
                        except Exception:
                            conf_f = 0.0

                        # EasyOCR box is in the same coordinate system as col_body_mid
                        pts = [(int(p[0]), int(p[1])) for p in box]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]

                        x1b = max(0, min(xs))
                        x2b = min(W_body_up - 1, max(xs))
                        y1b = max(0, min(ys))
                        y2b = min(y_bot - y_top - 1, max(ys))

                        # decide if this token is a footer candidate number
                        nums_for_overlay = self._extract_last_plausible_circuit_number(txt)
                        is_candidate = (
                            conf_f >= 0.50
                            and any(1 <= val <= self.MAX_CONTINUED_SECTION_CKT for val in nums_for_overlay)
                        )

                        color = (0, 0, 255) if is_candidate else (0, 255, 0)  # red for candidates, green otherwise

                        # draw rectangle
                        cv2.rectangle(
                            vis_mid,
                            (x1b, y1b),
                            (x2b, y2b),
                            color,
                            1,
                        )

                        # label with text + confidence
                        label_txt = f"{txt} ({conf_f:.2f})"
                        label_y = max(10, y1b - 4)
                        cv2.putText(
                            vis_mid,
                            label_txt,
                            (x1b, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

                    try:
                        cv2.imwrite(mid_path_overlay, vis_mid)
                        side_overlay_written[side_for_col] = True                        
                        print(
                            f"[BreakerFooterFinder] Saved CKT body mid80 overlay for col {idx}: {mid_path_overlay}"
                        )
                    except Exception as e:
                        print(
                            f"[BreakerFooterFinder] Failed to write CKT body mid80 overlay image: {e}"
                        )

        # --- 5c) choose footer token / panel size from collected candidates ---
        footer_y: Optional[int] = None
        token_y: Optional[int] = None
        token_val: Optional[int] = None
        panel_size: Optional[int] = None

        # ------------------------------------------------------------
        # STEP 1: run the ORIGINAL normal-mode footer logic unchanged
        # ------------------------------------------------------------
        normal_footer_y: Optional[int] = None
        normal_token_y: Optional[int] = None
        normal_token_val: Optional[int] = None
        normal_panel_size: Optional[int] = None

        normal_mode_candidates = list(footer_token_candidates)

        cutoff_info = self._find_footer_candidate_cutoff_y(
            footer_token_candidates=footer_token_candidates,
            all_numeric_candidates=all_numeric_candidates,
        )

        if cutoff_info is not None:
            cutoff_y = float(cutoff_info["cutoff_y"])

            normal_mode_candidates = [
                c for c in normal_mode_candidates
                if float(c.get("y_page", 0.0)) <= cutoff_y
            ]

            if self.debug:
                print(
                    "[BreakerFooterFinder] Footer candidate cutoff selected: "
                    f"reason={cutoff_info.get('reason')}, "
                    f"cutoff_y={cutoff_y:.1f}, "
                    f"details={cutoff_info}"
                )

        # Sanity-check OCR values against their approximate physical row.
        # This only rejects extreme outliers; it never replaces the OCR value.
        normal_mode_candidates = (
            self._filter_normal_candidates_by_expected_row(
                candidates=normal_mode_candidates,
                header_bottom_y=header_bottom_y,
                max_value_delta=4,
            )
        )

        if normal_mode_candidates:
            values_present = {c["val"] for c in normal_mode_candidates}

            if self.debug:
                print("[BreakerFooterFinder] footer_token_candidates:")
                for c in footer_token_candidates:
                    print(
                        f"  val={c['val']} conf={c['conf']:.2f} "
                        f"y_page={c['y_page']:.1f} side={c['side']} col={c['col_idx']} "
                        f"text='{c['raw_text']}'"
                    )
                print(f"[BreakerFooterFinder] values_present={sorted(values_present)}")

            chosen_size: Optional[int] = None
            chosen_vals: Optional[set[int]] = None

            # Require at least 2 matching values in a bucket before we trust it.
            min_bucket_hits = 2

            for size in self.PANEL_SIZE_ORDER:
                vals_in_bucket = values_present & self.PANEL_FOOTER_MAP[size]
                if len(vals_in_bucket) >= min_bucket_hits:
                    chosen_size = size
                    chosen_vals = vals_in_bucket
                    break

            if chosen_size is None and self.debug:
                print(
                    "[BreakerFooterFinder] No matching panel size for footer tokens; "
                    f"values_present={sorted(values_present)}"
                )

            if chosen_size is not None and chosen_vals:
                size_cands = [
                    c for c in normal_mode_candidates
                    if c["val"] in self.PANEL_FOOTER_MAP[chosen_size]
                ]

                best = max(
                    size_cands,
                    key=lambda c: (c["y_page"], c["conf"])
                )

                normal_token_y = int(round(best["y_page"]))
                normal_token_val = int(best["val"])
                normal_panel_size = chosen_size

                footer_line_y = self._find_footer_line_from_anchor(
                    gray_lines,
                    normal_token_y,
                    search_down_px=60,
                )

                if footer_line_y is not None:
                    normal_footer_y = footer_line_y
                else:
                    normal_footer_y = normal_token_y

                if self.debug:
                    print(
                        "[BreakerFooterFinder] Normal-mode footer selected: "
                        f"panel_size={chosen_size}, "
                        f"vals_seen={sorted(chosen_vals)}, "
                        f"val={normal_token_val}, conf={best['conf']:.2f}, "
                        f"token_y={normal_token_y}, footer_y={normal_footer_y}, "
                        f"side={best['side']}"
                    )

                snapped_footer_y = None
                if gray_lines is not None:
                    snapped_footer_y = self._snap_footer_line_from_token(
                        gray_lines=gray_lines,
                        token_y=normal_token_y,
                    )

                if snapped_footer_y is not None:
                    normal_footer_y = snapped_footer_y

        # default to the ORIGINAL normal result
        footer_y = normal_footer_y
        token_y = normal_token_y
        token_val = normal_token_val
        panel_size = normal_panel_size

        # ------------------------------------------------------------
        # STEP 2: continued-section override ONLY if the top clearly starts high
        # ------------------------------------------------------------
        continued_start_num = self._infer_continued_section_start(
            all_numeric_candidates,
            H,
        )

        if self.debug:
            print(f"[BreakerFooterFinder] continued_start_num = {continued_start_num}")

        if continued_start_num is not None:
            continued_best = self._choose_continued_section_bottom_candidate(
                all_numeric_candidates=all_numeric_candidates,
                start_num=continued_start_num,
                page_height=H,
            )

            if continued_best is not None:
                continued_end_num = int(continued_best["val"])
                raw_section_size = continued_end_num - continued_start_num + 1
                continued_panel_size = self._round_up_to_standard_panel_size(raw_section_size)

                # Require at least 2 values in the detected continued span
                continued_vals_present = {
                    int(c["val"])
                    for c in all_numeric_candidates
                    if continued_start_num <= int(c["val"]) <= continued_end_num
                }
                continued_support_hits = len(continued_vals_present)

                if self.debug:
                    print(
                        "[BreakerFooterFinder] Continued-section candidate: "
                        f"start_num={continued_start_num}, "
                        f"end_num={continued_end_num}, "
                        f"raw_section_size={raw_section_size}, "
                        f"continued_panel_size={continued_panel_size}, "
                        f"support_hits={continued_support_hits}, "
                        f"vals_seen={sorted(continued_vals_present)}, "
                        f"token_y={int(round(continued_best['y_page']))}"
                    )

                if continued_panel_size is not None and continued_support_hits >= 2:
                    # OVERRIDE normal mode only when we clearly detected a continued section
                    token_y = int(round(continued_best["y_page"]))
                    token_val = continued_end_num
                    panel_size = continued_panel_size

                    snapped_footer_y = None
                    if gray_lines is not None:
                        snapped_footer_y = self._snap_footer_line_from_token(
                            gray_lines=gray_lines,
                            token_y=token_y,
                        )

                    if snapped_footer_y is not None:
                        footer_y = snapped_footer_y
                    else:
                        footer_y = token_y

                    dbg_marks.append((token_y, f"CONT_START={continued_start_num}"))
                    dbg_marks.append((token_y, f"CONT_END={continued_end_num}"))
                    dbg_marks.append((token_y, f"CONT_SIZE={continued_panel_size}"))
                    dbg_marks.append((footer_y, "FOOTER_LINE"))

                    if self.debug:
                        print(
                            "[BreakerFooterFinder] Continued-section OVERRIDE selected: "
                            f"panel_size={panel_size}, token_val={token_val}, "
                            f"token_y={token_y}, footer_y={footer_y}"
                        )

        # final marks for normal mode if we stayed there
        if continued_start_num is None and token_y is not None:
            dbg_marks.append((token_y, f"FOOTER_VAL={token_val}"))
            if footer_y is not None:
                dbg_marks.append((footer_y, "FOOTER_LINE"))

        # --- 6) Debug overlay: band + verticals + CKT boxes ---
        if self.debug:
            vis = cv2.cvtColor(band_lines, cv2.COLOR_GRAY2BGR)
            H_band, W_band = vis.shape[:2]

            cv2.rectangle(
                vis,
                (0, 0),
                (W_band - 1, H_band - 1),
                (0, 255, 255),
                1,
            )
            label = f"FOOTER HEADER BAND  y:[{y1},{y2})"
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

            # vertical lines
            for x in v_cols:
                xi = int(x)
                if 0 <= xi < W_band:
                    cv2.line(vis, (xi, 0), (xi, H_band - 1), (255, 0, 255), 1)
                    cv2.putText(
                        vis,
                        "HDR",
                        (xi + 2, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # highlight CKT columns
            for xl, xr in cct_cols:
                cv2.rectangle(
                    vis,
                    (int(xl), 0),
                    (int(xr) - 1, H_band - 1),
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    vis,
                    "CKT",
                    (int(xl) + 4, H_band - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]
            debug_img_overlay_path = os.path.join(
                debug_dir,
                f"{base}_footer_header_band_overlay.png",
            )
            try:
                cv2.imwrite(debug_img_overlay_path, vis)
            except Exception as e:
                print(
                    f"[BreakerFooterFinder] Failed to write footer header band overlay image: {e}"
                )

        return FooterResult(
            footer_y=footer_y,
            token_y=token_y,
            token_val=token_val,
            panel_size=panel_size,
            dbg_marks=dbg_marks,
            vlines_x=v_cols,
            cct_cols=cct_cols,
        )
