# AnchoringClasses/BreakerFooterFinder.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import os
import cv2
import numpy as np
from difflib import SequenceMatcher
import re

# Same values you use in HeaderBandScanner
_HDR_OCR_SCALE = 2.0           # or whatever you use there
_HDR_OCR_ALLOWLIST = None      # or your actual allowlist
_HDR_MIN_CONF = 0.35           # or your actual threshold

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
        84: {84, 83, 82, 81},
        72: {72, 71, 70, 69},
        66: {66, 65, 64, 63},
        54: {54, 53, 52, 51},
        42: {42, 41, 40, 39},
        30: {30, 29, 28, 27},
        18: {18, 17, 16, 15},
    }

    # search largest size first
    PANEL_SIZE_ORDER: List[int] = [84, 72, 66, 54, 42, 30, 18]

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

    def _score_header_columns(self, column_groups: List[Dict]) -> Dict:
        """
        Given column_groups (each with .items = OCR tokens), assign a semantic
        role per column: 'ckt', 'description', 'trip', 'poles', 'combo', or ignored.
        Also returns a panel-level layout:

          layout: 'combined'   -> trip + poles live in same "hero" column (per side)
                  'separated'  -> trip and poles live in different hero columns (per side)
                  'unknown'    -> anything else / insufficient signal
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

        def _is_like(word: str, targets: List[str], base_threshold: float = 0.78) -> bool:
            """
            General fuzzy match `word` against a list of canonical `targets`.
            Uses norm_word() on both sides and SequenceMatcher ratio.
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
            """
            w = norm_word(word)
            if not w:
                return False

            for t in targets:
                t_norm = norm_word(t)
                if not t_norm:
                    continue

                # Hero matching thresholds:
                # - very short tokens (<=4): need ~0.90+
                # - longer tokens: allow a bit more noise (~0.75+)
                if len(t_norm) <= 4:
                    threshold = 0.90
                else:
                    threshold = 0.75

                if SequenceMatcher(a=w, b=t_norm).ratio() >= threshold:
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
                is_sentence_like = len(pieces) >= 3

                for raw in pieces:
                    raw = raw.strip()
                    if not raw:
                        continue

                    w_norm = norm_word(raw)
                    if not w_norm:
                        continue

                    # Skip pure-numeric tokens (no letters)
                    if not any(ch.isalpha() for ch in w_norm):
                        continue

                    word_entries.append((raw, is_sentence_like))

            # Base structure for this column
            has_notes = False
            score = {
                "ckt": 0,
                "description": 0,
                "trip": 0,
                "poles": 0,
            }
            hero_trip_rank = 0
            hero_poles_rank = 0

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
                        "role": None,
                    }
                )
                continue

            # Score words + hero ranks
            for w_raw, is_sentence_like in word_entries:
                w_norm = norm_word(w_raw)
                if not w_norm:
                    continue

                # notes / remarks
                if _is_like(
                    w_raw,
                    ["NOTE", "NOTES", "REMARK", "REMARKS", "COMMENT", "COMMENTS"],
                    base_threshold=0.80,
                ):
                    has_notes = True

                if is_sentence_like:
                    continue

                # CKT / CCT / NO.
                if _is_like(
                    w_raw,
                    ["CKT", "CCT", "NO", "NO."],
                    base_threshold=0.95,
                ):
                    score["ckt"] += 3

                # DESCRIPTION / DESIGNATION / NAME
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
                    base_threshold=0.70,
                ):
                    score["description"] += 3

                # Trip hero ranks
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

                # Poles hero ranks
                if _hero_match(w_raw, ["POLE", "POLES", "PO.", "PO"]):
                    hero_poles_rank = max(hero_poles_rank, 4)
                elif w_norm == "P":
                    hero_poles_rank = max(hero_poles_rank, 1)

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
                    "role": None,
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

            for r in ("ckt", "description"):
                if s[r] > best_score_val:
                    best_score_val = s[r]
                    best_role = r

            if best_role is not None and best_score_val >= 2:
                info["role"] = best_role
            else:
                info["role"] = None

        # Hero candidates: only pure trip/poles columns
        hero_candidates = [
            info
            for info in col_infos
            if (not info["has_notes"])
               and info["role"] is None
               and not info.get("has_ckt_signal")
               and not info.get("has_desc_signal")
        ]

        if not hero_candidates:
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

        # Determine left/right side using geometric center
        global_left = min(info["x_left"] for info in hero_candidates)
        global_right = max(info["x_right"] for info in hero_candidates)
        center_x = 0.5 * (global_left + global_right)

        for info in col_infos:
            col_center = 0.5 * (info["x_left"] + info["x_right"])
            info["side"] = "left" if col_center < center_x else "right"

        # Pick best hero trip/poles per side
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

        # Panel-level layout decision
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

        # Hero-based assignment for trip/poles/combo
        if layout == "combined":
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
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                if tcol is not None and best_trip_rank[side] > 0:
                    if tcol["role"] is None:
                        tcol["role"] = "trip"

                pcol = best_poles_col[side]
                if pcol is not None and best_poles_rank[side] > 0:
                    if pcol["role"] is None and (tcol is None or pcol["index"] != tcol["index"]):
                        pcol["role"] = "poles"
        else:
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                if tcol is not None and best_trip_rank[side] > 0:
                    if tcol["role"] is None:
                        tcol["role"] = "trip"

                pcol = best_poles_col[side]
                if pcol is not None and best_poles_rank[side] > 0:
                    if pcol["role"] is None and (tcol is None or pcol["index"] != tcol["index"]):
                        pcol["role"] = "poles"

        # Build roles map + summaries
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
                    dets = self.reader.readtext(
                        col_band_up,
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

                # --- trim top and bottom 10% of the upresed crop ---
                H_body_up, W_body_up = col_body_up.shape[:2]
                trim = int(0.10 * H_body_up)
                y_top = max(0, trim)
                y_bot = max(y_top + 1, H_body_up - trim)

                col_body_mid = col_body_up[y_top:y_bot, :]
                if col_body_mid.size == 0:
                    if self.debug:
                        print(
                            f"[BreakerFooterFinder] Trimmed CKT body crop empty for col {idx}: "
                            f"H_body_up={H_body_up}, trim={trim}"
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
                        dets = self.reader.readtext(
                            col_body_mid,
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
                    nums = [int(m.group()) for m in re.finditer(r"\d+", str(txt) or "")]
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
                        if val in self.FOOTER_TOKEN_VALUES:
                            footer_token_candidates.append(
                                {
                                    "val": int(val),
                                    "conf": conf_f,
                                    "y_page": float(y_page_c),
                                    "x_page": float(x_page_c),
                                    "side": side,
                                    "col_idx": idx,
                                    "raw_text": str(txt),
                                }
                            )

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
                        nums_for_overlay = [int(m.group()) for m in re.finditer(r"\d+", str(txt) or "")]
                        is_candidate = (
                            conf_f >= 0.50
                            and any(val in self.FOOTER_TOKEN_VALUES for val in nums_for_overlay)
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

        if footer_token_candidates:
            values_present = {c["val"] for c in footer_token_candidates}

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

            # search largest first so we don't stop at 18 when 42/84 exist
            for size in self.PANEL_SIZE_ORDER:
                needed = self.PANEL_FOOTER_MAP[size]
                present = values_present & needed

                # require at least 2 numbers from this 4-number window
                if len(present) >= 2:
                    chosen_size = size
                    chosen_vals = present
                    break

            if chosen_size is not None and chosen_vals:
                size_cands = [
                    c for c in footer_token_candidates
                    if c["val"] in self.PANEL_FOOTER_MAP[chosen_size]
                ]

                best = max(
                    size_cands,
                    key=lambda c: (c["y_page"], c["conf"])
                )

                footer_y = int(round(best["y_page"]))
                token_y = footer_y
                token_val = int(best["val"])
                panel_size = chosen_size

                dbg_marks.append((footer_y, f"FOOTER_VAL={token_val}"))
                if self.debug:
                    print(
                        "[BreakerFooterFinder] Selected footer token: "
                        f"panel_size={chosen_size}, "
                        f"vals_seen={sorted(chosen_vals)}, "
                        f"val={token_val}, conf={best['conf']:.2f}, "
                        f"y_page={best['y_page']:.1f}, side={best['side']}"
                    )
            else:
                if self.debug:
                    print(
                        "[BreakerFooterFinder] No consistent panel size "
                        "found from footer token candidates."
                    )

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
