# OcrLibrary/BreakerTableParser6.py
from __future__ import annotations
import os, re, json, cv2, numpy as np
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

PARSER_VERSION = "BreakerParser6"

# --- simple OCR settings for header band ---
_HDR_OCR_SCALE        = 2.0          # up-res factor for header band OCR
_HDR_PAD_ROWS_TOP     = 50           # pixels above header_y
_HDR_PAD_ROWS_BOT     = 85           # pixels below header_y
_HDR_OCR_ALLOWLIST    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/()."
_HDR_MIN_CONF        = 0.50         # minimum OCR confidence to use a token for header scoring

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
            - may include v_grid_xs: list of x positions for long vertical grid lines
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
                "columnGroups": [],
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
                "columnGroups": [],
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
                "columnGroups": [],
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

        # --- 3b) group header tokens into column buckets using v_grid_xs ---
        column_groups: List[Dict] = []
        v_cols = analyzer_result.get("v_grid_xs") or []
        if v_cols and len(v_cols) >= 2 and tokens:
            v_cols_sorted = sorted(int(x) for x in v_cols)

            # build intervals: [x_left, x_right] = one column
            for i in range(len(v_cols_sorted) - 1):
                x_left = v_cols_sorted[i]
                x_right = v_cols_sorted[i + 1]
                column_groups.append(
                    {
                        "index": i + 1,
                        "x_left": int(x_left),
                        "x_right": int(x_right),
                        "items": [],
                    }
                )

            # assign tokens by x-center into these intervals (with small tolerance)
            TOL = 2  # px tolerance so items touching the grid line still count
            for tok in tokens:
                x1b, y1b, x2b, y2b = tok["box_band"]
                x_center = 0.5 * (x1b + x2b)
                for col in column_groups:
                    if (col["x_left"] - TOL) <= x_center <= (col["x_right"] + TOL):
                        col["items"].append(tok)
                        break  # stop at first matching column
        else:
            column_groups = []

        # --- 3c) normalize & score columns into semantic roles (ckt/desc/trip/poles) ---
        normalized_columns = self._score_header_columns(column_groups)

        # Debug printout to terminal
        if self.debug:
            print("[HeaderBandScanner] Column grouping in header band:")
            cols_summary = normalized_columns.get("columns", [])
            if not cols_summary:
                print("  (no vertical grid columns or no header tokens)")
            for col in cols_summary:
                role = col.get("role")
                ignored_reason = col.get("ignoredReason")
                if role:
                    role_str = role
                elif ignored_reason:
                    role_str = f"IGNORED({ignored_reason})"
                else:
                    role_str = "unknown"

                print(
                    f"  Column {col['index']} "
                    f"[{col['x_left']},{col['x_right']}] "
                    f"role={role_str}: {col['texts']}"
                )
            layout = normalized_columns.get("layout")
            if layout:
                print(f"[HeaderBandScanner] Panel layout: {layout}")

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

            # overlay the column grid lines on the header band too
            H_band, W_band = vis.shape[:2]
            for x in v_cols:
                xi = int(x)
                if 0 <= xi < W_band:
                    cv2.line(vis, (xi, 0), (xi, H_band - 1), (255, 0, 255), 1)
                    cv2.putText(
                        vis,
                        "COL",
                        (xi + 2, 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
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
            "columnGroups": column_groups,
            "normalizedColumns": normalized_columns,
        }

    def _score_header_columns(self, column_groups: List[Dict]) -> Dict:
        """
        Given column_groups (each with .items = OCR tokens), assign a semantic
        role per column: 'ckt', 'description', 'trip', 'poles', 'combo', or ignored.
        Also returns a convenient summary and a panel-level layout:

          layout: 'combined'   -> has combo columns, no (trip + poles) pairs
                  'separated'  -> has distinct trip and poles columns, no combos
                  'unknown'    -> anything else

        - Only ignores explicit NOTES/REMARKS/COMMENTS columns.
        - Handles common OCR confusions (0/O, 1/I/L/J, 5/S, 2/Z, 8/B).
        - Uses fuzzy matching so near-misses like 'Descrlptlon' still score.
        - Treats long, sentence-like chunks as inline notes: they can mark a
          column as "notes", but they do NOT drive header role scoring.
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
            Fuzzy match `word` against a list of canonical `targets`.

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

        # First pass: compute scores per column and stash everything
        col_infos: List[Dict] = []

        for col in column_groups:
            items = col.get("items", []) or []

            texts: List[str] = []
            # (word, is_sentence_like)
            word_entries: List[Tuple[str, bool]] = []

            # Build text + word list, but only from tokens with sufficient OCR confidence
            for tok in items:
                txt = str(tok.get("text", "")).strip()
                if not txt:
                    continue
                texts.append(txt)

                conf = float(tok.get("conf", 0.0))
                if conf < _HDR_MIN_CONF:
                    continue  # too noisy for scoring

                # Split into word-ish chunks
                pieces = [w for w in re.split(r"[\s/,;-]+", txt) if w.strip()]
                if not pieces:
                    continue

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

            # Base structure for this column
            has_notes = False
            score = {
                "ckt": 0,
                "description": 0,
                "trip": 0,
                "poles": 0,
            }

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
                        "role": None,  # final role filled later
                    }
                )
                continue

            # Score words
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
                # drive header-role scoring (treat as inline note).
                if is_sentence_like:
                    continue

                # --- CKT / CCT / NO. (very strict; basically exact) ---
                if _is_like(
                    w_raw,
                    ["CKT", "CCT", "NO", "NO."],
                    base_threshold=0.95,  # high threshold; 3-char token must be almost exact
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

                # --- TRIP / AMP(S) / BKR / SIZE / LOAD (≈1 error allowed) ---
                if _is_like(
                    w_raw,
                    ["TRIP", "TRIPPING", "AMP", "AMPS", "BREAKER", "BKR", "BRKR", "SIZE", "LOAD"],
                    base_threshold=0.74,  # TRIP len=4 → 1 mismatch ~0.75
                ):
                    score["trip"] += 2

                # --- POLES / P (1–2 errors allowed on POLE/POLES; 'P' stays strict) ---
                if w_norm == "P" or _is_like(
                    w_raw,
                    ["POLE", "POLES", "PO.", "PO"],
                    base_threshold=0.78,  # POLES len=5 → 1 mismatch ~0.8, 2 ~0.6
                ):
                    score["poles"] += 3

            col_infos.append(
                {
                    "index": col["index"],
                    "x_left": col["x_left"],
                    "x_right": col["x_right"],
                    "texts": texts,
                    "has_notes": has_notes,
                    "ignoredReason": "notes" if has_notes else None,
                    "score": score,
                    "role": None,  # fill in second pass
                }
            )

        # --- Panel-level layout decision (combined vs separated vs unknown) ---
        any_combo_candidate = False
        any_trip_only = False
        any_poles_only = False

        for info in col_infos:
            if info["has_notes"]:
                continue
            s = info["score"]
            trip_pos = s["trip"] > 0
            poles_pos = s["poles"] > 0

            if trip_pos and poles_pos:
                any_combo_candidate = True
            elif trip_pos:
                any_trip_only = True
            elif poles_pos:
                any_poles_only = True

        if any_combo_candidate and not (any_trip_only or any_poles_only):
            layout = "combined"
        elif (any_trip_only and any_poles_only) and not any_combo_candidate:
            layout = "separated"
        elif any_combo_candidate and (any_trip_only or any_poles_only):
            # Ambiguous, but presence of combo columns is a strong hint
            layout = "combined"
        else:
            layout = "unknown"

        # --- Second pass: finalize per-column roles, enforcing layout rules ---
        role_to_index: Dict[str, int] = {}
        summaries: List[Dict] = []

        for info in col_infos:
            score = info["score"]
            has_notes = info["has_notes"]
            ignored_reason = info["ignoredReason"]
            role = None

            if not has_notes:
                trip_pos = score["trip"] > 0
                poles_pos = score["poles"] > 0

                # If this column has BOTH trip & poles signal, treat as combo
                if trip_pos and poles_pos:
                    if layout in ("combined", "unknown"):
                        role = "combo"
                    else:
                        # separated layout: don't use combo; will fall back to best single role
                        pass

                if role is None:
                    # Fall back to best single role (existing behavior)
                    best_role = None
                    best_score = 0
                    for r, val in score.items():
                        if val > best_score:
                            best_score = val
                            best_role = r

                    if best_role is not None and best_score >= 2:
                        role = best_role

                # Enforce mutual exclusivity depending on layout
                if layout == "combined" and role in ("trip", "poles"):
                    role = None  # only combo is allowed for trip/poles info
                if layout == "separated" and role == "combo":
                    role = None  # no combo in strictly separated layouts

                # --- Combined-layout fallback: rescue low-conf AMP+POLES columns ---
                # Only runs if we *still* don't have a role and we're in a combined layout.
                if role is None and layout == "combined" and info.get("texts"):
                    col_has_amp = False
                    col_has_pole = False

                    for txt in info["texts"]:
                        for raw in re.split(r"[\s/,;-]+", txt):
                            raw = raw.strip()
                            if not raw:
                                continue

                            w_norm = norm_word(raw)
                            if not w_norm:
                                continue

                            # Same rule: ignore pure-numeric tokens here too.
                            if not any(ch.isalpha() for ch in w_norm):
                                continue

                            if _is_like(
                                raw,
                                ["AMP", "AMPS"],
                                base_threshold=0.70,
                            ):
                                col_has_amp = True

                            if _is_like(
                                raw,
                                ["POLE", "POLES", "PO.", "PO"],
                                base_threshold=0.70,
                            ):
                                col_has_pole = True

                    if col_has_amp and col_has_pole:
                        role = "combo"

            # Track first index per role
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
            "roles": role_to_index,   # e.g. {"ckt": 5, "description": 1, "trip": 3, "poles": 4} or {"combo": 3}
            "columns": summaries,     # per-column details
            "layout": layout,         # 'combined' | 'separated' | 'unknown'
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
