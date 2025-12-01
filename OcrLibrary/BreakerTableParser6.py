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
_HDR_MIN_CONF         = 0.50         # minimum OCR confidence to use a token for header scoring

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
        # --- separate gray for OCR vs line detection ---
        raw_gray      = analyzer_result.get("gray", None)          # with grid lines
        gridless_gray = analyzer_result.get("gridless_gray", None) # after degridding

        # use gridless for OCR if available, otherwise fall back to raw
        gray_ocr   = gridless_gray if gridless_gray is not None else raw_gray
        gray_lines = raw_gray if raw_gray is not None else gray_ocr

        header_y = analyzer_result.get("header_y")
        src_path = analyzer_result.get("src_path")

        H = W = None
        if gray_ocr is not None:
            H, W = gray_ocr.shape

        header_bottom_y = analyzer_result.get("header_bottom_y")

        debug_dir = self._ensure_debug_dir(analyzer_result)
        debug_img_raw_path = None
        debug_img_overlay_path = None

        # If we don't have the basics, bail gracefully
        if gray_ocr is None or gray_lines is None or header_y is None or H is None:
            if self.debug:
                print("[HeaderBandScanner] Missing gray or header_y; skipping header scan.")
            return {
                "band_y1": None,
                "band_y2": None,
                "tokens": [],
                "debugImageRaw": None,
                "debugImageOverlay": None,
                "columnGroups": [],
                "normalizedColumns": {
                    "roles": {},
                    "columns": [],
                    "layout": "unknown",
                },
            }

        # --- 1) define the header band in page coordinates ---
        if (
            isinstance(header_bottom_y, (int, float))
            and isinstance(header_y, (int, float))
            and header_bottom_y > header_y + 4  # must be *below* header line
        ):
            y1 = max(0, int(header_y))
            y2 = min(H, int(header_bottom_y))
        else:
            # Fallback: legacy padding behaviour
            y1 = max(0, int(header_y) - _HDR_PAD_ROWS_TOP)
            y2 = min(H, int(header_y) + _HDR_PAD_ROWS_BOT)

        band_ocr   = gray_ocr[y1:y2, :]
        band_lines = gray_lines[y1:y2, :]

        if band_ocr.size == 0 or band_lines.size == 0:
            if self.debug:
                print("[HeaderBandScanner] Empty header band after crop; skipping.")
            return {
                "band_y1": y1,
                "band_y2": y2,
                "tokens": [],
                "debugImageRaw": None,
                "debugImageOverlay": None,
                "columnGroups": [],
                "normalizedColumns": {
                    "roles": {},
                    "columns": [],
                    "layout": "unknown",
                },
            }

        # --- 1b) save RAW cropped band to debug folder so you can see exactly what we used ---
        if self.debug:
            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]
            debug_img_raw_path = os.path.join(debug_dir, f"{base}_parser_header_band_raw.png")
            try:
                cv2.imwrite(debug_img_raw_path, band_ocr)
            except Exception as e:
                print(f"[HeaderBandScanner] Failed to write raw header band image: {e}")
                debug_img_raw_path = None

        # Always define these so there are no "referenced before assignment" issues
        tokens: List[Dict] = []
        column_groups: List[Dict] = []

        # --- 2) find header-local vertical lines (column dividers) ---
        header_v_cols = self._find_header_verticals(band_lines)
        v_cols = sorted(int(x) for x in header_v_cols)

        if self.debug:
            print(f"[HeaderBandScanner] header_v_cols (lines band) = {v_cols}")

        if self.reader is not None:
            H_band, W_band = band_ocr.shape[:2]

            if v_cols and len(v_cols) >= 2:
                # --- 2a) OCR per column strip between successive vertical lines ---
                for i in range(len(v_cols) - 1):
                    x_left = v_cols[i]
                    x_right = v_cols[i + 1]

                    # guard against degenerate / reversed intervals
                    if x_right <= x_left or x_left < 0 or x_right > W_band:
                        continue

                    # Initialize this column group up front
                    col_group = {
                        "index": len(column_groups) + 1,
                        "x_left": int(x_left),
                        "x_right": int(x_right),
                        "items": [],
                    }
                    column_groups.append(col_group)

                    # Crop this column strip from the OCR band
                    col_band = band_ocr[:, x_left:x_right]

                    if col_band.size == 0:
                        continue

                    # Up-res the column band
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
                            print(f"[HeaderBandScanner] OCR failed on header column {i}: {e}")
                        dets = []

                    # Map OCR boxes back to *full band* coordinates, and lock them to this column
                    for box, txt, conf in dets:
                        try:
                            conf_f = float(conf or 0.0)
                        except Exception:
                            conf_f = 0.0

                        # box in upscaled column space -> downscale back to column-band coords
                        pts_band_local = [
                            (
                                int(p[0] / _HDR_OCR_SCALE),
                                int(p[1] / _HDR_OCR_SCALE),
                            )
                            for p in box
                        ]
                        xs = [p[0] for p in pts_band_local]
                        ys = [p[1] for p in pts_band_local]

                        # local coords within this column strip
                        x1_local = max(0, min(xs))
                        x2_local = min(col_band.shape[1] - 1, max(xs))
                        y1b = max(0, min(ys))
                        y2b = min(H_band - 1, max(ys))

                        # convert to *band* coords by offsetting with x_left
                        x1b = x_left + x1_local
                        x2b = x_left + x2_local

                        # page-coordinates (add vertical offset)
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
                # --- 2b) Fallback: no reliable verticals -> OCR whole band as a single "column" ---
                if self.debug:
                    print("[HeaderBandScanner] No full-height verticals; OCRing whole header band.")

                band_up = cv2.resize(
                    band_ocr,
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
                        print(f"[HeaderBandScanner] OCR failed on header band (fallback): {e}")
                    dets = []

                col_group = {
                    "index": 1,
                    "x_left": 0,
                    "x_right": band_ocr.shape[1] - 1,
                    "items": [],
                }
                column_groups.append(col_group)

                H_band, W_band = band_ocr.shape[:2]

                for box, txt, conf in dets:
                    try:
                        conf_f = float(conf or 0.0)
                    except Exception:
                        conf_f = 0.0

                    pts_band = [
                        (
                            int(p[0] / _HDR_OCR_SCALE),
                            int(p[1] / _HDR_OCR_SCALE),
                        )
                        for p in box
                    ]
                    xs = [p[0] for p in pts_band]
                    ys = [p[1] for p in pts_band]
                    x1b, x2b = max(0, min(xs)), min(W_band - 1, max(xs))
                    y1b, y2b = max(0, min(ys)), min(H_band - 1, max(ys))

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

        # --- 3) normalize & score columns into semantic roles (ckt/desc/trip/poles) ---
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
            vis = cv2.cvtColor(band_lines, cv2.COLOR_GRAY2BGR)

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

            # Use same header-local lines we used for OCR
            header_v_cols_dbg = v_cols

            # Draw header-local lines in magenta
            for x in header_v_cols_dbg:
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

            base = os.path.splitext(os.path.basename(src_path or "panel"))[0]
            debug_img_overlay_path = os.path.join(
                debug_dir,
                f"{base}_parser_header_band_overlay.png",
            )
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

    def _find_header_verticals(self, band: np.ndarray) -> List[int]:
        """
        Detect vertical grid lines inside the *header band only*.

        Strategy:
          - Binarize the band.
          - Morphologically open with a tall vertical kernel to keep vertical strokes.
          - Connected-components over the result.
          - Keep only components that:
              * are tall enough (>= ~85% of band height), and
              * start near the top and end near the bottom of the band, and
              * are thin (not blocks of fill).
          - Return sorted, de-duplicated X-center positions for those components.
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

            We *really* only want near-exact matches here to avoid things like
            'WIRE' being treated as 'LOAD' or 'POLE'.
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

        # ---------- Hero candidates: only pure trip/poles columns ----------
        # We NEVER allow a column that has any ckt/description signal to become
        # trip/poles/combo. That enforces "NAME beats BKR", "CKT beats everything",
        # etc.
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

            if role and role not in role_to_index:
                role_to_index[role] = info["index"]

            summaries.append(
                {
                    "index": info["index"],
                    "x_left": info["x_left"],
                    "x_right": info["x_right"],
                    "texts": info["texts"],  # now per-word
                    "role": role,
                    "ignoredReason": ignored_reason,
                }
            )

        return {
            "roles": role_to_index,
            "columns": summaries,
            "layout": layout,
        }

class SeparatedLayoutParser:
    """
    Handles parsing when the header layout is 'separated':
      - Trip and poles live in different hero columns (left/right).
      - For now, we only:
          * identify the trip/poles columns from the header scan
          * crop their body strips (header_bottom -> footer)
          * save debug images (raw + upscaled) per column
      - No breaker OCR / row parsing yet; that will be layered on later.
    """

    def __init__(self, *, debug: bool = False, reader=None):
        self.debug = bool(debug)
        self.reader = reader  # reserved for when we start OCR'ing the body

    def parse(self, analyzer_result: Dict, header_scan: Dict) -> Dict:
        """
        Given analyzer_result + header_scan with normalizedColumns.layout == 'separated',
        crop the body strips for all trip/poles columns and emit debug images.
        """
        # --- image sources (same pattern as HeaderBandScanner) ---
        raw_gray      = analyzer_result.get("gray", None)
        gridless_gray = analyzer_result.get("gridless_gray", None)

        gray_body = gridless_gray if gridless_gray is not None else raw_gray
        if gray_body is None:
            if self.debug:
                print("[SeparatedLayoutParser] Missing gray_body; cannot crop body columns.")
            return {
                "layout": "separated",
                "bodyColumns": [],
                "detected_breakers": [],
            }

        H, W = gray_body.shape[:2]

        src_path = analyzer_result.get("src_path")
        src_dir  = analyzer_result.get("src_dir") or os.path.dirname(src_path or ".")
        debug_dir = analyzer_result.get("debug_dir") or os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(src_path or "panel"))[0]

        # --- header / footer anchors ---
        header_y         = analyzer_result.get("header_y")
        header_bottom_y  = analyzer_result.get("header_bottom_y")

        # Prefer explicit header_bottom_y, else fall back to headerScan band
        band_y2 = header_scan.get("band_y2")
        if isinstance(header_bottom_y, (int, float)) and isinstance(header_y, (int, float)):
            body_y_top = max(0, int(header_bottom_y))
        elif isinstance(band_y2, (int, float)):
            body_y_top = max(0, int(band_y2))
        elif isinstance(header_y, (int, float)):
            body_y_top = max(0, int(header_y))
        else:
            body_y_top = 0

        # Footer top (defensive against different analyzer versions)
        footer_y = analyzer_result.get("footer_y")
        if not isinstance(footer_y, (int, float)):
            footer_y = analyzer_result.get("footer_top_y")
        if not isinstance(footer_y, (int, float)):
            footer_y = analyzer_result.get("footer_band_y1")
        if not isinstance(footer_y, (int, float)):
            footer_y = H  # fallback: run to bottom of page

        body_y_bottom = min(H, int(footer_y))

        if self.debug:
            print(
                f"[SeparatedLayoutParser] Body Y-range: [{body_y_top}, {body_y_bottom}) "
                f"(H={H})"
            )

        if body_y_bottom <= body_y_top + 4:
            if self.debug:
                print("[SeparatedLayoutParser] Body band too small; skipping body columns.")
            return {
                "layout": "separated",
                "bodyColumns": [],
                "detected_breakers": [],
            }

        normalized = header_scan.get("normalizedColumns") or {}
        if normalized.get("layout") != "separated":
            # Guard: caller should only invoke this when layout == 'separated'
            if self.debug:
                print(
                    "[SeparatedLayoutParser] Warning: called with layout "
                    f"{normalized.get('layout')}; expected 'separated'."
                )
            return {
                "layout": normalized.get("layout", "unknown"),
                "bodyColumns": [],
                "detected_breakers": [],
            }

        cols_summary = normalized.get("columns", []) or []

        # We want all columns that are trip or poles (both sides)
        wanted_roles = {"trip", "poles"}

        body_columns: List[Dict] = []

        for col in cols_summary:
            role = col.get("role")
            if role not in wanted_roles:
                continue

            x_left  = max(0, int(col.get("x_left", 0)))
            x_right = min(W - 1, int(col.get("x_right", W - 1)))
            if x_right <= x_left + 1:
                continue

            body_strip = gray_body[body_y_top:body_y_bottom, x_left:x_right]
            if body_strip.size == 0:
                continue

            raw_path = None
            up_path  = None

            if self.debug:
                # Raw body strip
                raw_path = os.path.join(
                    debug_dir,
                    f"{base}_parser_body_sep_col{col['index']}_{role}_raw.png",
                )
                try:
                    cv2.imwrite(raw_path, body_strip)
                except Exception as e:
                    print(
                        f"[SeparatedLayoutParser] Failed to write body raw col "
                        f"{col['index']} ({role}): {e}"
                    )
                    raw_path = None

                # Up-res body strip (same scale factor as header OCR)
                try:
                    body_up = cv2.resize(
                        body_strip,
                        None,
                        fx=_HDR_OCR_SCALE,
                        fy=_HDR_OCR_SCALE,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    up_path = os.path.join(
                        debug_dir,
                        f"{base}_parser_body_sep_col{col['index']}_{role}_up.png",
                    )
                    cv2.imwrite(up_path, body_up)
                except Exception as e:
                    print(
                        f"[SeparatedLayoutParser] Failed to write body up col "
                        f"{col['index']} ({role}): {e}"
                    )
                    up_path = None

            body_columns.append(
                {
                    "index": col["index"],
                    "role": role,
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_top": body_y_top,
                    "y_bottom": body_y_bottom,
                    "debugImageRaw": raw_path,
                    "debugImageUp": up_path,
                }
            )

        if self.debug:
            print(
                f"[SeparatedLayoutParser] Extracted {len(body_columns)} body columns "
                f"for trip/poles."
            )

        # No breaker parsing yet; just prepping geometry + debug images.
        return {
            "layout": "separated",
            "bodyColumns": body_columns,
            "detected_breakers": [],
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
        self._separated_parser = SeparatedLayoutParser(debug=self.debug, reader=self.reader)        

    def parse_from_analyzer(self, analyzer_result: Dict) -> Dict:
        """
        Entry point used by the API.

        Right now:
          - Reads spaces from analyzer_result (corrected if available)
          - Runs the header-band OCR scan
          - If header layout is 'separated', runs SeparatedLayoutParser
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

        normalized = header_scan.get("normalizedColumns") or {}
        layout = normalized.get("layout", "unknown")

        separated_scan: Optional[Dict] = None
        detected_breakers: List[Dict] = []

        if layout == "separated":
            if self.debug:
                print("[BreakerTableParser] Layout 'separated' → using SeparatedLayoutParser.")
            separated_scan = self._separated_parser.parse(analyzer_result, header_scan)
            detected_breakers = separated_scan.get("detected_breakers", [])
        else:
            if self.debug:
                print(f"[BreakerTableParser] Layout '{layout}' → no body parser yet.")

        if self.debug:
            print(
                f"[BreakerTableParser] Header band y:[{header_scan.get('band_y1')},"
                f"{header_scan.get('band_y2')}) tokens={len(header_scan.get('tokens', []))}"
            )

        # Minimal, legacy-compatible result + new separated path metadata
        return {
            "parserVersion": PARSER_VERSION,
            "name": None,  # API will overwrite with deduped panel name
            "spaces": int(spaces),
            "detected_breakers": detected_breakers,
            "headerScan": header_scan,
            "separatedScan": separated_scan,
        }
