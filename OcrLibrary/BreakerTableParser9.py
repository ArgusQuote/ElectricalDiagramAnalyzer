# OcrLibrary/BreakerTableParser9.py
from __future__ import annotations
import os, re, cv2, numpy as np
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

PARSER_VERSION = "BreakerParser9"

_HDR_OCR_SCALE        = 2.0
_HDR_OCR_ALLOWLIST    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/().#"
_HDR_MIN_CONF         = 0.40
 
def _prep_gray_like_analyzer12(src_path: str) -> Optional[np.ndarray]:
    """
    Recreate the same gray image that BreakerTableAnalyzer12 uses:

      - BGR -> gray
      - CLAHE
      - upscale to min height 1600 px

    This ensures header_y / footer_y coordinates still line up.
    """
    if not src_path:
        return None

    try:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        H, W = g.shape
        if H < 1600:
            s = 1600.0 / H
            g = cv2.resize(
                g,
                (int(W * s), int(H * s)),
                interpolation=cv2.INTER_CUBIC,
            )
        return g
    except Exception:
        return None

def _ensure_gray(analyzer_result: Dict) -> Optional[np.ndarray]:
    """
    Make sure analyzer_result has a 'gray' image.

    - If 'gray' already exists, return it.
    - Otherwise, reconstruct it from src_path using the same prep as Analyzer12,
      store it back into analyzer_result['gray'], and return it.
    """
    gray = analyzer_result.get("gray")
    if gray is not None:
        return gray

    src_path = analyzer_result.get("src_path")
    gray = _prep_gray_like_analyzer12(src_path)
    if gray is not None:
        analyzer_result["gray"] = gray
    return gray

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
            - expects keys: src_path, header_y, header_bottom_y
            - if 'gray' is missing, we will rebuild it from src_path
        """
        # --- get gray for OCR + line detection ---
        raw_gray = analyzer_result.get("gray")
        if raw_gray is None:
            raw_gray = _ensure_gray(analyzer_result)

        gray_ocr   = raw_gray
        gray_lines = raw_gray

        header_y        = analyzer_result.get("header_y")
        src_path        = analyzer_result.get("src_path")
        header_bottom_y = analyzer_result.get("header_bottom_y")

        H = W = None
        if gray_ocr is not None:
            H, W = gray_ocr.shape

        debug_dir = self._ensure_debug_dir(analyzer_result)
        debug_img_raw_path = None
        debug_img_overlay_path = None

        # If we don't have the basics, bail gracefully
        if gray_ocr is None or gray_lines is None or header_y is None or H is None:
            if self.debug:
                print("[HeaderBandScanner] Missing gray/header_y; skipping header scan.")
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
                "error": "Missing gray image or header_y; cannot scan header band.",
            }

        # --- 1) define the header band in page coordinates (MUST have valid header_bottom_y) ---
        if not isinstance(header_bottom_y, (int, float)) or not isinstance(header_y, (int, float)):
            if self.debug:
                print(
                    "[HeaderBandScanner] Missing header_bottom_y/header_y; "
                    "cannot determine header band. Bailing."
                )
                print(f"  header_y={header_y!r}, header_bottom_y={header_bottom_y!r}")
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
                "error": "Missing header_bottom_y from analyzer; cannot determine header band.",
            }

        y1 = max(0, int(header_y))
        y2 = min(H, int(header_bottom_y))

        # Require a non-trivial band height
        if y2 <= y1 + 4:
            if self.debug:
                print(
                    "[HeaderBandScanner] Header band too small or invalid: "
                    f"y1={y1}, y2={y2}, H={H}. Bailing."
                )
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
                "error": "Invalid header band height from analyzer; cannot scan header.",
            }

        band_ocr   = gray_ocr[y1:y2, :]
        band_lines = gray_lines[y1:y2, :]

        # --- 1b) save RAW cropped band to debug folder ---
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
                if self.debug:
                    print("[HeaderBandScanner] No full-height verticals; skipping header OCR.")

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
                _is_like(t, ["CKT", "CCT"], base_threshold=0.95) for t in texts
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
                #   TRIP/TRIPPING (5) > AMP/AMPS (4) > SIZE (3) > BREAKER/BKR/BRKR/CB (2)
                if _hero_match(w_raw, ["TRIP", "TRIPPING"]):
                    hero_trip_rank = max(hero_trip_rank, 5)
                elif _hero_match(w_raw, ["AMP", "AMPS"]):
                    hero_trip_rank = max(hero_trip_rank, 4)
                elif _hero_match(w_raw, ["SIZE"]):
                    hero_trip_rank = max(hero_trip_rank, 3)
                elif _hero_match(w_raw, ["BREAKER", "BKR", "BRKR", "CB"]):
                    hero_trip_rank = max(hero_trip_rank, 2)

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

        # Implied-combo detection
        # If a side has a trip hero but no poles hero, treat that side's trip column as an implied combo (trip+poles) column.
        implied_combo_sides = set()
        for side in ("left", "right"):
            if best_trip_rank[side] > 0 and best_poles_rank[side] == 0:
                implied_combo_sides.add(side)

        if self.debug and implied_combo_sides:
            print(
                "[HeaderBandScanner] Implied combo layout on sides: "
                f"{sorted(implied_combo_sides)}"
            )

        if combo_side_exists or implied_combo_sides:
            layout = "combined"
        elif any_trip_hero and any_poles_hero:
            layout = "separated"
        else:
            layout = "unknown"

        # ---------- Hero-based assignment for trip/poles/combo (per side) ----------
        # We never override an existing 'ckt' or 'description' role.

        if layout == "combined":
            # Per side:
            #   Case 1: explicit combo (trip & poles heroes on same column)
            #   Case 2: implied combo (trip hero but no poles hero on that side)
            for side in ("left", "right"):
                tcol = best_trip_col[side]
                pcol = best_poles_col[side]

                # Case 1: explicit combo
                if (
                    tcol is not None
                    and pcol is not None
                    and tcol["index"] == pcol["index"]
                    and best_trip_rank[side] > 0
                    and best_poles_rank[side] > 0
                ):
                    if tcol["role"] is None:
                        tcol["role"] = "combo"
                    continue

                # Case 2: implied combo (trip hero only, no poles hero)
                if (
                    best_trip_rank[side] > 0
                    and best_poles_rank[side] == 0
                    and tcol is not None
                    and tcol["role"] is None
                ):
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

class SeparatedLayoutParser:
    """
    Handles parsing when the header layout is 'separated':
      - Trip and poles live in different hero columns (left/right).
      - Uses horizontal grid lines (within a reference body strip) to define row bands.
      - OCRs the trip/poles body strips ROW-BY-ROW using those bands.
      - Associates each row's amps + poles and emits per-breaker records
        plus an aggregate (amps/poles) histogram.
    """

    def __init__(self, *, debug: bool = False, reader=None):
        self.debug = bool(debug)
        self.reader = reader

    def _infer_body_rows_from_tokens(
        self,
        tokens_by_col_index: Dict[int, List[Dict]],
        body_y_top: int,
        body_y_bottom: int,
    ):
        """
        OLD: token-clustering-based row inference. Currently unused by parse(),
        which now relies on horizontal row-divider lines. Kept for possible
        future fallback / experiments.
        """
        ys = []

        for col_tokens in tokens_by_col_index.values():
            for tok in (col_tokens or []):
                try:
                    x1, y1, x2, y2 = tok["box_page"]
                except Exception:
                    continue
                y_center = 0.5 * (y1 + y2)
                # Keep only tokens that are actually within the body band
                if y_center < body_y_top or y_center > body_y_bottom:
                    continue
                ys.append(float(y_center))

        if not ys:
            if self.debug:
                print("[SeparatedLayoutParser] No token Y-centers found; no body rows inferred.")
            return []

        ys.sort()

        # If there is only one y center, treat the entire band as one row
        if len(ys) == 1:
            return [(int(body_y_top), int(body_y_bottom))]

        # Compute gaps between successive centers
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        gaps_sorted = sorted(gaps)
        # Simple median
        mid = len(gaps_sorted) // 2
        if len(gaps_sorted) % 2 == 0:
            median_gap = 0.5 * (gaps_sorted[mid - 1] + gaps_sorted[mid])
        else:
            median_gap = gaps_sorted[mid]

        # Define a max gap threshold to decide "new row" vs "same row"
        body_height = max(1, body_y_bottom - body_y_top)
        # Clamp the threshold into a reasonable range
        min_gap = 5.0
        max_gap = max(20.0, 0.15 * body_height)
        gap_threshold = max(min_gap, min(median_gap * 1.5, max_gap))

        if self.debug:
            print(
                f"[SeparatedLayoutParser] token Y-centers: count={len(ys)}, "
                f"median_gap={median_gap:.2f}, gap_threshold={gap_threshold:.2f}"
            )

        # Cluster ys by gap_threshold
        clusters = [[ys[0]]]
        for y in ys[1:]:
            if y - clusters[-1][-1] <= gap_threshold:
                clusters[-1].append(y)
            else:
                clusters.append([y])

        # Compute cluster centers
        centers = [int(round(sum(c) / len(c))) for c in clusters]
        centers.sort()

        if self.debug:
            print(
                f"[SeparatedLayoutParser] inferred row centers={centers} "
                f"(body_y_top={body_y_top}, body_y_bottom={body_y_bottom})"
            )

        # Convert centers into row bands [top, bottom)
        row_spans = []
        for idx, c in enumerate(centers):
            if idx == 0:
                top = body_y_top
            else:
                top = int((centers[idx - 1] + c) / 2)

            if idx + 1 < len(centers):
                bottom = int((c + centers[idx + 1]) / 2)
            else:
                bottom = body_y_bottom

            if bottom > top + 3:
                row_spans.append((top, bottom))

        if self.debug:
            print(f"[SeparatedLayoutParser] inferred {len(row_spans)} body row bands from tokens.")

        return row_spans

    def _row_text_for_column(self, row_top: int, row_bottom: int, col_tokens):
        """
        For a given column + row band, stitch together all tokens that fall
        in that vertical band, ordered left→right.
        """
        if not col_tokens:
            return ""

        parts = []
        for tok in col_tokens:
            x1, y1, x2, y2 = tok["box_page"]
            y_center = 0.5 * (y1 + y2)
            if y_center < row_top or y_center >= row_bottom:
                continue
            parts.append((x1, tok["text"]))

        if not parts:
            return ""

        parts.sort(key=lambda t: t[0])
        return " ".join(p[1] for p in parts).strip()

    def _detect_row_divider_lines_in_strip(self, strip_gray):
        """
        Given a single body strip in GRAY (gridless),
        detect strong horizontal 'row divider' lines that span
        ~95%+ of the strip width.

        Returns:
            List of y-centers (ints) in STRIP-LOCAL coordinates.
        """
        import cv2
        import numpy as np  # noqa: F401

        if strip_gray is None or strip_gray.size == 0:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        if H_strip <= 0 or W_strip <= 0:
            return []

        # 1) Binarize (invert so lines are white on black)
        blur = cv2.GaussianBlur(strip_gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        # 2) Emphasize horizontal strokes with a wide, thin kernel
        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(15, int(0.40 * W_strip)),  # fairly wide
                1,                             # very thin vertically
            ),
        )
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

        # 3) Connected components to find horizontal blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            h_candidates,
            connectivity=8,
        )
        if num_labels <= 1:
            return []

        # Lines must span >= 95% of the strip width and be thin
        min_full_w = int(0.95 * W_strip)
        max_thick = max(2, int(0.03 * H_strip))

        ys_raw = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w < min_full_w:
                continue
            if h > max_thick:
                continue

            yc = y + h // 2
            ys_raw.append(int(yc))

        if not ys_raw:
            return []

        ys_raw.sort()

        # Collapse near-duplicate line centers
        merged = []
        MERGE_PX = 3
        for y in ys_raw:
            if not merged or abs(y - merged[-1]) > MERGE_PX:
                merged.append(y)

        if self.debug:
            print(
                f"[SeparatedLayoutParser] Detected {len(merged)} long horizontal lines "
                f"in strip (H={H_strip}, W={W_strip})."
            )

        return merged

    def _compute_row_bands_in_strip(self, strip_gray):
        """
        Given a body-strip in GRAY (gridless), use the detected long
        horizontal lines to define row bands in STRIP-LOCAL coordinates.

        Behavior:
          - Treat regions:
                top_of_strip → first_line,
                each line_i → line_{i+1},
                last_line → bottom_of_strip
            as potential row bands.
          - Use a small padding inside each segment so we avoid the core of
            the grid line, but do NOT over-trim (to keep single characters
            near the line, like a lone '1').

        Returns:
            List of (row_top, row_bottom) in STRIP-LOCAL coordinates.
        """
        import cv2  # noqa: F401

        if strip_gray is None or strip_gray.size == 0:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        if H_strip <= 0 or W_strip <= 0:
            return []

        line_ys = self._detect_row_divider_lines_in_strip(strip_gray)
        if not line_ys:
            # No lines at all -> treat whole strip as one band
            return [(0, H_strip)]

        # Sort and clamp
        line_ys = sorted(int(y) for y in line_ys)
        line_ys = [max(0, min(H_strip - 1, y)) for y in line_ys]

        # Build raw segments:
        #   [0 -> first_line], [line_i -> line_{i+1}], [last_line -> H_strip]
        segments = []
        prev = 0
        for y in line_ys:
            if y > prev:
                segments.append((prev, y))
            prev = y
        if H_strip > prev:
            segments.append((prev, H_strip))

        bands = []
        prev_bottom = 0
        for (top, bottom) in segments:
            if bottom <= top + 3:
                continue

            span = bottom - top
            # Smaller padding: 2% of span, min 1 px
            pad = max(1, int(0.02 * span))

            # Slightly "inflate" compared to the raw segment, but keep ordering
            row_top = max(prev_bottom, top + pad - 1)
            row_bottom = min(H_strip, bottom - pad + 1)

            if row_bottom > row_top + 2:
                bands.append((row_top, row_bottom))
                prev_bottom = row_bottom

        if self.debug:
            print(
                f"[SeparatedLayoutParser] _compute_row_bands_in_strip: "
                f"lines={line_ys} → segments={segments} → {len(bands)} row bands"
            )

        if not bands:
            # Ultra-defensive fallback
            return [(0, H_strip)]

        return bands

    def _ocr_body_column_by_rows(
        self,
        strip_gray,
        x_left: int,
        body_y_top: int,
        row_bands_strip,
    ):
        """
        OCR a single trip/poles body strip ROW-BY-ROW.

        Inputs:
          - strip_gray: gridless gray crop for this body column
                        shape (H_strip, W_strip)
          - x_left:     column's left X in PAGE coordinates
          - body_y_top: Y at which this strip starts in PAGE coordinates
          - row_bands_strip: list of (row_top, row_bottom) in STRIP-LOCAL coords

        Returns:
          Flat list of OCR tokens, each with box_page coordinates.
        """
        import cv2

        if strip_gray is None or strip_gray.size == 0 or self.reader is None:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        tokens = []

        for (row_top, row_bottom) in row_bands_strip:
            row_top = max(0, min(H_strip - 1, int(row_top)))
            row_bottom = max(row_top + 1, min(H_strip, int(row_bottom)))
            if row_bottom <= row_top:
                continue

            row_gray = strip_gray[row_top:row_bottom, :]
            if row_gray.size == 0:
                continue

            # Up-res this single row band
            row_up = cv2.resize(
                row_gray,
                None,
                fx=_HDR_OCR_SCALE,
                fy=_HDR_OCR_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )

            try:
                dets = self.reader.readtext(
                    row_up,
                    detail=1,
                    paragraph=False,
                    allowlist=_HDR_OCR_ALLOWLIST,
                    mag_ratio=1.0,
                    contrast_ths=0.05,
                    adjust_contrast=0.7,
                    text_threshold=0.4,
                    low_text=0.25,
                )
            except Exception:
                dets = []

            for box, txt, conf in dets:
                # length-aware confidence thresholding (same as combined)
                txt_clean = str(txt or "").strip()
                if not txt_clean:
                    continue

                try:
                    conf_f = float(conf or 0.0)
                except Exception:
                    conf_f = 0.0

                min_conf = _HDR_MIN_CONF
                if len(txt_clean) == 1:
                    # Relax threshold a bit for single-character tokens (e.g. '1')
                    min_conf = _HDR_MIN_CONF * 0.7

                if conf_f < min_conf:
                    continue

                # box is in UPSCALED row coords -> map back to row-local
                pts_local = [
                    (
                        int(p[0] / _HDR_OCR_SCALE),
                        int(p[1] / _HDR_OCR_SCALE),
                    )
                    for p in box
                ]
                xs = [p[0] for p in pts_local]
                ys = [p[1] for p in pts_local]

                x1_row = max(0, min(W_strip - 1, min(xs)))
                x2_row = max(0, min(W_strip - 1, max(xs)))
                y1_row = max(0, min((row_bottom - row_top) - 1, min(ys)))
                y2_row = max(0, min((row_bottom - row_top) - 1, max(ys)))

                # Map to STRIP-LOCAL coords
                x1_strip = x1_row
                x2_strip = x2_row
                y1_strip = row_top + y1_row
                y2_strip = row_top + y2_row

                # Map to PAGE coords
                x1_page = x_left + x1_strip
                x2_page = x_left + x2_strip
                y1_page = body_y_top + y1_strip
                y2_page = body_y_top + y2_strip

                tokens.append(
                    {
                        "text": txt_clean,
                        "conf": conf_f,
                        "box_page": [int(x1_page), int(y1_page), int(x2_page), int(y2_page)],
                    }
                )

        return tokens

    def _parse_trip_value(self, text: str):
        """
        Extract an amp rating (e.g. '20', '20A', '20 AMP') from trip text.
        Returns an int or None.
        """
        import re

        if not text:
            return None

        t = text.upper()
        # If clearly marked SPARE, ignore
        if "SPARE" in t or "SPACE" in t:
            return None

        m = re.search(r"(\d{1,4})", t)
        if not m:
            return None

        try:
            val = int(m.group(1))
        except Exception:
            return None

        if val <= 0:
            return None

        return val

    def _parse_poles_value(self, text: str):
        """
        Extract a pole count (1/2/3) from poles text.
        Returns 1,2,3 or None.
        """
        import re

        if not text:
            return None

        t = text.upper()
        if "SPARE" in t or "SPACE" in t:
            return None

        # Strong patterns first
        if "3P" in t or "3-P" in t or "3 POLE" in t:
            return 3
        if "2P" in t or "2-P" in t or "2 POLE" in t:
            return 2
        if "1P" in t or "1-P" in t or "1 POLE" in t:
            return 1
        if "SP" in t:  # SP = single pole
            return 1

        # Fallback: bare digit 1/2/3
        m = re.search(r"\b([123])\b", t)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

        return None

    def parse(self, analyzer_result: dict, header_scan: dict) -> dict:
        """
        Given analyzer_result + header_scan with normalizedColumns.layout == 'separated',
        we:
          - crop body strips for trip/poles columns
          - compute row bands *independently per side* from horizontal row-divider lines
          - OCR each trip/poles column ROW-BY-ROW using that side's bands
          - associate each row's amps + poles per side
          - accumulate breaker counts
        """
        import os
        import cv2
        from typing import Dict

        # --- image sources ---
        raw_gray = analyzer_result.get("gray")
        if raw_gray is None:
            raw_gray = _ensure_gray(analyzer_result)

        gray_body  = raw_gray
        gray_lines = raw_gray

        if gray_body is None or gray_lines is None:
            if self.debug:
                print("[SeparatedLayoutParser] Missing gray_body/gray_lines; cannot crop body columns.")
            return {
                "layout": "separated",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
            }

        H, W = gray_body.shape[:2]

        src_path = analyzer_result.get("src_path")
        src_dir  = analyzer_result.get("src_dir") or os.path.dirname(src_path or ".")
        debug_dir = analyzer_result.get("debug_dir") or os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(src_path or "panel"))[0]

        # --- header anchor / body top (MUST have header_bottom_y) ---
        header_y        = analyzer_result.get("header_y")
        header_bottom_y = analyzer_result.get("header_bottom_y")
        footer_y        = analyzer_result.get("footer_y")

        if not isinstance(header_bottom_y, (int, float)):
            if self.debug:
                print(
                    "[SeparatedLayoutParser] Missing header_bottom_y; "
                    "cannot determine body top. Bailing."
                )
                print(f"  header_y={header_y!r}, header_bottom_y={header_bottom_y!r}")
            return {
                "layout": "separated",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
                "error": "Missing header_bottom_y from analyzer; cannot determine body band.",
            }

        if not isinstance(footer_y, (int, float)):
            if self.debug:
                print(
                    "[SeparatedLayoutParser] Missing footer_y; "
                    "cannot determine body bottom. Bailing."
                )
                print(f"  footer_y={footer_y!r}")
            return {
                "layout": "separated",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
                "error": "Missing footer_y from analyzer; cannot determine body band.",
            }

        body_y_top    = max(0, int(header_bottom_y))
        body_y_bottom = int(footer_y)

        if self.debug:
            print(
                f"[SeparatedLayoutParser] Body Y-range: [{body_y_top}, {body_y_bottom}) "
                f"(H={H})  (using header_bottom_y→footer_y as body band)"
            )

        if body_y_bottom <= body_y_top + 4:
            if self.debug:
                print("[SeparatedLayoutParser] Body band too small; skipping body columns.")
            return {
                "layout": "separated",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
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
                "breakerCounts": {},
            }

        cols_summary = normalized.get("columns", []) or []

        # We want all columns that are trip or poles (both sides)
        wanted_roles = {"trip", "poles"}

        body_columns = []

        # --- collect body strips for trip/poles (geometry only) ---
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

            body_columns.append(
                {
                    "index": col["index"],
                    "role": role,
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_top": body_y_top,
                    "y_bottom": body_y_bottom,
                    "debugImageOverlay": None,  # filled in below when debug=True
                    # 'side' assigned below
                }
            )

        if self.debug:
            print(
                f"[SeparatedLayoutParser] Extracted {len(body_columns)} body columns "
                f"for trip/poles."
            )

        if not body_columns:
            return {
                "layout": "separated",
                "bodyColumns": body_columns,
                "detected_breakers": [],
                "breakerCounts": {},
            }

        # --- assign columns to left/right sides based on X center ---
        role_cols = [c for c in body_columns if c["role"] in wanted_roles]
        global_left = min(c["x_left"] for c in role_cols)
        global_right = max(c["x_right"] for c in role_cols)
        center_x = 0.5 * (global_left + global_right)

        for col in role_cols:
            col_center = 0.5 * (col["x_left"] + col["x_right"])
            col["side"] = "left" if col_center < center_x else "right"

        by_side = {
            "left": {"trip": None, "poles": None},
            "right": {"trip": None, "poles": None},
        }

        for col in role_cols:
            side = col["side"]
            role = col["role"]
            # First seen per (side, role) wins
            if by_side[side][role] is None:
                by_side[side][role] = col["index"]

        if self.debug:
            print("[SeparatedLayoutParser] trip/poles columns by side:", by_side)

        # --- determine row bands *per side* from that side's reference column ---
        row_bands_by_side = {"left": None, "right": None}
        row_spans_by_side = {"left": [], "right": []}

        for side in ("left", "right"):
            side_cols = [c for c in role_cols if c["side"] == side]
            if not side_cols:
                continue

            # Prefer trip column on that side as reference, else any poles column
            ref_col = None
            for c in side_cols:
                if c["role"] == "trip":
                    ref_col = c
                    break
            if ref_col is None:
                for c in side_cols:
                    if c["role"] == "poles":
                        ref_col = c
                        break

            if ref_col is None:
                continue  # nothing usable on this side

            ref_strip = gray_body[
                ref_col["y_top"]:ref_col["y_bottom"],
                ref_col["x_left"]:ref_col["x_right"],
            ]
            if ref_strip.size == 0:
                row_bands_strip = [(0, ref_col["y_bottom"] - ref_col["y_top"])]
            else:
                row_bands_strip = self._compute_row_bands_in_strip(ref_strip)
                if not row_bands_strip:
                    row_bands_strip = [(0, ref_col["y_bottom"] - ref_col["y_top"])]

            row_bands_by_side[side] = row_bands_strip
            row_spans_by_side[side] = [
                (body_y_top + top_s, body_y_top + bottom_s)
                for (top_s, bottom_s) in row_bands_strip
            ]

            if self.debug:
                print(
                    f"[SeparatedLayoutParser] Side='{side}' using "
                    f"{len(row_spans_by_side[side])} row bands from horizontal lines."
                )

        # --- OCR each trip/poles column ROW-BY-ROW using that column's side bands ---
        tokens_by_col_index: Dict[int, List[Dict]] = {}

        for col in body_columns:
            idx   = col["index"]
            x_l   = col["x_left"]
            x_r   = col["x_right"]
            y_top = col["y_top"]
            y_bot = col["y_bottom"]
            side  = col.get("side", "left")

            row_bands_strip = row_bands_by_side.get(side)
            if not row_bands_strip:
                tokens_by_col_index[idx] = []
                continue

            strip = gray_body[y_top:y_bot, x_l:x_r]
            if strip.size == 0:
                tokens_by_col_index[idx] = []
                continue

            tokens_by_col_index[idx] = self._ocr_body_column_by_rows(
                strip_gray=strip,
                x_left=x_l,
                body_y_top=y_top,
                row_bands_strip=row_bands_strip,
            )

        if not tokens_by_col_index:
            return {
                "layout": "separated",
                "bodyColumns": body_columns,
                "detected_breakers": [],
                "breakerCounts": {},
            }

        # --- Build OCR overlays for each body column (debug only) ---
        if self.debug:
            for col in body_columns:
                idx   = col["index"]
                role  = col["role"]
                x_l   = col["x_left"]
                x_r   = col["x_right"]
                y_top = col["y_top"]
                y_bot = col["y_bottom"]

                col_tokens = tokens_by_col_index.get(idx, [])
                if not col_tokens:
                    continue

                # Crop the strip again for visualization only
                strip = gray_body[y_top:y_bot, x_l:x_r]
                if strip.size == 0:
                    continue

                H_strip, W_strip = strip.shape[:2]
                vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

                # Draw a border + label for the column
                cv2.rectangle(
                    vis,
                    (0, 0),
                    (vis.shape[1] - 1, vis.shape[0] - 1),
                    (0, 255, 255),
                    1,
                )
                lbl = f"BODY {role.upper()}  col={idx}"
                cv2.putText(
                    vis,
                    lbl,
                    (8, max(16, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Draw detected row-divider lines (for debug) in BLUE
                line_ys = self._detect_row_divider_lines_in_strip(strip)
                for yc in line_ys:
                    yc_int = max(0, min(H_strip - 1, int(yc)))
                    cv2.line(
                        vis,
                        (0, yc_int),
                        (W_strip - 1, yc_int),
                        (255, 0, 0),  # BLUE in BGR
                        1,
                    )
                    cv2.putText(
                        vis,
                        "ROW",
                        (4, max(10, yc_int - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Draw each OCR token (convert from page coords to strip-local coords)
                for tok in col_tokens:
                    x1p, y1p, x2p, y2p = tok["box_page"]
                    text = tok.get("text", "")
                    conf = tok.get("conf", 0.0)

                    # local coords within the strip
                    x1 = max(0, min(W_strip - 1, x1p - x_l))
                    x2 = max(0, min(W_strip - 1, x2p - x_l))
                    y1 = max(0, min(H_strip - 1, y1p - y_top))
                    y2 = max(0, min(H_strip - 1, y2p - y_top))

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    label = f"{text} ({conf:.2f})"
                    ty = y1 - 4 if y1 - 4 > 10 else y2 + 12
                    cv2.putText(
                        vis,
                        label,
                        (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                overlay_path = os.path.join(
                    debug_dir,
                    f"{base}_parser_body_sep_col{idx}_{role}_overlay.png",
                )
                try:
                    cv2.imwrite(overlay_path, vis)
                    col["debugImageOverlay"] = overlay_path
                except Exception as e:
                    print(
                        f"[SeparatedLayoutParser] Failed to write body overlay col "
                        f"{idx} ({role}): {e}"
                    )
                    col["debugImageOverlay"] = None

        detected_breakers = []
        breaker_counts: Dict[str, int] = {}

        # --- walk rows and associate amps + poles *per side* ---
        for side in ("left", "right"):
            trip_idx = by_side[side].get("trip")
            poles_idx = by_side[side].get("poles")
            side_row_spans = row_spans_by_side.get(side) or []

            if trip_idx is None or poles_idx is None or not side_row_spans:
                continue

            trip_tokens = tokens_by_col_index.get(trip_idx, [])
            poles_tokens = tokens_by_col_index.get(poles_idx, [])

            for row_idx, (row_top, row_bottom) in enumerate(side_row_spans):
                trip_text = self._row_text_for_column(row_top, row_bottom, trip_tokens)
                poles_text = self._row_text_for_column(row_top, row_bottom, poles_tokens)

                amps = self._parse_trip_value(trip_text)
                poles = self._parse_poles_value(poles_text)

                if amps is None or poles is None:
                    continue

                key = f"{amps}A_{poles}P"
                breaker_counts[key] = breaker_counts.get(key, 0) + 1

                detected_breakers.append(
                    {
                        "side": side,
                        "rowIndex": row_idx,  # index is per-side now
                        "rowTop": row_top,
                        "rowBottom": row_bottom,
                        "amperage": int(amps),
                        "poles": int(poles),
                        "tripText": trip_text,
                        "polesText": poles_text,
                    }
                )

        if self.debug:
            print(
                f"[SeparatedLayoutParser] detected_breakers={len(detected_breakers)}, "
                f"unique combos={len(breaker_counts)}"
            )

        return {
            "layout": "separated",
            "bodyColumns": body_columns,
            "detected_breakers": detected_breakers,
            "breakerCounts": breaker_counts,
        }

class CombinedLayoutParser:
    """
    Handles parsing when the header layout is 'combined':
      - Trip + poles live together in a single combo column (per side).
      - Identifies body rows from the grid lines.
      - OCRs the combo body strips (split into top/bottom halves).
      - For each row, parses a combined "amps/poles" text like:
            20A 1P
            20A1P
            20 A 1 P
            20A/1P
            20/1
            20-1
            1 P 20 A
        into (amps, poles), with guards:
          * poles ∈ {1,2,3}
          * amps > 0 and amps % 5 == 0   (amps always end in 0 or 5)
    """

    def __init__(self, *, debug: bool = False, reader=None):
        self.debug = bool(debug)
        self.reader = reader  # shared EasyOCR instance (same as header)

    def _infer_body_rows_from_tokens(
        self,
        tokens_by_col_index: Dict[int, List[Dict]],
        body_y_top: int,
        body_y_bottom: int,
    ):
        """
        Infer body row bands purely from OCR token positions (gridless body),
        instead of using horizontal grid lines.

        Shared logic with SeparatedLayoutParser but kept separate for clarity.
        """
        ys = []

        for col_tokens in tokens_by_col_index.values():
            for tok in (col_tokens or []):
                try:
                    x1, y1, x2, y2 = tok["box_page"]
                except Exception:
                    continue
                y_center = 0.5 * (y1 + y2)
                if y_center < body_y_top or y_center > body_y_bottom:
                    continue
                ys.append(float(y_center))

        if not ys:
            if self.debug:
                print("[CombinedLayoutParser] No token Y-centers found; no body rows inferred.")
            return []

        ys.sort()

        if len(ys) == 1:
            return [(int(body_y_top), int(body_y_bottom))]

        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        gaps_sorted = sorted(gaps)
        mid = len(gaps_sorted) // 2
        if len(gaps_sorted) % 2 == 0:
            median_gap = 0.5 * (gaps_sorted[mid - 1] + gaps_sorted[mid])
        else:
            median_gap = gaps_sorted[mid]

        body_height = max(1, body_y_bottom - body_y_top)
        min_gap = 5.0
        max_gap = max(20.0, 0.15 * body_height)
        gap_threshold = max(min_gap, min(median_gap * 1.5, max_gap))

        if self.debug:
            print(
                f"[CombinedLayoutParser] token Y-centers: count={len(ys)}, "
                f"median_gap={median_gap:.2f}, gap_threshold={gap_threshold:.2f}"
            )

        clusters = [[ys[0]]]
        for y in ys[1:]:
            if y - clusters[-1][-1] <= gap_threshold:
                clusters[-1].append(y)
            else:
                clusters.append([y])

        centers = [int(round(sum(c) / len(c))) for c in clusters]
        centers.sort()

        if self.debug:
            print(
                f"[CombinedLayoutParser] inferred row centers={centers} "
                f"(body_y_top={body_y_top}, body_y_bottom={body_y_bottom})"
            )

        row_spans = []
        for idx, c in enumerate(centers):
            if idx == 0:
                top = body_y_top
            else:
                top = int((centers[idx - 1] + c) / 2)

            if idx + 1 < len(centers):
                bottom = int((c + centers[idx + 1]) / 2)
            else:
                bottom = body_y_bottom

            if bottom > top + 3:
                row_spans.append((top, bottom))

        if self.debug:
            print(f"[CombinedLayoutParser] inferred {len(row_spans)} body row bands from tokens.")

        return row_spans

    def _ocr_body_column(
        self,
        gray_body,
        body_y_top: int,
        body_y_bottom: int,
        x_left: int,
        x_right: int,
    ):
        """
        OCR a single combo body strip and return tokens in PAGE coordinates.

        UPDATED:
          - No splitting into halves; OCR the full strip at once.
        """
        import cv2

        if gray_body is None or self.reader is None:
            return []

        H, W = gray_body.shape[:2]
        body_y_top = max(0, min(H - 1, int(body_y_top)))
        body_y_bottom = max(body_y_top + 1, min(H, int(body_y_bottom)))
        x_left = max(0, min(W - 1, int(x_left)))
        x_right = max(x_left + 1, min(W, int(x_right)))

        band = gray_body[body_y_top:body_y_bottom, x_left:x_right]
        if band.size == 0:
            return []

        H_band, W_band = band.shape[:2]
        if H_band <= 0:
            return []

        # Up-res entire band
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
        except Exception:
            dets = []

        tokens = []

        for box, txt, conf in dets:
            try:
                conf_f = float(conf or 0.0)
            except Exception:
                conf_f = 0.0

            if conf_f < _HDR_MIN_CONF:
                continue

            # OCR box is in upscaled band coordinates → map back to band
            pts_local = [
                (
                    int(p[0] / _HDR_OCR_SCALE),
                    int(p[1] / _HDR_OCR_SCALE),
                )
                for p in box
            ]
            xs = [p[0] for p in pts_local]
            ys = [p[1] for p in pts_local]

            x1_band = max(0, min(W_band - 1, min(xs)))
            x2_band = max(0, min(W_band - 1, max(xs)))
            y1_band = max(0, min(H_band - 1, min(ys)))
            y2_band = max(0, min(H_band - 1, max(ys)))

            # Map band coords into PAGE coords
            x1_page = x_left + x1_band
            x2_page = x_left + x2_band
            y1_page = body_y_top + y1_band
            y2_page = body_y_top + y2_band

            tokens.append(
                {
                    "text": str(txt or "").strip(),
                    "conf": conf_f,
                    "box_page": [int(x1_page), int(y1_page), int(x2_page), int(y2_page)],
                }
            )

        return tokens

    def _row_text_for_column(self, row_top: int, row_bottom: int, col_tokens):
        """
        For a given column + row band, stitch together all tokens that fall
        in that vertical band, ordered left→right.
        """
        if not col_tokens:
            return ""

        parts = []
        for tok in col_tokens:
            x1, y1, x2, y2 = tok["box_page"]
            y_center = 0.5 * (y1 + y2)
            if y_center < row_top or y_center >= row_bottom:
                continue
            parts.append((x1, tok["text"]))

        if not parts:
            return ""

        parts.sort(key=lambda t: t[0])
        return " ".join(p[1] for p in parts).strip()

    def _detect_row_divider_lines_in_strip(self, strip_gray):
        """
        Given a single body combo-strip in GRAY (gridless),
        detect strong horizontal 'row divider' lines that span
        ~95%+ of the strip width.

        Returns:
            List of y-centers (ints) in STRIP-LOCAL coordinates.
        """
        import cv2
        import numpy as np  # noqa: F401

        if strip_gray is None or strip_gray.size == 0:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        if H_strip <= 0 or W_strip <= 0:
            return []

        # 1) Binarize (invert so lines are white on black)
        blur = cv2.GaussianBlur(strip_gray, (3, 3), 0)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )

        # 2) Emphasize horizontal strokes with a wide, thin kernel
        Kh = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(15, int(0.40 * W_strip)),  # fairly wide
                1,                             # very thin vertically
            ),
        )
        h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

        # 3) Connected components to find horizontal blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            h_candidates,
            connectivity=8,
        )
        if num_labels <= 1:
            return []

        # Lines must span >= 95% of the strip width and be thin
        min_full_w = int(0.95 * W_strip)
        max_thick = max(2, int(0.03 * H_strip))

        ys_raw = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w < min_full_w:
                continue
            if h > max_thick:
                continue

            yc = y + h // 2
            ys_raw.append(int(yc))

        if not ys_raw:
            return []

        ys_raw.sort()

        # Collapse near-duplicate line centers
        merged = []
        MERGE_PX = 3
        for y in ys_raw:
            if not merged or abs(y - merged[-1]) > MERGE_PX:
                merged.append(y)

        if self.debug:
            print(
                f"[CombinedLayoutParser] Detected {len(merged)} long horizontal lines "
                f"in strip (H={H_strip}, W={W_strip})."
            )

        return merged

    def _compute_row_bands_in_strip(self, strip_gray):
        """
        Given a combo body-strip in GRAY (gridless), use the detected long
        horizontal lines to define row bands in STRIP-LOCAL coordinates.

        NEW BEHAVIOR:
          - We treat regions:
                top_of_strip → first_line,
                each line_i → line_{i+1},
                last_line → bottom_of_strip
            as potential row bands.
          - This way, we still get a top row and bottom row even if there
            is no grid line at the very top or very bottom of the body.

        Returns:
            List of (row_top, row_bottom) in STRIP-LOCAL coordinates.
        """
        import cv2  # noqa: F401

        if strip_gray is None or strip_gray.size == 0:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        if H_strip <= 0 or W_strip <= 0:
            return []

        line_ys = self._detect_row_divider_lines_in_strip(strip_gray)
        if not line_ys:
            # No lines at all -> treat whole strip as one band
            return [(0, H_strip)]

        # Sort and clamp
        line_ys = sorted(int(y) for y in line_ys)
        line_ys = [max(0, min(H_strip - 1, y)) for y in line_ys]

        # Build raw segments:
        # [0 -> first_line], [line_i -> line_{i+1}], [last_line -> H_strip]
        segments = []
        prev = 0
        for y in line_ys:
            if y > prev:
                segments.append((prev, y))
            prev = y

        if H_strip > prev:
            segments.append((prev, H_strip))

        bands = []
        prev_bottom = 0
        for (top, bottom) in segments:
            if bottom <= top + 3:
                continue

            span = bottom - top
            # Smaller padding than before:2% of span, min 1 px
            pad = max(1, int(0.02 * span))

            # Slightly "inflate" compared to the raw segment, but keep ordering
            row_top = max(prev_bottom, top + pad - 1)
            row_bottom = min(H_strip, bottom - pad + 1)

            if row_bottom > row_top + 2:
                bands.append((row_top, row_bottom))
                prev_bottom = row_bottom

        if self.debug:
            print(
                f"[CombinedLayoutParser] _compute_row_bands_in_strip: "
                f"lines={line_ys} → segments={segments} → {len(bands)} row bands"
            )

        if not bands:
            # Ultra-defensive fallback
            return [(0, H_strip)]

        return bands

    def _ocr_body_column_by_rows(
        self,
        strip_gray,
        x_left: int,
        body_y_top: int,
        row_bands_strip,
    ):
        """
        OCR a single combo body strip ROW-BY-ROW.

        Inputs:
          - strip_gray: gridless gray crop for this combo column
                        shape (H_strip, W_strip)
          - x_left:     column's left X in PAGE coordinates
          - body_y_top: Y at which this strip starts in PAGE coordinates
          - row_bands_strip: list of (row_top, row_bottom) in STRIP-LOCAL coords

        Returns:
          Flat list of OCR tokens, each with box_page coordinates.
        """
        import cv2

        if strip_gray is None or strip_gray.size == 0 or self.reader is None:
            return []

        H_strip, W_strip = strip_gray.shape[:2]
        tokens = []

        for (row_top, row_bottom) in row_bands_strip:
            row_top = max(0, min(H_strip - 1, int(row_top)))
            row_bottom = max(row_top + 1, min(H_strip, int(row_bottom)))
            if row_bottom <= row_top:
                continue

            row_gray = strip_gray[row_top:row_bottom, :]
            if row_gray.size == 0:
                continue

            # Up-res this single row band
            row_up = cv2.resize(
                row_gray,
                None,
                fx=_HDR_OCR_SCALE,
                fy=_HDR_OCR_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )

            try:
                dets = self.reader.readtext(
                    row_up,
                    detail=1,
                    paragraph=False,
                    allowlist=_HDR_OCR_ALLOWLIST,
                    mag_ratio=1.0,
                    contrast_ths=0.05,
                    adjust_contrast=0.7,
                    text_threshold=0.4,
                    low_text=0.25,
                )
            except Exception:
                dets = []

            for box, txt, conf in dets:
                # --- NEW: length-aware confidence thresholding ---
                txt_clean = str(txt or "").strip()
                if not txt_clean:
                    continue

                try:
                    conf_f = float(conf or 0.0)
                except Exception:
                    conf_f = 0.0

                # Default threshold
                min_conf = _HDR_MIN_CONF
                # But if it's a single character (e.g. '1'), allow a bit lower
                if len(txt_clean) == 1:
                    min_conf = _HDR_MIN_CONF * 0.7

                if conf_f < min_conf:
                    continue
                # --- END NEW LOGIC ---

                # box is in UPSCALED row coords -> map back to row-local
                pts_local = [
                    (
                        int(p[0] / _HDR_OCR_SCALE),
                        int(p[1] / _HDR_OCR_SCALE),
                    )
                    for p in box
                ]
                xs = [p[0] for p in pts_local]
                ys = [p[1] for p in pts_local]

                x1_row = max(0, min(W_strip - 1, min(xs)))
                x2_row = max(0, min(W_strip - 1, max(xs)))
                y1_row = max(0, min((row_bottom - row_top) - 1, min(ys)))
                y2_row = max(0, min((row_bottom - row_top) - 1, max(ys)))

                # Map to STRIP-LOCAL coords
                x1_strip = x1_row
                x2_strip = x2_row
                y1_strip = row_top + y1_row
                y2_strip = row_top + y2_row

                # Map to PAGE coords
                x1_page = x_left + x1_strip
                x2_page = x_left + x2_strip
                y1_page = body_y_top + y1_strip
                y2_page = body_y_top + y2_strip

                tokens.append(
                    {
                        "text": txt_clean,
                        "conf": conf_f,
                        "box_page": [int(x1_page), int(y1_page), int(x2_page), int(y2_page)],
                    }
                )

        return tokens

    def _normalize_digitish_chars(self, s: str) -> str:
        """
        Normalize characters commonly mis-OCR'd inside numeric runs:

          - S / Z / 5 / $ -> '5'
          - 0 / O / D / Q / G -> '0'
          - 1 / I / L / J / ! / | -> '1'

        We apply this only to the combo text, and SPARE/SPACE detection is
        done on the raw string *before* normalization.
        """
        mapping_5 = set("5S$Z")
        mapping_0 = set("0ODQG")
        mapping_1 = set("1ILJ!|")

        out = []
        for ch in s:
            cu = ch.upper()
            if cu in mapping_5:
                out.append("5")
            elif cu in mapping_0:
                out.append("0")
            elif cu in mapping_1:
                out.append("1")
            else:
                out.append(cu)
        return "".join(out)

    def _is_valid_amps(self, val: int) -> bool:
        """
        Guard for valid amps:
          - > 0
          - ends in 0 or 5
          - optionally, >= 10 (to avoid silly 5A parses)
        """
        if val is None:
            return False
        if val <= 0:
            return False
        if val < 10:
            return False
        if val % 5 != 0:
            return False
        return True
    
    def _decode_amp_digits(self, digits: str) -> Optional[int]:
        """
        Normalize amperage digit strings coming from combo cells.

        Handles:
          - '20'   -> 20
          - '201'  -> 20   (slash misread as '1')
          - '20/'  -> 20   (after char normalization -> '201')
          - '20I'/'20L'/'20J'/'20T' -> 20  (via normalization → '201')
          - '251'  -> 25   (one junk digit inside)
        """
        if not digits:
            return None

        # First try straight parse
        try:
            val = int(digits)
        except ValueError:
            val = None

        if val is not None and self._is_valid_amps(val):
            return val

        # Classic "lost slash" pattern at the end, e.g.:
        #   '201' -> '20', '151' -> '15', '251' -> '25'
        if len(digits) >= 3 and digits.endswith("1"):
            try:
                val2 = int(digits[:-1])
            except ValueError:
                val2 = None
            if val2 is not None and self._is_valid_amps(val2):
                return val2

        # General: remove exactly one junk digit and see if we get a valid amp
        for i in range(len(digits)):
            candidate = digits[:i] + digits[i + 1 :]
            if not candidate:
                continue
            try:
                val2 = int(candidate)
            except ValueError:
                continue
            if self._is_valid_amps(val2):
                return val2

        return None

    def _decode_pole_digits(self, digits: str) -> Optional[int]:
        """
        Normalize pole digit strings.

        Handles ONLY:
        - '1'    -> 1
        - '2'    -> 2
        - '3'    -> 3
        - '11'   -> 1  (leading '1' is misread slash)
        - '12'   -> 2
        - '13'   -> 3

        Anything else ('20', '15', '101', etc.) is rejected.
        """
        if not digits:
            return None

        # Single digit is simple
        if len(digits) == 1:
            try:
                d = int(digits)
            except ValueError:
                return None
            return d if d in (1, 2, 3) else None

        # 2-digit pattern where leading '1' is our fake slash
        if len(digits) == 2 and digits[0] == "1" and digits[1] in "123":
            return int(digits[1])

        # All other multi-digit cases are considered invalid for poles
        return None

    def _validate_amp_pole_pair(self, amp_str: str, pole_str: str):
        """
        Given raw amp/pole substrings (already isolated by regex),
        normalize to digits and enforce:
          - poles ∈ {1,2,3} (with 11/12/13 cleanup)
          - amps % 5 == 0 and > 0 (with '201' / extra-digit cleanup)

        Always returns a 2-tuple: (amps or None, poles or None).
        """
        amp_digits = "".join(ch for ch in amp_str if ch.isdigit())
        pole_digits = "".join(ch for ch in pole_str if ch.isdigit())

        if not amp_digits or not pole_digits:
            return None, None

        amps = self._decode_amp_digits(amp_digits)
        if amps is None:
            return None, None

        poles = self._decode_pole_digits(pole_digits)
        if poles is None:
            return None, None

        return amps, poles

    def _parse_combo_cell(self, text: str):
        """
        Parse a combined 'amps/poles' text blob into a list of (amps, poles) pairs.

        Handles:
          - 20A 1P
          - 20A1P
          - 20 A 1 P
          - 20A/1P
          - 20/1
          - 20-1
          - 1 P 20 A
          - 20 13  (→ 20 A / 3 P)
          - 20 12  (→ 20 A / 2 P)
          - 20 1 1 (→ 20 A / 1 P)

        Plus "lost slash" dense-digit cases like:
          - 2071  → 20 / 1
          - 25012 → 250 / 2

        Returns:
          list of (amps:int, poles:int). Empty list if nothing valid.
        """
        import re

        if not text:
            return []

        raw = text.upper().strip()

        # Ignore clearly marked spares/spaces
        if "SPARE" in raw or "SPACE" in raw:
            return []

        # Normalize mis-OCR'd chars (S/Z→5, O/Q/D/G→0, I/L/J/|/!→1)
        norm = self._normalize_digitish_chars(raw)

        pairs = []

        # --- 1) Explicit amp/pole regex patterns (amps-first and poles-first) ---

        amp_first_re = re.compile(
            r"""
            (?P<amp>\d{1,4})      # 1-4 digits (amps)
            \s*A?                 # optional 'A'
            \s*[/\-]?\s*          # optional separator '/', '-'
            (?P<pole>\d{1,2})     # 1-2 digits (poles: 1, 2, 3, 11,12,13, etc.)
            \s*P?                 # optional 'P'
            """,
            re.VERBOSE,
        )

        pole_first_re = re.compile(
            r"""
            (?P<pole>\d{1,2})     # 1-2 digits (poles)
            \s*P?                 # optional 'P'
            \s*[/\-]?\s*          # optional separator '/', '-'
            (?P<amp>\d{1,4})      # 1-4 digits (amps)
            \s*A?                 # optional 'A'
            """,
            re.VERBOSE,
        )

        for pattern in (amp_first_re, pole_first_re):
            for m in pattern.finditer(norm):
                amp_str = m.group("amp")
                pole_str = m.group("pole")
                amps, poles = self._validate_amp_pole_pair(amp_str, pole_str)
                if amps is not None and poles is not None:
                    pairs.append((amps, poles))

        if pairs:
            # Found at least one explicit combo; don't run other heuristics.
            return pairs

        # --- 2) Dense-digit "lost slash" heuristic (single pair max) ---

        digits = "".join(ch for ch in norm if ch.isdigit())
        if len(digits) >= 2:
            # Last digit is candidate pole; rest is amps-ish blob
            pole_candidate = self._decode_pole_digits(digits[-1:])
            left = digits[:-1]
            amp_candidate = self._decode_amp_digits(left)

            if pole_candidate is not None and amp_candidate is not None:
                return [(amp_candidate, pole_candidate)]

        # --- 3) Token-level heuristic for split tokens like '20 13', '20 1 1' ---

        tokens = [t for t in re.split(r"\s+", norm) if t]

        amp_candidates = []
        pole_candidates = []

        for tok in tokens:
            # Skip obvious junk like bare '/', '-', etc.
            if not any(ch.isdigit() for ch in tok):
                continue

            digit_str = "".join(ch for ch in tok if ch.isdigit())
            if not digit_str:
                continue

            amp_val = self._decode_amp_digits(digit_str)
            if amp_val is not None:
                amp_candidates.append(amp_val)

            pole_val = self._decode_pole_digits(digit_str)
            if pole_val is not None:
                pole_candidates.append(pole_val)

        if amp_candidates and pole_candidates:
            # Heuristic:
            #   - Use the *largest* amp (usually the multi-digit one like 20, 30, 50)
            #   - Use the *last* pole candidate (often from '13' or trailing '1')
            best_amp = max(amp_candidates)
            best_pole = pole_candidates[-1]
            return [(best_amp, best_pole)]

        # Nothing safely parsed
        return []

    def parse(self, analyzer_result: dict, header_scan: dict) -> dict:
        """
        Given analyzer_result + header_scan with normalizedColumns.layout == 'combined',
        we:
          - crop body strips for combo columns
          - compute row bands *independently per side* from horizontal row-divider lines
          - OCR each combo column ROW-BY-ROW using that side's bands
          - for each row+side, parse combined amps/poles text
          - accumulate breaker counts
        """
        import os
        import cv2
        from typing import Dict, Optional

        # --- image sources ---
        raw_gray = analyzer_result.get("gray")
        if raw_gray is None:
            raw_gray = _ensure_gray(analyzer_result)

        gray_body  = raw_gray
        gray_lines = raw_gray

        if gray_body is None or gray_lines is None:
            if self.debug:
                print("[CombinedLayoutParser] Missing gray_body/gray_lines; cannot crop body columns.")
            return {
                "layout": "combined",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
            }

        H, W = gray_body.shape[:2]

        src_path = analyzer_result.get("src_path")
        src_dir  = analyzer_result.get("src_dir") or os.path.dirname(src_path or ".")
        debug_dir = analyzer_result.get("debug_dir") or os.path.join(src_dir, "debug")
        if self.debug:
            os.makedirs(debug_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(src_path or "panel"))[0]

        # --- header anchor / body top (MUST have header_bottom_y) ---
        header_y        = analyzer_result.get("header_y")
        header_bottom_y = analyzer_result.get("header_bottom_y")
        footer_y        = analyzer_result.get("footer_y")

        if not isinstance(header_bottom_y, (int, float)):
            if self.debug:
                print(
                    "[CombinedLayoutParser] Missing header_bottom_y; "
                    "cannot determine body top. Bailing."
                )
                print(f"  header_y={header_y!r}, header_bottom_y={header_bottom_y!r}")
            return {
                "layout": "combined",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
                "error": "Missing header_bottom_y from analyzer; cannot determine body band.",
            }

        if not isinstance(footer_y, (int, float)):
            if self.debug:
                print(
                    "[CombinedLayoutParser] Missing footer_y; "
                    "cannot determine body bottom. Bailing."
                )
                print(f"  footer_y={footer_y!r}")
            return {
                "layout": "combined",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
                "error": "Missing footer_y from analyzer; cannot determine body band.",
            }

        body_y_top    = max(0, int(header_bottom_y))
        body_y_bottom = int(footer_y)

        if self.debug:
            print(
                f"[CombinedLayoutParser] Body Y-range: [{body_y_top}, {body_y_bottom}) "
                f"(H={H})  (using header_bottom_y→footer_y as body band)"
            )

        if body_y_bottom <= body_y_top + 4:
            if self.debug:
                print("[CombinedLayoutParser] Body band too small; skipping body columns.")
            return {
                "layout": "combined",
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
            }

        normalized = header_scan.get("normalizedColumns") or {}
        if normalized.get("layout") != "combined":
            # Guard: caller should only invoke this when layout == 'combined'
            if self.debug:
                print(
                    "[CombinedLayoutParser] Warning: called with layout "
                    f"{normalized.get('layout')}; expected 'combined'."
                )
            return {
                "layout": normalized.get("layout", "unknown"),
                "bodyColumns": [],
                "detected_breakers": [],
                "breakerCounts": {},
            }

        cols_summary = normalized.get("columns", []) or []

        # We want all columns that are combo (both sides)
        wanted_roles = {"combo"}

        body_columns = []

        # --- collect combo body columns (geometry only) ---
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

            body_columns.append(
                {
                    "index": col["index"],
                    "role": role,
                    "x_left": x_left,
                    "x_right": x_right,
                    "y_top": body_y_top,
                    "y_bottom": body_y_bottom,
                    "debugImageOverlay": None,  # filled when debug=True
                    # 'side' assigned below
                }
            )

        if self.debug:
            print(
                f"[CombinedLayoutParser] Extracted {len(body_columns)} body combo columns."
            )

        if not body_columns:
            return {
                "layout": "combined",
                "bodyColumns": body_columns,
                "detected_breakers": [],
                "breakerCounts": {},
            }

        # --- assign columns to left/right sides based on X center ---
        role_cols = [c for c in body_columns if c["role"] in wanted_roles]
        global_left = min(c["x_left"] for c in role_cols)
        global_right = max(c["x_right"] for c in role_cols)
        center_x = 0.5 * (global_left + global_right)

        by_side: Dict[str, Optional[int]] = {"left": None, "right": None}
        for col in role_cols:
            col_center = 0.5 * (col["x_left"] + col["x_right"])
            side = "left" if col_center < center_x else "right"
            col["side"] = side
            # First combo column per side wins
            if by_side[side] is None:
                by_side[side] = col["index"]

        if self.debug:
            print("[CombinedLayoutParser] combo columns by side:", by_side)

        # --- determine row bands *per side* from that side's reference combo column ---
        row_bands_by_side = {"left": None, "right": None}
        row_spans_by_side = {"left": [], "right": []}

        for side in ("left", "right"):
            side_cols = [c for c in role_cols if c.get("side") == side]
            if not side_cols:
                continue

            # Single role: combo. Use the first combo column on this side as reference.
            ref_col = side_cols[0]

            ref_strip = gray_body[
                ref_col["y_top"]:ref_col["y_bottom"],
                ref_col["x_left"]:ref_col["x_right"],
            ]
            if ref_strip.size == 0:
                row_bands_strip = [(0, ref_col["y_bottom"] - ref_col["y_top"])]
            else:
                row_bands_strip = self._compute_row_bands_in_strip(ref_strip)
                if not row_bands_strip:
                    row_bands_strip = [(0, ref_col["y_bottom"] - ref_col["y_top"])]

            row_bands_by_side[side] = row_bands_strip
            row_spans_by_side[side] = [
                (body_y_top + top_s, body_y_top + bottom_s)
                for (top_s, bottom_s) in row_bands_strip
            ]

            if self.debug:
                print(
                    f"[CombinedLayoutParser] Side='{side}' using "
                    f"{len(row_spans_by_side[side])} row bands from horizontal lines."
                )

        tokens_by_col_index: Dict[int, list] = {}

        # --- OCR each combo column ROW-BY-ROW using that column's side bands ---
        for col in body_columns:
            idx   = col["index"]
            x_l   = col["x_left"]
            x_r   = col["x_right"]
            y_top = col["y_top"]
            y_bot = col["y_bottom"]
            side  = col.get("side", "left")

            row_bands_strip = row_bands_by_side.get(side)
            if not row_bands_strip:
                tokens_by_col_index[idx] = []
                continue

            strip = gray_body[y_top:y_bot, x_l:x_r]
            if strip.size == 0:
                tokens_by_col_index[idx] = []
                continue

            tokens_by_col_index[idx] = self._ocr_body_column_by_rows(
                strip_gray=strip,
                x_left=x_l,
                body_y_top=y_top,
                row_bands_strip=row_bands_strip,
            )

        if not tokens_by_col_index:
            return {
                "layout": "combined",
                "bodyColumns": body_columns,
                "detected_breakers": [],
                "breakerCounts": {},
            }

        # --- Build OCR overlays for each body column (debug only) ---
        if self.debug:
            for col in body_columns:
                idx   = col["index"]
                role  = col["role"]
                x_l   = col["x_left"]
                x_r   = col["x_right"]
                y_top = col["y_top"]
                y_bot = col["y_bottom"]

                col_tokens = tokens_by_col_index.get(idx, [])
                if not col_tokens:
                    continue

                # Crop the strip again for visualization only
                strip = gray_body[y_top:y_bot, x_l:x_r]
                if strip.size == 0:
                    continue

                H_strip, W_strip = strip.shape[:2]
                vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

                # Draw a border + label for the column
                cv2.rectangle(
                    vis,
                    (0, 0),
                    (vis.shape[1] - 1, vis.shape[0] - 1),
                    (0, 255, 255),
                    1,
                )
                lbl = f"BODY COMBO  col={idx}"
                cv2.putText(
                    vis,
                    lbl,
                    (8, max(16, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Draw detected row-divider lines (for debug) in BLUE
                line_ys = self._detect_row_divider_lines_in_strip(strip)
                for yc in line_ys:
                    yc_int = max(0, min(H_strip - 1, int(yc)))
                    cv2.line(
                        vis,
                        (0, yc_int),
                        (W_strip - 1, yc_int),
                        (255, 0, 0),  # BLUE in BGR
                        1,
                    )
                    cv2.putText(
                        vis,
                        "ROW",
                        (4, max(10, yc_int - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                # Draw each OCR token (convert from page coords to strip-local coords)
                for tok in col_tokens:
                    x1p, y1p, x2p, y2p = tok["box_page"]
                    text = tok.get("text", "")
                    conf = tok.get("conf", 0.0)

                    # local coords within the strip
                    x1 = max(0, min(W_strip - 1, x1p - x_l))
                    x2 = max(0, min(W_strip - 1, x2p - x_l))
                    y1 = max(0, min(H_strip - 1, y1p - y_top))
                    y2 = max(0, min(H_strip - 1, y2p - y_top))

                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    label = f"{text} ({conf:.2f})"
                    ty = y1 - 4 if y1 - 4 > 10 else y2 + 12
                    cv2.putText(
                        vis,
                        label,
                        (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                overlay_path = os.path.join(
                    debug_dir,
                    f"{base}_parser_body_combined_col{idx}_overlay.png",
                )
                try:
                    cv2.imwrite(overlay_path, vis)
                    col["debugImageOverlay"] = overlay_path
                except Exception as e:
                    print(
                        f"[CombinedLayoutParser] Failed to write body overlay col "
                        f"{idx} ({role}): {e}"
                    )
                    col["debugImageOverlay"] = None

        detected_breakers = []
        breaker_counts: Dict[str, int] = {}

        # --- walk rows and parse combo text *per side* ---
        for side in ("left", "right"):
            col_idx = by_side.get(side)
            side_row_spans = row_spans_by_side.get(side) or []
            if col_idx is None or not side_row_spans:
                continue

            col_tokens = tokens_by_col_index.get(col_idx, [])

            for row_idx, (row_top, row_bottom) in enumerate(side_row_spans):
                combo_text = self._row_text_for_column(row_top, row_bottom, col_tokens)
                combo_pairs = self._parse_combo_cell(combo_text)
                if not combo_pairs:
                    continue

                if self.debug:
                    print(
                        f"[CombinedLayoutParser] side={side} row={row_idx} "
                        f"text='{combo_text}' -> {combo_pairs}"
                    )

                for amps, poles in combo_pairs:
                    key = f"{amps}A_{poles}P"
                    breaker_counts[key] = breaker_counts.get(key, 0) + 1

                    detected_breakers.append(
                        {
                            "side": side,
                            "rowIndex": row_idx,  # per-side row index
                            "rowTop": row_top,
                            "rowBottom": row_bottom,
                            "amperage": int(amps),
                            "poles": int(poles),
                            "comboText": combo_text,
                        }
                    )

        if self.debug:
            print(
                f"[CombinedLayoutParser] detected_breakers={len(detected_breakers)}, "
                f"unique combos={len(breaker_counts)}"
            )

        return {
            "layout": "combined",
            "bodyColumns": body_columns,
            "detected_breakers": detected_breakers,
            "breakerCounts": breaker_counts,
        }

class BreakerTableParser:
    """
    Parser6 orchestrator.
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
        self._combined_parser  = CombinedLayoutParser(debug=self.debug, reader=self.reader)

    def parse_from_analyzer(self, analyzer_result: Dict) -> Dict:
        """
        Entry point used by the API.

        Right now:
          - Reads spaces from analyzer_result (corrected if available)
          - Runs the header-band OCR scan
          - If header layout is 'separated', runs SeparatedLayoutParser
          - Aggregates detected breakers into breakerCounts
          - Prints a human-readable summary to the terminal (when debug=True)
          - Returns JSON-safe parser result
        """
        if not isinstance(analyzer_result, dict):
            analyzer_result = {}

        # --- spaces: from analyzer 'spaces', or fall back to 'panel_size' (Analyzer12) ---
        raw_spaces = analyzer_result.get("spaces")
        if raw_spaces is None:
            raw_spaces = analyzer_result.get("panel_size")

        if raw_spaces is None:
            spaces = 0
        else:
            try:
                spaces = int(raw_spaces)
            except (TypeError, ValueError):
                spaces = 0

        header_scan = self._header_scanner.scan(analyzer_result)

        normalized = header_scan.get("normalizedColumns") or {}
        layout = normalized.get("layout", "unknown")

        separated_scan: Optional[Dict] = None
        combined_scan: Optional[Dict] = None
        detected_breakers: List[Dict] = []

        if layout == "separated":
            if self.debug:
                print("[BreakerTableParser] Layout 'separated' → using SeparatedLayoutParser.")
            separated_scan = self._separated_parser.parse(analyzer_result, header_scan)
            detected_breakers = separated_scan.get("detected_breakers", []) or []
        elif layout == "combined":
            if self.debug:
                print("[BreakerTableParser] Layout 'combined' → using CombinedLayoutParser.")
            combined_scan = self._combined_parser.parse(analyzer_result, header_scan)
            detected_breakers = combined_scan.get("detected_breakers", []) or []
        else:
            if self.debug:
                print(f"[BreakerTableParser] Layout '{layout}' → no body parser yet.")

        # --- Aggregate breaker counts (amps + poles) ---
        breaker_counts: Dict[str, int] = {}
        for br in detected_breakers:
            amps = br.get("amperage")
            poles = br.get("poles")
            if amps is None or poles is None:
                continue
            a = int(amps)
            p = int(poles)
            key = f"{p}P_{a}A"
            breaker_counts[key] = breaker_counts.get(key, 0) + 1

        if self.debug:
            print(
                f"[BreakerTableParser] Header band y:[{header_scan.get('band_y1')},"
                f"{header_scan.get('band_y2')}) tokens={len(header_scan.get('tokens', []))}"
            )

        # --- Build final JSON-safe result ---
        result = {
            "parserVersion": PARSER_VERSION,
            "name": None,  # API will overwrite with deduped panel name
            "spaces": spaces,
            "detected_breakers": detected_breakers,
            "breakerCounts": breaker_counts,
            "headerScan": header_scan,
            "separatedScan": separated_scan,
            "combinedScan": combined_scan,
        }

        # --- Human-readable terminal summary (only when debug=True) ---
        if self.debug:
            self._print_terminal_summary(analyzer_result, result)

        return result

    def _print_terminal_summary(self, analyzer_result: Dict, result: Dict) -> None:
        """
        Print a simple, human-readable summary that mirrors the final JSON, per panel:

          Panel name - LIA
          Amps - 125A, main breaker amps - Unknown, volts - 480V, spaces - 42
          Breakers
            1 P, 20 A, count - 30
            3 P, 100 A, count - 4
            ...

        No per-breaker rows are printed here; only the aggregated tally.
        """

        # ---- Source image / crop ----
        src_path = analyzer_result.get("src_path") or analyzer_result.get("image_path")
        if src_path:
            src_name = os.path.basename(src_path)
        else:
            src_name = "Unknown source"

        # ---- Panel-level metadata ----

        # Name: prefer parser result, then analyzer, then header attrs if present
        name = (
            result.get("name")
            or analyzer_result.get("panel_name")
            or analyzer_result.get("name")
        )

        header_attrs = analyzer_result.get("header_attrs") or {}
        if not name:
            name = header_attrs.get("name")

        if not name:
            name = "Unknown"

        # Amps / main breaker amps / voltage:

        # Panel bus amps
        panel_amps = (
            analyzer_result.get("panel_amps")
            or analyzer_result.get("amps")
            or analyzer_result.get("rating_amps")
            or header_attrs.get("amperage")
        )

        # Main breaker amps
        main_breaker_amps = (
            analyzer_result.get("main_breaker_amps")
            or analyzer_result.get("main_amps")
            or analyzer_result.get("main_rating_amps")
        )

        # Voltage
        volts = (
            analyzer_result.get("voltage")
            or analyzer_result.get("volts")
            or analyzer_result.get("system_voltage")
            or header_attrs.get("voltage")
        )

        spaces = result.get("spaces")

        # Convert missing values to "Unknown" strings for nice printing
        def _fmt(v: Optional[object], suffix: str = "") -> str:
            if v is None:
                return "Unknown"
            try:
                if suffix:
                    return f"{int(v)}{suffix}"
                return str(v)
            except Exception:
                return str(v)

        panel_amps_str = _fmt(panel_amps, "A")
        main_amps_str = _fmt(main_breaker_amps, "A")
        volts_str = _fmt(volts, "V")
        spaces_str = _fmt(spaces)

        breaker_counts: Dict[str, int] = result.get("breakerCounts") or {}

        # ---- Pretty print ----
        print()
        print("====================================")
        print("Breakers")

        if not breaker_counts:
            print("  (none detected)")
            print("====================================")
            return

        # sort by poles, then amps numerically if possible
        def _sort_key(item):
            key, _count = item
            m = re.match(r"(\d+)P_(\d+)A", key)
            if not m:
                return (9999, 9999, key)
            p = int(m.group(1))
            a = int(m.group(2))
            return (p, a, key)

        for key, count in sorted(breaker_counts.items(), key=_sort_key):
            m = re.match(r"(\d+)P_(\d+)A", key)
            if m:
                poles = int(m.group(1))
                amps = int(m.group(2))
                print(f"  {poles} P, {amps} A, count - {count}")
            else:
                # Fallback if key format ever changes
                print(f"  {key}, count - {count}")

        print("====================================")
