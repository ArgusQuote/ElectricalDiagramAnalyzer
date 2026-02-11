# OcrLibrary/PanelHeaderParserV6.py
import os, re, cv2, json, numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False


class PanelParser:
    """
    Panel Header Parser V4 (revised association)
    - Robust label/value association with wrong-context penalties
    - Handles 65kA / 65000 A / 480Y/277V / 277/480V / 208/120, etc.
    - Clamps header to label cluster to avoid breaker table bleed
    - Debug overlay shows ranked candidates per role
    """

    # ====== Name noise (avoid picking nouns as the panel name) ======
    _NAME_STOPWORDS = {
        "NEW","EXISTING","TYPICAL","TYPE","SYSTEM","DISTRIBUTION","NORMAL","EMERGENCY",
        "CRITICAL","LIGHTING","POWER","PANEL", "PANELBOARD", "BOARD",
        "RATING","INTERRUPTING","AIC","KAIC","SCCR","SYMMETRICAL","ASYMMETRICAL","RMS",
        "AMPACITY","LOAD","CAPACITY","FAULT","SHORT","CURRENT","AVAILABLE",
        "AMP","AMPS","VOLT","VOLTS","V","KA","KVA","KV","HZ","HERTZ",
        "PHASE","PHASES","PH","Ø","WIRE","WIRES","CONDUCTOR","CONDUCTORS","POLE","POLES",
        "NEMA","ENCLOSURE","INDOOR","OUTDOOR","WEATHERPROOF","SURFACE","FLUSH",
        "MOUNTING","WIDTH","DEPTH","HEIGHT","SECTIONS",
        "BUS","BUSS","BUSBAR","MATERIAL","ALUMINUM","AL","ALUM","COPPER","CU",
        "MAIN","MCB","MLO","SERVICE","SERVICE ENTRANCE","SE","S.E.","LUG","LUGS",
        "FEED","FEEDS","FEEDER","FED","FROM","BY","FEED THRU","FEED-THRU","FEEDTHRU",
        "GROUND","GROUNDED","GND","EARTH","NEUTRAL","NEUT","N",
        "GFI","GFCI","AFCI","SHUNT","TRIP","PROT","PROTECTION","OPTIONS","DATA",
        "I-LINE","ILINE","I","LINE","QO","HOM","SQUARE","SCHNEIDER","EATON","SIEMENS",
        "NOTES","TABLE","SCHEDULE","SIZE","RANGE","CATALOG","CAT","CAT.","DWG","REV","DATE",
        "ACCESSORY","ACCESSORIES",
        # Added for robustness: common header/drawing words that shouldn't be names
        "STAGE","PROPOSED","FUTURE","ALTERNATE","REVISION",
        "SHEET","DRAWING","DETAIL","SECTION","ELEVATION","PLAN","SPEC","SPECIFICATION",
        "CIRCUIT","BREAKER","DEVICE","EQUIPMENT","APPARATUS","UNIT","MODULE",
        "DESCRIPTION","LOCATION","AREA","ROOM","FLOOR","LEVEL","BUILDING",
        "SEE","REFER","NOTE","PER","AS","SHOWN","INDICATED","REQUIRED",
        "GENERAL","STANDARD","SPECIAL","CUSTOM","MODIFIED","REVISED",
    }

    # ====== Value stopwords (tokens containing these should NOT be voltage/amps/AIC values) ======
    _VALUE_STOPWORDS = {
        # Words that indicate a token is contextual text, not an electrical value
        "STAGE","TYPICAL","NEW","EXISTING","PROPOSED","FUTURE","ALTERNATE",
        "SEE","REFER","NOTE","NOTES","PER","AS","SHOWN","INDICATED",
        "SHEET","DRAWING","DWG","DETAIL","SECTION","PLAN","ELEVATION",
        "SCHEDULE","TABLE","CIRCUIT","DESCRIPTION","LOCATION","AREA",
        "GENERAL","STANDARD","SPECIAL","CUSTOM","MODIFIED","REVISED",
        "EQUIPMENT","APPARATUS","DEVICE","UNIT","MODULE","SYSTEM",
        "FLOOR","LEVEL","BUILDING","ROOM","SPACE",
    }

    # ====== Label families (regex) ======
    _LABELS = {
        "VOLTAGE": [
            r"\bVOLT(AGE|S)?\b", r"\bVOLTS?\b", r"\bV\b",
            r"\bPH(ASE)?\b", r"\bWIRE(S)?\b", r"\bØ\b"
        ],
        "BUS": [
            r"\bBUS{1,2}\b", r"\bBUS{1,2}\s*RATING\b", r"\bPANEL\s*RATING\b", r"\bAMPACITY\b", r"\bRATING\b"
        ],
        "MAIN": [r"\bMAIN\s*(RATING|BREAKER|DEVICE|TYPE)\b", r"\bMAIN\s*(TYPE|RATING|BREAKER|DEVICE)\b",
                 r"\b(?:MCB|M\W*C\W*B)\b",
                 r"\bMLO\b", r"\bMAIN\s*LUGS?\b", r"\bMAIN\s*TYPE\b", r"\bMAINS?\b", r"\bMAIN\b"
        ],
        "AIC": [
            r"\bA\.?\s*I\.?\s*C\.?\b", r"\bAIC\b", r"\bKAIC\b", r"\bSCCR\b",
            r"\bINTERRUPTING\s*RATING\b", r"\bAVAILABLE\s*FAULT\s*CURRENT\b", r"\bFAULT\s*CURRENT\b",
            r"\bSYMMETRICAL\b"
        ],
        "NAME": [r"\bPANEL\s*DESIGNATION\b", r"\bDESIGNATION\b", r"\bPANEL(BOARD)?\b", r"\bBOARD\b", r"\bPANEL\s*:?\b", r"\bDISTRIBUTION\s*PANEL\b"],
        "WRONG": [
            # Original patterns
            r"\bNOTES?\b", r"\bTABLE\b", r"\bSCHEDULE\b", r"\bSIZE\s*\(??A\)?\b", r"\bCIRCUIT\b",
            r"\bCAT(ALOG)?\b", r"\bDWG\b", r"\bREV\b", r"\bDATE\b",
            # NEW: References/instructions that indicate context, not values
            r"\bSEE\s+\w+\b",           # "SEE SHEET", "SEE NOTE", etc.
            r"\bREFER\s+TO\b",          # "REFER TO"
            r"\bPER\s+\w+\b",           # "PER SPEC", "PER PLAN"
            # NEW: Drawing/construction terms
            r"\bSTAGE\b", r"\bTYPICAL\b", r"\bFUTURE\b", r"\bPROPOSED\b", r"\bEXISTING\b",
            r"\bSHEET\b", r"\bDRAWING\b", r"\bDETAIL\b", r"\bSECTION\b", r"\bELEVATION\b",
            # NEW: Description/location terms
            r"\bDESCRIPTION\b", r"\bLOCATION\b", r"\bAREA\b", r"\bROOM\b", r"\bFLOOR\b",
            # NEW: General header/title words
            r"\bGENERAL\b", r"\bSPECIAL\b", r"\bSTANDARD\b", r"\bTYPE\b",
            r"\bEQUIPMENT\b", r"\bAPPARATUS\b",
        ],
    }

    # ====== Role weights (blend of value-shape, label-affinity, side, ctx, penalties) ======
    # NOTE: W_wrong increased to penalize candidates near WRONG labels more strongly
    _WEIGHTS = {
        "VOLTAGE": dict(W_shape=0.55, W_conf=0.15, W_lbl=0.22, W_side=0.12, W_ctx=0.07, W_wrong=0.18, W_y=0.00),
        "BUS":     dict(W_shape=0.55, W_conf=0.15, W_lbl=0.18, W_side=0.09, W_ctx=0.05, W_wrong=0.20, W_y=0.00),
        "MAIN":    dict(W_shape=0.55, W_conf=0.15, W_lbl=0.20, W_side=0.11, W_ctx=0.05, W_wrong=0.20, W_y=0.00),
        "AIC":     dict(W_shape=0.60, W_conf=0.12, W_lbl=0.22, W_side=0.12, W_ctx=0.08, W_wrong=0.18, W_y=0.00),
        "NAME":    dict(W_shape=0.55, W_conf=0.10, W_lbl=0.15, W_side=0.12, W_ctx=0.00, W_wrong=0.12, W_y=0.35)
    }

    # Base thresholds - raised slightly for more conservative detection
    _THRESH = {"VOLTAGE": 0.58, "BUS": 0.56, "MAIN": 0.56, "AIC": 0.56, "NAME": 0.50}
    
    # Higher thresholds used when no supporting label is present (dynamic thresholding)
    _THRESH_NO_LABEL = {"VOLTAGE": 0.65, "BUS": 0.62, "MAIN": 0.62, "AIC": 0.62, "NAME": 0.55}

    _SIGMA_PX = 80.0

    def __init__(self, debug: bool = False, voltage_first_number_only: bool = True):
        self.debug = debug
        self.voltage_first_number_only = voltage_first_number_only
        self.reader = None
        if _HAS_OCR:
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)

        # Optional external config (JSONC)
        self.labels_cfg = {}
        try:
            cfg_path = os.path.join(os.path.dirname(__file__), "labels_config.jsonc")
            if os.path.exists(cfg_path):
                raw = open(cfg_path, "r", encoding="utf-8").read()
                raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
                raw = re.sub(r"(?m)//.*$", "", raw)
                raw = re.sub(r",(\s*[}\]])", r"\1", raw)
                self.labels_cfg = json.loads(raw) or {}
        except Exception:
            pass

        self.last_main_type = None

    def _trim_voltage_to_allowed(self, txt: str | None) -> int | None:
        """
        Return ONLY one of {120, 208, 240, 480, 600} if present.

        Rules:
        - For 120/240 → choose 120 (intentional special-case)
        - For common wye pairs like 120/208 or 277/480 → choose the higher system voltage (208, 480)
        - Otherwise:
            - if voltage_first_number_only=True → choose the first occurring allowed voltage
            - else → choose the highest occurring allowed voltage
        """
        if not txt:
            return None
        import re

        s = self._normalize_digits(str(txt).upper())
        s = self._normalize_voltage_text(s)

        # Explicit special-case: 120/240 should normalize to 120
        if re.search(r'(?<!\d)120\s*/\s*240(?!\d)', s):
            return 120

        # If this token contains an explicit voltage pair, prefer the higher system voltage.
        # Examples: "120/208 WYE" -> 208, "277/480" -> 480, "208Y/120" -> 208
        pair = re.search(r'(?<!\d)(\d{3})\s*[YV]?\s*/\s*(\d{3})(?!\d)', s)
        if pair:
            a = int(pair.group(1))
            b = int(pair.group(2))
            allowed = {120, 208, 240, 480, 600}

            if a in allowed and b in allowed:
                return max(a, b)

        # Otherwise collect allowed voltages in appearance order
        allowed_vals = ("600", "480", "240", "208", "120")

        matches: list[tuple[int, int]] = []
        for v in allowed_vals:
            m = re.search(rf'(?<!\d){v}(?!\d)', s)
            if m:
                matches.append((m.start(), int(v)))
                continue
            pos = s.find(v)
            if pos != -1:
                matches.append((pos, int(v)))

        if not matches:
            return None

        matches.sort(key=lambda t: t[0])
        vals_in_order = [v for _, v in matches]

        if self.voltage_first_number_only:
            return vals_in_order[0]

        return max(vals_in_order)

    def parse_panel(
        self,
        image_path: str,
        y_band: Optional[Tuple[int,int]] = None,
        header_y: Optional[int] = None,
        header_y_ratio: Optional[float] = None,
    ) -> Dict:
        image_path = os.path.abspath(os.path.expanduser(image_path))
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Preprocess
        prep_full = self._prep_for_ocr(img)
        H, W = prep_full.shape[:2]
        # keep geometry for downstream scoring / left-bias
        self._last_full_W = W
        self._last_band_top = 0

        # ----- Determine header band -----
        if header_y_ratio is not None:
            y1, y2 = 0, int(max(1, min(H - 1, round(header_y_ratio * H))))
            SAFETY = 8
            by2 = min(H - 1, y2 + SAFETY)
        elif header_y is not None:
            y1, y2 = 0, int(max(1, min(H - 1, header_y)))
            SAFETY = 8
            by2 = min(H - 1, y2 + SAFETY)
        else:
            if y_band is not None:
                y1, y2 = int(max(0, min(H - 2, y_band[0]))), int(max(1, min(H - 1, y_band[1])))
                if y2 <= y1:
                    y1, y2 = 0, H - 1
            else:
                y1, y2 = 0, H - 1
            BAND_PAD_BOTTOM = max(28, int(0.02 * H))
            by2 = min(H - 1, y2 + BAND_PAD_BOTTOM)

        self._last_band_top = y1
        prep = prep_full[y1:by2, :]
        band_offset_y = y1

        # Multi-pass OCR with varied parameters for better accuracy
        if self.reader is None and _HAS_OCR:
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)

        all_results = []
        
        # Pass 1: Standard - good baseline for most text
        try:
            det1 = self.reader.readtext(
                prep, detail=1, paragraph=False,
                mag_ratio=1.6, contrast_ths=0.05, adjust_contrast=0.7,
                text_threshold=0.4, low_text=0.3,
            )
            all_results.append(list(det1))
        except Exception:
            pass
        
        # Pass 2: Inverted - catches white text on dark backgrounds
        try:
            inv = cv2.bitwise_not(prep)
            det2 = self.reader.readtext(
                inv, detail=1, paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,()/:- kKVvYØø ",
                mag_ratio=1.9, contrast_ths=0.05, adjust_contrast=0.7,
                text_threshold=0.4, low_text=0.25
            )
            all_results.append(list(det2))
        except Exception:
            pass
        
        # Pass 3: Higher magnification - better for small text
        try:
            det3 = self.reader.readtext(
                prep, detail=1, paragraph=False,
                mag_ratio=2.2, contrast_ths=0.08, adjust_contrast=0.8,
                text_threshold=0.5, low_text=0.35,
            )
            all_results.append(list(det3))
        except Exception:
            pass
        
        # Pass 4: Adaptive threshold - helps with varying background brightness
        try:
            thresh = cv2.adaptiveThreshold(
                prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            det4 = self.reader.readtext(
                thresh, detail=1, paragraph=False,
                mag_ratio=1.8, contrast_ths=0.05, adjust_contrast=0.5,
                text_threshold=0.45, low_text=0.3,
            )
            all_results.append(list(det4))
        except Exception:
            pass
        
        # Pass 5: High-confidence pass with standard allowlist
        # NOTE: Previously excluded I/L/O to avoid 1/l/0 confusion, but this caused
        # worse regressions by garbling words like "PANEL", "LIGHTING", "EXISTING".
        # The _normalize_digits() function handles digit confusion instead.
        try:
            det5 = self.reader.readtext(
                prep, detail=1, paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,()/:- kKVvYØø ",
                mag_ratio=1.6, contrast_ths=0.05, adjust_contrast=0.7,
                text_threshold=0.45, low_text=0.3,
            )
            all_results.append(list(det5))
        except Exception:
            pass
        
        # Merge all OCR results using confidence-weighted merging
        detailed = self._merge_ocr_results(all_results)

        # Normalize tokens (absolute coords)
        items = []
        h_band, w_band = prep.shape[:2]
        for entry in detailed:
            try:
                box, txt, conf = entry
            except Exception:
                continue
            if not txt:
                continue
            xs = [int(max(0, min(w_band - 1, p[0]))) for p in box]
            ys = [int(max(0, min(h_band - 1, p[1]))) for p in box]
            x1, x2 = min(xs), max(xs)
            yb1, yb2 = min(ys) + band_offset_y, max(ys) + band_offset_y
            items.append({
                "text": str(txt),
                "conf": float(conf or 0.0),
                "x1": x1, "y1": yb1, "x2": x2, "y2": yb2,
                "xc": 0.5*(x1+x2), "yc": 0.5*(yb1+yb2)
            })
        items.sort(key=lambda d: (d["y1"], d["x1"]))

        # Lines & cap
        tokens = [((it["x1"], it["y1"], it["x2"], it["y2"]), it["text"], it["conf"]) for it in items]
        lines = self._group_into_lines(tokens)
        header_cap = max(1, int(self.labels_cfg.get("headerMaxLines", 15)))
        lines = lines[:header_cap]

        # ===== Association pipeline =====
        labels_map = self._collect_label_candidates(items)
        value_cands = self._collect_value_candidates(items)
        # --- Simple NAME injector (top 1–2 lines) ---
        simple_name = self._simple_name_from_top(lines)
        if simple_name:
            value_cands.setdefault("NAME", []).append(simple_name)

        # Clamp candidate values to above/beside the header label cluster (avoid table bleed)
        header_labels = (labels_map.get("VOLTAGE", []) +
                         labels_map.get("AIC", []) +
                         labels_map.get("MAIN", []) +
                         labels_map.get("BUS", []))
        if header_labels:
            lbl_heights = [abs(L["y2"] - L["y1"]) for L in header_labels]
            med_h = float(np.median(lbl_heights)) if lbl_heights else 18.0
            header_bottom = max(L["y2"] for L in header_labels) + int(1.2 * med_h)
            header_bottom = min(header_bottom, by2)
            for _role in ("BUS", "MAIN", "AIC", "VOLTAGE", "NAME"):
                value_cands[_role] = [c for c in value_cands.get(_role, []) if c["y2"] <= header_bottom]

        main_mode = self._scan_main_mode(items)  # "MLO" / "MCB" / None
        band_top = y1

        ranked_map = {}
        chosen_map = {}
        ROLE_ORDER = ("VOLTAGE", "AIC", "BUS", "MAIN", "NAME")

        # ===== Consume-as-we-go =====
        used = set()

        def _cand_key(c: dict) -> tuple:
            """
            Stable identity for a candidate across roles/pools.
            Use bbox + normalized text.
            """
            if not c:
                return None
            txt = str(c.get("text", "") or "").strip().upper()
            # normalize OCR digit confusions so the same token matches across passes
            txt = self._normalize_digits(txt)
            return (
                int(round(float(c.get("x1", -1)))),
                int(round(float(c.get("y1", -1)))),
                int(round(float(c.get("x2", -1)))),
                int(round(float(c.get("y2", -1)))),
                txt,
            )

        def _is_used(c: dict) -> bool:
            k = _cand_key(c)
            return (k is not None) and (k in used)

        def _mark_used(c: dict) -> None:
            k = _cand_key(c)
            if k is not None:
                used.add(k)

        def _unmark_used(c: dict) -> None:
            k = _cand_key(c)
            if k is not None and k in used:
                used.remove(k)

        def _pick_role_consuming(role: str, ranked: list) -> dict | None:
            """
            Like _pick_role, but skips candidates already used by earlier roles.
            Uses dynamic thresholding: higher threshold when no supporting label exists.
            """
            # Dynamic thresholding: use higher threshold if no label for this role
            has_label = bool(labels_map.get(role))
            if has_label:
                thr = self._THRESH.get(role, 0.5)
            else:
                thr = self._THRESH_NO_LABEL.get(role, self._THRESH.get(role, 0.5))
            
            for c in (ranked or []):
                if float(c.get("rank", 0.0)) < thr:
                    break
                if _is_used(c):
                    continue
                return c
            return None

        def _set_role(role: str, cand: dict | None, allow_share_with: str | None = None) -> None:
            """
            Assign chosen_map[role] and update the used set.
            If allow_share_with is provided, the candidate may already be used
            ONLY by that other role (same exact key).
            """
            # remove previous reservation for this role (if we are changing it)
            prev = chosen_map.get(role)
            if prev:
                # If prev is also used by allow_share_with, don't unmark it here (rare)
                _unmark_used(prev)

            if not cand:
                chosen_map[role] = None
                return

            k = _cand_key(cand)
            if k is None:
                chosen_map[role] = cand
                return

            if k in used:
                if allow_share_with:
                    other = chosen_map.get(allow_share_with)
                    if other and _cand_key(other) == k:
                        # allowed share
                        chosen_map[role] = cand
                        return
                # not allowed to reuse → reject
                chosen_map[role] = None
                return

            chosen_map[role] = cand
            used.add(k)

        for role in ROLE_ORDER:
            ranked = self._score_candidates(role, value_cands.get(role, []), labels_map, band_top, main_mode)
            ranked_map[role] = ranked

            picked = _pick_role_consuming(role, ranked)
            # consume immediately so later roles can't reuse it
            if picked:
                _set_role(role, picked)
            else:
                chosen_map[role] = None

        def _looks_like_electrical_value(s: str) -> bool:
            if not s:
                return False
            t = self._normalize_digits(str(s).upper()).strip()
            t = t.replace("Α", "A").replace("А", "A")  # unicode A -> ASCII A

            if re.search(r"(?<!\d)\d{1,4}\s*(A\.?|AMPS?\.?)\b", t):
                return True
            if re.search(r"(?<!\d)[1-6]\d{2,3}\s*[YV]?\s*/\s*[1-6]?\d{2,3}(?!\d)", t):
                return True
            if re.search(r"(?<!\d)[1-6]\d{2,3}\s*V\b", t):
                return True
            if re.search(r"(?<!\d)\d{2,3}\s*(KAIC|AIC|KA|K)\b", t):
                return True
            if re.search(r"(?<!\d)\d{2,3}[,]?\d{3}(?!\d)", t):
                return True
            if re.fullmatch(r"\d{1,4}", t):
                return True
            return False

        if chosen_map.get("NAME") and _looks_like_electrical_value(chosen_map["NAME"].get("text", "")):
            chosen_map["NAME"] = None

        # ---- AIC FALLBACK: if nothing chosen, scan whole band for kA-like tokens (22K, 22KAIC, 22AIC, etc.) ----
        if not chosen_map.get("AIC"):
            SMALL_KA = {10, 14, 18, 22, 25, 30, 35, 42, 50, 65, 100, 125, 200}
            best = None

            for it in items:
                raw = str(it.get("text", "") or "")
                if not raw:
                    continue

                up = self._normalize_digits(raw.upper())
                # Remove spaces so we can catch "22 K", "22KAIC", "JAC:22K", etc.
                t_nos = re.sub(r"\s+", "", up)

                # Match 2–3 digit kA-style values with K/KA/AIC/KAIC suffix
                mk = re.search(r'(?<!\d)(\d{2,3})(?:KAIC|AIC|KA|K)\b', t_nos)
                if not mk:
                    continue

                val = int(mk.group(1))

                # Be conservative: only accept typical interrupt ratings
                if val not in SMALL_KA and not (10 <= val <= 100):
                    continue

                cand = {
                    "x1": it["x1"], "y1": it["y1"], "x2": it["x2"], "y2": it["y2"],
                    "xc": it["xc"], "yc": it["yc"],
                    "conf": float(it.get("conf", 0.5)),
                    "text": raw,
                    "shape": 0.82,  # decent but not perfect (fallback)
                    "ctx": 0.05,
                    "rank": 0.60,   # mid-level rank so debug output is sensible
                }

                if best is None:
                    best = cand
                else:
                    # Prefer higher OCR confidence; on tie, take the one higher up on the page
                    if cand["conf"] > best["conf"] or (
                        cand["conf"] == best["conf"] and cand["y1"] < best["y1"]
                    ):
                        best = cand

            if best is not None:
                chosen_map["AIC"] = best

        # ===== Unit-first amps role assignment (explicit A/AMPS drive detection; labels split roles) =====
        # Gather explicit-unit amps candidates from BOTH BUS and MAIN pools
        # (because _collect_value_candidates() can classify a token as MAIN-only or BUS-only)
        _unit_pool = (value_cands.get("BUS") or []) + (value_cands.get("MAIN") or [])
        # Roles that must never be stolen from by BUS/MAIN reassignment
        _protected_roles = ("VOLTAGE", "AIC", "NAME")

        # De-dupe (BUS and MAIN pools may contain separate dict copies of the same token)
        unit_amps = []
        _seen = set()
        for c in _unit_pool:
            if not bool(c.get("has_unit")):
                continue
            k = (
                int(c.get("x1", -1)), int(c.get("y1", -1)),
                int(c.get("x2", -1)), int(c.get("y2", -1)),
                str(c.get("text", "")).strip().upper(),
            )
            if k in _seen:
                continue
            _seen.add(k)
            unit_amps.append(c)

        unit_amps.sort(key=lambda c: (-float(c.get("conf", 0.0)), c["y1"], c["x1"]))

        def _role_affinity(c):
            return (self._label_affinity("BUS", c, labels_map),
                    self._label_affinity("MAIN", c, labels_map))

        has_main_label = bool(labels_map.get("MAIN"))

        if len(unit_amps) >= 2:
            # Pick one for BUS and one for MAIN using label affinity; break ties by y/x
            scored = []
            for c in unit_amps:
                aff_b, aff_m = _role_affinity(c)
                scored.append((c, aff_b - aff_m, aff_b, aff_m))

            # Best BUS-leaning that is not already used by earlier roles
            bus_sorted = sorted(
                scored,
                key=lambda t: (-(t[1]), -t[2], t[0]["y1"], t[0]["x1"])
            )

            bus_pick = None
            for (c, _, _, _) in bus_sorted:
                # don't steal a token already used by VOLTAGE/AIC/NAME
                if _is_used(c):
                    # allow reuse only if it's currently used by BUS/MAIN (we're about to replace anyway),
                    # so don't allow stealing from protected roles.
                    if any(chosen_map.get(r) and _cand_key(chosen_map[r]) == _cand_key(c) for r in _protected_roles):
                        continue
                bus_pick = c
                break

            if bus_pick is not None:
                # MAIN pick from remaining, MAIN-leaning, also respecting consumption
                scored_main = [t for t in scored if t[0] is not bus_pick]
                main_sorted = sorted(
                    scored_main,
                    key=lambda t: (-(t[3] - t[2]), -t[3], t[0]["y1"], t[0]["x1"])
                )

                main_pick = None
                for (c, _, _, _) in main_sorted:
                    if _is_used(c):
                        if any(chosen_map.get(r) and _cand_key(chosen_map[r]) == _cand_key(c) for r in _protected_roles):
                            continue
                    main_pick = c
                    break

                # Apply with used-set updates (no sharing allowed here)
                if bus_pick is not None:
                    _set_role("BUS", bus_pick)
                if main_pick is not None:
                    _set_role("MAIN", main_pick)

        elif len(unit_amps) == 1:
            only = unit_amps[0]
            mode_upper = (main_mode or "").upper()

            if mode_upper == "MLO":
                # MLO: panel has no main device → only BUS rating is meaningful.
                _set_role("BUS", only)
                _set_role("MAIN", None)

            elif mode_upper == "MCB":
                # Allow BUS and MAIN to share the same candidate ONLY in explicit MCB mode
                _set_role("BUS", only)
                _set_role("MAIN", only, allow_share_with="BUS")

            else:
                # No explicit mode. If there is *no* MAIN label anywhere,
                # treat this single amps token as BUS-only – do NOT invent a MAIN.
                if not has_main_label:
                    _set_role("BUS", only)
                    _set_role("MAIN", None)
                else:
                    # There is a MAIN label somewhere: use affinity to decide role.
                    aff_b, aff_m = _role_affinity(only)
                    if aff_m > aff_b + 0.08:
                        _set_role("MAIN", only)
                        # leave BUS as-is (do not invent/override BUS here)
                    else:
                        _set_role("BUS", only)
                        # leave MAIN as-is (do not invent/override MAIN here)

        else:
            # No explicit-unit amps → keep ranked picks but still apply mode enforcement
            pass

        # Final mode enforcement (last word)
        if (main_mode or "").upper() == "MLO":
            chosen_map["MAIN"] = None
        # If both still None and we had any amps tokens, BUS fallback will run below as before.

        # ===== NAME fallback: only if we have a strong designation/name label present =====
        if not chosen_map.get("NAME"):
            has_name_label = bool(labels_map.get("NAME"))

            # NOTE: you currently do NOT have "DESIGNATION" in _LABELS, so this will always be False.
            # Keep it here if you plan to add it later, otherwise remove it.
            has_designation_label = bool(labels_map.get("DESIGNATION"))

            if has_name_label or has_designation_label:
                name_cands = list(value_cands.get("NAME") or [])
                if name_cands:
                    name_cands.sort(key=lambda c: (c["y1"], c["x1"]))
                    cand0 = dict(name_cands[0])
                    cand0.setdefault("rank", 0.51)
                    _set_role("NAME", cand0)

            # else: leave NAME as None (unknown) – do NOT do ultra-conservative scan

            else:
                # ultra-conservative: scan top 3 lines for a short, clean alphanum token
                picked = None
                top_lines = lines[: min(3, len(lines))]
                for ln in top_lines:
                    for (_, tok, _) in ln["tokens"]:
                        up = (tok or "").strip().upper().strip(":")
                        if not up:
                            continue
                        if up in self._NAME_STOPWORDS:
                            continue
                        if re.fullmatch(r"[A-Z0-9][A-Z0-9._\-\/]{0,10}", up):
                            picked = {
                                "text": up,
                                "x1": ln["rect"][0],
                                "y1": ln["rect"][1],
                                "x2": ln["rect"][2],
                                "y2": ln["rect"][3],
                                "xc": 0.5 * (ln["rect"][0] + ln["rect"][2]),
                                "yc": 0.5 * (ln["rect"][1] + ln["rect"][3]),
                                "conf": 0.50,
                                "rank": 0.51,
                            }
                            break
                    if picked:
                        break
                if picked:
                    _set_role("NAME", picked)

        # ===== BUS fallback: if no BUS chosen, promote best amperage-looking value =====
        if not chosen_map.get("BUS"):

            # If we have a MAIN value but absolutely no BUS value-shapes,
            # treat MAIN as the BUS rating too (typical when only "50 AMP MCB" is printed).
            if chosen_map.get("MAIN") and not (value_cands.get("BUS") or ranked_map.get("BUS")):
                chosen_map["BUS"] = chosen_map["MAIN"]

            # If still no BUS after that, try to promote the best amperage-looking candidate
            if not chosen_map.get("BUS"):
                bus_ranked = list(ranked_map.get("BUS") or [])
                main_ranked = list(ranked_map.get("MAIN") or [])

                def _amps_from_text(t: str) -> Optional[int]:
                    m = re.search(r"\b([6-9]\d|[1-9]\d{2,3})\s*(A|AMPS?)\b", (t or "").upper())
                    return int(m.group(1)) if m else None

                # Prefer explicit-unit amps first (then fall back to any amps-looking)
                unit_first = [
                    c for c in (bus_ranked + main_ranked)
                    if _amps_from_text(c.get("text", "")) is not None
                ]
                pool = unit_first if unit_first else (bus_ranked + main_ranked)
                # keep only entries that *actually* parse as amps in [60..1200]
                pool = [c for c in pool if _amps_from_text(c.get("text", "")) is not None]

                if pool:
                    # Prefer: higher rank → higher confidence; tiebreak by top-most then left-most
                    pool.sort(
                        key=lambda c: (
                            -float(c.get("rank", 0.0)),
                            c.get("y1", 1e9),
                            c.get("x1", 1e9),
                        )
                    )
                    chosen_map["BUS"] = pool[0]

            # If still nothing, scan ALL tokens permissively for an amps-looking value
            if not chosen_map.get("BUS"):
                # Reuse header_bottom if available; else allow everything in the band
                try:
                    _header_bottom = header_bottom
                except NameError:
                    _header_bottom = by2

                best = None
                for it in items:
                    raw = (it.get("text","") or "")
                    up = raw.upper()

                    # ---- Guard: don't treat voltage-looking tokens like "120/208 Wye" as BUS ----
                    txtD = self._normalize_digits(up)
                    txtN = self._normalize_voltage_text(txtD)

                    pair = re.search(
                        r'\b([1-6]\d{2,3})\s*[YV]?[\/]\s*([1-6]?\d{2,3})\s*V?\b',
                        txtN,
                    )
                    has_volty_ctx = bool(
                        re.search(r'\b(WYE|DELTA|PH|PHASE|Ø|VOLT|VOLTS|V)\b', txtN)
                    )
                    single = (
                        re.search(r'(?<!\d)([1-6]\d{2,3})(?!\d)', txtN)
                        if has_volty_ctx
                        else None
                    )
                    strong_voltage_token = bool(pair) or bool(single)
                    if strong_voltage_token:
                        # e.g. "120/208 Wye" → skip as potential BUS
                        continue

                    # accept "###A"/"### A" or a clean bare number 60..1200
                    m = re.search(r"(?<!\d)([1-9]\d{1,3})(?!\d)\s*(?:A|AMPS?)\b", up)
                    n = None
                    if m:
                        n = int(m.group(1))
                    else:
                        m2 = re.search(r"(?<!\d)([6-9]\d|[1-9]\d{2,3})(?!\d)\b", up)
                        if m2:
                            n = int(m2.group(1))
                    if n is None or not (60 <= n <= 1200):
                        continue

                    # keep it in the header band region
                    if it["y2"] > _header_bottom:
                        continue

                    # pick the top-most, then left-most
                    if best is None or (it["y1"], it["x1"]) < (best["y1"], best["x1"]):
                        best = it

                if best is not None:
                    chosen_map["BUS"] = {
                        "text": best["text"], "x1": best["x1"], "y1": best["y1"],
                        "x2": best["x2"], "y2": best["y2"], "xc": best["xc"], "yc": best["yc"],
                        "conf": float(best.get("conf", 0.5)), "rank": 0.50  # diagnostic rank
                    }

        # ===== NEW: Cross-validate all picks for plausibility =====
        chosen_map = self._cross_validate_picks(chosen_map, labels_map)

        # ===== Convert chosen → normalized outputs =====
        def _to_int(s):
            try: return int(str(s).replace(",", "").strip())
            except Exception: return None

        # NAME
        name = (chosen_map["NAME"]["text"] if chosen_map["NAME"] else "") or ""
        name = self._normalize_name_output(name)

        # VOLTAGE: we already have a chosen candidate; now trim any junk.
        voltage_val = None
        if chosen_map["VOLTAGE"]:
            raw_v = chosen_map["VOLTAGE"]["text"]
            voltage_val = self._trim_voltage_to_allowed(raw_v)

        # --- Voltage fallback: accept lone 120/208/240/480 tokens if no explicit voltage found ---
        # Only do this if we had at least one VOLTAGE label somewhere, so we don't
        # accidentally treat random numbers as system voltage on drawings with no voltage header.
        if voltage_val is None and labels_map.get("VOLTAGE"):
            try:
                _header_bottom = header_bottom
            except NameError:
                _header_bottom = by2

            lone_hits: List[int] = []
            for it in items:
                # Stay within the header region (avoid breaker table numbers)
                if it["y2"] > _header_bottom:
                    continue

                raw_txt = str(it.get("text", "") or "")
                s_txt = self._normalize_digits(raw_txt.upper()).strip()

                # Only accept pure numeric tokens that are exactly one of 120/208/240/480
                # (no units like "A"/"AMPS" and never 600 so we don't confuse with 600A panels).
                m = re.fullmatch(r"(120|208|240|480)", s_txt)
                if not m:
                    continue

                lone_hits.append(int(m.group(1)))

            if lone_hits:
                if self.voltage_first_number_only:
                    # Pick the first one we encountered in scan order
                    voltage_val = lone_hits[0]
                else:
                    # Existing behavior
                    voltage_val = max(lone_hits)

        # BUS (accept with or without unit)
        bus_amp = None
        if chosen_map["BUS"]:
            tU = self._normalize_digits(chosen_map["BUS"]["text"].upper())
            # allow 2–4 digit amps (10–9999), we’ll clamp later
            m = re.search(r"\b([1-9]\d{1,3})\s*(?:A|AMPS?)\b", tU)
            if m:
                bus_amp = _to_int(m.group(1))
            else:
                m2 = re.search(r'(?<!\d)([1-9]\d{1,3})(?!\d)', tU)
                if m2:
                    raw_n = _to_int(m2.group(1))
                    # OCR noise fix: "225A" often misread as "2254" (A→4/6/8)
                    # If number > 1200 and ends in 4/6/8, try dropping last digit
                    if raw_n and raw_n > 1200 and str(raw_n)[-1] in "468":
                        bus_amp = raw_n // 10  # Drop last digit (e.g., 2254 → 225)
                    else:
                        bus_amp = raw_n

        # MAIN
        main_amp = None
        self.last_main_type = None
        if chosen_map["MAIN"]:
            txtU0 = chosen_map["MAIN"]["text"].upper()
            txtU  = self._normalize_digits(txtU0)
            # allow 2–4 digit amps (10–9999), we’ll clamp later
            m = re.search(r"\b([1-9]\d{1,3})\s*(?:A|AMPS?)\b", txtU)
            if m:
                main_amp = _to_int(m.group(1))
            else:
                m2 = re.search(r'(?<!\d)([1-9]\d{1,3})(?!\d)', txtU)
                if m2:
                    main_amp = _to_int(m2.group(1))
            if re.search(r"\b(MLO|MAIN\s*LUGS?)\b", txtU):
                self.last_main_type = "MLO"
            elif re.search(r"\b(MCB|M\W*C\W*B|MAIN\s*BREAKER)\b", txtU):
                self.last_main_type = "MCB"

        # AIC → kA (65kA, 65 kA, 65,000 A, 65000)
        int_rating_ka = None
        if chosen_map["AIC"]:
            t_raw = chosen_map["AIC"]["text"].upper()
            t_fix = self._normalize_digits(t_raw)
            t_nos = t_fix.replace(" ", "")
            # e.g. 65kA
            mk = re.search(r"\b(\d{2,3})K?A\b", t_nos)
            if mk and "KA" in t_nos:
                int_rating_ka = int(mk.group(1))
            else:
                m = re.search(r"\b(\d{2,3}[,]?\d{3})\s*(?:A|KA)?\b", t_fix)
                if m:
                    val = int(m.group(1).replace(",", ""))
                    if (val % 1000) == 0:
                        int_rating_ka = max(1, val // 1000)
                else:
                    # discrete small kA fallback (e.g. chosen token is just "65")
                    SMALL_KA = {10, 14, 18, 22, 25, 30, 35, 42, 50, 65, 100, 125, 200}
                    m_small2 = re.search(r'(?<!\d)(\d{2,3})(?!\d)', t_fix)
                    if m_small2:
                        n = int(m_small2.group(1))
                        if n in SMALL_KA and not re.search(r'\bA(MPS?)?\b', t_fix):
                            int_rating_ka = n


        # Prefer the explicit tag on the CHOSEN MAIN value (if present).
        # If that’s absent, fall back to the global scan.
        mode = ((self.last_main_type or main_mode or "")).upper()
        if mode == "MLO":
            main_amp_out = None                          # never report a main breaker when MLO
        elif mode == "MCB":
            main_amp_out = main_amp or bus_amp           # breaker present; allow fallback to bus if unlabeled
        else:
            main_amp_out = main_amp                      # unknown mode: only use an explicit MAIN value
        
        # --- Normalize to plain ints for rules engine (no units) ---
        voltage_i = self._to_int_or_none(voltage_val)
        bus_i     = self._to_int_or_none(bus_amp)
        main_i    = self._to_int_or_none(main_amp_out)
        aic_i     = self._to_int_or_none(int_rating_ka)

        # Snap any detected bus/main rating below 100A up to 100A
        AMP_FLOOR = 100
        if bus_i is not None and bus_i < AMP_FLOOR:
            bus_i = AMP_FLOOR
        if main_i is not None and main_i < AMP_FLOOR:
            main_i = AMP_FLOOR

        # Build attrs, omitting MAIN entirely for MLO
        attrs = {
            "amperage": bus_i,
            "voltage": voltage_i,
            "intRating": aic_i,          # kA as plain int
            "detected_breakers": [],
        }
        if mode != "MLO":
            attrs["mainBreakerAmperage"] = main_i

        result = {
            "type": "panelboard",
            "name": name,
            "attrs": attrs,
        }

        # ---- expose winning boxes for downstream review overlays ----
        def _rect4(it):
            return [int(it["x1"]), int(it["y1"]), int(it["x2"]), int(it["y2"])]

        winning = {}
        if chosen_map.get("NAME"):
            winning["name"] = _rect4(chosen_map["NAME"])
        if chosen_map.get("VOLTAGE"):
            winning["voltage"] = _rect4(chosen_map["VOLTAGE"])
        if chosen_map.get("BUS"):
            winning["bus"] = _rect4(chosen_map["BUS"])
        if chosen_map.get("MAIN"):
            winning["main"] = _rect4(chosen_map["MAIN"])
        if chosen_map.get("AIC"):
            winning["aic"] = _rect4(chosen_map["AIC"])

        # coords are in the coordinate system of prep_full (after _prep_for_ocr resize)
        result["winningBoxes"] = winning
        result["boxImageShape"] = {"w": int(W), "h": int(H)}

        # ===== Debug overlay =====
        if self.debug:
            # ===== FULL TOKEN TRACE (labels, value-shapes, and candidate ranks) =====
            print("\n==== PANEL HEADER RAW TRACE ====")

            # Map role -> human label
            _role_human = {
                "NAME": "name",
                "VOLTAGE": "volts",
                "BUS": "bus amps",
                "MAIN": "main amps",
                "AIC": "AIC",
                "WRONG": "not of interest",
            }

            # Build a lookup of ranks for quick “did this token become a candidate?”
            def _key(c):
                return (int(round(c["x1"])), int(round(c["y1"])),
                        int(round(c["x2"])), int(round(c["y2"])), (c.get("text","")).strip())
            role_idx = {r:{} for r in ("NAME","VOLTAGE","BUS","MAIN","AIC")}
            for r in role_idx:
                for c in ranked_map.get(r, []) or []:
                    role_idx[r][_key(c)] = float(c.get("rank", 0.0))

            def _rank_for(it, role):
                k = _key(it)
                if k in role_idx[role]:
                    return role_idx[role][k]
                # bbox proximity fallback (±4px)
                x1,y1,x2,y2 = int(round(it["x1"])), int(round(it["y1"])), int(round(it["x2"])), int(round(it["y2"]))
                best = None
                for (kx1,ky1,kx2,ky2,kt), r in role_idx[role].items():
                    if abs(x1-kx1)<=4 and abs(y1-ky1)<=4 and abs(x2-kx2)<=4 and abs(y2-ky2)<=4:
                        best = r if best is None else max(best, r)
                return best

            # Precompile label regex
            comp_labels = {role:[re.compile(rx, re.I) for rx in self._LABELS.get(role, [])]
                        for role in ("NAME","VOLTAGE","BUS","MAIN","AIC","WRONG")}

            for it in items:
                raw = (it["text"] or "").strip()
                if not raw:
                    continue
                up = raw.upper()

                # Determine label roles this token matches
                label_roles = [role for role, rxs in comp_labels.items() if any(rx.search(raw) for rx in rxs)]

                # Determine value “shapes” (diagnostic only; permissive)
                value_roles = []
                # Amps: "###A"/"### A" or bare 60..1200
                if re.search(r"(?<!\d)([1-9]\d{1,3})(?!\d)\s*(?:A|AMPS?)\b[.)]?", up):
                    value_roles.extend(["BUS","MAIN"])
                elif re.search(r"(?<!\d)([6-9]\d|[1-9]\d{2,3})(?!\d)\b", up):
                    value_roles.extend(["BUS","MAIN"])

                # Voltage: 480Y/277, 208/120, or single 120..600 (optional V)
                if re.search(r"\b(\d{3,4})\s*[YV]?[\/]\s*(\d{2,4})\s*V?\b", up) or \
                re.search(r"\b([1-6]\d{2})\s*V?\b", up):
                    value_roles.append("VOLTAGE")

                # AIC: 65kA or 65,000 A etc.
                up_nos = up.replace(" ", "")
                if re.search(r"\b\d{2,3}[kK]A\b", up_nos) or \
                re.search(r"\b(\d{2,3}[,]?\d{3})\s*(?:A|KA)\b", up):
                    value_roles.append("AIC")

                # Name-ish: short alphanum
                if re.fullmatch(r"[A-Z0-9][A-Z0-9._\-\/]{0,12}", up):
                    value_roles.append("NAME")

                # Candidate ranks (if any) for each role
                ranks_bits = []
                for role in ("NAME","VOLTAGE","BUS","MAIN","AIC"):
                    rr = _rank_for(it, role)
                    if rr is not None:
                        ranks_bits.append(f"{_role_human[role]}:{rr:.2f}")

                # Build human text like: "Volts" - label for volts 88%
                parts = []
                if label_roles:
                    for r in label_roles:
                        human = _role_human.get(r, r.lower())
                        if r == "WRONG":
                            parts.append(f"label for {human}")
                        else:
                            parts.append(f"label for {human}")
                if value_roles:
                    # de-dup while preserving order
                    seen = set()
                    vr = [v for v in value_roles if not (v in seen or seen.add(v))]
                    for r in vr:
                        human = _role_human.get(r, r.lower())
                        parts.append(f"value for {human}")

                if not parts:
                    parts.append("not of interest")

                desc = " / ".join(parts)
                ranks_txt = f" | candidates: {', '.join(ranks_bits)}" if ranks_bits else ""
                print(f'"{raw}" - {desc} ({int(round(it["conf"]*100))}%)' + ranks_txt)

            print("\n==== PANEL HEADER CLASSIFY (revised) ====")
            print(f"Band: y=[0,{by2}]  items={len(items)}")
            for role in ("NAME","VOLTAGE","BUS","MAIN","AIC"):
                lst = ranked_map.get(role, [])
                print(f"\n[{role}] candidates (top 10):")
                if not lst:
                    print("  (none)")
                for c in lst[:10]:
                    p = c.get("_parts", {})
                    print(f'  - "{c["text"]}" '
                          f'@({int(c["x1"])},{int(c["y1"])}) '
                          f'conf={c["conf"]:.2f}  '
                          f'shape={p.get("shape",0):.2f} '
                          f'lbl={p.get("lbl",0):.2f} side={p.get("side",0):.2f} '
                          f'ctx={p.get("ctx",0):.2f} wrong={p.get("wrong",0):.2f} '
                          f'ybias={p.get("y",0):.2f}  →  rank={c["rank"]:.2f}')

            print("\nFinal picks:")
            for role in ("NAME","VOLTAGE","BUS","MAIN","AIC"):
                it = chosen_map.get(role)
                if not it:
                    print(f"  {role}: None")
                else:
                    r = float(it.get("rank", 0.0))
                    print(f'  {role}: "{it["text"]}" '
                          f'@({int(it["x1"])},{int(it["y1"])}) '
                          f'conf={it["conf"]:.2f} rank={r:.2f}')
            print("\nNormalized output:")
            print(f'  name="{name}"  voltage={voltage_i}  bus={bus_i}  main={main_i}  aic={aic_i}')
            print("=============================================")
            self._write_overlay_with_band(image_path, prep_full, (y1, by2), items, ranked_map, chosen_map)

        return result
 
    # --------- Helpers ---------
    def _scan_main_mode(self, items: List[dict]) -> Optional[str]:
        txt = " ".join(str(it.get("text","")) for it in items).upper()
        has_mlo = bool(re.search(r"\b(MLO|MAIN\s*LUGS?)\b", txt))
        has_mcb = bool(re.search(r"\b(MCB|M\W*C\W*B)\b", txt))

        # MLO always wins if present anywhere
        if has_mlo:
            return "MLO"
        if has_mcb:
            return "MCB"
        return None

    def _normalize_name_output(self, s: str) -> str:
        import re
        u = (s or "").strip()
        u = re.sub(r'^(?:PANEL(?:BOARD)?\b\s*:?)\s*', '', u, flags=re.I)  # drop ONLY the leading label
        u = re.sub(r'\s{2,}', ' ', u).strip()
        return u

    def _simple_name_from_top(self, lines) -> Optional[dict]:
        """
        Robust name extraction:
        0) NEW: big + centered + name-shaped token wins immediately
        1) Handle SINGLE-TOKEN '...Panel: NAME' (colon inside same token)
        2) Handle MULTI-TOKEN '... Panel : NAME ...' (window to the right)
        3) Fallbacks (label-without-colon, short ID)
        Prefers names that contain at least one letter to avoid picking bare numbers like '3'.
        Also guards against header words like 'SCHEDULE'/'DESIGNATION' (and OCR-typo variants).
        """
        import re
        import difflib

        if not lines:
            return None

        LABEL_WORDS = {"PANEL", "PANELBOARD", "BOARD", "PNL"}
        HARD_STOPS = {
            "SYSTEM","DISTRIBUTION","VOLT","VOLTS","VOLTAGE","V","PHASE","PHASES","PH","Ø",
            "WIRE","WIRES","RATING","BUS","MAIN","MCB","MLO","AIC","KAIC","SCCR","FAULT",
            "MOUNTING","FEED","FEEDS","FED","FROM","BY","FEED-THRU","FEEDTHRU","FEED THRU",
            "NUMBER","NO.","ENCLOSURE","TYPE","NOTES","TABLE","SCHEDULE","DATE","REV","NEUTRAL"
        }

        VOLT_RE = re.compile(r'\b([1-6]\d{2,3})\s*[YV]?[\/]?\s*([1-6]?\d{2,3})\s*V?\b')
        AMPS_RE = re.compile(r'\b([6-9]\d|[1-9]\d{2,3})\s*(A|AMPS?)\b', re.I)

        def _mk_val_by_rects(rects, texts, confs, conf_hint=None, shape=0.98, ctx=0.38):
            r1, r2 = rects[0], rects[-1]
            x1, y1 = min(r1[0], r2[0]), min(r1[1], r2[1])
            x2, y2 = max(r1[2], r2[2]), max(r1[3], r2[3])
            text = re.sub(r"\s+", " ", " ".join((t or "").strip() for t in texts)).strip()
            conf = float(conf_hint if conf_hint is not None else (sum(confs) / max(1, len(confs))))
            return {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "xc": 0.5 * (x1 + x2), "yc": 0.5 * (y1 + y2),
                "conf": conf, "text": text,
                "shape": shape, "ctx": ctx,
            }

        def _has_letter(s: str) -> bool:
            return any(ch.isalpha() for ch in (s or ""))

        def _median(vals):
            if not vals:
                return 0.0
            vals = sorted(vals)
            n = len(vals)
            mid = n // 2
            if n % 2:
                return float(vals[mid])
            return 0.5 * (vals[mid - 1] + vals[mid])

        def _looks_like_header_word(up: str) -> bool:
            """
            Treat OCR-near-misses of words like SCHEDULE/DESIGNATION/PANELBOARD as 'header words',
            not as names. Examples: 'scheduie', 'desicnation', etc.
            
            IMPORTANT: Don't reject panel names like "EXPANEL", "LP1", "PANEL-A" etc.
            Names that have prefixes (EX-, LP, etc.) should be allowed even if they contain
            "PANEL" or other bad words.
            """
            base = re.sub(r'[^A-Z]', '', (up or "").upper())
            if not base:
                return False
            
            bad_words = ["SCHEDULE", "DESIGNATION", "DESIGN", "PANELBOARD", "PANEL"]
            for w in bad_words:
                # Only consider it a header word if:
                # 1. The candidate is close in length to the bad word (±1 char)
                # 2. AND the similarity ratio is high (>=0.74)
                # This prevents "EXPANEL" (7 chars) from matching "PANEL" (5 chars)
                # but still catches OCR mistakes like "PANEI", "PANNEL", "SCHEDUIE"
                if abs(len(base) - len(w)) > 1:
                    continue
                ratio = difflib.SequenceMatcher(a=base, b=w).ratio()
                if ratio >= 0.74:
                    return True
            return False

        top = lines[: min(3, len(lines))]

        # ===== PASS -1: big + centered + name-shaped token =====
        # We look at the top 2–3 lines, find tokens that are much taller than the rest
        # and roughly centered horizontally. These are treated as "designation" style IDs
        # like C2B, LP-1, etc.
        all_h = []
        for ln in top:
            for (r, t, c) in ln["tokens"]:
                all_h.append(max(1, r[3] - r[1]))
        med_h = _median(all_h) or 1.0

        big_center_candidates = []
        full_W = getattr(self, "_last_full_W", None)

        if full_W and med_h > 0:
            for ln in top:
                for (r, t, c) in ln["tokens"]:
                    h = max(1, r[3] - r[1])
                    # "Big": taller than ~1.6x median line height
                    if h < 1.6 * med_h:
                        continue
                    raw = (t or "").strip()
                    up = raw.upper().strip(":")
                    if not up:
                        continue
                    if up in self._NAME_STOPWORDS:
                        continue
                    if _looks_like_header_word(up):
                        continue
                    if VOLT_RE.search(up) or AMPS_RE.search(up):
                        continue
                    if not _has_letter(up):
                        continue
                    # fairly short ID-like token: C2B, LP-1, etc.
                    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._\-/]{0,10}", up):
                        continue

                    # "Centered": horizontal center between ~25% and 75% of page width
                    cx = 0.5 * (r[0] + r[2])
                    if not (0.25 * full_W <= cx <= 0.75 * full_W):
                        continue

                    score = (h / med_h) * float(c or 0.7)
                    big_center_candidates.append((score, r, raw, c))

            if big_center_candidates:
                # Pick the strongest "big centered" candidate and return immediately
                big_center_candidates.sort(key=lambda z: -z[0])
                _, rect, text, conf = big_center_candidates[0]
                return _mk_val_by_rects([rect], [text], [conf],
                                        conf_hint=float(conf or 0.8),
                                        shape=0.99, ctx=0.45)

        # Handles OCR where the panel label *and* the name are fused into a single token.
        # Examples:
        #   "PANEL: LP-1"
        #   "NEW PANEL C"
        #   "EXISTING PNL-2 NORMAL"
        # 
        # IMPORTANT: Multiple OCR passes can produce duplicate/noisy detections of the same
        # text region. We collect all candidates and pick the best one rather than returning
        # the first match. Prefer tokens with ":" (proper format) and higher confidence.
        panel_name_candidates = []
        
        for ln in top:
            for (r, t, c) in ln["tokens"]:
                raw = (t or "").strip()
                if not raw:
                    continue
                up = raw.upper()
                # Only care about tokens that clearly contain a panel label
                if "PANEL" not in up and "PNL" not in up:
                    continue

                # Find PANEL / PANELBOARD / PNL anywhere, then take the text to the right as the name tail.
                m = re.search(r'(?:PANEL(?:BOARD)?|PNL)\b[:\s\-]*', raw, flags=re.I)
                if not m:
                    continue

                # Everything after "PANEL"/"PNL" is the tail
                tail = raw[m.end():]
                # Drop parenthetical annotations like "(NEW)", "(EXISTING)"
                tail = re.sub(r'\([^)]*\)', '', tail)
                tail = tail.strip()
                if not tail:
                    continue

                # Only take the first chunk to avoid "LP-1A NORMAL POWER"
                first = tail.split()[0].strip(" -:'\"")
                if not first:
                    continue

                first_up = first.upper()
                if first_up in self._NAME_STOPWORDS or _looks_like_header_word(first_up):
                    continue
                # Require at least one letter so we don't pick bare "2"
                if not _has_letter(first):
                    continue
                # Short, ID-shaped: LP-1, P6, LP1A, C, etc.
                if not re.fullmatch(r"[A-Z0-9][A-Z0-9._\-/]{0,12}", first_up):
                    continue

                # Score this candidate: prefer tokens with ":" (proper format), higher confidence,
                # and cleaner text (no special chars like ~)
                has_colon = ":" in raw
                has_noise = bool(re.search(r'[~`!@#$%^&*]', raw))  # OCR garbage indicators
                conf_val = float(c or 0.5)
                
                # Score: colon=+1.0, no_noise=+0.5, confidence=+conf
                score = (1.0 if has_colon else 0.0) + (0.5 if not has_noise else 0.0) + conf_val
                
                panel_name_candidates.append({
                    "rect": r, "first": first, "conf": c, "score": score, "raw": raw
                })
        
        # Pick the best candidate
        if panel_name_candidates:
            panel_name_candidates.sort(key=lambda x: -x["score"])
            best = panel_name_candidates[0]
            return _mk_val_by_rects(
                [best["rect"]], [best["first"]], [best["conf"]],
                conf_hint=float(best["conf"] or 0.8),
                shape=0.98,
                ctx=0.42,
            )

        # ===== PASS 0: SINGLE-TOKEN "…Panel: NAME" =====
        for ln in top:
            for (r, t, c) in ln["tokens"]:
                raw = (t or "").strip()
                up = raw.upper()
                if ":" in up:
                    parts = up.split(":", 1)
                    left, right = parts[0], parts[1]
                    if any(lbl in left.split() for lbl in LABEL_WORDS):
                        right_clean = re.sub(r"[^A-Za-z0-9._\-/\s]", "", right).strip()
                        if (right_clean and _has_letter(right_clean)
                                and not VOLT_RE.search(right_clean)
                                and not AMPS_RE.search(right_clean)):
                            if _looks_like_header_word(right_clean.upper()):
                                continue
                            return _mk_val_by_rects(
                                [r], [right_clean], [c],
                                conf_hint=float(c or 0.75),
                                shape=0.98, ctx=0.40
                            )

        # ===== PASS 1: MULTI-TOKEN "… Panel : NAME …" =====
        for ln in top:
            toks = ln["tokens"]
            colon_idx = None
            for i, (r, t, c) in enumerate(toks):
                if ":" in (t or ""):
                    has_panel_left = any(
                        ((tt or "").strip().upper().strip(":") in LABEL_WORDS)
                        for (_, tt, _) in toks[max(0, i - 4): i + 1]
                    )
                    if has_panel_left:
                        colon_idx = i
                        break
            if colon_idx is None:
                continue

            picked_rects, picked_texts, picked_confs = [], [], []
            MAX_TOKENS = 6
            for j in range(colon_idx + 1, len(toks)):
                (r, t, c) = toks[j]
                raw = (t or "").strip()
                up = raw.upper().strip()
                if ":" in raw:
                    break
                if up in HARD_STOPS:
                    break
                if VOLT_RE.search(up) or AMPS_RE.search(up):
                    break
                if up and up not in self._NAME_STOPWORDS:
                    if _looks_like_header_word(up):
                        continue
                    if (_has_letter(up)
                            and (re.fullmatch(r"[A-Z0-9][A-Z0-9._\-/]{0,24}", up) or len(up) <= 16)):
                        picked_rects.append(r)
                        picked_texts.append(raw)
                        picked_confs.append(float(c or 0.7))
                if len(picked_rects) >= MAX_TOKENS:
                    break
            if picked_rects:
                return _mk_val_by_rects(picked_rects, picked_texts, picked_confs)

            # Fallback on this line: first short ID after colon that has a letter
            for j in range(colon_idx + 1, len(toks)):
                (r, t, c) = toks[j]
                up = (t or "").strip().upper().strip(":")
                if (up and _has_letter(up)
                        and up not in self._NAME_STOPWORDS
                        and not _looks_like_header_word(up)
                        and re.fullmatch(r"[A-Z0-9][A-Z0-9._\-/]{1,12}", up)):
                    return _mk_val_by_rects(
                        [r], [t], [c],
                        conf_hint=float(c or 0.7),
                        shape=0.96, ctx=0.35
                    )

        # ===== PASS 2: Label present but no colon → first short ID to its right =====
        for ln in top:
            toks = ln["tokens"]
            for i, (r, t, c) in enumerate(toks):
                up = (t or "").strip().upper().strip(":")
                if up in LABEL_WORDS:
                    for j in range(i + 1, len(toks)):
                        (r2, t2, c2) = toks[j]
                        up2 = (t2 or "").strip().upper().strip(":")
                        if (up2 and _has_letter(up2)
                                and up2 not in self._NAME_STOPWORDS
                                and not _looks_like_header_word(up2)
                                and re.fullmatch(r"[A-Z0-9][A-Z0-9._\-/]{1,12}", up2)):
                            return _mk_val_by_rects(
                                [r2], [t2], [c2],
                                conf_hint=float(c2 or 0.7),
                                shape=0.96, ctx=0.35
                            )

        # ===== PASS 3: Last resort: first short token on top lines that has a letter =====
        for ln in top:
            for (r, t, conf) in ln["tokens"]:
                up = (t or "").strip().upper().strip(":")
                if (up and _has_letter(up)
                        and up not in self._NAME_STOPWORDS
                        and not _looks_like_header_word(up)
                        and re.fullmatch(r"[A-Z0-9][A-Z0-9._\-/]{0,12}", up)):
                    return _mk_val_by_rects(
                        [r], [t], [conf],
                        conf_hint=float(conf or 0.6),
                        shape=0.92, ctx=0.20
                    )

        return None

    def _label_affinity(self, role: str, it: dict, labels_map: dict) -> float:
        import math
        Ls = labels_map.get(role, [])
        if not Ls:
            return 0.0
        def dist(a, b): return math.hypot(a["xc"]-b["xc"], a["yc"]-b["yc"])
        dmin = min(dist(it, L) for L in Ls)
        return math.exp(-dmin / self._SIGMA_PX)

    def _cross_validate_picks(self, chosen_map: dict, labels_map: dict) -> dict:
        """
        Sanity-check and correct chosen values using electrical domain knowledge.
        
        Features:
        1. Value snapping: Correct common OCR misreads to valid electrical values
        2. Range validation: Reject values outside plausible ranges
        3. Context validation: Require supporting labels for ambiguous values
        4. Conflict resolution: Handle cases where same token is picked for multiple roles
        
        Returns modified chosen_map with corrections and invalid picks set to None.
        """
        import re
        
        # Valid electrical values for snapping
        VALID_VOLTAGES = {120, 208, 240, 277, 480, 600}
        VALID_BUS_AMPS = {60, 100, 125, 150, 200, 225, 250, 300, 400, 600, 800, 1000, 1200, 1600, 2000}
        VALID_AIC = {10, 14, 18, 22, 25, 35, 42, 50, 65, 100, 200}
        
        def _snap_to_valid(value: int, valid_set: set, tolerance: float = 0.12) -> int | None:
            """
            Snap a detected value to the nearest valid value if within tolerance.
            
            Examples:
            - 200 -> 208 (within 4% of 208)
            - 22 -> 22 (exact match for AIC)
            - 220 -> 225 (within 2.2% of 225)
            """
            if value is None:
                return None
            
            # First check for exact match
            if value in valid_set:
                return value
            
            # Find nearest valid value
            best_match = None
            best_diff = float('inf')
            
            for v in valid_set:
                diff = abs(value - v)
                rel_diff = diff / max(v, 1)
                
                if rel_diff <= tolerance and diff < best_diff:
                    best_diff = diff
                    best_match = v
            
            return best_match
        
        def _extract_voltage(cand: dict) -> int | None:
            if not cand:
                return None
            txt = str(cand.get("text", "") or "").upper()
            txt = self._normalize_digits(txt)
            
            # Try to find standard voltage values
            for v in (600, 480, 277, 240, 208, 120):
                if str(v) in txt:
                    return v
            
            # Try to extract any 3-digit number and snap it
            m = re.search(r"(?<!\d)([1-6]\d{2})(?!\d)", txt)
            if m:
                raw_v = int(m.group(1))
                snapped = _snap_to_valid(raw_v, VALID_VOLTAGES, tolerance=0.08)
                if snapped and self.debug:
                    if snapped != raw_v:
                        print(f"[ValueSnap] Voltage {raw_v} -> {snapped}")
                return snapped
            return None
        
        def _extract_amps(cand: dict) -> int | None:
            if not cand:
                return None
            txt = str(cand.get("text", "") or "").upper()
            txt = self._normalize_digits(txt)
            m = re.search(r"(?<!\d)([1-9]\d{1,3})(?!\d)", txt)
            if m:
                raw_a = int(m.group(1))
                snapped = _snap_to_valid(raw_a, VALID_BUS_AMPS, tolerance=0.12)
                if snapped and self.debug:
                    if snapped != raw_a:
                        print(f"[ValueSnap] Amps {raw_a} -> {snapped}")
                return snapped if snapped else raw_a  # Return raw if no snap match
            return None
        
        def _extract_aic(cand: dict) -> int | None:
            if not cand:
                return None
            txt = str(cand.get("text", "") or "").upper()
            txt = self._normalize_digits(txt)
            # kA form: 65kA, 22KAIC
            m = re.search(r"(\d{2,3})(?:KAIC|AIC|KA|K)\b", txt.replace(" ", ""))
            if m:
                raw_aic = int(m.group(1))
                snapped = _snap_to_valid(raw_aic, VALID_AIC, tolerance=0.15)
                if snapped and self.debug:
                    if snapped != raw_aic:
                        print(f"[ValueSnap] AIC {raw_aic} -> {snapped}")
                return snapped if snapped else raw_aic
            # Large form: 65,000 -> 65kA
            m2 = re.search(r"(\d{2,3})[,]?000", txt)
            if m2:
                raw_aic = int(m2.group(1))
                snapped = _snap_to_valid(raw_aic, VALID_AIC, tolerance=0.15)
                return snapped if snapped else raw_aic
            return None
        
        # ===== Apply value snapping and update candidate text =====
        
        # Snap voltage
        if chosen_map.get("VOLTAGE"):
            v_volts = _extract_voltage(chosen_map["VOLTAGE"])
            if v_volts and v_volts in VALID_VOLTAGES:
                # Update the snapped value in the candidate for downstream use
                chosen_map["VOLTAGE"]["_snapped_value"] = v_volts
            else:
                if self.debug:
                    print(f"[CrossValidate] Rejecting non-standard voltage: {v_volts}")
                chosen_map["VOLTAGE"] = None
        
        # Snap bus amps
        if chosen_map.get("BUS"):
            v_bus = _extract_amps(chosen_map["BUS"])
            if v_bus:
                chosen_map["BUS"]["_snapped_value"] = v_bus
        
        # Snap main amps
        if chosen_map.get("MAIN"):
            v_main = _extract_amps(chosen_map["MAIN"])
            if v_main:
                chosen_map["MAIN"]["_snapped_value"] = v_main
        
        # Snap AIC
        if chosen_map.get("AIC"):
            v_aic = _extract_aic(chosen_map["AIC"])
            if v_aic:
                chosen_map["AIC"]["_snapped_value"] = v_aic
        
        # Re-extract after snapping
        v_volts = chosen_map["VOLTAGE"].get("_snapped_value") if chosen_map.get("VOLTAGE") else None
        v_bus = chosen_map["BUS"].get("_snapped_value") if chosen_map.get("BUS") else None
        v_main = chosen_map["MAIN"].get("_snapped_value") if chosen_map.get("MAIN") else None
        v_aic = chosen_map["AIC"].get("_snapped_value") if chosen_map.get("AIC") else None
        
        # Rule 2: Validate voltage has some supporting context
        if chosen_map.get("VOLTAGE") and not labels_map.get("VOLTAGE"):
            cand = chosen_map["VOLTAGE"]
            txt = str(cand.get("text", "") or "").upper()
            # Must have explicit voltage syntax (V, Y, /, WYE, DELTA, etc.)
            has_explicit_syntax = bool(
                re.search(r"\dV\b|Y/|\bWYE\b|\bDELTA\b|/\d{2,3}", txt)
            )
            if not has_explicit_syntax:
                if self.debug:
                    print(f"[CrossValidate] Rejecting voltage without label support: {txt}")
                chosen_map["VOLTAGE"] = None
        
        # Rule 3: AIC must be in reasonable range (10-200 kA)
        if chosen_map.get("AIC") and v_aic is not None:
            if v_aic < 10 or v_aic > 200:
                if self.debug:
                    print(f"[CrossValidate] Rejecting out-of-range AIC: {v_aic}kA")
                chosen_map["AIC"] = None
        
        # Rule 4: Bus amps should be reasonable (60-4000A for most panels)
        if chosen_map.get("BUS") and v_bus is not None:
            if v_bus < 60 or v_bus > 4000:
                if self.debug:
                    print(f"[CrossValidate] Rejecting out-of-range bus amps: {v_bus}A")
                chosen_map["BUS"] = None
        
        # Rule 5: If BUS and VOLTAGE are the same token text but parsed differently, reject one
        if chosen_map.get("BUS") and chosen_map.get("VOLTAGE"):
            bus_txt = str(chosen_map["BUS"].get("text", "") or "").strip()
            volt_txt = str(chosen_map["VOLTAGE"].get("text", "") or "").strip()
            if bus_txt == volt_txt:
                # Same token got picked for both - keep whichever has better context
                bus_has_label = bool(labels_map.get("BUS"))
                volt_has_label = bool(labels_map.get("VOLTAGE"))
                if volt_has_label and not bus_has_label:
                    if self.debug:
                        print(f"[CrossValidate] Same token as BUS/VOLTAGE - keeping VOLTAGE")
                    chosen_map["BUS"] = None
                elif bus_has_label and not volt_has_label:
                    if self.debug:
                        print(f"[CrossValidate] Same token as BUS/VOLTAGE - keeping BUS")
                    chosen_map["VOLTAGE"] = None
        
        return chosen_map

    def _is_sentence_context(self, token: dict, items: list, med_h: float) -> bool:
        """
        Returns True if this token appears to be part of a sentence-like context.
        
        Indicators of sentence context:
        - Token is surrounded by multiple words on the same line
        - Neighboring tokens are common sentence words (SEE, REFER, NOTE, PER, etc.)
        - Token appears in a run of 3+ closely-spaced words
        
        This helps reject candidates like "STAGE" in "SEE PANEL STAGE 2".
        """
        if not items or not token:
            return False
        
        # Sentence indicator words that suggest nearby tokens are part of instructions
        SENTENCE_INDICATORS = {
            "SEE", "REFER", "NOTE", "NOTES", "PER", "AS", "SHOWN", "INDICATED",
            "FOR", "THE", "THIS", "THAT", "WITH", "FROM", "TO", "AND", "OR",
            "IS", "ARE", "BE", "WILL", "SHALL", "MUST", "MAY", "CAN",
            "ALL", "EACH", "EVERY", "ANY", "OTHER", "SAME", "SIMILAR",
            "PROVIDE", "INSTALL", "CONNECT", "MOUNT", "LOCATE", "VERIFY",
        }
        
        tx1, ty1, tx2, ty2 = token["x1"], token["y1"], token["x2"], token["y2"]
        t_cy = (ty1 + ty2) / 2.0
        t_height = max(1, ty2 - ty1)
        
        # Find tokens on the same line (vertically overlapping)
        same_line_tokens = []
        for it in items:
            if it is token:
                continue
            iy1, iy2 = it["y1"], it["y2"]
            i_cy = (iy1 + iy2) / 2.0
            
            # Check vertical overlap / same line
            v_overlap = max(0, min(ty2, iy2) - max(ty1, iy1))
            if v_overlap > 0.3 * t_height or abs(i_cy - t_cy) < 0.6 * med_h:
                same_line_tokens.append(it)
        
        if len(same_line_tokens) < 2:
            return False
        
        # Check if any neighboring tokens are sentence indicators
        for neighbor in same_line_tokens:
            n_text = str(neighbor.get("text", "") or "").strip().upper()
            # Check each word in the neighbor token
            for word in n_text.split():
                word_clean = word.strip(".,;:!?()-")
                if word_clean in SENTENCE_INDICATORS:
                    return True
        
        # Check if we're in a run of 3+ words (suggests a sentence/phrase)
        # Sort by x position and check for closely-spaced tokens
        same_line_tokens.append(token)
        same_line_tokens.sort(key=lambda t: t["x1"])
        
        # Count consecutive closely-spaced tokens
        consecutive_count = 1
        for i in range(1, len(same_line_tokens)):
            prev = same_line_tokens[i-1]
            curr = same_line_tokens[i]
            gap = curr["x1"] - prev["x2"]
            # If gap is small (less than 2x median height), tokens are part of same phrase
            if gap < 2.0 * med_h:
                consecutive_count += 1
            else:
                consecutive_count = 1
            
            # If we have 4+ consecutive tokens, this looks like a sentence
            if consecutive_count >= 4:
                return True
        
        return False

    def _collect_label_candidates(self, items: list) -> dict:
        import re
        role_map = {k: [] for k in ("VOLTAGE","BUS","MAIN","AIC","NAME","WRONG")}
        comp = {role: [re.compile(rx, re.I) for rx in rxs] for role, rxs in self._LABELS.items()}
        for it in items:
            txt = str(it["text"])
            for role, rxs in comp.items():
                if any(rx.search(txt) for rx in rxs):
                    role_map[role].append({
                        "x1": it["x1"], "y1": it["y1"], "x2": it["x2"], "y2": it["y2"],
                        "xc": it["xc"], "yc": it["yc"], "conf": float(it["conf"]), "text": txt
                    })
        return role_map

    def _collect_value_candidates(self, items: list) -> dict:
        import re
        out = {k: [] for k in ("NAME","VOLTAGE","BUS","MAIN","AIC")}
        heights = [abs(it2["y2"] - it2["y1"]) for it2 in items] or [1]
        med_h = float(np.median(heights)) if hasattr(np, "median") else (sum(heights) / len(heights))

        _VOLTY_WORDS = {"V", "VOLTS", "VOLT", "VOLTAGE", "WYE", "DELTA", "PHASE", "PH", "WIRES", "WIRE", "Ø"}

        # precompute AIC/SCCR label positions for adjacency checks
        aic_labels: list[dict] = []
        aic_label_rxs = [re.compile(rx, re.I) for rx in self._LABELS.get("AIC", [])]
        for it2 in items:
            t2 = str(it2.get("text", "") or "")
            if any(rx.search(t2) for rx in aic_label_rxs):
                aic_labels.append(it2)

        # Precompute VOLTAGE label positions for proximity checks
        voltage_labels: list[dict] = []
        voltage_label_rxs = [re.compile(rx, re.I) for rx in self._LABELS.get("VOLTAGE", [])]
        for it2 in items:
            t2 = str(it2.get("text", "") or "")
            if any(rx.search(t2) for rx in voltage_label_rxs):
                voltage_labels.append(it2)

        def _has_nearby_voltage_label(it_check: dict) -> bool:
            """Check if a VOLTAGE label is within _SIGMA_PX distance."""
            import math
            if not voltage_labels:
                return False
            for lbl in voltage_labels:
                d = math.hypot(it_check["xc"] - lbl["xc"], it_check["yc"] - lbl["yc"])
                if d <= self._SIGMA_PX * 1.5:  # 1.5x for a bit more tolerance
                    return True
            return False

        for it in items:
            raw = str(it["text"] or "")
            txt = raw.upper()
            txtD = self._normalize_digits(txt)
            conf = float(it["conf"])
            x1,y1,x2,y2,xc,yc = it["x1"],it["y1"],it["x2"],it["y2"],it["xc"],it["yc"]

            # --- NEW: Check for VALUE_STOPWORDS - skip tokens containing these ---
            has_value_stopword = any(sw in txt.split() for sw in self._VALUE_STOPWORDS)

            # --- NEW: Check for sentence-like context ---
            # Tokens that appear in sentences (e.g., "SEE PANEL STAGE 2") should not be values
            is_in_sentence = self._is_sentence_context(it, items, med_h)

            # VOLTAGE (pairs or single; tolerate OCR typos and missing slash)
            txtN = self._normalize_voltage_text(txtD)

            # --- Guards: don't let AIC-like or explicit-amp tokens become VOLTAGE ---

            # AIC-like: 10k..100k numbers, optionally with A/KA (e.g., "18000", "65,000A")
            aic_like = bool(re.search(r"\b(\d{2,3}[,]?\d{3})\s*(?:A|KA)?\b", txtD))

            # Amperage-like: "125 A", "225A", "400 AMPS"
            amps_like = bool(re.search(r"\b([1-9]\d{1,3})\s*(A\.?|AMPS?\.?)\b", txtD))

            # Voltage-ish context words/letters
            has_volty_ctx = bool(
                re.search(r'\b(WYE|DELTA|PH|PHASE|Ø|VOLT|VOLTS|V)\b', txtN)
            )

            # Explicit voltage syntax indicators (strong signal)
            has_explicit_voltage_syntax = (
                "/" in txtN or                                      # Pair separator: 480/277
                re.search(r'\d[YV]\b', txtN) or                     # Y or V suffix: 480Y, 208V
                re.search(r'\b(WYE|DELTA)\b', txtN)                 # Explicit wye/delta
            )

            # Recognize "pair" voltages like "480/277", "208/120", "480Y/277V", etc.
            pair = re.search(
                r'\b([1-6]\d{2,3})\s*[YV]?[\/]?\s*([1-6]?\d{2,3})\s*V?\b',
                txtN,
            )

            # If it looks AIC-like and has no clear voltage context or separators,
            # kill the pair match so things like "18000" don't become VOLTAGE.
            if pair and aic_like and not has_volty_ctx and "/" not in txtN and "Y" not in txtN and "V" not in txtN:
                pair = None

            # NEW: Reject voltage candidates that contain VALUE_STOPWORDS or are in sentence context
            # BUT: If we have explicit voltage syntax (/, Y, V, WYE, DELTA), override these guards
            if pair and not has_explicit_voltage_syntax and (has_value_stopword or is_in_sentence):
                pair = None

            # Single-voltage detection (e.g., "480V", "208", "600V")
            # STRENGTHENED: Require EITHER:
            #   - Explicit voltage syntax (V suffix, etc.), OR
            #   - Nearby VOLTAGE label (within SIGMA_PX distance)
            # This prevents random 3-digit numbers from being detected as voltage.
            single = None
            if not aic_like and not amps_like:
                # If we have explicit voltage syntax, override stopword/sentence checks
                should_check = has_explicit_voltage_syntax or (not has_value_stopword and not is_in_sentence)
                if should_check:
                    if has_explicit_voltage_syntax or _has_nearby_voltage_label(it):
                        single = re.search(r'(?<!\d)([1-6]\d{2,3})(?!\d)', txtN)
                    elif has_volty_ctx:
                        if _has_nearby_voltage_label(it):
                            single = re.search(r'(?<!\d)([1-6]\d{2,3})(?!\d)', txtN)

            if pair or single:
                shape = 0.92 if pair else 0.62
                ctx = 0.15 if has_volty_ctx else 0.0
                # Boost shape score if we have explicit syntax
                if has_explicit_voltage_syntax:
                    shape = min(1.0, shape + 0.08)
                out["VOLTAGE"].append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "xc": xc, "yc": yc,
                    "conf": conf,
                    "text": raw,
                    "shape": min(1.0, shape),
                    "ctx": min(1.0, ctx),
                })

            # -----------------------------------------------------------
            # BUS/MAIN (accept "###A", "### A", and bare "###" in ranges)
            # But:
            #   - allow smaller mains (e.g., 50A) when explicitly tied to MCB/MAIN
            #   - DO NOT treat voltage-looking tokens (120/208 Wye, 480Y/277V) as amps
            # -----------------------------------------------------------

            m_with_unit = re.search(r"\b([1-9]\d{1,3})\s*(A\.?|AMPS?\.?)\b", txtD)  # 10..9999 A w/ unit
            m_bare_num  = re.search(r"(?<!\d)([1-9]\d{1,3})(?!\d)", txtD)           # bare 2–4 digits

            # Strong "this is voltage" signal for amps suppression
            strong_voltage_token = bool(pair) or (
                bool(single) and has_volty_ctx
            )
            if not strong_voltage_token:
                # Extra guards: slash or Wye/Delta/PH context implies voltage
                if "/" in txtN or re.search(r"\b(WYE|DELTA|PH|PHASE|Ø)\b", txtN):
                    strong_voltage_token = True

            # Explicit main-breaker context (relax lower bound here)
            main_ctxt = bool(re.search(r"\b(MCB|MAIN\s*BREAKER|MAIN\s*DEVICE)\b", txt))

            cand = None
            # NEW: Skip BUS/MAIN candidates that contain VALUE_STOPWORDS or are in sentence context
            if not has_value_stopword and not is_in_sentence and m_with_unit:
                n = int(m_with_unit.group(1))
                # For explicit main-breaker tokens, allow 30A–4000A.
                # For everything else, keep the 60A floor to avoid branch circuits.
                lo = 30 if main_ctxt else 60
                if lo <= n <= 4000:
                    cand = {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "xc": xc, "yc": yc,
                        "conf": conf,
                        "text": raw,
                        "shape": 0.90,
                        "ctx": 0.12,
                        "has_unit": True,
                    }

            elif not has_value_stopword and not is_in_sentence and m_bare_num and not strong_voltage_token:
                n = int(m_bare_num.group(1))
                # allow bare 60..1200; rely on label affinity to disambiguate from AIC/others
                # (bare 50, 40, etc. are *not* accepted here to avoid table circuits)
                
                # OCR noise fix: "225A" often misread as "2254" (A→4), "2256" (A→6), "2258" (A→8)
                # If number > 1200, ends in 4/6/8, and token contains BUS/RATING, try dropping last digit
                has_bus_hint = bool(re.search(r'\b(BUS|RATING)\b', txt))
                if n > 1200 and has_bus_hint and str(n)[-1] in "468":
                    n = int(str(n)[:-1])  # Drop last digit (e.g., 2254 → 225)
                
                if 60 <= n <= 1200:
                    # small score because it's unlabeled; still promotable by BUS fallback
                    cand = {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "xc": xc, "yc": yc,
                        "conf": conf,
                        "text": raw,
                        "shape": 0.68,
                        "ctx": 0.04,
                        "has_unit": False,
                    }

            if cand is not None:
                up_full = txt  # already uppercased version of raw text

                # If the token mentions BUS but not MAIN/MCB, treat it as BUS-only.
                has_bus_word  = bool(re.search(r'\bBUS\b', up_full))
                # If the token explicitly mentions MAIN or M.C.B/MCB, treat it as MAIN-only.
                has_main_word = bool(re.search(r'\bMAIN\b|\bM\W*C\W*B\b', up_full))

                if has_bus_word and not has_main_word:
                    # Pure bus rating → only BUS
                    out["BUS"].append(dict(cand))
                elif has_main_word and not has_bus_word:
                    # Pure main rating → only MAIN
                    out["MAIN"].append(dict(cand))
                else:
                    # Ambiguous or unlabeled → still let scoring decide between BUS/MAIN
                    out["BUS"].append(dict(cand))
                    out["MAIN"].append(dict(cand))

            # AIC (10k..100k) and KA forms: 65kA, 65 kA
            # Normalize spaces for kA form matching
            # NEW: Skip AIC candidates that contain VALUE_STOPWORDS or are in sentence context
            if not has_value_stopword and not is_in_sentence:
                t_nos = txtD.replace(" ", "")
                mk = re.search(r"\b(\d{2,3})(?:KAIC|AIC|KA|K)\b", t_nos)
                if mk:
                    val_ka = int(mk.group(1))
                    if 10 <= val_ka <= 100:
                        shape = 0.90
                        ctx = 0.10
                        out["AIC"].append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "xc": xc, "yc": yc,
                            "conf": conf,
                            "text": raw,
                            "shape": shape,
                            "ctx": ctx,
                        })
                else:
                    AIC = re.search(r"\b(\d{2,3}[,]?\d{3})\s*(?:A|KA)?\b", txtD)  # e.g. 65000, 65,000A, 29,000A
                    if AIC:
                        val = int(AIC.group(1).replace(",", ""))
                        if 10000 <= val <= 100000:
                            # ---- Reject non-thousand-rounded values like 29,114 ----
                            if (val % 1000) != 0:
                                pass
                            else:
                                shape = 0.85 + (0.10 if "," in AIC.group(1) else 0.0)
                                ctx = 0.08 if re.search(r"\b(SYMMETRICAL|AIC|A\.?\s*I\.?\s*C\.?|SCCR)\b", txt) else 0.0
                                out["AIC"].append({
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                    "xc": xc, "yc": yc,
                                    "conf": conf,
                                    "text": raw,
                                    "shape": min(1.0, shape),
                                    "ctx": ctx,
                                })
                    else:
                        SMALL_KA = {10, 14, 18, 22, 25, 30, 35, 42, 50, 65, 100, 125, 200}
                        # Look for a plain 2–3 digit number
                        m_small = re.search(r'(?<!\d)(\d{2,3})(?!\d)', txtD)
                        if m_small:
                            n = int(m_small.group(1))
                            if n in SMALL_KA and not re.search(r'\bA(MPS?)?\b', txtD):
                                # Must be horizontally to the right of an AIC label and on roughly the same row
                                for L in aic_labels:
                                    yov = max(0, min(y2, L["y2"]) - max(y1, L["y1"]))  # vertical overlap
                                    dx = x1 - L["x2"]                                   # distance to the right
                                    if yov > 0 and 0 <= dx <= 3 * med_h:
                                        shape = 0.88
                                        ctx = 0.20  # extra context because it's directly tied to AIC label
                                        out["AIC"].append({
                                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                            "xc": xc, "yc": yc,
                                            "conf": conf,
                                            "text": raw,
                                            "shape": shape,
                                            "ctx": ctx
                                        })
                                        break  # only need to prove adjacency to one label

            # NAME candidates (short, alphanum, higher & larger)
            up = raw.strip().upper().strip(":")
            if not up:
                continue

            # ---- NEW: never allow electrical numeric fields to be NAME ----
            # amps like 225A / 225 AMPS
            if re.search(r"\b\d{1,4}\s*(?:A\.?|AMPS?\.?)\b", up):
                continue

            # voltage like 208Y/120, 480/277, 480V
            if re.search(r"\b[1-6]\d{2,3}\s*[YV]?\s*/\s*[1-6]?\d{2,3}\b", up) or re.search(r"\b[1-6]\d{2,3}\s*V\b", up):
                continue

            # AIC like 42K, 65KAIC, 65K, 65000A
            if re.search(r"\b\d{2,3}\s*(?:KAIC|AIC|KA|K)\b", up) or re.search(r"\b\d{2,3}[,]?\d{3}\b", up):
                continue

            # require at least one letter (prevents "3", "42", etc.)
            if not any(ch.isalpha() for ch in up):
                continue

            if up in self._NAME_STOPWORDS or up in _VOLTY_WORDS:
                continue

            # Don't allow NAME-label tokens (PANEL, PANELBOARD, BOARD) to be values
            if any(re.search(rx, up, re.I) for rx in self._LABELS["NAME"]):
                continue

            if re.fullmatch(r"[A-Z0-9][A-Z0-9._\-\/]{0,10}", up):
                # Don't allow NAME-label tokens (PANEL, PANELBOARD, BOARD) to be values
                if any(re.search(rx, up, re.I) for rx in self._LABELS["NAME"]):
                    continue
                if re.fullmatch(r"[A-Z0-9][A-Z0-9._\-\/]{0,10}", up):
                    is_short_id = bool(re.fullmatch(r"[A-Z]{0,2}\d{1,2}", up)) or len(up) <= 4
                    is_all_caps = up.isupper() and any(ch.isalpha() for ch in up)
                    if is_short_id or is_all_caps:
                        h_norm = min(1.5, max(0.5, (it["y2"] - it["y1"]) / max(1.0, med_h)))
                        base = 0.58 + 0.18 * (1.0 if is_short_id else 0.0)
                        shape = min(1.0, base + 0.10 * (h_norm - 1.0))

                        # context: small boosts for explicit panel words *and* top-left placement
                        ctx = 0.05 if re.search(r"\b(PNL|PANEL)\b", raw) else 0.0
                        try:
                            W_full = getattr(self, "_last_full_W", None)
                            band_top = getattr(self, "_last_band_top", 0)
                        except Exception:
                            W_full, band_top = None, 0

                        if W_full:
                            # left-edge boost if within left 20% and in top 18% of SIGMA above band top
                            left_edge = (it["x1"] <= 0.20 * W_full)
                            top_edge  = ((it["y1"] - band_top) <= 0.18 * self._SIGMA_PX)
                            if left_edge and top_edge:
                                ctx += 0.12
                            elif left_edge:
                                ctx += 0.08
                        out["NAME"].append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "xc": xc, "yc": yc,
                            "conf": conf,
                            "text": up,
                            "shape": shape,
                            "ctx": ctx
                        })

        # De-dup near-identical detections from multi-pass OCR
        def _amps_value(text: str) -> Optional[int]:
            t = self._normalize_digits(str(text).upper())
            m = re.search(r"(?<!\d)([1-9]\d{1,3})(?!\d)", t)
            return int(m.group(1)) if m else None

        def _iou(a: dict, b: dict) -> float:
            ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
            bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / float(area_a + area_b - inter)

        for role, lst in out.items():
            if not lst:
                continue

            merged: List[dict] = []
            # sort top-to-bottom, left-to-right for deterministic merging
            for c in sorted(lst, key=lambda d: (d["y1"], d["x1"])):
                if not merged:
                    merged.append(c)
                    continue

                last = merged[-1]

                # For BUS/MAIN/AIC, consider duplicates when:
                #   - bboxes strongly overlap, AND
                #   - numeric value matches.
                if role in ("BUS", "MAIN", "AIC"):
                    same_place = _iou(c, last) >= 0.70
                    n_c = _amps_value(c.get("text", ""))
                    n_l = _amps_value(last.get("text", ""))
                    same_num = (n_c is not None and n_c == n_l)
                    is_dup = same_place and same_num
                else:
                    # For NAME/VOLTAGE we keep the old, stricter rule:
                    # same text and very close x-center.
                    is_dup = (
                        c["text"] == last["text"]
                        and abs(c["xc"] - last["xc"]) < 18
                    )

                if is_dup:
                    # Keep the higher-confidence version
                    if float(c.get("conf", 0.0)) > float(last.get("conf", 0.0)):
                        merged[-1] = c
                else:
                    merged.append(c)

            out[role] = merged

        return out

    def _score_candidates(self, role: str, cands: list, labels_map: dict, band_top: int, main_mode: Optional[str] = None) -> list:
        import math
        baseW = dict(self._WEIGHTS[role])

        if role in ("BUS","MAIN"):
            if main_mode == "MLO":
                if role == "MAIN":
                    baseW["W_shape"] *= 0.70
                    baseW["W_lbl"]   *= 1.20
                else:
                    baseW["W_shape"] *= 1.05
                    baseW["W_lbl"]   *= 1.20
            elif main_mode == "MCB":
                baseW["W_shape"] *= 1.05
                baseW["W_lbl"]   *= 1.05
            else:
                baseW["W_lbl"]   *= 1.20

        same_labels = labels_map.get(role, [])
        wrong_labels = labels_map.get("WRONG", [])

        def dist(a, b):
            return math.hypot(float(a["xc"] - b["xc"]), float(a["yc"] - b["yc"]))

        ranked = []
        for c in cands:
            W = dict(baseW)  # fresh per candidate

            if role in ("BUS","MAIN") and bool(c.get("has_unit")):
                W["W_shape"] = W["W_shape"] + 0.15
                W["W_lbl"]   = W["W_lbl"]   - 0.06

            # optional: keep weights sane (no negatives, no crazy sum)
            for k in ("W_shape","W_conf","W_lbl","W_side","W_ctx","W_wrong","W_y"):
                W[k] = max(0.0, float(W.get(k, 0.0)))

            # label affinity (nearest same-role label)
            S_lbl = 0.0
            if same_labels:
                dmin = min(dist(c, L) for L in same_labels)
                S_lbl = math.exp(-dmin / self._SIGMA_PX)

            # right-of-label preference when vertically overlapping
            S_side = 0.0
            for L in same_labels:
                yov = max(0, min(c["y2"], L["y2"]) - max(c["y1"], L["y1"]))
                if yov > 0 and c["x1"] > L["x2"]:
                    S_side = max(S_side, 1.0)

            # wrong-label penalty
            S_wrong = 0.0
            if wrong_labels:
                dmin_w = min(dist(c, L) for L in wrong_labels)
                S_wrong = math.exp(-dmin_w / self._SIGMA_PX)

            # NAME top bias
            S_y = 0.0
            if role == "NAME":
                y = float(c["yc"])
                S_y = max(0.0, min(1.0, 1.0 - (y - band_top) / max(1.0, 1.0 * self._SIGMA_PX)))

            # --- Role-keyword bias: push "BUS..." toward BUS - and "MAIN..." toward MAIN ---
            tu = str(c.get("text", "")).upper()
            hint = 0.0
            if role == "BUS":
                if re.search(r"\bBUS\b|\bBUS\s*RATING\b|\bPANEL\s*RATING\b", tu):
                    hint += 1.0
                if re.search(r"\bMAIN\b|\bMCB\b|\bMAIN\s*(BREAKER|DEVICE|LUGS?)\b", tu):
                    hint -= 1.0
            elif role == "MAIN":
                if re.search(r"\bMAIN\b|\bMCB\b|\bMAIN\s*(BREAKER|DEVICE|LUGS?)\b", tu):
                    hint += 1.0
                if re.search(r"\bBUS\b|\bBUS\s*RATING\b|\bPANEL\s*RATING\b", tu):
                    hint -= 1.0

            # Scale the bias
            W_hint = 0.22

            TOTAL = (
                W["W_shape"] * float(c.get("shape", 0.0))
                + W["W_conf"]  * float(c.get("conf", 0.0))
                + W["W_lbl"]   * float(S_lbl)
                + W["W_side"]  * float(S_side)
                + W["W_ctx"]   * float(c.get("ctx", 0.0))
                - W["W_wrong"] * float(S_wrong)
                + W["W_y"]     * float(S_y)
                + W_hint       * float(hint)
            )
            ranked.append({
                **c,
                "rank": max(0.0, min(1.0, TOTAL)),
                "_parts": dict(
                    shape=float(c.get("shape",0.0)),
                    conf=float(c.get("conf",0.0)),
                    lbl=float(S_lbl), side=float(S_side),
                    ctx=float(c.get("ctx",0.0)),
                    wrong=float(S_wrong), y=float(S_y),
                    weights=W
                )
            })

        ranked.sort(key=lambda d: (-d["rank"], -float(d.get("conf", 0.0)), d["y1"], d["x1"]))
        return ranked

    def _pick_role(self, role: str, ranked: list, has_label: bool = True) -> dict | None:
        """
        Pick the best candidate for a role if it meets the threshold.
        Uses dynamic thresholding: higher threshold when no supporting label exists.
        """
        if has_label:
            thr = self._THRESH.get(role, 0.5)
        else:
            thr = self._THRESH_NO_LABEL.get(role, self._THRESH.get(role, 0.5))
        return ranked[0] if ranked and ranked[0]["rank"] >= thr else None

    def _normalize_voltage_text(self, s: str) -> str:
        """
        Make OCR-y voltage strings matchable:
        - fix I/l between digits -> '/'
        - fix letter O between digits -> '0'
        - treat '1' between two 3-digit voltages as a '/'
        - collapse punctuation around the pair
        - allow forms like '208Y120V' (missing slash)
        """
        u = (s or "").upper()

        # Replace I or l BETWEEN digits with '/'
        u = re.sub(r'(?<=\d)[IL](?=\d)', '/', u)

        # Replace letter O BETWEEN digits with zero
        u = re.sub(r'(?<=\d)[Oo](?=\d)', '0', u)

        # Generic glue fix: '1201240' → '120/240', '2081120' → '208/120', etc.
        # Pattern: 3 digits, optional spaces, '1', optional spaces, 3 digits
        # Only convert when BOTH sides are known voltage values.
        ALLOWED_VOLTS = {"120", "208", "240", "277", "347", "480", "600"}

        def _glued_pair_repl(m):
            left = m.group(1)
            right = m.group(2)
            if left in ALLOWED_VOLTS and right in ALLOWED_VOLTS:
                return f"{left}/{right}"
            # If it doesn't look like a voltage pair, leave it alone
            return m.group(0)

        u = re.sub(r'(?<!\d)(\d{3})\s*1\s*(\d{3})(?!\d)', _glued_pair_repl, u)

        # Normalize common separators/spaces
        u = u.replace(',', ' ')
        u = u.replace('.', ' ')
        u = re.sub(r'\s+', ' ', u)

        return u

    def _normalize_digits(self, s: str) -> str:
        """
        Normalize common OCR character confusions in numeric contexts.
        
        Handles:
        - O/o -> 0 (between digits or before A/AMPS)
        - l/I/| -> 1 (between digits)
        - S -> 5 (at start of number, common OCR error)
        - B -> 8 (before two digits, e.g., B00 -> 800)
        - Z -> 2 (between digits)
        - Unicode A variants -> ASCII A
        """
        u = s or ""
        
        # O/o → 0 in numeric contexts (iterative to handle OO, OOO, etc.)
        # Run multiple times to catch cases like 6OO -> 60O -> 600
        for _ in range(3):  # Max 3 iterations should handle any practical case
            prev = u
            # O/o between digits → 0  (e.g., 2O8 → 208, 6o0 → 600)
            u = re.sub(r'(?<=\d)[Oo](?=\d)', '0', u)
            # O/o at end of number (before non-digit or end) → 0 (e.g., 48O → 480)
            u = re.sub(r'(?<=\d)[Oo](?=\D|$)', '0', u)
            if u == prev:
                break  # No more changes
        
        # l/I/| between digits → 1 (e.g., 2I5 → 215, 12l0 → 1210)
        u = re.sub(r'(?<=\d)[lI|](?=\d)', '1', u)
        
        # S at start of 2-3 digit number → 5 (e.g., S00 → 500, S08 → 508)
        # Only when followed by digits that would make a valid electrical value
        u = re.sub(r'\bS(?=\d{2,3}\b)', '5', u)
        
        # B before two digits → 8 (e.g., B00 → 800, B25 → 825)
        u = re.sub(r'\bB(?=\d{2}\b)', '8', u)
        
        # Z between digits → 2 (e.g., 1Z0 → 120, 20Z → 202)
        u = re.sub(r'(?<=\d)Z(?=\d)', '2', u)
        
        # O/o after a digit and before A/AMPS.
        # Also handle runs like "4OOA" -> "400A", "22SA" -> "225A"
        def _oo_to_zeros(m):
            return "0" * len(m.group(0))
        u = re.sub(r'(?<=\d)[Oo]+(?=\s*(?:A|AMPS?)\b)', _oo_to_zeros, u, flags=re.I)
        
        # S before A/AMPS after digits → 5 (e.g., 22SA → 225A)
        u = re.sub(r'(?<=\d)S(?=\s*(?:A|AMPS?)\b)', '5', u, flags=re.I)
        
        # Normalize Unicode A variants (Greek alpha Α, Cyrillic А) → ASCII A
        u = u.replace("Α", "A").replace("А", "A")
        
        # Normalize Unicode digit lookalikes
        # Fullwidth digits ０-９ → ASCII 0-9
        for i in range(10):
            u = u.replace(chr(0xFF10 + i), str(i))
        
        return u

    def _merge_ocr_results(self, results_list: List[List]) -> List:
        """
        Merge multiple OCR pass results using confidence-weighted selection.
        
        When multiple passes detect text in the same region, keep the detection
        with the highest confidence. This helps when one pass reads a character
        correctly but with lower confidence, and another misreads it with similar
        confidence.
        
        Args:
            results_list: List of OCR result lists, each containing (box, text, conf) tuples
        
        Returns:
            Merged list of (box, text, conf) tuples
        """
        if not results_list:
            return []
        
        merged = {}
        
        def _bbox_key(box) -> tuple:
            """Create a key for bbox, rounded to nearest 16px for matching nearby detections."""
            if not box or len(box) < 4:
                return None
            try:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                # Round to nearest 16px to catch slightly offset detections across OCR passes
                x1 = int(round(min(xs) / 16) * 16)
                y1 = int(round(min(ys) / 16) * 16)
                x2 = int(round(max(xs) / 16) * 16)
                y2 = int(round(max(ys) / 16) * 16)
                return (x1, y1, x2, y2)
            except Exception:
                return None
        
        def _text_similarity(t1: str, t2: str) -> float:
            """Simple text similarity check."""
            if not t1 or not t2:
                return 0.0
            t1_norm = self._normalize_digits(t1.upper().strip())
            t2_norm = self._normalize_digits(t2.upper().strip())
            if t1_norm == t2_norm:
                return 1.0
            # Check if one is substring of other
            if t1_norm in t2_norm or t2_norm in t1_norm:
                return 0.8
            return 0.0
        
        for results in results_list:
            if not results:
                continue
            for entry in results:
                try:
                    box, text, conf = entry
                except Exception:
                    continue
                
                if not text or not box:
                    continue
                
                key = _bbox_key(box)
                if key is None:
                    continue
                
                conf = float(conf or 0.0)
                
                # Check if we already have a detection at this location
                if key in merged:
                    existing_box, existing_text, existing_conf = merged[key]
                    
                    # If same text (after normalization), keep higher confidence
                    if _text_similarity(text, existing_text) >= 0.8:
                        if conf > existing_conf:
                            merged[key] = (box, text, conf)
                    else:
                        # Different text - keep the one with higher confidence
                        # But boost confidence if multiple passes agree on digit patterns
                        text_norm = self._normalize_digits(text.upper())
                        existing_norm = self._normalize_digits(existing_text.upper())
                        
                        # Check if they extract the same numbers
                        nums_new = re.findall(r'\d+', text_norm)
                        nums_existing = re.findall(r'\d+', existing_norm)
                        
                        if nums_new == nums_existing and nums_new:
                            # Same numbers, keep higher confidence version
                            if conf > existing_conf:
                                merged[key] = (box, text, conf)
                        elif conf > existing_conf + 0.1:
                            # Significantly higher confidence, replace
                            merged[key] = (box, text, conf)
                else:
                    merged[key] = (box, text, conf)
        
        return list(merged.values())

    def _write_overlay_with_band(
        self,
        image_path: str,
        gray_full: np.ndarray,
        y_band: Tuple[int, int],
        items: List[dict],
        ranks_map: Dict[str, List[dict]],
        chosen_map: Dict[str, Optional[dict]],
    ) -> None:
        import os
        import cv2
        import numpy as np

        if gray_full is None or gray_full.size == 0:
            return

        vis = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR) if len(gray_full.shape) == 2 else gray_full.copy()
        H, W = vis.shape[:2]
        by1 = int(max(0, min(H - 2, y_band[0])))
        by2 = int(max(by1 + 1, min(H - 1, y_band[1])))

        YELLOW = (0, 255, 255)
        WHITE  = (255, 255, 255)
        BAND   = (60, 60, 60)
        COLORS = {
            "NAME":    (255, 128,   0),
            "VOLTAGE": (255, 200, 100),
            "BUS":     (  0, 200,   0),
            "MAIN":    (180,   0, 180),
            "AIC":     (  0,   0, 255),
        }

        def put_text(img, text, org, color, scale=0.55, thick=2):
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        def box(img, x1, y1, x2, y2, color, thick=2):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, int(thick))

        # Band
        box(vis, 0, by1, W - 1, by2, BAND, 2)
        put_text(vis, "HEADER OCR BAND", (12, max(20, by1 - 6)), BAND, 0.60, 2)

        # All detections (yellow)
        for it in (items or []):
            x1, y1, x2, y2 = int(it["x1"]), int(it["y1"]), int(it["x2"]), int(it["y2"])
            box(vis, x1, y1, x2, y2, YELLOW, 1)
            label = f'{str(it.get("text",""))[:22]} ({float(it.get("conf",0.0)):.2f})'
            put_text(vis, label, (x1, max(14, y1 - 4)), YELLOW, 0.45, 1)

        # Ranked candidates per role
        MAX_PER_ROLE = 14
        for role, color in COLORS.items():
            lst = (ranks_map.get(role) or [])[:MAX_PER_ROLE]
            for i, c in enumerate(lst):
                x1, y1, x2, y2 = int(c["x1"]), int(c["y1"]), int(c["x2"]), int(c["y2"])
                box(vis, x1, y1, x2, y2, color, 2)
                tag = f"{role}#{i+1} r={float(c.get('rank',0.0)):.2f}"
                put_text(vis, tag, (x1, max(14, y1 - 6)), color, 0.55, 2)

        # Chosen picks
        for role, color in COLORS.items():
            chosen = chosen_map.get(role)
            if not chosen:
                continue
            x1, y1, x2, y2 = int(chosen["x1"]), int(chosen["y1"]), int(chosen["x2"]), int(chosen["y2"])
            box(vis, x1, y1, x2, y2, color, 3)
            cap = f"{role}: CHOSEN  r={float(chosen.get('rank',0.0)):.2f}"
            put_text(vis, cap, (x1, min(H - 6, y2 + 18)), color, 0.60, 2)

        # Legend
        legend = [
            "Legend:",
            "Yellow = OCR tokens",
            "NAME (orange) candidates",
            "VOLTAGE (light orange) candidates",
            "BUS (green) candidates",
            "MAIN (magenta) candidates",
            "AIC (red) candidates",
            "Thick box = chosen",
        ]
        lx, ly, pad, line_h, box_w = 12, 26, 6, 18, 340
        box_h = pad * 2 + line_h * len(legend)
        cv2.rectangle(vis, (lx - pad, ly - 18), (lx - pad + box_w, ly - 18 + box_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (lx - pad, ly - 18), (lx - pad + box_w, ly - 18 + box_h), (60, 60, 60), 1)
        for i, line in enumerate(legend):
            put_text(vis, line, (lx, ly + i * line_h), WHITE, 0.48, 1)

        base_dir = os.path.dirname(image_path) or "."
        debug_dir = os.path.join(base_dir, "debug")
        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception:
            pass
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(debug_dir, f"{base}_header_overlay.png")
        try:
            cv2.imwrite(out_path, vis)
            print(f"[HEADER] Overlay written: {out_path}")
        except Exception as e:
            print(f"[HEADER] Failed to write overlay: {e}")

    def _prep_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for better OCR accuracy.
        
        Steps:
        1. Grayscale conversion
        2. Sharpening to improve edge clarity
        3. CLAHE for adaptive contrast enhancement
        4. Morphological cleanup for thin text
        5. Bilateral filter for noise reduction while preserving edges
        6. High-quality upscaling for small images
        """
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sharpening to improve edge clarity (helps with blurry text)
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        g = cv2.filter2D(g, -1, sharpen_kernel)
        
        # CLAHE with slightly lower clip limit for less noise amplification
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        g = clahe.apply(g)
        
        # Morphological cleanup - helps with thin/broken text characters
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel_morph)
        
        # Bilateral filter for noise reduction while preserving edges
        g = cv2.bilateralFilter(g, d=5, sigmaColor=35, sigmaSpace=35)
        
        # High-quality upscaling with INTER_LANCZOS4 (better for text than CUBIC)
        h, w = g.shape
        if h < 1400 or w < 1400:
            scale = max(1400 / h, 1400 / w, 1.8)
            g = cv2.resize(g, (int(w * scale), int(h * scale)), 
                           interpolation=cv2.INTER_LANCZOS4)
        return g

    @staticmethod
    def _to_int_or_none(s: str):
        if s is None: return None
        try:
            return int(str(s).replace(",", "").strip())
        except Exception:
            return None

    def _voltage_to_int(self, s: str):
        if not s:
            return None
        nums = [int(n) for n in re.findall(r'(?<!\d)(\d{2,4})(?!\d)', str(s))]
        if not nums:
            return None
        if 120 in nums and 240 in nums:
            return 120
        return max(nums)

    @staticmethod
    def _aic_to_ka(s: str):
        if not s:
            return None
        try:
            amps = int(str(s).replace(",", ""))
            return max(1, amps // 1000)
        except Exception:
            m = re.search(r"\d{4,6}", str(s).replace(",", ""))
            return int(m.group(0)) // 1000 if m else None

    @staticmethod
    def _bbox_to_rect(bbox) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    def _group_into_lines(self, tokens: List[Tuple[Tuple[int, int, int, int], str, float]]):
        if not tokens:
            return []
        toks = sorted(tokens, key=lambda t: (((t[0][1] + t[0][3]) / 2.0), t[0][0]))
        lines = []
        for r, t, c in toks:
            cy = (r[1] + r[3]) / 2.0
            placed = False
            for line in lines:
                ly1, ly2 = line['rect'][1], line['rect'][3]
                lcy = (ly1 + ly2) / 2.0
                lh = max(1, min(ly2 - ly1, r[3] - r[1]))
                if abs(cy - lcy) <= 0.6 * lh:
                    line['tokens'].append((r, t, c))
                    x1 = min(line['rect'][0], r[0])
                    y1 = min(line['rect'][1], r[1])
                    x2 = max(line['rect'][2], r[2])
                    y2 = max(line['rect'][3], r[3])
                    line['rect'] = (x1, y1, x2, y2)
                    placed = True
                    break
            if not placed:
                lines.append({'rect': r, 'tokens': [(r, t, c)]})
        for line in lines:
            line['tokens'].sort(key=lambda tt: tt[0][0])
            line['text'] = " ".join([tt[1] for tt in line['tokens']])
        lines.sort(key=lambda L: L['rect'][1])
        return lines
