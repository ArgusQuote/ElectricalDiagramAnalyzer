# OcrLibrary/PanelHeaderParserV11.py
from __future__ import annotations
import os, re, cv2, json, difflib, numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

from OcrLibrary.ocr_timeout import readtext_with_timeout


class PanelParser:
    """
    Panel Header Parser V11
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
    }

    # ===== Label-led first, value-led fallback only if shape is very strong =====
    _GATE = {
        "LBL_MIN": {
            "VOLTAGE":   0.22,
            "BUS":       0.20,
            "MAIN":      0.20,
            "AIC":       0.20,
            "NAME":      0.15,
            "MOUNTING":  0.00,
            "ENCLOSURE": 0.00,
        },
        "SHAPE_STRONG": {
            "VOLTAGE":   0.88,
            "BUS":       0.82,
            "MAIN":      0.82,
            "AIC":       0.85,
            "NAME":      0.78,
            "MOUNTING":  0.70,
            "ENCLOSURE": 0.70,
        },
        "SHAPE_STRONG_NO_LABEL": {
            "VOLTAGE":   0.90,
            "BUS":       0.85,
            "MAIN":      0.85,
            "AIC":       0.88,
            "NAME":      0.82,
            "MOUNTING":  0.70,
            "ENCLOSURE": 0.70,
        },
        "CONF_MIN_NO_LABEL": {
            "VOLTAGE":   0.45,
            "BUS":       0.45,
            "MAIN":      0.45,
            "AIC":       0.40,
            "NAME":      0.45,
            "MOUNTING":  0.45,
            "ENCLOSURE": 0.45,
        },
        "MIN_SHAPE_ALWAYS": {
            "VOLTAGE":   0.60,
            "BUS":       0.65,
            "MAIN":      0.65,
            "AIC":       0.75,
            "NAME":      0.55,
            "MOUNTING":  0.60,
            "ENCLOSURE": 0.60,
        },
        "WRONG_MAX": 0.55,
    }

    # ====== Label families (regex) ======
    _LABELS = {
        "VOLTAGE": [
            r"\bVOLT(AGE|S)?\b", r"\bVOLTS?\b", r"\bVAC\b", r"\bVDC\b", r"\bV\b",
            r"\bPH(ASE)?\b", r"\bWIRE(S)?\b", r"\bØ\b"
        ],

        # BUS should be BUS-specific only
        "BUS": [
            r"\bBUS{1,2}\b",
            r"\bBUS{1,2}\s*RATING\b",
        ],

        "MAIN": [
            r"\bMAIN\s*(RATING|BREAKER|BKR|BRKR|DEVICE|TYPE)\b",
            r"\bMAIN\s+CIRCUIT\s*(BREAKER|BKR|BRKR)\b",
            r"\b(?:MCB|M\W*C\W*B)\b",
            r"\bMLO\b",
            r"\bMAIN\s*LUGS?\b",
            r"\bMAIN\s*TYPE\b",
            r"\bMAINS?\b",
            r"\bMAIN\b"
        ],

        "MOUNTING": [
            r"\bMOUNT(?:ING)?\b",
            r"\bM0UNT(?:ING)?\b",
        ],

        "ENCLOSURE": [
            r"\bENCLOSURE\b",
            r"\bENCL(?:OSURE)?\b",
        ],

        # Generic rating labels (used only when BUS/MAIN explicit labels are missing)
        "RATING": [
            r"\bPANEL\s*RATING\b",
            r"\bAMPACITY\b",
            r"\bRATING\b",
        ],

        "AIC": [
            r"\bA\.?\s*I\.?\s*C\.?\b", r"\bAIC\b", r"\bKAIC\b", r"\bSCCR\b",
            r"\bINTERRUPTING\s*RATING\b", r"\bAVAILABLE\s*FAULT\s*CURRENT\b", r"\bFAULT\s*CURRENT\b",
            r"\bSYMMETRICAL\b"
        ],
        "NAME": [
            r"\bPANEL\s*DESIGNATION\b",
            r"\bDESIGNATION\b",
            r"\bPANEL\s*MARK\b",

            # strong explicit label forms
            r"\bPANEL\s*:\b",
            r"\bPANELBOARD\s*:\b",
            r"\bBOARD\s*:\b",
            r"\bMARK\s*:\b",

            # weaker generic forms
            r"\bDISTRIBUTION\s*PANEL\b",
            r"\bPANEL\b",
        ],
        "WRONG": [
            r"\bNOTES?\b", r"\bTABLE\b", r"\bSCHEDULE\b", r"\bSIZE\s*\(??A\)?\b", r"\bCIRCUIT\b",
            r"\bCAT(ALOG)?\b", r"\bDWG\b", r"\bREV\b", r"\bDATE\b"
        ],
    }

    # ====== Role weights (blend of value-shape, label-affinity, side, ctx, penalties) ======
    _WEIGHTS = {
        "VOLTAGE": dict(W_shape=0.55, W_conf=0.15, W_lbl=0.22, W_side=0.12, W_ctx=0.07, W_wrong=0.12, W_y=0.00),
        "BUS":     dict(W_shape=0.55, W_conf=0.15, W_lbl=0.18, W_side=0.09, W_ctx=0.05, W_wrong=0.15, W_y=0.00),
        "MAIN":    dict(W_shape=0.55, W_conf=0.15, W_lbl=0.20, W_side=0.11, W_ctx=0.05, W_wrong=0.15, W_y=0.00),
        "AIC":     dict(W_shape=0.60, W_conf=0.12, W_lbl=0.22, W_side=0.12, W_ctx=0.08, W_wrong=0.12, W_y=0.00),
        "NAME":    dict(W_shape=0.55, W_conf=0.10, W_lbl=0.15, W_side=0.12, W_ctx=0.00, W_wrong=0.08, W_y=0.35),
        "MOUNTING": dict(W_shape=0.55, W_conf=0.12, W_lbl=0.24, W_side=0.14, W_ctx=0.08, W_wrong=0.12, W_y=0.00),
        "ENCLOSURE": dict(W_shape=0.55, W_conf=0.12, W_lbl=0.24, W_side=0.14, W_ctx=0.08, W_wrong=0.12, W_y=0.00),
    }

    VOLTAGE_CANONICAL_MAP = {
        120: [
            "120V",
            "120 VOLTS",
        ],
        120240: [
            "120/240V",
            "120/240",
            "240/120V",
            "240/120",
        ],
        208: [
            "208V",
            "208 VOLTS",
            "208/120V",
            "208/120",
            "208Y/120V",
            "208Y/120",
            "120/208V",
            "120/208",
            "120/208Y",
            "120V/208Y",
            "120/208 WYE",
            "208/120 WYE",
            "208Y120V",
            "120208",
            "208120",
        ],
        240: [
            "240V",
            "240 VOLTS",
        ],
        480: [
            "480V",
            "480 VOLTS",
            "480/277V",
            "480/277",
            "480Y/277V",
            "480Y/277",
            "277/480V",
            "277/480",
            "277/480 WYE",
            "480/277 WYE",
            "480Y277V",
            "277480",
            "480277",
        ],
        600: [
            "600V",
            "600 VOLTS",
            "600/347V",
            "600/347",
            "600Y/347V",
            "600Y/347",
            "347/600V",
            "347/600",
            "347/600 WYE",
            "600/347 WYE",
            "600Y347V",
            "347600",
            "600347",
        ],
    }

    VOLTAGE_OUTPUT_MAP = {
        120: 120,
        120240: 120,
        208: 208,
        240: 240,
        480: 480,
        600: 600,
    }

    _THRESH = {"VOLTAGE":0.55, "BUS":0.54, "MAIN":0.54, "AIC":0.54, "NAME":0.50, "MOUNTING": 0.52, "ENCLOSURE": 0.52,}

    _SIGMA_PX = 80.0

    def __init__(self, debug: bool = False, voltage_first_number_only: bool = True, reader=None):
        self.debug = debug
        self.voltage_first_number_only = voltage_first_number_only
        self.reader = reader
        if self.reader is None and _HAS_OCR:
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
        Return ONLY one of {120, 208, 240, 480, 600} using:
        1) strong normalization
        2) snap to closest canonical voltage family
        3) fallback direct parsing if needed
        """
        if not txt:
            return None

        s = self._normalize_digits(str(txt).upper())
        s = self._normalize_voltage_text(s)

        snapped = self._snap_voltage_text(s)
        if snapped is not None:
            return self.VOLTAGE_OUTPUT_MAP.get(snapped)

        allowed = {120, 208, 240, 480, 600}

        # direct explicit pair fallback
        pair = re.search(r'(?<!\d)(\d{3})\s*[YV]?\s*/\s*(\d{3})(?!\d)', s)
        if pair:
            a = int(pair.group(1))
            b = int(pair.group(2))
            if a in allowed and b in allowed:
                if {a, b} == {120, 240}:
                    return 120
                return max(a, b)

        # direct single-number fallback
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

        # OCR (normal + inverted)
        if self.reader is None and _HAS_OCR:
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)

        detailed = readtext_with_timeout(
            self.reader,
            prep, detail=1, paragraph=False,
            mag_ratio=1.6, contrast_ths=0.05, adjust_contrast=0.7,
            text_threshold=0.4, low_text=0.3,
        )
        try:
            inv = cv2.bitwise_not(prep)
            det2 = readtext_with_timeout(
                self.reader,
                inv, detail=1, paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,()/:-._\"'“”‘’ kKVvYØø ",
                mag_ratio=1.9, contrast_ths=0.05, adjust_contrast=0.7,
                text_threshold=0.4, low_text=0.25
            )
            detailed = list(detailed) + list(det2)
        except Exception:
            pass

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

        # --- Strong colon-led extractors for non-NAME header values ---
        for _role in ("VOLTAGE", "BUS", "MAIN", "AIC", "MOUNTING", "ENCLOSURE"):
            colon_val = self._extract_value_after_colon_label(lines, _role)
            if colon_val:
                value_cands.setdefault(_role, []).append(colon_val)

        # --- Simple NAME injector (top 1–2 lines) ---
        simple_name = self._simple_name_from_top(lines)
        if simple_name:
            value_cands.setdefault("NAME", []).append(simple_name)

        special_header_type = self._detect_special_header_type(items, lines)

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
            for _role in ("BUS", "MAIN", "AIC", "VOLTAGE", "MOUNTING", "ENCLOSURE", "NAME"):
                value_cands[_role] = [c for c in value_cands.get(_role, []) if c["y2"] <= header_bottom]

        main_mode = self._scan_main_mode(items)  # "MLO" / "MCB" / None
        band_top = y1

        ranked_map = {}
        chosen_map = {}
        ROLE_ORDER = ("VOLTAGE", "AIC", "BUS", "MAIN", "MOUNTING", "ENCLOSURE", "NAME")

        # ===== Consume-as-we-go =====
        used = set()

        def _cand_key(c: dict) -> tuple:
            """
            Stable identity for a candidate across roles/pools.
            Quantize bbox a bit so normal+inverted OCR bbox drift collapses.
            """
            if not c:
                return None

            txt = str(c.get("text", "") or "").strip().upper()
            txt = self._normalize_digits(txt)

            q = 6  # px bucket to collapse multi-pass bbox jitter

            def Q(v):
                try:
                    return int(round(float(v))) // q
                except Exception:
                    return -1

            return (
                Q(c.get("x1", -1)),
                Q(c.get("y1", -1)),
                Q(c.get("x2", -1)),
                Q(c.get("y2", -1)),
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

        def _as_ranked(role: str, c: dict) -> dict:
            """
            If c came from value_cands, it may not have rank/_parts.
            Replace it with the corresponding ranked_map version (same token)
            so debug + gating + used-set behavior stays consistent.
            """
            if not c:
                return c

            ck = _cand_key(c)
            if ck is None:
                return c

            # 1) exact key match
            for rc in (ranked_map.get(role) or []):
                if _cand_key(rc) == ck:
                    return rc

            # 2) tolerant match: allow small bbox drift (quantization should already handle most)
            x1, y1, x2, y2, txt = ck
            best = None
            best_iou = 0.0

            def _iou_rect(a: dict, b: dict) -> float:
                ax1, ay1, ax2, ay2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
                bx1, by1, bx2, by2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0.0
                inter = (ix2 - ix1) * (iy2 - iy1)
                areaA = max(1.0, (ax2 - ax1) * (ay2 - ay1))
                areaB = max(1.0, (bx2 - bx1) * (by2 - by1))
                return inter / float(areaA + areaB - inter)

            for rc in (ranked_map.get(role) or []):
                if self._normalize_digits(str(rc.get("text", "")).upper()).strip() != txt:
                    continue
                iou = _iou_rect(c, rc)
                if iou > best_iou:
                    best_iou = iou
                    best = rc

            return best if best is not None and best_iou >= 0.70 else c

        def _is_colon_labeled(c: dict | None, role: str | None = None) -> bool:
            if not c:
                return False

            if c.get("fromColonLabel"):
                return True

            if not role:
                return False

            t = self._normalize_digits(str(c.get("text", "")).upper()).strip()
            t = re.sub(r"\s+", " ", t)

            amp_value = r"[1-9]\d{1,3}\s*(?:A|AMP|AMPS|AMPERES|AMPERE|MAP)?"

            if role == "BUS":
                return bool(re.search(
                    rf"\bBUSS?\b(?:\s+(?:AMPS?|AMPERES?|AMPERE|RATING|SIZE|RATED))*\s*:\s*{amp_value}\b",
                    t
                ))

            if role == "MAIN":
                return bool(
                    re.search(
                        rf"\bMAINS?\b(?:\s+(?:CIRCUIT|BREAKER|BRKR|BKR|DEVICE|AMPS?|AMPERES?|AMPERE|RATING|SIZE|RATED))*\s*:\s*{amp_value}\b",
                        t
                    )
                    or re.search(
                        rf"\bM\s*\.?\s*C\s*\.?\s*B\s*\.?\s*:\s*{amp_value}\b",
                        t
                    )
                )

            return False

        def _overlaps_chosen_name(c: dict | None) -> bool:
            """
            True if candidate overlaps the currently chosen NAME box.
            Used to stop later BUS/MAIN reassignment from stealing amp-like text
            out of a NAME-owned colon label/value line.
            """
            if not c:
                return False

            nm = chosen_map.get("NAME")
            if not nm:
                return False

            ax1, ay1, ax2, ay2 = float(c["x1"]), float(c["y1"]), float(c["x2"]), float(c["y2"])
            bx1, by1, bx2, by2 = float(nm["x1"]), float(nm["y1"]), float(nm["x2"]), float(nm["y2"])

            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return False

            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            iou = inter / float(area_a + area_b - inter)

            # also treat same-line horizontal overlap as suspicious, even if IoU is modest
            same_row = abs((ay1 + ay2) * 0.5 - (by1 + by2) * 0.5) <= max(18.0, 0.6 * min(ay2 - ay1, by2 - by1))
            horiz_overlap = min(ax2, bx2) - max(ax1, bx1)

            return (iou >= 0.18) or (same_row and horiz_overlap > 0)
        
        def _conflicts_with_role(role: str, text: str) -> bool:
            """
            Hard exclusions to prevent role confusion during value-led fallback.
            Example: don't let 100A become VOLTAGE, don't let 120/208 become BUS, etc.
            """
            if not text:
                return True

            t = self._normalize_digits(str(text).upper()).strip()
            tN = self._normalize_voltage_text(t)
            snapped_voltage = self._snap_voltage_text(tN)

            is_amps = bool(re.search(r"\b\d{1,4}\s*(A\.?|AMP\.?|AMPS?\.?)\b", t))
            is_aic  = (
                bool(re.search(r"\b\d{2,3}\s*(KAMP|KAIC|AIC|KA|K)\b", re.sub(r"\s+", "", t)))
                or bool(re.search(r"\b\d{2,3}[,]?\d{3}\b", t))
            )
            is_vpair = (
                snapped_voltage is not None
                or bool(re.search(r"\b[1-6]\d{2,3}\s*[YV]?\s*/\s*[1-6]?\d{2,3}\b", tN))
                or bool(re.search(r"\b[1-6]\d{2,3}\s*V\b", tN))
            )
            has_volt_words = bool(re.search(r"\b(VOLT|VOLTS|V)\b", tN))

            if role == "VOLTAGE":
                return is_amps or is_aic

            if role in ("BUS", "MAIN"):
                looks_voltage = is_vpair or (has_volt_words and not is_amps)
                return looks_voltage or is_aic

            if role == "AIC":
                # Only hard-conflict AIC with voltage-looking tokens.
                # DO NOT try to decide "10,000 A" vs "225A" here — _passes_gate() handles that using lbl affinity.
                return is_vpair or has_volt_words

            if role == "NAME":
                return is_amps or is_aic or is_vpair or has_volt_words

            if role == "MOUNTING":
                return is_amps or is_aic or is_vpair

            if role == "ENCLOSURE":
                return is_amps or is_aic or is_vpair

            return False

        def _passes_gate(role: str, c: dict, labels_map: dict) -> bool:
            """
            V9 gate:
            - If role labels exist: accept if label-led is good OR value-led is extremely strong + non-conflicting
            - If role labels do not exist: accept ONLY if value-led is extremely strong + confidence min + non-conflicting
            - Always avoid WRONG-context pull unless label affinity is strong
            """
            if not c:
                return False

            p = (c.get("_parts", {}) or {})
            lbl   = float(p.get("lbl", 0.0))
            shape = float(p.get("shape", 0.0))
            wrong = float(p.get("wrong", 0.0))
            conf  = float(c.get("conf", 0.0))

            # Bare 50 is risky because it can be a random schedule/header number.
            # Allow it only when strongly tied to BUS/MAIN/RATING label logic.
            if role in ("BUS", "MAIN") and c.get("requiresStrongAmpLabel"):
                if lbl < 0.60 and not bool(c.get("fromColonLabel")):
                    return False
    
            has_role_labels = bool(labels_map.get(role))

            LBL_MIN  = float(self._GATE["LBL_MIN"].get(role, 0.20))
            STRONG   = float(self._GATE["SHAPE_STRONG"].get(role, 0.85))
            STRONG_N = float(self._GATE["SHAPE_STRONG_NO_LABEL"].get(role, 0.88))
            CONF_N   = float(self._GATE["CONF_MIN_NO_LABEL"].get(role, 0.45))
            WRONG_MAX = float(self._GATE.get("WRONG_MAX", 0.55))

            MIN_SHAPE = float(self._GATE["MIN_SHAPE_ALWAYS"].get(role, 0.0))
            if shape < MIN_SHAPE:
                return False

            # Special-case: AIC can be written as amps ("10,000 A").
            # Allow trailing "A" ONLY when it's clearly tied to the AIC label (high lbl affinity)
            if role == "AIC":
                t = self._normalize_digits(str(c.get("text", "")).upper()).strip()

                # Strong AIC-only handling for amp-form interrupt ratings like:
                #   10,000 A
                #   22,000 A
                #   65,000 A
                #
                # These should ONLY be allowed when clearly tied to the AIC label.
                has_aic_words = bool(re.search(r"\b(AIC|KAIC|SCCR|INTERRUPTING|FAULT)\b", t))
                looks_like_aic_amps = bool(re.search(r"\b(\d{2,3}[,]?\d{3})\s*A\b", t))  # 10,000 A / 65000 A
                aic_amp_match = re.search(r"\b(\d{2,3}[,]?\d{3})\s*A\b", t)

                if aic_amp_match:
                    nA = int(aic_amp_match.group(1).replace(",", ""))

                    # Reject weird non-thousand-rounded values
                    if (nA % 1000) != 0:
                        return False

                    # Reject anything too small to be interrupting rating amps
                    if nA < 10000:
                        return False

                    # If this is the plain amps form (10,000 A) without AIC words,
                    # require VERY strong AIC label support or colon-led extraction.
                    if not has_aic_words:
                        if not (lbl >= 0.85 or bool(c.get("fromColonLabel"))):
                            return False

                # Reject normal amp ratings like 225A / 400A / 1200A from ever becoming AIC
                m_small_amps = re.search(r"(?<![\d,])([1-9]\d{1,3})(?![\d,])\s*A\b", t)
                if m_small_amps:
                    n_small = int(m_small_amps.group(1))
                    if n_small < 5000:
                        return False

            # Hard conflict check
            if _conflicts_with_role(role, str(c.get("text", ""))):
                return False

            # Wrong-context penalty: if too close to WRONG labels, only allow if label affinity is strong
            if wrong >= WRONG_MAX and lbl < (LBL_MIN + 0.10) and shape < 0.90:
                return False

            if has_role_labels:
                # Label-led acceptance
                if lbl >= LBL_MIN:
                    return True
                # Value-led override only if shape is extremely strong
                return shape >= STRONG
            else:
                # No labels: must be very strong and confident
                return (shape >= STRONG_N) and (conf >= CONF_N)

        def _pick_role_consuming(role: str, ranked: list) -> dict | None:
            """
            V9: Like _pick_role, but:
            - skips candidates already used
            - applies label-led / value-led gating
            - prefers returning None over wrong guesses
            """
            thr = self._THRESH.get(role, 0.5)
            for c in (ranked or []):
                if float(c.get("rank", 0.0)) < thr:
                    continue
                if _is_used(c):
                    continue
                # V9 gate: must pass label-led / value-led acceptance
                if not _passes_gate(role, c, labels_map):
                    continue
                if self.debug:
                    print(f"{role} CHECK", c.get("text"), c.get("rank"), "used=", _is_used(c), "gate=", _passes_gate(role, c, labels_map))

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
        if self.debug:
            print("AFTER LOOP NAME =", chosen_map.get("NAME", {}).get("text") if chosen_map.get("NAME") else None)
        
        # --- Colon evidence for BUS/MAIN ---
        # Colon labeling is a confidence signal, NOT a hard exclusivity requirement.
        # OCR may produce both "BUS AMPS: 100" and bare "100"; the chosen candidate
        # may be bare even though the visual label/value association had a colon.
        bus_pick = chosen_map.get("BUS")
        main_pick = chosen_map.get("MAIN")

        bus_colon = _is_colon_labeled(bus_pick, "BUS")
        main_colon = _is_colon_labeled(main_pick, "MAIN")

        # Do not clear BUS/MAIN purely because only one selected candidate is colon-labeled.
        # The colon-led extractor already boosts confidence through fromColonLabel/shape/ctx.

        # If NAME was claimed from a colon label, do not allow MAIN/BUS to reuse
        # overlapping text from that same name region.
        if _is_colon_labeled(chosen_map.get("NAME")):
            if chosen_map.get("MAIN") and _overlaps_chosen_name(chosen_map["MAIN"]):
                _set_role("MAIN", None)
            if chosen_map.get("BUS") and _overlaps_chosen_name(chosen_map["BUS"]):
                _set_role("BUS", None)

        def _looks_like_electrical_value_for_name(s: str) -> bool:
            """
            Return True only when the token is *clearly* an electrical rating/value
            (amps/volts/aic) and therefore should NOT be used as the panel NAME.

            Important: do NOT nuke short alphanum IDs like P1A, LP-1, C2B, etc.
            """
            if not s:
                return False

            t = self._normalize_digits(str(s).upper()).strip()
            t = t.replace("Α", "A").replace("А", "A")  # unicode A -> ASCII A

            # If it contains any letters (besides the unit letters in the patterns below),
            # treat it as name-shaped and do NOT auto-nuke.
            # Example: "P1A", "LP-1", "C2B", "PP1" should survive.
            if re.search(r"[B-Z]", t):   # letters other than A/V/K (units) -> definitely name-like
                return False

            # --- Hard electrical patterns ---
            # Amps (explicit unit)
            if re.search(r"(?<!\d)\d{1,4}\s*(A\.?|AMPS?\.?)\b", t):
                return True

            # Voltage pairs (with slash or Y/ V context)
            if re.search(r"(?<!\d)[1-6]\d{2,3}\s*[YV]?\s*/\s*[1-6]?\d{2,3}(?!\d)", t):
                return True

            # Single voltage with V unit
            if re.search(r"(?<!\d)[1-6]\d{2,3}\s*V\b", t):
                return True

            # AIC/kA tokens
            if re.search(r"(?<!\d)\d{2,3}\s*(KAMP|KAIC|AIC|KA|K)\b", t):
                return True

            # Large AIC-like raw amps (65,000 etc)
            if re.search(r"(?<!\d)\d{2,3}[,]?\d{3}(?!\d)", t):
                return True

            # Pure numeric tokens are almost never panel names (e.g., "18000", "125", "3")
            if re.fullmatch(r"\d{1,5}", t):
                return True

            return False

        # Only nuke NAME if it's clearly an electrical value token
        if chosen_map.get("NAME") and _looks_like_electrical_value_for_name(chosen_map["NAME"].get("text", "")):
            chosen_map["NAME"] = None

        if self.debug:
            print("AFTER ELECTRICAL-NUKE NAME =", chosen_map.get("NAME", {}).get("text") if chosen_map.get("NAME") else None)

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
                mk = re.search(r'(?<!\d)(\d{2,3})(?:KAMP|KAIC|AIC|KA|K)\b', t_nos)
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

            # If NAME was claimed from a colon-owned label/value, do not allow later
            # BUS/MAIN reassignment to reuse overlapping amp-looking text from that same source area.
            if _is_colon_labeled(chosen_map.get("NAME")) and _overlaps_chosen_name(c):
                continue

            k = _cand_key(c)   # <-- use the same identity logic as the "used" set
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
                # Never steal from a colon-owned NAME region
                if _is_colon_labeled(chosen_map.get("NAME")) and _overlaps_chosen_name(c):
                    continue

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
                bus_key = _cand_key(bus_pick)
                scored_main = [t for t in scored if _cand_key(t[0]) != bus_key]
                main_sorted = sorted(
                    scored_main,
                    key=lambda t: (-(t[3] - t[2]), -t[3], t[0]["y1"], t[0]["x1"])
                )

                main_pick = None
                for (c, _, _, _) in main_sorted:
                    # Never steal from a colon-owned NAME region
                    if _is_colon_labeled(chosen_map.get("NAME")) and _overlaps_chosen_name(c):
                        continue

                    if _is_used(c):
                        if any(chosen_map.get(r) and _cand_key(chosen_map[r]) == _cand_key(c) for r in _protected_roles):
                            continue
                    main_pick = c
                    break

                # Apply with used-set updates (no sharing allowed here)
                if bus_pick is not None:
                    _set_role("BUS", _as_ranked("BUS", bus_pick))
                if main_pick is not None:
                    _set_role("MAIN", _as_ranked("MAIN", main_pick))

        elif len(unit_amps) == 1:
            only = unit_amps[0]
            mode_upper = (main_mode or "").upper()
            aff_b = 0.0
            aff_m = 0.0

            # If we already found DISTINCT BUS and MAIN tokens earlier,
            # do NOT overwrite them with the "single token" fallback logic.
            already_distinct = (
                chosen_map.get("BUS") is not None
                and chosen_map.get("MAIN") is not None
                and _cand_key(chosen_map["BUS"]) != _cand_key(chosen_map["MAIN"])
            )
            if already_distinct:
                # keep existing BUS/MAIN picks
                pass
            else:
                if mode_upper == "MLO":
                    # MLO: panel has no main device → only BUS rating is meaningful.
                    _set_role("BUS", _as_ranked("BUS", only))
                    _set_role("MAIN", None)

                elif mode_upper == "MCB":
                    # MCB: Only share if one of the roles is missing.
                    # If both are already set (same token), leave it alone.
                    bus_set  = chosen_map.get("BUS")  is not None
                    main_set = chosen_map.get("MAIN") is not None

                    if not bus_set and not main_set:
                        # If we have evidence BOTH roles exist (both labels present),
                        # do NOT share a single amps token between BUS and MAIN.
                        has_bus_label  = bool(labels_map.get("BUS"))
                        has_main_label = bool(labels_map.get("MAIN"))

                        if has_bus_label and has_main_label:
                            aff_b, aff_m = _role_affinity(only)
                            if aff_m > aff_b + 0.05:
                                _set_role("MAIN", _as_ranked("MAIN", only))
                                _set_role("BUS", None)
                            else:
                                _set_role("BUS", _as_ranked("BUS", only))
                                _set_role("MAIN", None)
                        else:
                            _set_role("BUS", _as_ranked("BUS", only))
                            _set_role("MAIN", _as_ranked("MAIN", only), allow_share_with="BUS")

                    elif bus_set and not main_set:
                        # If both BUS and MAIN labels exist, do NOT auto-share a single amps token.
                        has_bus_label  = bool(labels_map.get("BUS"))
                        has_main_label = bool(labels_map.get("MAIN"))
                        if has_bus_label and has_main_label:
                            _set_role("MAIN", None)
                        else:
                            _set_role("MAIN", chosen_map["BUS"], allow_share_with="BUS")

                    elif main_set and not bus_set:
                        # Symmetric rule: don't auto-invent BUS from MAIN when both labels exist.
                        has_bus_label  = bool(labels_map.get("BUS"))
                        has_main_label = bool(labels_map.get("MAIN"))
                        if has_bus_label and has_main_label:
                            _set_role("BUS", None)
                        else:
                            _set_role("BUS", chosen_map["MAIN"])

                    else:
                        # both set (likely same token) → do nothing
                        pass

                else:
                    # No explicit mode. If there is *no* MAIN label anywhere,
                    # treat this single amps token as BUS-only – do NOT invent a MAIN.
                    if not has_main_label:
                        _set_role("BUS", _as_ranked("BUS", only))
                        _set_role("MAIN", None)
                    else:
                        # There is a MAIN label somewhere: use affinity to decide role.
                        if aff_m > aff_b + 0.08:
                            _set_role("MAIN", _as_ranked("MAIN", only))
                            # leave BUS as-is (do not invent/override BUS here)
                        else:
                            _set_role("BUS", _as_ranked("BUS", only))
                            # leave MAIN as-is (do not invent/override MAIN here)

        else:
            # No explicit-unit amps → keep ranked picks but still apply mode enforcement
            pass

        # Re-check colon evidence AFTER the unit-first amps reassignment.
        # Colon is useful evidence, but it should not hard-clear the opposite role.
        # OCR can split the same printed field into both a colon-labeled candidate
        # and a bare-number candidate, so colon mismatch alone is not a reliable veto.
        bus_pick = chosen_map.get("BUS")
        main_pick = chosen_map.get("MAIN")

        bus_colon = _is_colon_labeled(bus_pick, "BUS")
        main_colon = _is_colon_labeled(main_pick, "MAIN")

        # Final mode enforcement (last word)
        if (main_mode or "").upper() == "MLO":
            chosen_map["MAIN"] = None

        # ===== Mode-aware BUS/MAIN reconciliation =====
        mode_upper = (main_mode or "").upper()

        def _norm_txt(c):
            return self._normalize_digits(str((c or {}).get("text", "")).upper())

        def _is_mains_rating(c):
            t = _norm_txt(c)
            return bool(re.search(r'\bMAINS?\s*RATING\b|\bMAIN\s*RATING\b', t))

        def _is_main_device_rating(c):
            t = _norm_txt(c)
            return bool(
                re.search(
                    r'\bMCB\b|'
                    r'\bM\W*C\W*B\b|'
                    r'\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b|'
                    r'\bMAIN\s*DEVICE\b',
                    t
                )
            )

        def _looks_amp_like(c):
            if not c:
                return False
            t = _norm_txt(c)
            if self._looks_like_aic_or_ka_text(t) or self._looks_like_non_amp_context_text(t):
                return False
            if self._snap_voltage_text(self._normalize_voltage_text(t)) is not None:
                return False
            return bool(
                re.search(r"\b([1-9]\d{1,3})\s*(A|AMP|AMPS|MAP)\b", t)
                or re.search(r'(?<!\d)([6-9]\d|[1-9]\d{2,3})(?!\d)', t)
            )

        bus_ranked = list(ranked_map.get("BUS") or [])
        main_ranked = list(ranked_map.get("MAIN") or [])

        if mode_upper == "MLO":
            # No main breaker on MLO panels
            _set_role("MAIN", None)

            # Prefer explicit Mains Rating as BUS
            mlo_bus = None
            for c in bus_ranked + main_ranked:
                if _is_mains_rating(c) and _looks_amp_like(c):
                    mlo_bus = _as_ranked("BUS", c)
                    break

            if mlo_bus is not None:
                _set_role("BUS", mlo_bus)
            else:
                # fallback: best amp-looking candidate becomes BUS
                for c in bus_ranked + main_ranked:
                    if _looks_amp_like(c):
                        _set_role("BUS", _as_ranked("BUS", c))
                        break

        elif mode_upper == "MCB":
            # Prefer MCB/Main Breaker rating for MAIN
            mcb_main = None
            for c in main_ranked + bus_ranked:
                if _is_main_device_rating(c) and _looks_amp_like(c):
                    mcb_main = _as_ranked("MAIN", c)
                    break

            if mcb_main is not None:
                _set_role("MAIN", mcb_main)

            # Prefer Mains Rating for BUS
            mcb_bus = None
            for c in bus_ranked + main_ranked:
                if _is_mains_rating(c) and _looks_amp_like(c):
                    # don't reuse the MAIN token if a distinct mains rating exists
                    if not chosen_map.get("MAIN") or _cand_key(c) != _cand_key(chosen_map["MAIN"]):
                        mcb_bus = _as_ranked("BUS", c)
                        break

            if mcb_bus is not None:
                _set_role("BUS", mcb_bus)

            # If BUS still missing, allow fallback from MAIN only when there's just one real amp source
            if not chosen_map.get("BUS") and chosen_map.get("MAIN"):
                distinct_amp_keys = []
                seen_amp = set()
                for c in bus_ranked + main_ranked:
                    if not _looks_amp_like(c):
                        continue
                    k = _cand_key(c)
                    if k not in seen_amp:
                        seen_amp.add(k)
                        distinct_amp_keys.append(k)

                if len(distinct_amp_keys) == 1:
                    _set_role("BUS", chosen_map["MAIN"], allow_share_with="MAIN")

        # ===== NAME fallback: only if we have a strong designation/name label present =====
        if not chosen_map.get("NAME"):
            has_name_label = bool(labels_map.get("NAME"))

            if has_name_label:
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
                    u = self._normalize_digits(str(t).upper())

                    # reject AIC/kA-looking or non-rating context text before extracting bare numbers
                    if self._looks_like_aic_or_ka_text(u) or self._looks_like_non_amp_context_text(u):
                        return None

                    # reject voltage-looking text
                    uN = self._normalize_voltage_text(u)
                    if (
                        self._snap_voltage_text(uN) is not None
                        or re.search(r'(?<!\d)([1-6]\d{2,3})\s*[YV]?\s*/\s*([1-6]?\d{2,3})(?!\d)', uN)
                        or re.search(r'\b(VOLT|VOLTS|VOLTAGE|VAC|VDC)\b', u)
                        or re.search(r'(?<!\d)[1-6]\d{2,3}\s*V\b', uN)
                    ):
                        return None

                    m = re.search(r"\b([1-9]\d{1,3})\s*(A|AMP|AMPS|MAP)\b", u)
                    if m:
                        return int(m.group(1))

                    m2 = re.search(r'(?<!\d)([6-9]\d|[1-9]\d{2,3})(?!\d)', u)
                    if m2:
                        return int(m2.group(1))

                    return None

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

                    # ---- Guard: don't treat AIC/kA-looking tokens like "65k" as BUS ----
                    # ---- Guard: don't treat location/source text like "Location: ELECTRICAL 123" as BUS ----
                    txtD = self._normalize_digits(up)
                    if self._looks_like_aic_or_ka_text(txtD) or self._looks_like_non_amp_context_text(txtD):
                        continue

                    # ---- Guard: don't treat voltage-looking tokens like "120/208 Wye" as BUS ----
                    txtN = self._normalize_voltage_text(txtD)

                    pair = re.search(
                        r'\b([1-6]\d{2,3})\s*[YV]?[\/]\s*([1-6]?\d{2,3})\s*V?\b',
                        txtN,
                    )
                    has_volty_ctx = bool(
                        re.search(r'\b(WYE|DELTA|PH|PHASE|Ø|VOLT|VOLTS|VAC|VDC|V)\b', txtN)
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
                    m = re.search(r"(?<!\d)([1-9]\d{1,3})(?!\d)\s*(?:A|AMP|AMPS)\b", up)
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

        # ===== Scrub "border line sucked into number" errors (e.g., 1150A -> 150A) =====
        # Run ONLY on chosen BUS/MAIN after the full selection pipeline is complete.
        if chosen_map.get("BUS"):
            chosen_map["BUS"] = self._scrub_leading_line_digit_for_role(
                "BUS",
                chosen_map.get("BUS"),
                ranked_map,
                prep_full,
                plausible_min=60,
                plausible_max=1200,
                require_support=True,
            )

        if chosen_map.get("MAIN"):
            chosen_map["MAIN"] = self._scrub_leading_line_digit_for_role(
                "MAIN",
                chosen_map.get("MAIN"),
                ranked_map,
                prep_full,
                plausible_min=30,   # allow smaller mains if printed explicitly
                plausible_max=4000, # the code allows up to 4000A in some contexts
                require_support=True,
            )

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

        # BUS (accept with or without unit, but never from voltage-looking or AIC/kA-looking text)
        bus_amp = None
        if chosen_map["BUS"]:
            tU = self._normalize_digits(chosen_map["BUS"]["text"].upper())
            tN = self._normalize_voltage_text(tU)

            looks_like_voltage = (
                self._snap_voltage_text(tN) is not None
                or bool(re.search(r'(?<!\d)([1-6]\d{2,3})\s*[YV]?\s*/\s*([1-6]?\d{2,3})(?!\d)', tN))
                or bool(re.search(r'\b(VOLT|VOLTS|VOLTAGE|VAC|VDC)\b', tU))
                or bool(re.search(r'(?<!\d)[1-6]\d{2,3}\s*V\b', tN))
            )

            looks_like_aic_or_ka = self._looks_like_aic_or_ka_text(tU)
            looks_like_non_amp_context = self._looks_like_non_amp_context_text(tU)

            if not looks_like_voltage and not looks_like_aic_or_ka and not looks_like_non_amp_context:
                m = re.search(r"\b([1-9]\d{1,3})\s*(?:A|AMP|AMPS)\b", tU)
                if m:
                    bus_amp = _to_int(m.group(1))
                else:
                    m2 = re.search(r'(?<!\d)([1-9]\d{1,3})(?!\d)', tU)
                    if m2:
                        bus_amp = _to_int(m2.group(1))

        # MAIN
        main_amp = None
        self.last_main_type = None
        if chosen_map["MAIN"]:
            txtU0 = chosen_map["MAIN"]["text"].upper()
            txtU  = self._normalize_digits(txtU0)
            txtN  = self._normalize_voltage_text(txtU)

            looks_like_voltage = (
                self._snap_voltage_text(txtN) is not None
                or bool(re.search(r'(?<!\d)([1-6]\d{2,3})\s*[YV]?\s*/\s*([1-6]?\d{2,3})(?!\d)', txtN))
                or bool(re.search(r'\b(VOLT|VOLTS|VOLTAGE|VAC|VDC)\b', txtU))
                or bool(re.search(r'(?<!\d)[1-6]\d{2,3}\s*V\b', txtN))
            )

            looks_like_aic_or_ka = self._looks_like_aic_or_ka_text(txtU)
            looks_like_non_amp_context = self._looks_like_non_amp_context_text(txtU)

            if not looks_like_voltage and not looks_like_aic_or_ka and not looks_like_non_amp_context:
                m = re.search(r"\b([1-9]\d{1,3})\s*(?:A|AMP|AMPS|MAP)\b", txtU)
                if m:
                    main_amp = _to_int(m.group(1))
                else:
                    m2 = re.search(r'(?<!\d)([1-9]\d{1,3})(?!\d)', txtU)
                    if m2:
                        main_amp = _to_int(m2.group(1))

            if re.search(r"\b(MLO|MAIN\s*LUGS?)\b", txtU):
                self.last_main_type = "MLO"
            elif re.search(
                r"\bMCB\b|"
                r"\bM\W*C\W*B\b|"
                r"\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b",
                txtU
            ):
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

        mounting_style = None
        mounting_enclosure = None
        if chosen_map.get("MOUNTING"):
            mounting_style, mounting_enclosure = self._normalize_mounting_output(
                chosen_map["MOUNTING"]["text"]
            )

        enclosure_val = None
        if mounting_enclosure is None and chosen_map.get("ENCLOSURE"):
            enclosure_val = self._normalize_enclosure_output(
                chosen_map["ENCLOSURE"]["text"]
            )

        # Prefer the explicit tag on the CHOSEN MAIN value (if present).
        # If that’s absent, fall back to the global scan.
        mode = ((self.last_main_type or main_mode or "")).upper()

        # IMPORTANT: main_amp_out must be assigned in ALL paths
        main_amp_out = None

        if mode == "MLO":
            main_amp_out = None  # never report a main breaker when MLO

        elif mode == "MCB":
            # Only fall back to BUS when there is NO explicit MAIN label anywhere.
            # If MAIN is labeled but we couldn't read a distinct main rating, leave it unknown.
            if main_amp is not None:
                main_amp_out = main_amp
            else:
                main_amp_out = (bus_amp if not labels_map.get("MAIN") else None)

        else:
            # Unknown/unspecified mode:
            # If we read a main rating, use it; otherwise leave it unknown.
            # (Do NOT auto-copy BUS into MAIN here.)
            main_amp_out = main_amp if main_amp is not None else None
        
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
            "intRating": aic_i,
            "detected_breakers": [],
        }

        if mode != "MLO":
            attrs["mainBreakerAmperage"] = main_i

        if mounting_style is not None:
            attrs["trimStyle"] = mounting_style

        if mounting_enclosure is not None:
            attrs["enclosure"] = mounting_enclosure
        elif enclosure_val is not None:
            attrs["enclosure"] = enclosure_val

        result = {
            "type": "panelboard",
            "name": name,
            "attrs": attrs,
        }

        if special_header_type:
            result["specialHeaderType"] = special_header_type
            result["panelNote"] = special_header_type.get("note")

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
        if chosen_map.get("MOUNTING"):
            winning["mounting"] = _rect4(chosen_map["MOUNTING"])
        if chosen_map.get("ENCLOSURE"):
            winning["enclosure"] = _rect4(chosen_map["ENCLOSURE"])
        if chosen_map.get("AIC"):
            winning["aic"] = _rect4(chosen_map["AIC"])

        # coords are in the coordinate system of prep_full (after _prep_for_ocr resize)
        result["winningBoxes"] = winning
        result["boxImageShape"] = {"w": int(W), "h": int(H)}

        # ===== Debug overlay =====
        if self.debug:
            # ===== FULL TOKEN TRACE (labels, value-shapes, and candidate ranks) =====
            print("\n==== PANEL HEADER RAW TRACE ====")

            # Map role
            _role_human = {
                "NAME": "name",
                "VOLTAGE": "volts",
                "BUS": "bus amps",
                "MAIN": "main amps",
                "MOUNTING": "mounting",
                "ENCLOSURE": "enclosure",
                "AIC": "AIC",
                "WRONG": "not of interest",
            }

            # Build a lookup of ranks for quick “did this token become a candidate?”
            def _key(c):
                return (int(round(c["x1"])), int(round(c["y1"])),
                        int(round(c["x2"])), int(round(c["y2"])), (c.get("text","")).strip())
            role_idx = {r:{} for r in ("NAME","VOLTAGE","BUS","MAIN","MOUNTING","ENCLOSURE","AIC")}
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
                        for role in ("NAME","VOLTAGE","BUS","MAIN","MOUNTING","ENCLOSURE","AIC","WRONG")}

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

                # Mounting
                if re.search(r"\b(SURFACE|FLUSH|RECESSED)\b", up) or re.search(r"\bNEMA\s*1\b|\bNEMA\s*3R\b|\bN1\b|\bN3R\b", up):
                    value_roles.append("MOUNTING")

                # Enclosure
                if re.search(r"\bNEMA\s*1\b|\bNEMA\s*3R\b|\bN1\b|\bN3R\b|\bTYPE\s*1\b|\bTYPE\s*3R\b", up):
                    value_roles.append("ENCLOSURE")

                # Name-ish: short alphanum
                if re.fullmatch(r"[A-Z0-9][A-Z0-9._\-\/]{0,12}", up):
                    value_roles.append("NAME")

                # Candidate ranks (if any) for each role
                ranks_bits = []
                for role in ("NAME","VOLTAGE","BUS","MAIN","MOUNTING","ENCLOSURE","AIC"):
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
            for role in ("NAME","VOLTAGE","BUS","MAIN","MOUNTING","ENCLOSURE","AIC"):
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
            for role in ("NAME","VOLTAGE","BUS","MAIN","MOUNTING","ENCLOSURE","AIC"):
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
        has_mcb = bool(
            re.search(
                r"\b("
                r"MCB|"
                r"M\W*C\W*B|"
                r"MAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)"
                r")\b",
                txt,
            )
        )

        # MLO always wins if present anywhere
        if has_mlo:
            return "MLO"
        if has_mcb:
            return "MCB"
        return None

    def _normalize_name_output(self, s: str) -> str:
        import re
        u = (s or "").strip()
        u = self._strip_quotes(u)
        u = re.sub(r'^(?:PANEL(?:BOARD)?\b\s*:?)\s*', '', u, flags=re.I)  # drop ONLY the leading label
        u = re.sub(r'\s{2,}', ' ', u).strip()
        return u

    def _normalize_mounting_output(self, s: str) -> tuple[Optional[str], Optional[str]]:
        if not s:
            return None, None

        u = self._normalize_digits(str(s).upper())
        u = re.sub(r"\s+", " ", u).strip()

        trim_style = None
        enclosure = None

        if re.search(r"\b(FLUSH|RECESSED)\b", u):
            trim_style = "FLUSH"
        elif re.search(r"\bSURFACE\b", u):
            trim_style = "SURFACE"

        if re.search(r"\bNEMA\s*3R\b|\bN3R\b|\b(OUTDOOR|WEATHERPROOF)\b", u):
            enclosure = "Nema3R"
        elif re.search(r"\bNEMA\s*1\b|\bN1\b", u):
            enclosure = "Nema1"

        # Nema3R makes flush/surface irrelevant for downstream panel use
        if enclosure == "Nema3R":
            trim_style = None

        return trim_style, enclosure
    
    def _normalize_enclosure_output(self, s: str) -> Optional[str]:
        if not s:
            return None

        u = self._normalize_digits(str(s).upper())
        u = re.sub(r"\s+", " ", u).strip()

        if re.search(r"\bNEMA\s*3R\b|\bN3R\b|\bTYPE\s*3R\b|\b(OUTDOOR|WEATHERPROOF)\b", u):
            return "Nema3R"
        if re.search(r"\bNEMA\s*1\b|\bN1\b|\bTYPE\s*1\b", u):
            return "Nema1"

        return None
    
    def _normalize_special_header_text(self, s: str) -> str:
        import re

        u = str(s or "").upper()

        # basic cleanup
        u = u.replace("&", " AND ")
        u = u.replace("/", " ")
        u = u.replace("-", " ")
        u = u.replace("_", " ")

        # collapse whitespace
        u = re.sub(r"[^A-Z0-9\s]", " ", u)
        u = re.sub(r"\s+", " ", u).strip()

        return u


    def _special_header_word_family(
        self,
        word: str
    ) -> str:
        """
        Normalize common singular, plural, and abbreviated
        schedule-title words into stable family tokens.
        """
        w = str(
            word or ""
        ).upper().strip()

        if not w:
            return ""

        if w in {
            "SWITCHBOARD",
            "SWITCHBOARDS",
            "SWBD",
            "SWBDS",
        }:
            return "SWITCHBOARD"

        if w in {
            "LIGHT",
            "LIGHTS",
            "LIGHTING",
        }:
            return "LIGHT"

        if w in {
            "FIXTURE",
            "FIXTURES",
        }:
            return "FIXTURE"

        if w in {
            "SCHEDULE",
            "SCHEDULES",
        }:
            return "SCHEDULE"

        if w in {
            "CONDUIT",
            "CONDUITS",
        }:
            return "CONDUIT"

        if w in {
            "EQUIPMENT",
            "EQUIPMENTS",
            "EQUIP",
            "EQPT",
        }:
            return "EQUIPMENT"

        if w in {
            "MECHANICAL",
            "MECH",
        }:
            return "MECHANICAL"

        if w in {
            "HVAC",
        }:
            return "HVAC"

        if w in {
            "DISCONNECT",
            "DISCONNECTS",
            "DISC",
            "DISCS",
        }:
            return "DISCONNECT"

        if w in {
            "SAFETY",
        }:
            return "SAFETY"

        if w in {
            "SWITCH",
            "SWITCHES",
        }:
            return "SWITCH"

        if w in {
            "LABOR",
            "LABOUR",
        }:
            return "LABOR"

        if w in {
            "MOTOR",
            "MOTORS",
        }:
            return "MOTOR"

        if w in {
            "TRANSFORMER",
            "TRANSFORMERS",
            "XFMR",
            "XFMRS",
        }:
            return "TRANSFORMER"

        if w in {
            "GENERATOR",
            "GENERATORS",
            "GEN",
            "GENS",
        }:
            return "GENERATOR"

        if w in {
            "WIREWAY",
            "WIREWAYS",
            "WWA",
            "WWB",
            "WWC",
        }:
            return "WIREWAY"

        if w in {
            "INVERTER",
            "INVERTERS",
            "INV",
        }:
            return "INVERTER"

        if w in {
            "INTERIOR",
            "EXTERIOR",
        }:
            return w

        return w


    def _special_header_token_set(self, s: str) -> set[str]:
        txt = self._normalize_special_header_text(s)
        toks = txt.split()
        return {self._special_header_word_family(tok) for tok in toks if tok.strip()}

    def _detect_special_header_type(self, items: list[dict], lines) -> Optional[dict]:
        """
        Detect obvious non-panel / schedule header families using normalized concept matching.

        Examples matched as the same family:
        - SWITCHBOARD / SWITCHBOARDS / SWBD
        - LIGHT / LIGHTS / LIGHTING
        - FIXTURE / FIXTURES
        - SCHEDULE / SCHEDULES
        - INVERTER / INVERTERS / INV

        Guard against relational/source text like:
        - FED FROM: MAIN SWBD
        - FED BY: SWBD-A
        - SOURCE: INVERTER INV-1
        """
        import re
 
        FAMILY_RULES = [
            {
                "kind": "switchboard",
                "note": "Switchboard detected",
                "required_any": [
                    {"SWITCHBOARD"},
                ],
            },

            {
                "kind": "lighting_schedule",
                "note": "Lighting schedule detected",
                "required_any": [
                    {"LIGHT", "SCHEDULE"},
                    {"FIXTURE", "SCHEDULE"},
                    {
                        "LIGHT",
                        "FIXTURE",
                        "SCHEDULE",
                    },
                ],
            },

            {
                "kind": "conduit_schedule",
                "note": "Conduit schedule detected",
                "required_any": [
                    {"CONDUIT", "SCHEDULE"},
                ],
            },

            # More-specific schedule rules must stay
            # above the generic equipment rule.
            {
                "kind": "mechanical_schedule",
                "note": "Mechanical schedule detected",
                "required_any": [
                    {
                        "MECHANICAL",
                        "SCHEDULE",
                    },
                    {
                        "HVAC",
                        "SCHEDULE",
                    },
                    {
                        "MECHANICAL",
                        "EQUIPMENT",
                        "SCHEDULE",
                    },
                    {
                        "HVAC",
                        "EQUIPMENT",
                        "SCHEDULE",
                    },
                ],
            },

            {
                "kind": "disconnect_schedule",
                "note": "Disconnect schedule detected",
                "required_any": [
                    {
                        "DISCONNECT",
                        "SCHEDULE",
                    },
                    {
                        "SAFETY",
                        "SWITCH",
                        "SCHEDULE",
                    },
                ],
            },

            {
                "kind": "motor_schedule",
                "note": "Motor schedule detected",
                "required_any": [
                    {"MOTOR", "SCHEDULE"},
                    {
                        "MOTOR",
                        "EQUIPMENT",
                        "SCHEDULE",
                    },
                ],
            },

            {
                "kind": "transformer_schedule",
                "note": "Transformer schedule detected",
                "required_any": [
                    {
                        "TRANSFORMER",
                        "SCHEDULE",
                    },
                    {
                        "TRANSFORMER",
                        "EQUIPMENT",
                        "SCHEDULE",
                    },
                ],
            },

            {
                "kind": "generator_schedule",
                "note": "Generator schedule detected",
                "required_any": [
                    {
                        "GENERATOR",
                        "SCHEDULE",
                    },
                    {
                        "GENERATOR",
                        "EQUIPMENT",
                        "SCHEDULE",
                    },
                ],
            },

            {
                "kind": "labor_schedule",
                "note": "Labor schedule detected",
                "required_any": [
                    {"LABOR", "SCHEDULE"},
                ],
            },

            # Keep this after mechanical, motor,
            # transformer, and generator schedules.
            {
                "kind": "equipment_schedule",
                "note": "Equipment schedule detected",
                "required_any": [
                    {
                        "EQUIPMENT",
                        "SCHEDULE",
                    },
                ],
            },

            {
                "kind": "wireway",
                "note": "Wireway detected",
                "required_any": [
                    {"WIREWAY"},
                ],
            },

            {
                "kind": "inverter",
                "note": "Inverter detected",
                "required_any": [
                    {"INVERTER"},
                    {
                        "LIGHT",
                        "INVERTER",
                    },
                ],
            },
        ]

        NEGATIVE_CONTEXT_PATTERNS = [
            r"\bFED\s+FROM\b",
            r"\bFED\s+BY\b",
            r"\bSUPPLY\s+FROM\b",
            r"\bSUPPLY\s+BY\b",
            r"\bSUPPLIED\s+FROM\b",
            r"\bSUPPLIED\s+BY\b",
            r"\bSOURCE\b",
            r"\bCONNECTED\s+TO\b",
            r"\bMAIN\s+TYPE\b",
            r"\bMAINS\s+TYPE\b",
            r"\bMAINS?\s+RATING\b",
        ]

        def _token_height(it: dict) -> int:
            try:
                return max(1, int(it["y2"] - it["y1"]))
            except Exception:
                return 1

        def _line_text(ln) -> str:
            return " ".join((tok or "").strip() for (_, tok, _) in ln.get("tokens", []))

        def _is_name_label_line(cand: dict) -> bool:
            """
            True when a joined line appears to be describing the name/title/designation
            of the schedule/object, not general header metadata.

            Examples that should count:
            NAME: SWITCHBOARD XYZ
            PANEL NAME: INVERTER A
            PANEL DESIGNATION: WW-A
            SWITCHBOARD: 14A
            WIREWAY: WW-A
            INVERTER: INV-1
            """
            text = self._normalize_special_header_text(cand.get("text", ""))

            if not text:
                return False

            # Generic name/title labels
            if re.search(r"\b(NAME|PANEL\s+NAME|PANEL\s+DESIGNATION|DESIGNATION|TITLE)\b", text):
                return True

            # Special family used as the label itself, e.g. "SWITCHBOARD: 14A"
            # Colon is removed by normalize, so this sees "SWITCHBOARD 14A".
            # That is fine because this function is only used after the family rule matched.
            if re.search(r"\b(SWITCHBOARD|SWBD|WIREWAY|WWA|WWB|WWC|INVERTER|INV)\b", text):
                # Reject known metadata/source phrases
                if re.search(r"\b(SUPPLY\s+FROM|SUPPLIED\s+FROM|FED\s+FROM|FED\s+BY|SOURCE|MAINS?\s+TYPE|MAIN\s+TYPE|CONNECTED\s+TO)\b", text):
                    return False
                return True

            # Known schedule families are naturally
            # title phrases, including multi-word titles
            # such as MECHANICAL EQUIPMENT SCHEDULE.
            token_set = set(
                cand.get("token_set") or []
            )

            if not token_set:
                token_set = (
                    self._special_header_token_set(
                        text
                    )
                )

            schedule_title_families = {
                "LIGHT",
                "FIXTURE",
                "CONDUIT",
                "MECHANICAL",
                "HVAC",
                "EQUIPMENT",
                "DISCONNECT",
                "SAFETY",
                "MOTOR",
                "TRANSFORMER",
                "GENERATOR",
                "LABOR",
            }

            if (
                "SCHEDULE" in token_set
                and token_set.intersection(
                    schedule_title_families
                )
            ):
                return True

            return False

        def _has_negative_context(text: str, kind: str, source_type: str = "line", source_line=None, source_item=None) -> bool:
            """
            Reject special-header hits when the text is relational instead of title-like.

            Important case:
            FED FROM: .... MAIN SWBD
            where the matched token may only be "MAIN SWBD" and not contain the
            relational phrase itself.
            """
            up = self._normalize_special_header_text(text)

            # Explicit schedule-title families are not
            # relational source references.
            if str(
                kind or ""
            ).strip().lower().endswith(
                "_schedule"
            ):
                return False

            # 1) direct text match
            if any(re.search(rx, up, flags=re.I) for rx in NEGATIVE_CONTEXT_PATTERNS):
                return True

            # 2) if this came from a whole joined line, inspect the whole line text
            if source_type == "line" and source_line is not None:
                full_line = self._normalize_special_header_text(_line_text(source_line))
                if any(re.search(rx, full_line, flags=re.I) for rx in NEGATIVE_CONTEXT_PATTERNS):
                    return True

            # 3) if this came from a strong single token, inspect nearby same-row line context
            if source_type == "token" and source_item is not None and source_line is not None:
                toks = source_line.get("tokens", []) or []
                if toks:
                    item_x1 = float(source_item.get("x1", 0.0))
                    item_yc = float(source_item.get("yc", 0.0))
                    item_h = max(1.0, float(source_item.get("y2", 0.0)) - float(source_item.get("y1", 0.0)))

                    left_context_tokens = []
                    for (r, tok, conf) in toks:
                        rx1, ry1, rx2, ry2 = r
                        tok_yc = 0.5 * (ry1 + ry2)

                        # same-row-ish tokens to the LEFT of the matched token
                        if abs(tok_yc - item_yc) <= max(18.0, item_h * 0.8) and rx2 <= item_x1:
                            left_context_tokens.append((rx1, tok))

                    if left_context_tokens:
                        left_context_tokens.sort(key=lambda z: z[0])
                        left_text = self._normalize_special_header_text(" ".join(t for _, t in left_context_tokens))

                        if any(re.search(rx, left_text, flags=re.I) for rx in NEGATIVE_CONTEXT_PATTERNS):
                            return True

            return False

        heights = [_token_height(it) for it in (items or [])] or [1]
        med_h = float(np.median(heights)) if hasattr(np, "median") else (sum(heights) / len(heights))
        top_y = min((float(it.get("yc", 0.0)) for it in (items or [])), default=0.0)

        def _looks_name_like(it: dict) -> bool:
            txt = self._normalize_special_header_text(it.get("text", ""))
            if not txt:
                return False

            h = _token_height(it)
            yc = float(it.get("yc", 0.0))

            explicit_name_context = bool(
                re.search(r"\b(NAME|PANEL|PANELBOARD|BOARD|DESIGNATION)\s*:", txt)
            )
            size_bias = h >= (med_h * 1.15)
            top_bias = yc <= (top_y + med_h * 4.0)

            return explicit_name_context or size_bias or top_bias

        def _find_source_line_for_item(it: dict):
            if not lines:
                return None

            item_yc = float(it.get("yc", 0.0))
            best_line = None
            best_dy = None

            for ln in lines[:8]:
                rect = ln.get("rect")
                if not rect:
                    continue
                ly1, ly2 = rect[1], rect[3]
                line_yc = 0.5 * (ly1 + ly2)
                dy = abs(item_yc - line_yc)
                if best_dy is None or dy < best_dy:
                    best_dy = dy
                    best_line = ln

            return best_line

        candidates = []

        # joined header lines
        for ln in (lines or [])[:8]:
            joined = _line_text(ln).strip()
            if joined:
                candidates.append({
                    "text": joined,
                    "token_set": self._special_header_token_set(joined),
                    "source_type": "line",
                    "source_line": ln,
                    "source_item": None,
                })

        # strong individual tokens
        for it in (items or []):
            txt = str(it.get("text", "") or "").strip()
            if txt and _looks_name_like(it):
                src_line = _find_source_line_for_item(it)
                candidates.append({
                    "text": txt,
                    "token_set": self._special_header_token_set(txt),
                    "source_type": "token",
                    "source_line": src_line,
                    "source_item": it,
                })

        # Prefer fuller candidates first
        candidates.sort(key=lambda x: (-len(x["token_set"]), -len(x["text"])))

        for cand in candidates:
            original_text = cand["text"]
            token_set = cand["token_set"]

            if not token_set:
                continue

            for rule in FAMILY_RULES:
                for req in rule["required_any"]:
                    if req.issubset(token_set):
                        if cand["source_type"] != "token" and not _is_name_label_line(cand):
                            continue

                        if _has_negative_context(
                            original_text,
                            rule["kind"],
                            source_type=cand["source_type"],
                            source_line=cand["source_line"],
                            source_item=cand["source_item"],
                        ):
                            continue

                        return {
                            "kind": rule["kind"],
                            "note": rule["note"],
                            "matched_text": original_text,
                            "matched_tokens": sorted(token_set),
                        }

        return None

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

        def _mk_val_by_rects(rects, texts, confs, conf_hint=None, shape=0.98, ctx=0.38, from_colon_label=False):
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
                "fromColonLabel": bool(from_colon_label),
            }

        def _has_letter(s: str) -> bool:
            return any(ch.isalpha() for ch in (s or ""))

        def _is_id_shaped_name(s: str) -> bool:
            """
            Allow panel names like:
              LP-1
              EP-2
              200A/NIP-2
              2OOA/NIP-2
            even if they contain something amp-looking, as long as they are clearly
            mixed name/id text rather than a pure electrical value.
            """
            if not s:
                return False
            up = str(s).upper().strip()

            # must contain at least one letter somewhere
            if not _has_letter(up):
                return False

            # slash/hyphen/dot IDs are common panel names
            if re.search(r"[\/\-.]", up):
                return True

            # compact mixed alphanum IDs
            if re.fullmatch(r"[A-Z0-9][A-Z0-9._\-/]{1,24}", up):
                return True

            return False
        
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
            """
            base = re.sub(r'[^A-Z]', '', (up or "").upper())
            if not base:
                return False
            bad_words = ["SCHEDULE", "DESIGNATION", "DESIGN", "PANELBOARD", "PANEL"]
            for w in bad_words:
                ratio = difflib.SequenceMatcher(a=base, b=w).ratio()
                if ratio >= 0.74:  # fairly forgiving; 1–2 OCR mistakes still match
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
                        if right_clean and _has_letter(right_clean):
                            if _looks_like_header_word(right_clean.upper()):
                                continue
                            if VOLT_RE.search(right_clean):
                                continue

                            # Allow amp-looking text ONLY when it is clearly an ID-shaped panel name
                            # like "200A/NIP-2", not a plain electrical value.
                            if AMPS_RE.search(right_clean) and not _is_id_shaped_name(right_clean):
                                continue

                            return _mk_val_by_rects(
                                [r], [right_clean], [c],
                                conf_hint=float(c or 0.75),
                                shape=1.00,
                                ctx=0.85,
                                from_colon_label=True,
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
                joined_name = " ".join(picked_texts).strip()

                # If the joined colon-name contains amp-like text, still allow it when it is
                # clearly ID-shaped, e.g. "200A/NIP-2".
                if VOLT_RE.search(joined_name):
                    pass
                else:
                    if not AMPS_RE.search(joined_name) or _is_id_shaped_name(joined_name):
                        return _mk_val_by_rects(
                            picked_rects,
                            picked_texts,
                            picked_confs,
                            shape=1.00,
                            ctx=0.85,
                            from_colon_label=True,
                        )

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
                        shape=0.96,
                        ctx=0.35,
                        from_colon_label=True,
                    )

        # Handles OCR where the panel label *and* the name are fused into a single token.
        # Examples:
        #   "PANEL: LP-1"
        #   "NEW PANEL C"
        #   "EXISTING PNL-2 NORMAL"
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

                tail_up = tail.upper()
                if re.search(r"\b(WIRING|SCHEDULE|NOTES|DESIGNATION|INTERRUPTING|RATING)\b", tail_up):
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

                # Build a synthetic NAME candidate using the original bbox
                return _mk_val_by_rects(
                    [r], [first], [c],
                    conf_hint=float(c or 0.8),
                    shape=1.00,
                    ctx=0.85
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

    def _extract_value_after_colon_label(self, lines, role: str) -> Optional[dict]:
        """
        Strong colon-led extractor for header roles.
        If we see LABEL: value, or LABEL : value, return the value to the right
        as a synthetic high-confidence candidate.

        Used for roles like VOLTAGE / BUS / MAIN / AIC / MOUNTING / ENCLOSURE.
        """
        import re

        ROLE_LABEL_WORDS = {
            "VOLTAGE": {"VOLTAGE", "VOLT", "VOLTS", "VAC", "VDC"},
            "BUS": {"BUS", "BUSS", "AMPERES", "AMPERE", "AMPS", "AMP"},
            "MAIN": {"MAIN", "MAINS", "MCB", "MLO", "AMPERES", "AMPERE", "AMPS", "AMP"},
            "AIC": {"AIC", "KAIC", "SCCR", "INTERRUPTING", "RATING"},
            "MOUNTING": {"MOUNTING", "MOUNT"},
            "ENCLOSURE": {"ENCLOSURE", "ENCL", "NEMA", "TYPE"},
        }

        LABEL_WORDS = ROLE_LABEL_WORDS.get(role, set())
        if not LABEL_WORDS or not lines:
            return None

        top = lines[: min(6, len(lines))]

        def _mk_val_by_rects(rects, texts, confs, conf_hint=None, shape=1.00, ctx=0.95):
            r1, r2 = rects[0], rects[-1]
            x1, y1 = min(r1[0], r2[0]), min(r1[1], r2[1])
            x2, y2 = max(r1[2], r2[2]), max(r1[3], r2[3])
            text = re.sub(r"\s+", " ", " ".join((t or "").strip() for t in texts)).strip()
            conf = float(conf_hint if conf_hint is not None else (sum(confs) / max(1, len(confs))))
            return {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "xc": 0.5 * (x1 + x2), "yc": 0.5 * (y1 + y2),
                "conf": conf,
                "text": text,
                "shape": shape,
                "ctx": ctx,
                "fromColonLabel": True,
            }

        def _is_good_for_role(role: str, s: str) -> bool:
            if not s:
                return False

            # MAIN and AIC may legitimately be marked N/A.
            if role in ("MAIN", "AIC") and self._is_na_placeholder(s):
                return False

            up = self._normalize_digits(str(s).upper()).strip()
            upN = self._normalize_voltage_text(up)

            if role == "VOLTAGE":
                return self._snap_voltage_text(upN) is not None

            if role == "BUS":
                if re.search(
                    r"\bMCB\b|"
                    r"\bM\W*C\W*B\b|"
                    r"\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b|"
                    r"\bMAIN\s*DEVICE\b",
                    up
                ):
                    return False

                if re.search(r"\b([1-9]\d{1,3})\s*(A|AMPS?)\b", up):
                    return True

                return bool(re.search(r'(?<!\d)([6-9]\d|[1-9]\d{2,3})(?!\d)', up))

            if role == "MAIN":
                # MAIN value candidates should be amperage-bearing, not bare type words.
                if re.search(r"\b([1-9]\d{1,3})\s*(A|AMP|AMPS|MAP)\b", up):
                    return True

                # allow explicit main-device phrases only when they also carry a number
                if re.search(
                    r"\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b|"
                    r"\bMAIN\s*(?:DEVICE|LUGS?)\b",
                    up
                ) and re.search(
                    r'(?<!\d)([6-9]\d|[1-9]\d{2,3})(?!\d)',
                    up
                ):
                    return True

                return False

            if role == "AIC":
                t_nos = re.sub(r"\s+", "", up)
                if re.search(r'(?<!\d)(\d{2,3})(KAMP|KAIC|AIC|KA|K)\b', t_nos):
                    return True
                if re.search(r"\b(\d{2,3}[,]?\d{3})\s*(A|KA)?\b", up):
                    return True
                return False

            if role == "MOUNTING":
                return bool(re.search(r"\b(SURFACE|FLUSH|RECESSED|OUTDOOR|WEATHERPROOF)\b", up))

            if role == "ENCLOSURE":
                return bool(re.search(
                    r"\bNEMA\s*1\b|\bNEMA\s*3R\b|\bN1\b|\bN3R\b|\bTYPE\s*1\b|\bTYPE\s*3R\b|\b(OUTDOOR|WEATHERPROOF)\b",
                    up
                ))

            return False

        def _looks_like_new_label_start(tok_up: str) -> bool:
            if not tok_up:
                return False

            # generic stop if another header-like label starts
            generic = {
                "PANEL", "PANELBOARD", "BOARD",
                "VOLTAGE", "VOLT", "VOLTS",
                "BUS", "BUSS",
                "MAIN", "MAINS", "MCB", "MLO",
                "AIC", "KAIC", "SCCR", "INTERRUPTING", "RATING",
                "MOUNTING", "MOUNT",
                "ENCLOSURE", "ENCL", "NEMA", "TYPE",
                "LOCATION", "SUPPLY", "FROM",
                "SPD",
            }
            return tok_up.strip(":") in generic

        # ----- PASS 0: fused single-token "VOLTAGE: 208Y/120V" -----
        for ln in top:
            for (r, t, c) in ln["tokens"]:
                raw = (t or "").strip()
                if ":" not in raw:
                    continue

                left, right = raw.split(":", 1)
                left_words = set(re.findall(r"[A-Z0-9]+", left.upper()))
                if not (left_words & LABEL_WORDS):
                    continue

                right_clean = re.sub(r"\s+", " ", right).strip(" -")
                if _is_good_for_role(role, right_clean):
                    return _mk_val_by_rects(
                        [r], [right_clean], [c],
                        conf_hint=float(c or 0.80),
                    )

        # ----- PASS 1: split-token "VOLTAGE : 208Y/120V" -----
        for ln in top:
            toks = ln["tokens"]
            colon_idx = None

            for i, (r, t, c) in enumerate(toks):
                if ":" not in (t or ""):
                    continue

                left_words = set()
                for (_, tt, _) in toks[max(0, i - 4): i + 1]:
                    left_words.update(re.findall(r"[A-Z0-9]+", (tt or "").upper()))

                if left_words & LABEL_WORDS:
                    colon_idx = i
                    break

            if colon_idx is None:
                continue

            picked_rects, picked_texts, picked_confs = [], [], []
            for j in range(colon_idx + 1, min(len(toks), colon_idx + 7)):
                (r2, t2, c2) = toks[j]
                raw2 = (t2 or "").strip()
                if not raw2:
                    continue

                up2 = raw2.upper().strip()

                if ":" in raw2:
                    break
                if _looks_like_new_label_start(up2):
                    break

                picked_rects.append(r2)
                picked_texts.append(raw2)
                picked_confs.append(float(c2 or 0.70))

                joined = " ".join(picked_texts).strip()
                if _is_good_for_role(role, joined):
                    return _mk_val_by_rects(
                        picked_rects,
                        picked_texts,
                        picked_confs,
                        conf_hint=max(picked_confs) if picked_confs else 0.80,
                    )

        return None

    def _label_affinity(self, role: str, it: dict, labels_map: dict) -> float:
        import math

        Ls = labels_map.get(role, [])

        # If BUS/MAIN has no explicit labels, allow fallback to generic RATING labels
        if not Ls and role in ("BUS", "MAIN"):
            Ls = labels_map.get("RATING", [])

        if not Ls:
            return 0.0

        def dist(a, b): 
            return math.hypot(a["xc"]-b["xc"], a["yc"]-b["yc"])

        dmin = min(dist(it, L) for L in Ls)
        return math.exp(-dmin / self._SIGMA_PX)

    def _collect_label_candidates(self, items: list) -> dict:
        import re
        role_map = {k: [] for k in ("VOLTAGE","BUS","MAIN","MOUNTING","ENCLOSURE","RATING","AIC","NAME","WRONG")}
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
        out = {k: [] for k in ("NAME","VOLTAGE","BUS","MAIN","AIC","MOUNTING","ENCLOSURE")}
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

        for it in items:
            raw = str(it["text"] or "")
            txt = raw.upper()
            txtD = self._normalize_digits(txt)
            conf = float(it["conf"])
            is_na_placeholder = self._is_na_placeholder(raw)
            x1,y1,x2,y2,xc,yc = it["x1"],it["y1"],it["x2"],it["y2"],it["xc"],it["yc"]

            # VOLTAGE (pairs or single; tolerate OCR typos and missing slash)
            txtN = self._normalize_voltage_text(txtD)

            # --- Guards: don't let AIC-like or explicit-amp tokens become VOLTAGE ---

            # AIC-like: 10k..100k numbers, optionally with A/KA (e.g., "18000", "65,000A")
            aic_like = bool(re.search(r"\b(\d{2,3}[,]?\d{3})\s*(?:A|KA)?\b", txtD))

            # Amperage-like: "125 A", "225A", "400 AMPS"
            amps_like = bool(re.search(r"\b([1-9]\d{1,3})\s*(A\.?|AMPS?\.?)\b", txtD))

            # Voltage-ish context words/letters
            has_volty_ctx = bool(
                re.search(r'\b(WYE|DELTA|PH|PHASE|Ø|VOLT|VOLTS|VAC|VDC|V)\b', txtN)
            )

            # Old regex checks still help for clean cases
            pair = re.search(
                r'\b([1-6]\d{2,3})\s*[YV]?[\/]?\s*([1-6]?\d{2,3})\s*V?\b',
                txtN,
            )

            single = None
            if not aic_like and not amps_like and has_volty_ctx:
                single = re.search(r'(?<!\d)([1-6]\d{2,3})(?!\d)', txtN)

            snapped_voltage = None
            has_voltage_label_word = bool(re.search(r'\bVOLTAGE\b|\bVOLTS?\b|\bVAC\b|\bVDC\b|\bV\b', txt))
            has_voltage_numberish = bool(re.search(r'[IL\!\|]?\d{2,3}|\d{2,3}[IL\!\|]?', txtN))
            has_voltage_separator = ("/" in txtN) or ("Y" in txtN)

            if not aic_like and not amps_like and (has_volty_ctx or has_voltage_label_word or has_voltage_numberish or has_voltage_separator):
                snapped_voltage = self._snap_voltage_text(txtN)

            # MOUNTING value candidates
            up_mount = raw.strip().upper()
            up_mount = self._normalize_digits(up_mount)

            has_mount_word = bool(re.search(r"\b(SURFACE|FLUSH|RECESSED)\b", up_mount))
            has_nema_word = bool(re.search(r"\bNEMA\s*1\b|\bNEMA\s*3R\b|\bN1\b|\bN3R\b", up_mount))
            has_outdoor_word = bool(re.search(r"\b(OUTDOOR|WEATHERPROOF)\b", up_mount))

            if has_mount_word or has_nema_word or has_outdoor_word:
                shape = 0.0
                if has_mount_word:
                    shape += 0.72
                if has_nema_word or has_outdoor_word:
                    shape += 0.16

                # extra context if token itself includes mounting-ish wording
                ctx = 0.0
                if re.search(r"\b(SURFACE|FLUSH|RECESSED)\b", up_mount):
                    ctx += 0.10
                if re.search(r"\bNEMA\s*1\b|\bNEMA\s*3R\b|\bN1\b|\bN3R\b|\b(OUTDOOR|WEATHERPROOF)\b", up_mount):
                    ctx += 0.05

                out["MOUNTING"].append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "xc": xc, "yc": yc,
                    "conf": conf,
                    "text": raw,
                    "shape": min(1.0, shape),
                    "ctx": min(1.0, ctx),
                })

            # ENCLOSURE value candidates
            has_enclosure_word = bool(re.search(
                r"\bNEMA\s*1\b|\bNEMA\s*3R\b|\bN1\b|\bN3R\b|\bTYPE\s*1\b|\bTYPE\s*3R\b|\b(OUTDOOR|WEATHERPROOF)\b",
                up_mount
            ))

            if has_enclosure_word:
                shape = 0.72
                ctx = 0.10

                out["ENCLOSURE"].append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "xc": xc, "yc": yc,
                    "conf": conf,
                    "text": raw,
                    "shape": min(1.0, shape),
                    "ctx": min(1.0, ctx),
                })

            # If it looks AIC-like and has no clear voltage context or separators,
            # kill the pair match so things like "18000" don't become VOLTAGE.
            if pair and aic_like and not has_volty_ctx and "/" not in txtN and "Y" not in txtN and "V" not in txtN:
                pair = None

            if pair or single or snapped_voltage is not None:
                if pair or snapped_voltage in (120240, 208, 480, 600):
                    shape = 0.92
                else:
                    shape = 0.62

                ctx = 0.15 if has_volty_ctx else 0.0
                if ":" in raw and re.search(r'\bVOLTAGE\s*:', raw.upper()):
                    ctx += 0.10
                elif re.search(r'\bVOLTAGE\b', raw.upper()):
                    ctx += 0.05

                out["VOLTAGE"].append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "xc": xc, "yc": yc,
                    "conf": conf,
                    "text": raw,
                    "shape": min(1.0, shape),
                    "ctx": min(1.0, ctx),
                })

            # -----------------------------------------------------------
            # BUS/MAIN
            # Only allow amperage-looking values.
            # NEVER allow voltage-looking strings like:
            #   208/120V
            #   480Y/277V
            #   Volts: 208/120V
            #   480V
            #
            # Also NEVER allow AIC/kA-looking strings like:
            #   65k
            #   65KAIC
            #   AIC: 65
            #   Available Fault Current (A): 65k
            # -----------------------------------------------------------

            reject_as_amp_aic_like = self._looks_like_aic_or_ka_text(txtD)
            reject_as_amp_non_amp_context = self._looks_like_non_amp_context_text(txtD)
            reject_as_amp = reject_as_amp_aic_like or reject_as_amp_non_amp_context

            m_with_unit = None if reject_as_amp else re.search(
                r"\b([1-9]\d{1,3})\s*(A\.?|AMP\.?|AMPS?\.?)\b",
                txtD
            )

            m_bare_num = None if reject_as_amp else re.search(
                r"(?<!\d)([1-9]\d{1,3})(?!\d)",
                txtD
            )

            # strong voltage guards
            has_slash_voltage = "/" in txtN
            has_voltage_word = bool(re.search(r"\b(VOLT|VOLTS|VOLTAGE|VAC|VDC)\b", txt))
            has_phase_voltage_ctx = bool(re.search(r"\b(WYE|DELTA|PH|PHASE|Ø)\b", txtN))
            has_v_suffix = bool(re.search(r"(?<!\d)[1-6]\d{2,3}\s*V\b", txtN))
            has_voltage_pair = bool(re.search(r'(?<!\d)([1-6]\d{2,3})\s*[YV]?\s*/\s*([1-6]?\d{2,3})(?!\d)', txtN))
            snapped_voltage = self._snap_voltage_text(txtN) is not None

            strong_voltage_token = any([
                has_slash_voltage,
                has_voltage_word,
                has_phase_voltage_ctx,
                has_v_suffix,
                has_voltage_pair,
                snapped_voltage,
            ])

            # Explicit main-breaker context
            main_ctxt = bool(
                re.search(
                    r"\bMCB\b|"
                    r"\bM\W*C\W*B\b|"
                    r"\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b|"
                    r"\bMAIN\s*DEVICE\b|"
                    r"\bMAINS?\s*RATING\b",
                    txt
                )
            )
            m_main_map = re.search(r"\b([1-9]\d{1,3})\s*MAP\b", txtD)

            # N/A is allowed to mean "no MAIN value", but it should not
            # interfere with BUS parsing or imply MLO.
            if is_na_placeholder and main_ctxt:
                m_with_unit = None
                m_bare_num = None
                m_main_map = None

            cand = None
            cand_main_only = False

            # If it looks like voltage, never allow it into BUS/MAIN candidate pools
            if strong_voltage_token:
                cand = None

            elif m_with_unit or m_main_map:
                if m_with_unit:
                    n = int(m_with_unit.group(1))
                else:
                    n = int(m_main_map.group(1))

                if n is not None:
                    lo = 30 if (main_ctxt or m_main_map) else 50
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
                        if m_main_map and not m_with_unit:
                            cand_main_only = True

            elif m_bare_num:
                n = int(m_bare_num.group(1))

                if 50 <= n <= 1200:
                    cand = {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "xc": xc, "yc": yc,
                        "conf": conf,
                        "text": raw,
                        "shape": 0.68,
                        "ctx": 0.04,
                        "has_unit": False,
                        "requiresStrongAmpLabel": bool(n == 50),
                    }

            if cand is not None:
                up_full = txt

                has_bus_word = bool(re.search(r'\bBUS\b', up_full))

                # Explicit main-device wording should stay MAIN-only
                has_main_device_word = bool(
                    re.search(
                        r'\bMCB\b|'
                        r'\bM\W*C\W*B\b|'
                        r'\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b|'
                        r'\bMAIN\s*DEVICE\b',
                        up_full
                    )
                )

                # "Mains Rating" is the one ambiguous case we want to preserve for later logic
                has_mains_rating_word = bool(
                    re.search(r'\bMAINS?\s*RATING\b|\bMAIN\s*RATING\b', up_full)
                )

                if cand_main_only:
                    out["MAIN"].append(dict(cand))

                elif has_bus_word and not has_main_device_word:
                    out["BUS"].append(dict(cand))

                elif has_main_device_word and not has_bus_word:
                    # MCB Rating / Main Breaker / Main Device stay MAIN-only
                    out["MAIN"].append(dict(cand))

                elif has_mains_rating_word and not has_main_device_word:
                    # Keep this available to both roles for later reconciliation.
                    # We do NOT want to lose it on MLO jobs where it may need to become BUS.
                    out["BUS"].append(dict(cand))
                    out["MAIN"].append(dict(cand))

                else:
                    out["BUS"].append(dict(cand))
                    out["MAIN"].append(dict(cand))

            # AIC (10k..100k) and KA forms: 65kA, 65 kA
            # Normalize spaces for kA form matching
            t_nos = txtD.replace(" ", "")
            mk = None if is_na_placeholder else re.search(
                r"\b(\d{2,3})(?:KAMP|KAIC|AIC|KA|K)\b",
                t_nos
            )
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
                AIC = None if is_na_placeholder else re.search(
                    r"\b(\d{2,3}[,]?\d{3})\s*(?:A|KA)?\b",
                    txtD
                )
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
                    m_small = None if is_na_placeholder else re.search(
                        r'(?<!\d)(\d{2,3})(?!\d)',
                        txtD
                    )
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
            up = self._strip_quotes(up)
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
            if re.search(r"\b\d{2,3}\s*(?:KAMP|KAIC|AIC|KA|K)\b", up) or re.search(r"\b\d{2,3}[,]?\d{3}\b", up):
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

        def _voltage_key(text: str) -> Optional[str]:
            """
            Canonical voltage key for deduping multi-pass OCR voltage candidates.
            """
            if not text:
                return None

            snapped_family = self._snap_voltage_text(text)
            if snapped_family is not None:
                return str(snapped_family)

            t = self._normalize_digits(str(text).upper())
            t = self._normalize_voltage_text(t)

            m = re.search(r'(?<!\d)(\d{3,4})\s*[YV]?\s*/\s*(\d{2,4})(?!\d)', t)
            if m:
                return f"{m.group(1)}/{m.group(2)}"

            m2 = re.search(r'(?<!\d)(\d{3,4})(?!\d)', t)
            if m2:
                return m2.group(1)

            return None

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

                elif role == "VOLTAGE":
                    same_place = _iou(c, last) >= 0.70
                    k_c = _voltage_key(c.get("text", ""))
                    k_l = _voltage_key(last.get("text", ""))
                    same_key = (k_c is not None and k_c == k_l)
                    is_dup = same_place and same_key

                else:
                    # NAME (keep stricter rule)
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

        # NEW: BUS/MAIN fall back to generic rating labels if explicit ones are absent
        if not same_labels and role in ("BUS", "MAIN"):
            same_labels = labels_map.get("RATING", [])

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
                if re.search(
                    r"\bMAIN\b|"
                    r"\bMCB\b|"
                    r"\bM\W*C\W*B\b|"
                    r"\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b|"
                    r"\bMAIN\s*(?:DEVICE|LUGS?)\b",
                    tu
                ):
                    hint -= 1.0
            elif role == "MAIN":
                if re.search(
                    r"\bMAIN\b|"
                    r"\bMCB\b|"
                    r"\bM\W*C\W*B\b|"
                    r"\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b|"
                    r"\bMAIN\s*(?:DEVICE|LUGS?)\b",
                    tu
                ):
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

    def _pick_role(self, role: str, ranked: list) -> dict | None:
        thr = self._THRESH.get(role, 0.5)
        return ranked[0] if ranked and ranked[0]["rank"] >= thr else None

    def _voltage_compare_form(self, s: str) -> str:
        """
        Collapse a voltage string into a comparison form so ugly OCR variants
        can be matched against canonical patterns.

        Examples:
            "120 V / 208 Y WYE /3P" -> "120208Y"
            "208Y/120V"             -> "208Y120"
            "208YII2OV"             -> "208Y1120" (after normalization)
        """
        if not s:
            return ""

        u = str(s).upper()
        u = self._normalize_digits(u)
        u = self._normalize_voltage_text(u)

        # normalize wording
        u = re.sub(r'\bVOLTS?\b', 'V', u)
        u = re.sub(r'\bWYE\b', 'Y', u)

        # drop phase / wire tail noise
        u = re.sub(r'\b\d+\s*P(H(ASE)?)?\b', ' ', u)      # 3P / 3PH / 3PHASE
        u = re.sub(r'\b\d+\s*W(IRE)?\b', ' ', u)          # 4W / 4WIRE
        u = re.sub(r'\bPH(ASE)?\b', ' ', u)
        u = re.sub(r'\bWIRE(S)?\b', ' ', u)

        # keep only digits, Y, V, slash
        u = re.sub(r'[^0-9YV/]', '', u)

        # V is not important for matching family shape
        u = u.replace('V', '')

        # slash is not important for compare form
        u = u.replace('/', '')

        return u.strip()


    def _canonical_voltage_compare_forms(self) -> dict:
        """
        Returns:
            {
                208: {"208", "208120", "208Y120", "120208", ...},
                480: {...},
                ...
            }
        """
        out = {}
        for family, variants in self.VOLTAGE_CANONICAL_MAP.items():
            forms = set()
            for v in variants:
                cf = self._voltage_compare_form(v)
                if cf:
                    forms.add(cf)
            out[family] = forms
        return out


    def _snap_voltage_text(self, txt: str | None) -> Optional[int]:
        """
        Snap OCR voltage text to the closest canonical voltage family key
        from VOLTAGE_CANONICAL_MAP / VOLTAGE_OUTPUT_MAP.

        Returns one of:
            120, 120240, 208, 240, 480, 600
        or None if no confident match.
        """
        if not txt:
            return None

        raw_cf = self._voltage_compare_form(txt)
        if not raw_cf:
            return None

        # must contain enough numeric structure to plausibly be a voltage
        digit_runs = re.findall(r'\d+', raw_cf)
        joined_digits = ''.join(digit_runs)

        if len(joined_digits) < 3:
            return None

        # require either:
        # - at least one 3-digit-ish run
        # - or a slash/pair style voltage shape after normalization
        # - or a wye marker with enough digits
        has_threeish = bool(re.search(r'\d{3}', raw_cf))
        has_pairish = bool(re.search(r'\d+[Y/]?\d+', raw_cf))
        has_wyeish = ('Y' in raw_cf and len(joined_digits) >= 6)

        if not (has_threeish or has_pairish or has_wyeish):
            return None

        canonical_forms = self._canonical_voltage_compare_forms()

        best_family = None
        best_score = -1.0

        raw_nums = set(re.findall(r'\d{3}', raw_cf))
        raw_has_y = 'Y' in raw_cf

        for family, forms in canonical_forms.items():
            family_best = -1.0

            for form in forms:
                score = difflib.SequenceMatcher(a=raw_cf, b=form).ratio()

                form_nums = set(re.findall(r'\d{3}', form))
                form_has_y = 'Y' in form

                # bonus for matching the same voltage numbers
                num_overlap = len(raw_nums & form_nums)
                score += 0.08 * num_overlap

                # bonus if both are wye-ish
                if raw_has_y and form_has_y:
                    score += 0.06

                # small bonus when lengths are close
                score -= 0.02 * abs(len(raw_cf) - len(form))

                if score > family_best:
                    family_best = score

            if family_best > best_score:
                best_score = family_best
                best_family = family

        # threshold: high enough to avoid random junk, low enough for OCR garbage
        if best_score >= 0.72:
            return best_family

        return None

    def _normalize_voltage_text(self, s: str) -> str:
        """
        Strong OCR cleanup for voltage strings before matching / snapping.

        Goals:
        - repair common OCR mistakes:
            O/Q/D -> 0
            Z -> 2
            I/L/|/! can behave like 1 or slash depending on context
        - normalize spaces / punctuation
        - collapse junk tails like /3, /3P, 3PH, 4W, etc.
        - preserve enough structure for snapping to canonical voltage families
        """
        u = (s or "").upper()

        # normalize words first
        u = re.sub(r'\bVOLTS?\b', 'V', u)
        u = re.sub(r'\bWYE\b', 'Y', u)
        u = re.sub(r'\bVAC\b', 'V', u)
        u = re.sub(r'\bVDC\b', 'V', u)

        # common OCR substitutions
        u = u.replace("Ø", "")
        u = u.replace("O", "0")
        u = u.replace("Q", "0")
        u = u.replace("D", "0")
        u = u.replace("Z", "2")

        # normalize separators and whitespace
        u = u.replace(",", " ")
        u = u.replace(".", " ")
        u = u.replace(":", " ")
        u = re.sub(r'\s+', ' ', u).strip()

        # remove common phase / wire tails that do not matter for system voltage
        u = re.sub(r'\b\d+\s*/\s*\d+\s*W(IRE)?\b', ' ', u)
        u = re.sub(r'\b\d+\s*W(IRE)?\b', ' ', u)
        u = re.sub(r'\b\d+\s*P(H(ASE)?)?\b', ' ', u)     # 3P / 3PH / 3PHASE
        u = re.sub(r'\bPH(ASE)?\b', ' ', u)
        u = re.sub(r'\bWIRE(S)?\b', ' ', u)

        # common glued voltage pairs
        u = re.sub(r'(?<!\d)1201?240(?!\d)', '120/240', u)
        u = re.sub(r'(?<!\d)2081?120(?!\d)', '208/120', u)
        u = re.sub(r'(?<!\d)4801?277(?!\d)', '480/277', u)
        u = re.sub(r'(?<!\d)6001?347(?!\d)', '600/347', u)

        # squeeze spaces again
        u = re.sub(r'\s+', ' ', u).strip()

        # remove spaces around slash
        u = re.sub(r'\s*/\s*', '/', u)

        # I/L/!/| between digits often means slash or 1
        # First: if we have 3 digits + [I/L/!/|] + 3 digits, it's probably a slash
        ALLOWED_VOLTS = {"120", "208", "240", "277", "347", "480", "600"}

        def _mid_sep_repl(m):
            left = m.group(1)
            sep = m.group(2)
            right = m.group(3)
            if left in ALLOWED_VOLTS and right in ALLOWED_VOLTS:
                return f"{left}/{right}"
            return f"{left}{sep}{right}"

        u = re.sub(r'(?<!\d)(\d{3})([IL\!\|])(\d{3})(?!\d)', _mid_sep_repl, u)

        # Also handle 3 digits + literal 1 + 3 digits when both sides are valid voltages
        def _glued_pair_repl(m):
            left = m.group(1)
            right = m.group(2)
            if left in ALLOWED_VOLTS and right in ALLOWED_VOLTS:
                return f"{left}/{right}"
            return m.group(0)

        u = re.sub(r'(?<!\d)(\d{3})\s*1\s*(\d{3})(?!\d)', _glued_pair_repl, u)

        # remaining I/L/!/| inside number runs are more likely 1 than slash
        u = re.sub(r'[IL\!\|]', '1', u)

        # normalize known compact forms like 208Y120V -> 208Y/120V
        u = re.sub(r'(?<!\d)(208|480|600)Y(120|277|347)V?(?!\d)', r'\1Y/\2V', u)

        # normalize reversed wye-ish forms like 120/208Y -> 120/208Y
        # no structural change needed, but keep slash clean
        u = re.sub(r'\s+', '', u)

        return u

    def _normalize_digits(self, s: str) -> str:
        u = s or ""
        # O/o between digits → 0  (e.g., 2O8 → 208, 6o0 → 600)
        u = re.sub(r'(?<=\d)[Oo](?=\d)', '0', u)

        # O/o after a digit and before A/AMPS.
        # Also handle runs like "4OOA" -> "400A"
        def _oo_to_zeros(m):
            return "0" * len(m.group(0))

        u = re.sub(r'(?<=\d)[Oo]+(?=\s*(?:A|AMPS?)\b)', _oo_to_zeros, u, flags=re.I)

        return u

    def _is_na_placeholder(self, text: str) -> bool:
        """
        Recognize common OCR variants of N/A.

        This helper is only used for fields where N/A is a legitimate
        "no value" result, currently MAIN and AIC.
        """
        if not text:
            return False

        t = str(text).upper().strip()

        # Normalize likely separator OCR variants.
        t = t.replace("\\", "/")
        t = t.replace("|", "/")
        t = t.replace("!", "/")

        # Remove formatting/separators.
        compact = re.sub(r"[\s./_\-]+", "", t)

        return compact in {
            "NA",
            "NIA",
            "N1A",
            "NLA",
        }

    def _looks_like_aic_or_ka_text(self, s: str) -> bool:
        """
        True when text should NEVER be used as BUS/MAIN amps.

        Blocks OCR junk like:
          65k
          65 k
          65KA
          65KAIC
          AIC: 65
          KAIC 65
          65,000 A
          Available Fault Current (A): 65000

        This is intentionally for amp rejection only.
        """
        if not s:
            return False

        t = self._normalize_digits(str(s).upper())
        t_spaced = re.sub(r"\s+", " ", t).strip()
        t_nos = re.sub(r"[\s\.\-_:()/]+", "", t_spaced)

        # Explicit AIC/fault-current language
        if re.search(r"\b(AIC|KAIC|SCCR|INTERRUPTING|FAULT\s*CURRENT|AVAILABLE\s*FAULT)\b", t_spaced):
            return True

        # OCR variants like A.I.C. / A L C / ALC can be ugly, so the unit suffix matters most.
        # 65K, 65KA, 65KAIC, 65AIC, 65KAMP
        if re.search(r"(?<!\d)(\d{1,4})(KAMP|KAIC|AIC|KA|K)(?![A-Z0-9])", t_nos):
            return True

        # Large interrupt-rating amps: 10,000 A / 65000 A / 65,000
        if re.search(r"(?<!\d)(\d{2,3}[,]?\d{3})\s*(A|KA)?\b", t_spaced):
            try:
                val = int(re.search(r"(?<!\d)(\d{2,3}[,]?\d{3})", t_spaced).group(1).replace(",", ""))
                if 10000 <= val <= 200000 and val % 1000 == 0:
                    return True
            except Exception:
                pass

        return False

    def _looks_like_non_amp_context_text(self, s: str) -> bool:
        """
        True only when text has a number and is clearly a non-rating context field.

        This is intentionally narrow. It should block obvious metadata/source/location
        strings from becoming BUS/MAIN amps, but it should NOT make normal amp parsing
        more strict.

        Blocks:
          Location: ELECTRICAL 123
          LOC: 123
          Room 123
          Source: MDP-1
          Supply From: MDP-1
          Fed From: SWBD-1
          Fed By: TX-2
          Serving RTU-3

        Does NOT block just because text contains:
          ELECTRICAL
          ELEC
          AREA
          PANEL
          POWER
        """
        if not s:
            return False

        t = self._normalize_digits(str(s).upper())
        t = re.sub(r"\s+", " ", t).strip()

        # Only care if there is actually a numeric value to steal.
        if not re.search(r"(?<!\d)\d{1,4}(?!\d)", t):
            return False

        # If the same text explicitly says BUS/MAIN/MCB/PANEL RATING, do not block it.
        if re.search(
            r"\b(BUS|BUSS|MAIN|MAINS|MCB|M\W*C\W*B|PANEL\s*RATING|AMPACITY)\b|"
            r"\bMAIN\s+(?:CIRCUIT\s+)?(?:BREAKER|BKR|BRKR)\b|"
            r"\bMAIN\s*DEVICE\b",
            t
        ):
            return False

        # Explicit non-rating context labels only.
        if re.search(
            r"\b(LOCATION|LOC\.?|ROOM|SOURCE|SUPPLY\s+FROM|SUPPLIED\s+FROM|FED\s+FROM|FED\s+BY|SERVING)\b",
            t
        ):
            return True

        return False

    def _strip_quotes(self, s: str) -> str:
        if not s:
            return s
        # remove straight + curly quotes
        return s.translate(str.maketrans({
            '"': '', "'": '',
            '“': '', '”': '',
            '‘': '', '’': '',
        }))

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
            "NAME":      (255, 128,   0),
            "VOLTAGE":   (255, 200, 100),
            "BUS":       (  0, 200,   0),
            "MAIN":      (180,   0, 180),
            "MOUNTING":  (  0, 165, 255),
            "ENCLOSURE": (255, 255,   0),
            "AIC":       (  0,   0, 255),
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
            "MOUNTING (orange-blue) candidates",
            "ENCLOSURE (yellow) candidates",
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
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        g = cv2.bilateralFilter(g, d=5, sigmaColor=35, sigmaSpace=35)
        h, w = g.shape
        if h < 1400 or w < 1400:
            scale = max(1400 / h, 1400 / w, 1.8)
            g = cv2.resize(g, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
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

    # ---------- Line-suck / border-line scrub helpers ----------
    def _extract_first_int_1to4(self, text: str) -> Optional[int]:
        if not text:
            return None
        t = self._normalize_digits(str(text).upper())
        m = re.search(r'(?<!\d)([1-9]\d{0,3})(?!\d)', t)
        return int(m.group(1)) if m else None

    def _has_left_border_line_component(
        self,
        gray_full: np.ndarray,
        bbox: Tuple[int, int, int, int],
        *,
        left_frac: float = 0.22,
        min_h_frac: float = 0.80,
        max_w_frac: float = 0.12,
    ) -> bool:
        """
        Detect the classic error signature:
        a tall, skinny component on the LEFT edge of the token bbox
        (usually a panel border/grid line) that OCR interprets as a leading "1".
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            return False

        if gray_full is None or gray_full.size == 0:
            return False

        H, W = gray_full.shape[:2]
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return False

        crop = gray_full[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 10 or cw < 10:
            return False

        # binarize: make dark strokes become white (foreground)
        try:
            _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        except Exception:
            return False

        # optional: kill tiny specks
        bw = cv2.medianBlur(bw, 3)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num <= 1:
            return False

        left_limit = int(max(1, round(left_frac * cw)))
        min_h = float(min_h_frac) * float(ch)
        max_w = float(max_w_frac) * float(cw)

        # stats rows: [x, y, w, h, area]
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 6:
                continue

            # only consider components that are within the left portion of the crop
            cx = float(centroids[i][0])
            if cx > left_limit:
                continue

            # tall + skinny = likely border line
            if h >= min_h and w <= max_w:
                return True

        return False

    def _find_clean_alt_amp_in_ranked(
        self,
        ranked_list: List[dict],
        target_amp: int,
        *,
        min_conf: float = 0.0
    ) -> Optional[dict]:
        """
        Look for another candidate in the ranked list whose parsed amp equals target_amp.
        Returns the best match by confidence then rank.
        """
        best = None
        for c in (ranked_list or []):
            txt = c.get("text", "")
            n = self._extract_first_int_1to4(txt)
            if n is None or n != int(target_amp):
                continue
            if float(c.get("conf", 0.0)) < float(min_conf):
                continue
            if best is None:
                best = c
            else:
                if float(c.get("conf", 0.0)) > float(best.get("conf", 0.0)):
                    best = c
                elif float(c.get("conf", 0.0)) == float(best.get("conf", 0.0)) and float(c.get("rank", 0.0)) > float(best.get("rank", 0.0)):
                    best = c
        return best

    def _scrub_leading_line_digit_for_role(
        self,
        role: str,
        chosen: Optional[dict],
        ranked_map: Dict[str, List[dict]],
        gray_full: np.ndarray,
        *,
        plausible_min: int = 60,
        plausible_max: int = 1200,
        require_support: bool = True,
    ) -> Optional[dict]:
        """
        Fix cases like "1150 A" -> "150 A" caused by border line sucked into OCR token.
        Applies ONLY when:
          - suspicious (4 digits starting with '1' and >= 1000)
          - AND (alt candidate exists) OR (left-border line component detected)
        """
        if not chosen:
            return chosen

        raw_text = str(chosen.get("text", "") or "")
        n = self._extract_first_int_1to4(raw_text)
        if n is None:
            return chosen

        s = str(n)
        # suspicious pattern: 4 digits starting with 1 and >= 1000 (1150, 1600, 1100, 1250...)
        if not (len(s) == 4 and s.startswith("1") and n >= 1000):
            return chosen

        # proposed clean alt by dropping the leading '1'
        alt_n = None
        try:
            alt_n = int(s[1:])
        except Exception:
            alt_n = None

        if alt_n is None or not (plausible_min <= alt_n <= plausible_max):
            return chosen

        # Support signal A: exists elsewhere among ranked candidates for this role
        alt_cand = self._find_clean_alt_amp_in_ranked(ranked_map.get(role, []), alt_n)

        # Support signal B: geometry shows a left-edge border line component
        bbox = (chosen.get("x1", 0), chosen.get("y1", 0), chosen.get("x2", 0), chosen.get("y2", 0))
        has_line = self._has_left_border_line_component(gray_full, bbox)

        if require_support and (alt_cand is None) and (not has_line):
            # suspicious but we couldn't prove it's a line-suck
            return chosen

        # If we have an alt candidate, prefer it directly (more consistent + keeps bbox honest)
        if alt_cand is not None:
            if getattr(self, "debug", False):
                print(f"[SCRUB:{role}] Using alternate candidate {n} -> {alt_n} via ranked_map match")
            return alt_cand

        # Otherwise, patch the chosen token text (keep bbox, keep rank/conf)
        patched = dict(chosen)
        patched.setdefault("_raw_text", raw_text)

        # Replace only the first occurrence of the suspicious number in the text
        patched_text = raw_text
        patched_text = re.sub(rf'(?<!\d){re.escape(s)}(?!\d)', str(alt_n), patched_text, count=1)
        patched["text"] = patched_text

        return patched
