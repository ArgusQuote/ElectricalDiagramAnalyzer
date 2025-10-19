# OcrLibrary/PanelHeaderParserV4.py  (Drop-in replacement)
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
        "AMP","AMPS","A","VOLT","VOLTS","V","KA","KVA","KV","HZ","HERTZ",
        "PHASE","PHASES","PH","Ø","WIRE","WIRES","CONDUCTOR","CONDUCTORS","POLE","POLES",
        "NEMA","ENCLOSURE","INDOOR","OUTDOOR","WEATHERPROOF","SURFACE","FLUSH",
        "MOUNTING","WIDTH","DEPTH","HEIGHT","SECTIONS",
        "BUS","BUSS","BUSBAR","MATERIAL","ALUMINUM","AL","ALUM","COPPER","CU",
        "MAIN","MCB","MLO","SERVICE","SERVICE ENTRANCE","SE","S.E.","LUG","LUGS",
        "FEED","FEEDS","FEEDER","FED","FROM","BY","FEED THRU","FEED-THRU","FEEDTHRU",
        "GROUND","GROUNDED","GND","EARTH","NEUTRAL","NEUT","N",
        "GFI","GFCI","AFCI","SHUNT","TRIP","PROT","PROTECTION","OPTIONS","DATA",
        "I-LINE","ILINE","I","LINE","QO","HOM","SQUARE","SCHNEIDER","EATON","SIEMENS",
        "NOTES","TABLE","SCHEDULE","SIZE","RANGE","CATALOG","CAT","CAT.","DWG","REV","DATE"
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
        "MAIN": [r"\bMAIN\s*(RATING|BREAKER|DEVICE|TYPE)\b", r"\bMAIN\s*(TYPE|RATING|BREAKER|DEVICE)\b", r"\bMCB\b",
                 r"\bMLO\b", r"\bMAIN\s*LUGS?\b", r"\bMAIN\s*TYPE\b", r"\bMAINS?\b"
        ],
        "AIC": [
            r"\bA\.?\s*I\.?\s*C\.?\b", r"\bAIC\b", r"\bKAIC\b", r"\bSCCR\b",
            r"\bINTERRUPTING\s*RATING\b", r"\bAVAILABLE\s*FAULT\s*CURRENT\b", r"\bFAULT\s*CURRENT\b",
            r"\bSYMMETRICAL\b"
        ],
        "NAME": [r"\bPANEL(BOARD)?\b", r"\bBOARD\b", r"\bPANEL\s*:?\b", r"\bDISTRIBUTION\s*PANEL\b"],
        "WRONG": [
            r"\bNOTES?\b", r"\bTABLE\b", r"\bSCHEDULE\b", r"\bSIZE\s*\(??A\)?\b", r"\bCIRCUIT\b",
            r"\bCAT(ALOG)?\b", r"\bDWG\b", r"\bREV\b", r"\bDATE\b"
        ],
    }

    # ====== Role weights (blend of value-shape, label-affinity, side, ctx, penalties) ======
    _WEIGHTS = {
        "VOLTAGE": dict(W_shape=0.60, W_conf=0.15, W_lbl=0.15, W_side=0.03, W_ctx=0.07, W_wrong=0.12, W_y=0.00),
        "BUS":     dict(W_shape=0.55, W_conf=0.15, W_lbl=0.20, W_side=0.08, W_ctx=0.04, W_wrong=0.15, W_y=0.00),
        "MAIN":    dict(W_shape=0.55, W_conf=0.15, W_lbl=0.22, W_side=0.10, W_ctx=0.05, W_wrong=0.15, W_y=0.00),
        "AIC":     dict(W_shape=0.60, W_conf=0.12, W_lbl=0.22, W_side=0.06, W_ctx=0.08, W_wrong=0.12, W_y=0.00),
        "NAME":    dict(W_shape=0.55, W_conf=0.10, W_lbl=0.15, W_side=0.12, W_ctx=0.00, W_wrong=0.08, W_y=0.35)
    }

    _THRESH = {"VOLTAGE":0.55, "BUS":0.54, "MAIN":0.54, "AIC":0.54, "NAME":0.50}

    # ====== Non-panel label hits (quick gate) ======
    _NOT_PANEL_LABELS = {
        "WIREWAY",
        "WW",
        "FEEDER",
        "MECHANICAL",
        "CHART",
        "FEEDER",
        "TRANSFORMER",
        "XFMR",
        "TRANSFMR",
        "GENERATOR",
        "DISCONNECT",
        "EQUIPMENT",
        "DETAIL",
        "ELEVATION",
        "MECHANICAL",
        "MECH",
        "METER"
    }

    def _line_hits_not_panel(self, lines) -> tuple[bool, str]:
        """
        Scan the top few lines for clear 'not panel' signals.
        Return (hit, matched_token). Robust to spacing/case.
        """
        import re
        if not lines:
            return (False, "")
        top = lines[: min(4, len(lines))]  # only need a quick sniff test
        for ln in top:
            up = (ln.get("text") or "").upper()
            for token in self._NOT_PANEL_LABELS:
                # whole-word or phrase match; tolerate extra spaces
                t = re.sub(r"\s+", r"\\s+", token.strip())
                if re.search(rf"(^|\W){t}(\W|$)", up):
                    return (True, token)
        return (False, "")

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

    # ======= public =======
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

        detailed = self.reader.readtext(
            prep, detail=1, paragraph=False,
            mag_ratio=1.6, contrast_ths=0.05, adjust_contrast=0.7,
            text_threshold=0.4, low_text=0.3,
        )
        try:
            inv = cv2.bitwise_not(prep)
            det2 = self.reader.readtext(
                inv, detail=1, paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,()/:- kKVvYØø ",
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

        # --- Non-panel quick gate (bail before heavy association/scoring) ---
        hit_not_panel, matched = self._line_hits_not_panel(lines)
        if hit_not_panel:
            # Return a lightweight header result that marks this as a false positive.
            # Downstream UI will show name="false positive", attrs=Nones, and still keep the image.
            return {
                "type": "panelboard",
                "name": "false positive",
                "attrs": {
                    "amperage": None,
                    "voltage": None,
                    "intRating": None,
                    "mainBreakerAmperage": None,
                    "detected_breakers": [],   # ensure nothing downstream tries to use breakers
                    "_reason": f"non-panel label hit: {matched}",
                },
            }

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
        for role in ("NAME","VOLTAGE","BUS","MAIN","AIC"):
            ranked = self._score_candidates(role, value_cands.get(role, []), labels_map, band_top, main_mode)
            ranked_map[role] = ranked
            chosen_map[role] = self._pick_role(role, ranked)

        # BUS vs MAIN arbitration & main mode enforcement
        if main_mode == "MLO":
            chosen_map["MAIN"] = None
        elif main_mode is None:
            b, m = chosen_map.get("BUS"), chosen_map.get("MAIN")
            if b and m:
                # Prefer the candidate with stronger affinity to its own label
                aff_b = self._label_affinity("BUS", b, labels_map)
                aff_m = self._label_affinity("MAIN", m, labels_map)
                if abs(b["rank"] - m["rank"]) <= 0.06:
                    if aff_b > aff_m + 0.05:
                        chosen_map["MAIN"] = None
                    elif aff_m > aff_b + 0.05:
                        chosen_map["BUS"] = None
                # If still ambiguous, prefer BUS as the default rating holder
                if chosen_map["BUS"] and chosen_map["MAIN"]:
                    chosen_map["MAIN"] = None

        # ===== NAME fallback: pick top-left plausible token if none chosen =====
        if not chosen_map.get("NAME"):
            # prefer NAME candidates we already collected
            name_cands = list(value_cands.get("NAME") or [])
            if name_cands:
                # sort by top-most, then left-most
                name_cands.sort(key=lambda c: (c["y1"], c["x1"]))
                cand0 = dict(name_cands[0])
                cand0.setdefault("rank", 0.51)   # default rank for debug/readability
                chosen_map["NAME"] = cand0
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
                    chosen_map["NAME"] = picked

        # ===== BUS fallback: if no BUS chosen, promote best amperage-looking value =====
        if not chosen_map.get("BUS"):
            bus_ranked = list(ranked_map.get("BUS") or [])
            main_ranked = list(ranked_map.get("MAIN") or [])

            def _amps_from_text(t: str) -> Optional[int]:
                m = re.search(r"\b([6-9]\d|[1-9]\d{2,3})\s*(A|AMPS?)\b", (t or "").upper())
                return int(m.group(1)) if m else None

            # Consider sub-threshold BUS first, then MAIN as a donor
            pool = bus_ranked + main_ranked
            # keep only entries that *actually* parse as amps in [60..1200]
            pool = [c for c in pool if _amps_from_text(c.get("text","")) is not None]
            if pool:
                # Prefer: higher rank → higher confidence; tiebreak by top-most then left-most
                pool.sort(key=lambda c: (-float(c.get("rank", 0.0)), c.get("y1", 1e9), c.get("x1", 1e9)))
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

        # ===== Convert chosen → normalized outputs =====
        def _to_int(s):
            try: return int(str(s).replace(",", "").strip())
            except Exception: return None

        # NAME
        name = (chosen_map["NAME"]["text"] if chosen_map["NAME"] else "") or ""

        # VOLTAGE
        voltage_val = None
        if chosen_map["VOLTAGE"]:
            txtU = chosen_map["VOLTAGE"]["text"].upper()
            txtN = self._normalize_voltage_text(txtU)
            # Accept: 480Y/277, 480Y277, 277/480V, 208/120, etc. (slash optional; OCR fixes)
            m = re.search(r'\b([1-6]\d{2,3})\s*[YV]?[\/]?\s*([1-6]?\d{2,3})\s*V?\b', txtN)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                voltage_val = 120 if (120 in (a, b) and 240 in (a, b)) else max(a, b)
            else:
                # single like 480 V (or just 480)
                m2 = re.search(r'(?<!\d)([1-6]\d{2,3})(?!\d)', txtN)
                voltage_val = int(m2.group(1)) if m2 else None

        # BUS
        bus_amp = None
        if chosen_map["BUS"]:
            m = re.search(r"\b([6-9]\d|[1-9]\d{2,3})\s*(A|AMPS?)\b", chosen_map["BUS"]["text"].upper())
            if m: bus_amp = _to_int(m.group(1))

        # MAIN
        main_amp = None
        self.last_main_type = None
        if chosen_map["MAIN"]:
            txtU = chosen_map["MAIN"]["text"].upper()
            m = re.search(r"\b([6-9]\d|[1-9]\d{2,3})\s*(A|AMPS?)\b", txtU)
            if m: main_amp = _to_int(m.group(1))
            if re.search(r"\b(MLO|MAIN\s*LUGS?)\b", txtU): self.last_main_type = "MLO"
            elif re.search(r"\b(MCB|MAIN\s*BREAKER)\b", txtU): self.last_main_type = "MCB"

        # AIC → kA (65kA, 65 kA, 65,000 A, 65000)
        int_rating_ka = None
        if chosen_map["AIC"]:
            t = chosen_map["AIC"]["text"].upper().replace(" ", "")
            # e.g. 65kA
            mk = re.search(r"\b(\d{2,3})K?A\b", t)
            if mk and "KA" in t:
                int_rating_ka = int(mk.group(1))
            else:
                m = re.search(r"\b(\d{2,3}[,]?\d{3})\s*(?:A|KA)?\b", chosen_map["AIC"]["text"].upper())
                if m:
                    val = int(m.group(1).replace(",", ""))
                    int_rating_ka = max(1, val // 1000)

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

    # --------- Helpers for association ---------
    def _scan_main_mode(self, items: List[dict]) -> Optional[str]:
        txt = " ".join(str(it.get("text","")) for it in items).upper()
        has_mlo = bool(re.search(r"\b(MLO|MAIN\s*LUGS?)\b", txt))
        has_mcb = bool(re.search(r"\b(MCB|MAIN\s*(BREAKER|DEVICE))\b", txt))
        # If both appear somewhere on the page, defer to the chosen MAIN token later
        if has_mlo and has_mcb:
            return None
        if has_mlo:
            return "MLO"
        if has_mcb:
            return "MCB"
        return None

    def _simple_name_from_top(self, lines) -> Optional[dict]:
        """
        Prefer the top 1-2 lines. If a label word like PANEL/PANELBOARD/BOARD appears,
        choose the FIRST short alphanum token to its RIGHT on the same line.
        Else fall back to the first short alphanum token on those lines.
        """
        import re
        if not lines:
            return None

        LABEL_WORDS = {"PANEL", "PANELBOARD", "BOARD", "PNL"}
        top = lines[: min(2, len(lines))]

        def _mk_val(ln, tok_text, tok_rect=None, conf_hint=0.65, shape=0.95, ctx=0.30):
            # place box on the token if provided; else use whole line
            if tok_rect is None:
                x1,y1,x2,y2 = ln["rect"]
            else:
                x1,y1,x2,y2 = tok_rect
            return {
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "xc":0.5*(x1+x2),"yc":0.5*(y1+y2),
                "conf":conf_hint,"text":tok_text,
                "shape":shape,"ctx":ctx
            }

        # pass 1: “LABEL  ID” on same line → pick ID
        for ln in top:
            toks = ln["tokens"]  # list of (rect, text, conf)
            for i,(r,t,c) in enumerate(toks):
                up = (t or "").strip().upper().strip(":")
                if up in LABEL_WORDS:
                    # scan rightwards for the first short, clean ID
                    for j in range(i+1, len(toks)):
                        (r2, t2, c2) = toks[j]
                        up2 = (t2 or "").strip().upper().strip(":")
                        if up2 and up2 not in self._NAME_STOPWORDS and re.fullmatch(r"[A-Z0-9][A-Z0-9._\-\/]{0,12}", up2):
                            return _mk_val(ln, up2, tok_rect=r2, conf_hint=float(c2 or 0.7), shape=0.96, ctx=0.35)

        # pass 2: “PANEL[: ]ID” anywhere on the top lines
        for ln in top:
            m = re.search(r"\bPANEL(?:BOARD)?\b\s*:?\s*([A-Z0-9][A-Z0-9._\-\/]{1,12})", ln["text"].upper())
            if m:
                tok = m.group(1)
                if tok and tok not in self._NAME_STOPWORDS:
                    return _mk_val(ln, tok)

        # pass 3: first short alphanum token on those lines
        for ln in top:
            for (r,t,conf) in ln["tokens"]:
                up = (t or "").strip().upper().strip(":")
                if up and up not in self._NAME_STOPWORDS and re.fullmatch(r"[A-Z0-9][A-Z0-9._\-\/]{0,12}", up):
                    return _mk_val(ln, up, tok_rect=r, conf_hint=float(conf or 0.6), shape=0.92, ctx=0.20)

        return None

    def _label_affinity(self, role: str, it: dict, labels_map: dict) -> float:
        import math
        Ls = labels_map.get(role, [])
        if not Ls:
            return 0.0
        def dist(a, b): return math.hypot(a["xc"]-b["xc"], a["yc"]-b["yc"])
        dmin = min(dist(it, L) for L in Ls)
        return math.exp(-dmin / self._SIGMA_PX)

    # ---------- Collect label candidates ----------
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

    # ---------- Collect value candidates (shape-first) ----------
    def _collect_value_candidates(self, items: list) -> dict:
        import re
        out = {k: [] for k in ("NAME","VOLTAGE","BUS","MAIN","AIC")}
        heights = [abs(it2["y2"] - it2["y1"]) for it2 in items] or [1]
        med_h = float(np.median(heights)) if hasattr(np, "median") else (sum(heights) / len(heights))

        _VOLTY_WORDS = {"V", "VOLTS", "VOLT", "VOLTAGE", "WYE", "DELTA", "PHASE", "PH", "WIRES", "WIRE", "Ø"}

        for it in items:
            raw = str(it["text"] or "")
            txt = raw.upper()
            conf = float(it["conf"])
            x1,y1,x2,y2,xc,yc = it["x1"],it["y1"],it["x2"],it["y2"],it["xc"],it["yc"]

            # VOLTAGE (pairs or single; tolerate OCR typos and missing slash)
            txtN = self._normalize_voltage_text(txt)
            pair = re.search(r'\b([1-6]\d{2,3})\s*[YV]?[\/]?\s*([1-6]?\d{2,3})\s*V?\b', txtN)
            single = re.search(r'(?<!\d)([1-6]\d{2,3})(?!\d)', txtN)  # 3–4 digits, no trailing V required
            if pair or single:
                shape = 0.92 if pair else 0.62
                ctx = 0.15 if re.search(r'\b(WYE|DELTA|PH|PHASE|Ø|VOLT|VOLTS|V)\b', txtN) else 0.0
                out["VOLTAGE"].append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"xc":xc,"yc":yc,
                                       "conf":conf,"text":raw,"shape":min(1.0,shape),"ctx":min(1.0,ctx)})

            # BUS/MAIN (accept "###A", "### A", and bare "###" when in rating ranges)
            m_with_unit = re.search(r"\b([1-9]\d{1,3})\s*(A\.?|AMPS?\.?)\b", txt)  # 60..9999 A w/ unit
            m_bare_num  = re.search(r"(?<!\d)([1-9]\d{1,3})(?!\d)", txt)           # bare 2–4 digits

            cand = None
            if m_with_unit:
                n = int(m_with_unit.group(1))
                if 60 <= n <= 4000:
                    cand = {"x1":x1,"y1":y1,"x2":x2,"y2":y2,"xc":xc,"yc":yc,"conf":conf,
                            "text":raw, "shape":0.80, "ctx":0.12}
            elif m_bare_num:
                n = int(m_bare_num.group(1))
                # allow bare 60..1200; rely on label affinity to disambiguate from AIC/others
                if 60 <= n <= 1200:
                    # small score because it's unlabeled; still promotable by BUS fallback
                    cand = {"x1":x1,"y1":y1,"x2":x2,"y2":y2,"xc":xc,"yc":yc,"conf":conf,
                            "text":raw, "shape":0.68, "ctx":0.04}

            if cand is not None:
                out["BUS"].append(dict(cand))
                out["MAIN"].append(dict(cand))

            # AIC (10k..100k) and KA forms: 65kA, 65 kA
            # Normalize spaces for kA form matching
            t_nos = txt.replace(" ", "")
            mk = re.search(r"\b(\d{2,3})KA\b", t_nos)
            if mk:
                val_ka = int(mk.group(1))
                if 10 <= val_ka <= 100:
                    shape = 0.90
                    ctx = 0.10
                    out["AIC"].append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"xc":xc,"yc":yc,
                                       "conf":conf,"text":raw,"shape":shape,"ctx":ctx})
            else:
                AIC = re.search(r"\b(\d{2,3}[,]?\d{3})\s*(?:A|KA)?\b", txt)  # e.g. 65000, 65,000A
                if AIC:
                    val = int(AIC.group(1).replace(",", ""))
                    if 10000 <= val <= 100000:
                        shape = 0.85 + (0.10 if "," in AIC.group(1) else 0.0)
                        ctx = 0.08 if re.search(r"\b(SYMMETRICAL|AIC|A\.?\s*I\.?\s*C\.?|SCCR)\b", txt) else 0.0
                        out["AIC"].append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"xc":xc,"yc":yc,
                                           "conf":conf,"text":raw,"shape":min(1.0,shape),"ctx":ctx})

            # NAME candidates (short, alphanum, higher & larger)
            up = raw.strip().upper().strip(":")
            if not up or up in self._NAME_STOPWORDS or up in _VOLTY_WORDS:
                pass
            else:
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

        # De-dup by (role,text,near-x)
        for role in out.keys():
            merged = []
            for c in sorted(out[role], key=lambda d: (d["y1"], d["x1"])):
                if merged and c["text"] == merged[-1]["text"] and abs(c["xc"]-merged[-1]["xc"]) < 18:
                    if c["conf"] > merged[-1]["conf"]: merged[-1] = c
                else:
                    merged.append(c)
            out[role] = merged
        return out

    # ---------- Scoring & pick ----------
    def _score_candidates(self, role: str, cands: list, labels_map: dict, band_top: int, main_mode: Optional[str] = None) -> list:
        import math
        W = dict(self._WEIGHTS[role])

        # Dynamic tuning based on main mode
        if role in ("BUS","MAIN"):
            if main_mode == "MLO":
                if role == "MAIN":
                    W["W_shape"] *= 0.70; W["W_lbl"] *= 1.20
                else:
                    W["W_shape"] *= 1.05; W["W_lbl"] *= 1.20
            elif main_mode == "MCB":
                W["W_shape"] *= 1.05; W["W_lbl"] *= 1.05
            else:
                W["W_lbl"] *= 1.20

        same_labels = labels_map.get(role, [])
        wrong_labels = labels_map.get("WRONG", [])

        def dist(a, b):
            return math.hypot(float(a["xc"] - b["xc"]), float(a["yc"] - b["yc"]))

        ranked = []
        for c in cands:
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

            TOTAL = (
                W["W_shape"] * float(c.get("shape", 0.0))
                + W["W_conf"]  * float(c.get("conf", 0.0))
                + W["W_lbl"]   * float(S_lbl)
                + W["W_side"]  * float(S_side)
                + W["W_ctx"]   * float(c.get("ctx", 0.0))
                - W["W_wrong"] * float(S_wrong)
                + W["W_y"]     * float(S_y)
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

        ranked.sort(key=lambda d: (-d["rank"], d["y1"], d["x1"]))
        return ranked

    def _pick_role(self, role: str, ranked: list) -> dict | None:
        thr = self._THRESH.get(role, 0.5)
        return ranked[0] if ranked and ranked[0]["rank"] >= thr else None

    def _normalize_voltage_text(self, s: str) -> str:
        """
        Make OCR-y voltage strings matchable:
        - fix I/l between digits -> '/'
        - fix letter O between digits -> '0'
        - collapse punctuation around the pair
        - allow forms like '208Y120V' (missing slash)
        """
        u = (s or "").upper()

        # Replace I or l BETWEEN digits with '/'
        u = re.sub(r'(?<=\d)[IL](?=\d)', '/', u)

        # Replace letter O BETWEEN digits with zero
        u = re.sub(r'(?<=\d)O(?=\d)', '0', u)

        # Normalize common separators/spaces
        u = u.replace(',', ' ')
        u = u.replace('.', ' ')
        u = re.sub(r'\s+', ' ', u)

        return u

    # ======= overlay & legacy helpers =======
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

    # ======= support methods (kept) =======
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
