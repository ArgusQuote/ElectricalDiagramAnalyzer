import sys
import os
import re

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PartNumberSelector.PartNumberBuilder import mccb, nqPanelboard, nfPanelboard, iLinePanelboard, breakerSelector, blanks, eFrameBreaker, Disconnect, transformer as TransformerBuilder, loadcenter
import math 
 
INTERRUPTING_RATINGS = {
    'B': { 240:[25,65,100], 480:[18,35,65], 600:[14,18,25,65] },
    'H': { 240:[25,65,100,125,200], 480:[18,35,65,100,200], 600:[14,18,25,50,100] },
    'J': { 240:[25,65,100,125,200], 480:[18,35,65,100,200], 600:[14,18,25,50,100] },
    'Q': { 240:[10,25,65,100] },
    'L': { 240:[25,65,100,125,200], 480:[18,35,65,100,200], 600:[14,18,25,50,100] },
    'LL':{ 240:[42,65], 480:[30,35], 600:[22,25] },
    'M': { 240:[65,100], 480:[35,65], 600:[18,25] },
    'P': { 240:[65,100,125], 480:[35,65,50,100], 600:[18,25,50] },
    'R': { 240:[65,100,125], 480:[35,65,100], 600:[18,25,50,65] },
}

DEFAULT_INTERRUPTING_RATING = 22
def apply_interrupting_rating_defaults(raw: dict) -> dict:
    """
    Apply default interrupting rating when not detected or set to NONE.
    Returns the modified dictionary with notes about defaults applied.
    """
    raw = raw.copy()
    notes = raw.setdefault("_notes", [])
    
    # Check if interrupting rating is missing, None, or "NONE"
    int_rating = raw.get("intRating")
    
    if int_rating is None or _is_none_token(str(int_rating)) or int_rating == "":
        raw["intRating"] = DEFAULT_INTERRUPTING_RATING
        notes.append(f"No interrupting rating detected - system defaulted to {DEFAULT_INTERRUPTING_RATING}kAIC")
    
    return raw

HJ_2P_ALLOWED = {
    "H": {
        "D": {
            "AB": {15,20,25,30,35,40,45,50,60,70,80,90,100,110,125,150},
            "AC": {15,20,25,30,35,40,45,50,60,70,80,100,110,125,150},
            "BA": {15,25,30,35,45,60,90,110,125},
            "BC": {15,20,30,35,40,50,60,70,80,90,100,125,150},
            "CA": {15,20,30,35,40,45,50,60,70,80,100,110,125,150},
            "CB": {15,25,35,45,50,60,80,90,100,110,125},
        },
        "G": {
            "AB": {15,20,30,35,40,45,50,60,70,80,90,100,110,125,150},
            "AC": {15,20,25,30,35,40,45,50,60,70,80,90,100,125,150},
            "BA": {20,30,35,40,60,70,100,125,150},
            "BC": {15,20,30,40,45,50,60,80,100,110,125,150},
            "CA": {15,20,25,30,35,40,60,80,125,150},
            "CB": {20,25,30,35,40,45,70,80,90,100,110,125},
        },
        "J": {
            "AB": {15,20,25,30,35,40,45,50,60,80,90,100,110,125,150},
            "AC": {15,20,25,30,35,40,50,60,70,80,90,100,110,125,150},
            "BA": {15,20,30,35,60},
            "BC": {15,20,30,35,40,45,50,60,70,80,90,100,125,150},
            "CA": {15,20,25,40,45,50,60,70,80,100,150},
            "CB": {20,60,100,150},
        },
        "L": {
            "AB": {15,20,25,30,35,40,45,50,60,70,80,90,100,110,125,150},
            "AC": {15,20,35,45,60,70},
            "BA": {25,30,35,40,45,50,60,70,90,100,110,125,150},
            "BC": {15,20,25,30,35,40,45,70,110,125},
            "CA": {20,25,35,40,45,60,70,80,90,100,110,125,150},
            "CB": {15,20,25,30,35,40,45,50,60,70,80,90,100,110,125,150},
        },
    },
    "J": {
        "D": {
            "AB": {150,175,200,225,250},
            "AC": {150,200,225,250},
            "BC": {175,200,225,250},
            "CA": {225,250},
        },
        "G": {
            "AB": {150,175,200,225,250},
            "AC": {150,175,200,225,250},
            "BC": {200},
        },
        "J": {
            "AB": {150,175,200,225,250},
            "AC": {150,175,200,225,250},
            "BA": {250},
            "BC": {200,225,250},
            "CA": {225,250},
            "CB": {225,250},
        },
        "L": {
            "AB": {150,175,200,225,250},
            "AC": {200,225},
            "BA": {150,175,200,225,250},
            "BC": {225},
            "CA": {150,175,200,225,250},
            "CB": {150,175,200,225,250},
        },
    },
}

_2P_BASE_ORDER = ["AB", "AC", "BA", "BC", "CA", "CB"]

_CANON_PREF = {
    ("AB","BA"): "AB",
    ("AC","CA"): "AC",
    ("BC","CB"): "BC",
}

def canonicalize_pairs(allowed_pairs: list[str]) -> list[str]:
    s = set(allowed_pairs)
    chosen = []

    for a,b in [("AB","BA"),("AC","CA"),("BC","CB")]:
        if a in s and b in s:
            chosen.append(_CANON_PREF[(a,b)])
        elif a in s:
            chosen.append(a)
        elif b in s:
            chosen.append(b)
    return chosen

def pair_to_phases(pair: str) -> tuple[str,str]:
    return pair[0], pair[1]

def pick_balanced_pair(phase_load: dict[str, float], pairs: list[str], weight: float) -> str:
    best_pair, best_score = None, None
    for p in pairs:
        a,b = pair_to_phases(p)
        la = phase_load[a] + weight
        lb = phase_load[b] + weight
        lc = phase_load[[k for k in ("A","B","C") if k not in (a,b)][0]]
        spread = max(la, lb, lc) - min(la, lb, lc)
        if best_score is None or spread < best_score:
            best_score, best_pair = spread, p
    return best_pair

def hj_allowed_pairs(frame: str, intr_letter: str, amps: int):
    table = HJ_2P_ALLOWED.get(frame, {}).get(intr_letter, {})
    allowed = [pair for pair in _2P_BASE_ORDER if amps in table.get(pair, set())]
    return allowed

def _lc_snap_spaces(requested: int) -> int:
    for v in [4, 8, 12, 16, 24, 32, 40, 48, 60, 80, 84, 120]:
        if v >= max(0, int(requested or 0)):
            return v
    return 120

def _lc_cover_style(trim_style: str, enclosure: str) -> str:
    t = str(trim_style or "").upper()
    if t == "FLUSH":   return "Flush"
    if t == "SURFACE": return "Surface"
    return "Included"

def _lc_type_of_main_from_raw(raw: dict) -> str:
    return "M" if "mainBreakerAmperage" in raw else "L"

def _prefer_qo(material: str | None) -> bool:
    return str(material or "").upper() == "COPPER"

def ui_defaults():
    return {
        "panelboards": {
            "bussing_material":         "ALUMINUM",   # or "COPPER"
            "allow_plug_on_breakers":   True,         # True = okay, False = only bolt-on
            "rating_type":              "FULLY_RATED",# or "SERIES_RATED"
            "allow_feed_thru_lugs":     True,         # True = include FTL by default
            "default_trim_style":       "FLUSH",      # or "SURFACE"
            "enclosure":                "NEMA1",      # or "NEMA3R"
            "allow_square_d_spd":       True,         # True allows SPDs, False will not give SPD part numbers
        },
        "transformers": {
            "winding_material":   "ALUMINUM",   # ALUMINUM | COPPER
            "temperature_rating": 150,          # 150 | 115 | 80
            "default_type":       "3PHASESTANDARD",       # "3PHASESTANDARD" is first attempt then "K13","1PHASESTANDARD","WATCHDOG","RESIN"
            "weathershield":      False,        # dry-type weathershield for EX/EXN families
            "mounting":           "FLOOR",      # FLOOR or WALL or CEILING
            # RESIN-only default "enclosure" lives in 'resin_enclosure'
            "resin_enclosure":    "3R",         # 3R | SS | 4X  (builder passes as 'coreMaterial' key for RESIN rules)
        },
        "disconnects": {
            "allow_littlefuse":         True,          # future: LF fuses allowed
            "default_switch_type":      "GENERAL_DUTY",# starting point; we’ll escalate as needed
            "default_enclosure":        "NEMA1",       # UI default; OCR can override
            "default_fusible":          True,          # default to fusible unless OCR says otherwise
            "default_ground_required":  True,          # always give ground kit if available
            "default_solid_neutral":    True,          # give neutral kit when allowed/available
        },
    }

def loadcenter_defaults_from_panel_defaults() -> dict:
    pb = ui_defaults()["panelboards"]
    return {
        "allowPlugOn":     pb["allow_plug_on_breakers"],  # True/False
        "material":        pb["bussing_material"],        # "ALUMINUM" | "COPPER"
        "trimStyle":       pb["default_trim_style"],      # "FLUSH" | "SURFACE"
        "enclosure":       pb["enclosure"],               # "NEMA1" | "NEMA3R"
        "panelRatingType": pb["rating_type"],             # "FULLY_RATED" | "SERIES_RATED"
    }

def _deep_merge(base: dict, override: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return base
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

_DEFAULT_KEY_ALIASES = {
    "material": "bussing_material",
    "allowPlugOn": "allow_plug_on_breakers",
    "panelRatingType": "rating_type",
    "trimStyle": "default_trim_style",
    "nema": "enclosure",
}

def _apply_aliases(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        key = _DEFAULT_KEY_ALIASES.get(k, k)
        out[key] = _apply_aliases(v) if isinstance(v, dict) else v
    return out

class DefaultResolver:
    """
    Layers:
      1) Built-ins from ui_defaults()
      2) Job-level overrides (payload['defaults'])
      3) Item-level overrides (item['defaults']) [optional]
    """
    def __init__(self, job_defaults: dict | None, item_defaults: dict | None):
        self.job = _apply_aliases(job_defaults or {})
        self.item = _apply_aliases(item_defaults or {})

    def resolve(self, domain: str) -> dict:
        base = ui_defaults().get(domain, {})
        merged = _deep_merge(base, self.job.get("*", {}))
        merged = _deep_merge(merged, self.job.get(domain, {}))
        merged = _deep_merge(merged, self.item.get("*", {}))
        merged = _deep_merge(merged, self.item.get(domain, {}))
        return merged

class BaseEngine:
    def __init__(self, name: str, attrs: dict, resolver: "DefaultResolver | None" = None):
        self.name = name
        self.attrs = attrs.copy()
        self._defaults = resolver or DefaultResolver({}, {})

    def process(self) -> dict:
        raise NotImplementedError

ENGINE_REGISTRY = {}

_NONE_STR = "NONE"

def _is_none_token(val) -> bool:
    try:
        return isinstance(val, str) and val.strip().upper() == _NONE_STR
    except Exception:
        return False

def _scrub_none_tokens(obj):
    if isinstance(obj, dict):
        return {k: _scrub_none_tokens(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_none_tokens(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_scrub_none_tokens(v) for v in obj)
    if _is_none_token(obj):
        return None
    return obj

def _safe_int(v, default=0):
    try:
        if v in (None, "", "NONE"):
            return default
        return int(float(str(v)))
    except Exception:
        return default

def register_engine(item_type):
    def decorator(cls):
        ENGINE_REGISTRY[item_type] = cls
        return cls
    return decorator
import faulthandler, signal, time
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)

def process_job(payload) -> dict:
    job_defaults = {}
    if isinstance(payload, dict) and "items" in payload:
        items = payload.get("items", [])
        job_defaults = payload.get("defaults", {}) or {}
    else:
        items = payload

    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return {"error": "Payload must be a list of items or {'items':[...], 'defaults':{...}}."}

    results = {}
    for idx, itm in enumerate(items):
        if not isinstance(itm, dict):
            results[f"item_{idx+1}"] = {"error": "Item must be a dict with keys: type, name, attrs"}
            continue

        top_notes = itm.get("notes") or itm.get("Notes")
        
        # Apply defaults AFTER scrubbing None tokens
        raw_attrs = itm.get("attrs", {}) or {}
        raw_attrs = _scrub_none_tokens(raw_attrs)

        if itm.get("type") == "panelboard":
            raw_attrs = apply_interrupting_rating_defaults(raw_attrs)

        itm = {
            "type":  itm.get("type"),
            "name":  itm.get("name"),
            "attrs": raw_attrs,
            "defaults": itm.get("defaults", {}) or {},
        }

        etype = itm.get("type")
        if not isinstance(etype, str) or not etype.strip():
            results[(itm.get("name") or f"item_{idx+1}")] = {"error": "Missing or invalid 'type'."}
            continue

        raw_name = str(itm.get("name", "") or "").strip()

        top_notes = itm.get("notes") or itm.get("Notes")
        if top_notes:
            if not isinstance(top_notes, list):
                top_notes = [str(top_notes)]
            raw_attrs.setdefault("_notes", [])
            raw_attrs["_notes"].extend([str(n) for n in top_notes if n is not None])

        name = "name not detected" if _is_none_token(raw_name) or raw_name == "" else raw_name

        EngineCls = ENGINE_REGISTRY.get(etype)
        if not EngineCls:
            results[name] = {"error": f"Unknown item type '{etype}'"}
            continue

        resolver = DefaultResolver(job_defaults, itm["defaults"])
        engine = EngineCls(name, itm["attrs"], resolver)

        try:
            out = engine.process()
        except Exception as e:
            # Never let one panel crash the whole batch
            results[name] = {"error": f"Engine error for this item: {e.__class__.__name__}: {e}"}
            continue

        # Respect per-item skip without killing the batch
        if isinstance(out, dict) and out.get("_skipped") is True:
            reason = out.get("reason", "Missing/invalid required attributes.")
            # Keep the skip local to this panel
            results[name] = {"Skipped": reason}
            continue

        results[name] = out
    return results

def _family_screen(raw: dict, prefer_plug_on: bool) -> tuple[bool, bool, dict | None]:
    det = raw.get("detected_breakers", []) or []
    if not det:
        return True, True, None

    voltage     = _safe_int(raw.get("voltage"), 0)
    rating_type = (raw.get("panelRatingType") or "FULLY_RATED").upper()
    panel_k     = _safe_int(raw.get("intRating"), 0)
    allow_plug  = bool(raw.get("allowPlugOn", prefer_plug_on))

    def nf_max_amp(poles: int) -> int:
        if poles == 1: return 70
        if poles in (2, 3): return 125
        return 0

    def nf_ok(poles: int, amps: int) -> bool:
        if poles == 1: return 15 <= amps <= 70
        if poles in (2, 3): return 15 <= amps <= 125
        return False

    def supported_poles(base: str) -> set[int]:
        return {
            "QO": {1,2,3}, "QOB": {1,2,3},
            "QO-VH": {1,2,3}, "QOB-VH": {1,2,3},
            "QOH": {1,2},
            "QH": {1,2,3}, "QHB": {1,2,3},
        }[base]

    def choose_nq_base(poles: int) -> str | None:
        if rating_type == "SERIES_RATED":
            ladder = (["QO","QO-VH","QH","QHB"] if allow_plug
                    else ["QOB","QOB-VH","QOH","QH","QHB"])
            for b in ladder:
                if poles in supported_poles(b):
                    return b
            return ladder[-1]

        desired_at_240 = panel_k if (voltage == 208 or voltage == 240) else panel_k
        if desired_at_240 >= 65:
            base = "QH"
        elif desired_at_240 >= 42:
            base = "QOH"
        elif (voltage in (208, 240)) and desired_at_240 >= 25:
            base = "QH"
        elif desired_at_240 >= 22:
            base = "QO-VH" if allow_plug else "QOB-VH"
        else:
            base = "QO" if allow_plug else "QOB"

        if poles not in supported_poles(base):
            for b in {
                "QOH":   ["QH","QHB"],
                "QO":    ["QO-VH","QH","QHB"],
                "QOB":   ["QOB-VH","QOH","QH","QHB"],
                "QO-VH": ["QH","QHB"],
                "QOB-VH":["QOH","QH","QHB"],
            }.get(base, []):
                if poles in supported_poles(b):
                    return b
        return base

    default_k = {"QO":10,"QOB":10,"QO-VH":22,"QOB-VH":22,"QOH":42,"QH":65,"QHB":65}

    allow_nq = True
    allow_nf = True

    routing = {
        "forcedFamily": None,     # will be set to "I-LINE" if both fail
        "trigger": None,          # the first breaker that forces the decision
        "nq": {"allowed": True, "reason": None, "maxAmp": None, "base": None},
        "nf": {"allowed": True, "reason": None, "maxAmp": None},
    }

    for br in det:
        poles = _safe_int(br.get("poles"), 0)
        amps  = _safe_int(br.get("amperage"), 0)

        # ---- NQ screening (dynamic max amp from breakerSelector tables) ----
        if allow_nq:
            v = 240 if voltage == 208 else voltage
            if v > 240:
                allow_nq = False
                routing["nq"] = {"allowed": False, "reason": f"Voltage {voltage}V not supported for NQ", "maxAmp": None, "base": None}
            else:
                base = choose_nq_base(poles)
                if base is None:
                    allow_nq = False
                    routing["nq"] = {"allowed": False, "reason": f"No valid NQ breaker family for {poles}P", "maxAmp": None, "base": None}
                else:
                    req_k = panel_k if rating_type == "FULLY_RATED" else default_k.get(base, panel_k)
                    tmp = breakerSelector({
                        "breakerType": base,
                        "poles":       poles,
                        "amperage":    amps,
                        "interruptionRating": req_k,
                        "specialFeatures": "",
                        "iline":       False,
                    })
                    valid_amps = sorted(tmp.getAmperageOptionsByPoles().keys())
                    max_cap = (valid_amps[-1] if valid_amps else None)

                    ok = any(a >= amps for a in valid_amps)
                    if not ok:
                        allow_nq = False
                        routing["nq"] = {
                            "allowed": False,
                            "reason": f"Branch {poles}P {amps}A exceeds NQ capability for selected family",
                            "maxAmp": max_cap,
                            "base": base,
                        }

        # ---- NF screening (hard max; matches your nf_ok rules) ----
        if allow_nf:
            ok = nf_ok(poles, amps)
            if not ok:
                allow_nf = False
                routing["nf"] = {
                    "allowed": False,
                    "reason": f"Branch {poles}P {amps}A exceeds NF capability",
                    "maxAmp": nf_max_amp(poles),
                }

        # record the first breaker that eliminates BOTH
        if (not allow_nq) and (not allow_nf) and routing["trigger"] is None:
            routing["trigger"] = {"poles": poles, "amperage": amps}
            break

    if not (allow_nq or allow_nf):
        routing["forcedFamily"] = "I-LINE"
        return False, False, routing

    return allow_nq, allow_nf, routing

def _routing_bump_message(raw: dict) -> str | None:
    routing = raw.get("_routing") or {}
    if routing.get("forcedFamily") != "I-LINE":
        return None

    t = routing.get("trigger") or {}
    poles = t.get("poles")
    amps  = t.get("amperage")
    if not poles or not amps:
        return None

    nq = routing.get("nq") or {}
    nf = routing.get("nf") or {}

    nq_max  = nq.get("maxAmp")
    nf_max  = nf.get("maxAmp")
    nq_base = nq.get("base")

    parts = []
    parts.append(f"A branch breaker was detected at {amps}A ({poles}P).")

    # NQ detail
    if nq.get("allowed") is False:
        if nq_max is not None:
            base_txt = f" (family {nq_base})" if nq_base else ""
            parts.append(f"NQ limit for {poles}P is {nq_max}A{base_txt}.")
        elif nq.get("reason"):
            parts.append(f"NQ rejected: {nq['reason']}.")

    # NF detail
    if nf.get("allowed") is False:
        if nf_max is not None:
            parts.append(f"NF limit for {poles}P is {nf_max}A.")
        elif nf.get("reason"):
            parts.append(f"NF rejected: {nf['reason']}.")

    parts.append("System bumped panel selection to I-LINE.")
    return " ".join(parts)

@register_engine("panelboard")
class PanelboardEngine(BaseEngine):
    def process(self):
        defaults = self._defaults.resolve("panelboards")

        raw = self.attrs.copy()

        notes = raw.setdefault("_notes", [])
        MIN_BREAKER_AMP = 15
        MAX_BREAKER_AMP = 1200
        
        self._merge_gfi_counts_into_detected(raw)

        # --- sanitize & coalesce detected breakers (fix double-count) ---
        if isinstance(raw.get("detected_breakers"), list):
            validated = []
            for i, b in enumerate(raw["detected_breakers"]):
                if not isinstance(b, dict):
                    notes.append("A breaker was skipped: not a valid data structure.")
                    continue

                poles = _safe_int(b.get("poles"), 0)
                amps  = _safe_int(b.get("amperage"), 0)
                cnt   = _safe_int(b.get("count"), 1)
                feats = (b.get("specialFeatures") or "").strip()

                bad_fields = []
                if poles not in (1, 2, 3): bad_fields.append("poles")
                if amps  <= 0:              bad_fields.append("amperage")
                if cnt   <= 0:              bad_fields.append("count")
                if bad_fields:
                    notes.append(f"A breaker was skipped: missing or invalid {', '.join(bad_fields)}.")
                    continue

                # Standard amp window only
                if amps < MIN_BREAKER_AMP or amps > MAX_BREAKER_AMP:
                    notes.append(f"A breaker was skipped: detected amperage {amps}A outside supported 15-1200A.")
                    continue

                # Append exactly once after all checks
                validated.append({
                    "poles": poles,
                    "amperage": amps,
                    "count": cnt,
                    "specialFeatures": feats,
                })

            # Coalesce identical rows to avoid upstream dupes
            coalesced = {}
            for br in validated:
                key = (br["poles"], br["amperage"], br["specialFeatures"])
                coalesced[key] = coalesced.get(key, 0) + br["count"]

            raw["detected_breakers"] = [
                {"poles": k[0], "amperage": k[1], "specialFeatures": k[2], "count": c}
                for k, c in coalesced.items()
            ]

        # --- required attrs ---
        missing = []
        if raw.get("amperage") is None:
            missing.append("amperage")
        if raw.get("voltage") is None:
            missing.append("voltage")
        if raw.get("spaces")  is None:
            missing.append("spaces")

        is_potential_lc = (not raw.get("spd")) and _safe_int(raw.get("voltage"), 0) == 120
        # Only require intRating if it's not a loadcenter candidate AND it's still None after defaults were applied
        if not is_potential_lc and raw.get("intRating") in (None, ""):
            missing.append("intRating")

        if raw.get("mainBreakerAmperage") in (None, "", 0):
            raw.pop("mainBreakerAmperage", None)

        if missing:
            if set(missing) == {"intRating"} and is_potential_lc:
                pass
            else:
                return {"_skipped": True, "reason": f"Missing required attributes: {', '.join(missing)}"}

        # --- legacy key shims ---
        if "rating_type" in raw and "panelRatingType" not in raw:
            raw["panelRatingType"] = raw["rating_type"]
        if "allow_plug_on_breakers" in raw and "allowPlugOn" not in raw:
            raw["allowPlugOn"] = raw["allow_plug_on_breakers"]
        if "bussing_material" in raw and "material" not in raw:
            raw["material"] = raw["bussing_material"]
        if "trim_style" in raw and "trimStyle" not in raw:
            raw["trimStyle"] = raw["trim_style"]
        if "internal_spd" in raw and "spd" not in raw:
            raw["spd"] = raw["internal_spd"]

        # ---- defaults ----
        raw.setdefault("material",         defaults["bussing_material"])
        raw.setdefault("allowPlugOn",      defaults["allow_plug_on_breakers"])
        raw.setdefault("panelRatingType",  defaults["rating_type"])
        raw.setdefault("feedThruLugs",     defaults["allow_feed_thru_lugs"])
        raw.setdefault("trimStyle",        defaults["default_trim_style"])
        raw.setdefault("enclosure",        defaults["enclosure"])
        raw["_allowSqdSpd"] = bool(defaults.get("allow_square_d_spd", True))
        if not raw["_allowSqdSpd"]:
            raw.pop("spd", None)

        raw["material"]        = str(raw["material"]).upper()
        raw["trimStyle"]       = str(raw["trimStyle"]).upper()
        raw["panelRatingType"] = str(raw["panelRatingType"]).upper()
        raw["enclosure"]       = str(raw["enclosure"]).upper()

        # --- hard-correct +2 space detection glitch ---
        _spaces = _safe_int(raw.get("spaces"), 0)
        spaces_corrections = {
            20: 18,  # should be 18
            32: 30,  # should be 30
            44: 42,  # should be 42
            56: 54,  # should be 54
            68: 66,  # should be 66
            74: 72,  # should be 72
            86: 84,  # should be 84
        }
        if _spaces in spaces_corrections:
            corrected = spaces_corrections[_spaces]
            raw["spaces"] = corrected
            notes.append(f"Corrected detected spaces: {_spaces} → {corrected} (common OCR off-by-2).")

        # 1) Amperage ceiling for any branch = min(bus, main if present)
        bus_amps  = _safe_int(raw.get("amperage"), 0)
        mb_amps   = _safe_int(raw.get("mainBreakerAmperage"), 0)
        limit_amps = bus_amps if mb_amps in (0, None) else min(bus_amps, mb_amps)

        if limit_amps > 0 and isinstance(raw.get("detected_breakers"), list):
            kept = []
            over_limit_found = False
            for i, br in enumerate(raw["detected_breakers"]):
                b_amps = _safe_int(br.get("amperage"), 0)
                if b_amps > limit_amps:
                    notes.append(
                        f"A breaker was rejected: requested {b_amps}A exceeds "
                        f"panel/main limit of {limit_amps}A."
                    )
                    over_limit_found = True
                    continue
                kept.append(br)
            raw["detected_breakers"] = kept
            if over_limit_found:
                # single consolidated flag/note that propagates into the final panel result
                notes.append("Detected breakers exceeded panel/main rating — user review required.")
                raw["_needs_user_review"] = True

        # 2) Spaces sanity: compute BOTH classic slot math and I-LINE estimated spaces
        panel_spaces = _safe_int(raw.get("spaces"), 0)
        if panel_spaces > 0 and isinstance(raw.get("detected_breakers"), list):
            total_slots = 0
            for br in raw.get("detected_breakers", []):
                total_slots += max(0, _safe_int(br.get("poles"), 0) * _safe_int(br.get("count"), 1))

            iline_est = self._estimate_iline_required_spaces(
                raw.get("detected_breakers", []),
                voltage=_safe_int(raw.get("voltage"), 0),
            )

            # Choose the most forgiving interpretation; if neither fits, flag.
            candidates = [v for v in (total_slots, iline_est) if v and v > 0]
            needed = min(candidates) if candidates else 0
            if panel_spaces > 0 and needed > panel_spaces:
                raw["_overcapacity_spaces"] = int(needed - panel_spaces)
                raw.setdefault("_notes", []).append(
                    "too many breakers detected. user review required."
                )

        # 1-PHASE FAST-PATH: Try HOM/QO loadcenters for '120'
        lc_candidate = self._try_build_loadcenter_first(raw)
        if isinstance(lc_candidate, dict):
            return lc_candidate
        # else: fall back to the existing panelboard logic below

        # Use the presence of mainBreakerAmperage to determine main breaker panelboard
        has_mb = raw.get("mainBreakerAmperage") not in (None, "", 0)
        if has_mb:
            raw["typeOfMain"] = "MAIN BREAKER"
            return self._build_main_breaker(raw)
        else:
            raw.pop("mainBreakerAmperage", None)
            raw["typeOfMain"] = "MAIN LUG"
            return self._build_main_lug(raw)
        
    @staticmethod
    def _merge_gfi_counts_into_detected(raw: dict) -> None:
        gfi_counts = raw.get("gfiBreakerCounts") or raw.get("gfi_breaker_counts") or {}
        if not isinstance(gfi_counts, dict) or not gfi_counts:
            return

        det = raw.setdefault("detected_breakers", [])
        if not isinstance(det, list):
            det = []
            raw["detected_breakers"] = det

        for key, cnt in gfi_counts.items():
            m = re.match(r"(\d+)P_(\d+)A", str(key))
            if not m:
                continue
            poles = int(m.group(1))
            amps  = int(m.group(2))
            cnt   = _safe_int(cnt, 0)
            if cnt <= 0:
                continue
            det.append({"poles": poles, "amperage": amps, "count": cnt, "specialFeatures": "GFI"})

    def _nearest_iline_bucket(self, il, mode: str, amps: int, spaces: int, enclosure: str, material: str, trim: str):
        if amps <= 250:
            series_order = ["HCJ", "HCP", "HCR-U"]
        elif amps <= 800:
            series_order = ["HCP", "HCR-U"]
        else:
            series_order = ["HCR-U"]

        def scan_cfg(cfg, series_name):
            best = None
            for key, val in cfg.items():
                # key layouts: (type, A, S, ENCL, MAT[, TRIM[, SE]])
                type2, a2, s2, encl2, mat2 = key[:5]
                trim2 = key[5] if len(key) >= 6 else None
                if type2 != mode: 
                    continue
                if encl2 != enclosure: 
                    continue
                if enclosure != "NEMA3R" and trim and trim2 and trim2 != trim:
                    continue

                # - If user asked COPPER, only accept copper rows
                # - If user asked ALUMINUM:
                #    HCJ must be ALUMINUM
                #    HCP <= 600A must be ALUMINUM; >600A either is fine
                if material == "COPPER" and mat2 != "COPPER":
                    continue
                if material == "ALUMINUM":
                    if series_name == "HCJ" and mat2 != "ALUMINUM":
                        continue
                    if series_name == "HCP" and a2 <= 600 and mat2 != "ALUMINUM":
                        continue

                if a2 < amps or s2 < spaces:
                    continue

                cand = (a2 - amps, s2 - spaces, a2, s2, mat2)
                if best is None or cand < best:
                    best = cand
            if best:
                diffA, diffS, a2, s2, mat2 = best
                return (series_name, a2, s2, mat2)
            return None

        for series in series_order:
            cfg = (il.HCJ_configs if series == "HCJ" else
                il.HCP_configs if series == "HCP" else
                il.HCRU_configs)
            hit = scan_cfg(cfg, series)
            if hit:
                return hit

        if "HCR-U" not in series_order:
            series_order.append("HCR-U")
        cfg = il.HCRU_configs
        best_any = None
        for key, val in cfg.items():
            type2, a2, s2, encl2, mat2 = key[:5]
            trim2 = key[5] if len(key) >= 6 else None
            if type2 != mode or encl2 != enclosure:
                continue
            if enclosure != "NEMA3R" and trim and trim2 and trim2 != trim:
                continue
            if a2 < amps or s2 < spaces:
                continue
            cand = (a2 - amps, s2 - spaces, a2, s2, mat2)
            if best_any is None or cand < best_any:
                best_any = cand
        if best_any:
            _, _, a2, s2, mat2 = best_any
            return ("HCR-U", a2, s2, mat2)

        # truly nothing return a sane default that will still build
        return ("HCR-U", max(amps, 1200), max(spaces, 60), "ALUMINUM")

    def _iline_snap_spaces(self, il, series: str, mode: str, enclosure: str, material: str, trim: str, requested: int) -> int:
        requested = min(108, max(0, int(requested or 0)))
        enclosure = str(enclosure or "").upper()
        material  = str(material  or "").upper()
        trim      = str(trim      or "").upper()

        cfg = il.HCJ_configs if series == "HCJ" else il.HCP_configs if series == "HCP" else il.HCRU_configs
        valid = set()
        for key, val in cfg.items():
            type2, a2, s2, encl2, mat2 = key[:5]
            trim2 = key[5] if len(key) >= 6 else None
            if type2 != mode or encl2 != enclosure:
                continue
            if material == "COPPER" and mat2 != "COPPER":
                continue
            if material == "ALUMINUM":
                if series == "HCJ" and mat2 != "ALUMINUM":
                    continue
                if series == "HCP" and a2 <= 600 and mat2 != "ALUMINUM":
                    continue
            if enclosure != "NEMA3R" and trim and trim2 and trim2 != trim:
                continue
            valid.add(int(s2))
        if not valid:
            return requested
        snapped = min((s for s in valid if s >= requested), default=max(valid))
        return min(108, snapped)

    def _safe_generate_iline(self, panelAttrs):
        il = iLinePanelboard()
        out = il.generateILinePanelboardPartNumber(panelAttrs)
        if isinstance(out, str) and out.startswith("Invalid I-Line configuration"):
            series, bestA, bestS, bestMat = self._nearest_iline_bucket(
                il,
                mode=panelAttrs.get("typeOfMain", "MAIN LUG"),
                amps=_safe_int(panelAttrs.get("amperage"), 0),
                spaces=_safe_int(panelAttrs.get("spaces"), 0),
                enclosure=str(panelAttrs.get("enclosure","")).upper(),
                material=str(panelAttrs.get("material","")).upper(),
                trim=str(panelAttrs.get("trimStyle","")).upper()
            )
            panelAttrs["panelType"] = series
            panelAttrs["ilinePanelType"] = series
            panelAttrs["amperage"] = bestA
            panelAttrs["spaces"]   = bestS
            panelAttrs["material"] = bestMat
            out = il.generateILinePanelboardPartNumber(panelAttrs)
        return out

    def _largest_required_branch(self, raw: dict) -> int:
        brs = raw.get("detected_breakers", [])
        return max((_safe_int(b.get("amperage"), 0) for b in brs), default=0)

    def _compact_iline_blanks(self, iline_blanks: dict) -> dict | None:
        if not isinstance(iline_blanks, dict):
            return None
        items = []
        for it in iline_blanks.get("Items", []):
            try:
                pn = it.get("Part Number")
                qn = _safe_int(it.get("Quantity (packs)", it.get("Quantity", 0)), 0)
                if pn and qn > 0:
                    items.append({"Part Number": pn, "Quantity": qn})
            except Exception:
                continue
        return {"Items": items} if items else None

    def _estimate_iline_required_spaces(self, detected_breakers: list, voltage: int) -> int:

        frame_rules = {
            "B":  {"2": 3,  "3": 4.5, "amps": [15,20,25,30,35,40,45,50,60,70,80,90,100,110,125]},
            "Q":  {"2": 3,  "3": 4.5, "amps": [70,80,90,100,110,125,150,175,200,225,250]},
            "H":  {"2": 4.5,"3": 4.5, "amps_m":[15,20,25,30,35,40,45,50,60,70,80,90,100,110,125,150], "amps_e":[60,100,150]},
            "J":  {"2": 4.5,"3": 4.5, "amps_m":[150,175,200,225,250], "amps_e":[250]},
            "LL": {"2": 6,  "3": 6,   "amps":[125,150,175,200,225,250,300,350,400]},
            "L":  {"2": 6,  "3": 6,   "amps":[250,400,600], "amps_e":[250,400,600]},
            "M":  {"2": 9,  "3": 9,   "amps":[400,600]},
            "P":  {"3": 9,             "amps":[250,400,600,800,1000,1200]},
            "R":  {"3": 15,            "amps":[1000,1200]},
        }

        electronic_only = {"M","P","R"}
        threepole_only  = {"P","R"}

        def first_fit_frame(poles: int, amps: int, rating_voltage: int) -> tuple[str, float]:
            """
            Return (frame_letter, width_inches) choosing the smallest frame that works.
            Voltage & pole constraints enforced. If none, returns (None, 0.0).
            """
            order = ["Q","H","J","LL","L","M","P","R"]
            for frame in order:
                if frame == "Q" and rating_voltage > 240:
                    continue
                if frame in threepole_only and poles != 3:
                    continue
                if frame == "LL" and poles == 2:
                    continue
                fr = frame_rules.get(frame)
                if not fr or str(poles) not in fr:
                    continue
                amps_list = fr.get("amps") or fr.get("amps_m") or fr.get("amps_e") or []
                ok = next((a for a in amps_list if a >= amps), None)
                if ok is None:
                    continue
                return frame, float(fr[str(poles)])
            return None, 0.0

        rating_voltage = 240 if voltage == 208 else voltage
        total_inches = 0.0

        for br in (detected_breakers or []):
            poles = _safe_int(br.get("poles"), 0)
            amps  = _safe_int(br.get("amperage"), 0)
            cnt   = _safe_int(br.get("count"), 1)
            if poles not in (2,3) or cnt <= 0:
                continue
            frame, width_in = first_fit_frame(poles, amps, rating_voltage)
            if frame is None:
                continue
            total_inches += width_in * cnt

        return int(math.ceil(total_inches / 1.5))

    def _try_build_loadcenter_first(self, raw: dict):
        if raw.get("spd"):
            return None
        try:
            pb = ui_defaults()["panelboards"]
            raw.setdefault("allowPlugOn",     pb["allow_plug_on_breakers"])
            raw.setdefault("material",        pb["bussing_material"])
            raw.setdefault("trimStyle",       pb["default_trim_style"])
            raw.setdefault("enclosure",       pb["enclosure"])
            raw.setdefault("panelRatingType", pb["rating_type"])
        except Exception:
            pass

        voltage = _safe_int(raw.get("voltage"), 0)
        if voltage != 120:
            return None

        amps   = _safe_int(raw.get("amperage"), 0)
        spaces = _lc_snap_spaces(_safe_int(raw.get("spaces"), 0))
        if amps <= 0 or spaces <= 0:
            return None

        def _attempt_lc(family: str):
            attrs = {
                "loadCenterType": family,
                "enclosure": ("NEMA3R" if str(raw.get("enclosure","")).upper() == "NEMA3R" else "NEMA1"),
                "phasing": "1PHASE",
                "typeOfMain": _lc_type_of_main_from_raw(raw),
                "mainsRating": amps,
                "poleSpaces": spaces,
                "plugOnNeutral": bool(raw.get("allowPlugOn", True)),
                "coverStyle": _lc_cover_style(raw.get("trimStyle",""), str(raw.get("enclosure",""))),
                "valuePack": bool(raw.get("valuePack", False)),
                "groundBar": bool(raw.get("groundBar", False)),
                "specialApplication": "Standard",
                "quikGrip": bool(raw.get("quikGrip", False)),
                # QO is Copper bus; HOM is Aluminum bus
                "busMaterial": ("Copper" if family == "QO" else "Aluminum"),
            }
            try:
                built = loadcenter().generateLoadcenterPartNumber(attrs)
            except Exception:
                return None, None
            if isinstance(built, dict) and not ("Error" in built and built["Error"]):
                return built, attrs
            return None, None

        prefer_qo = _prefer_qo(raw.get("material"))
        family_order = ["QO", "Homeline"] if prefer_qo else ["Homeline", "QO"]

        built = None
        lc_attrs = None
        for fam in family_order:
            b, a = _attempt_lc(fam)
            if b:
                built, lc_attrs = b, a
                break
        if not built:
            return None

        series = "QO" if lc_attrs["loadCenterType"] == "QO" else "HOM"
        panel_like = {
            "Panel Family": "LOADCENTER",
            "Series": series,
            "spaces": lc_attrs["poleSpaces"],
        }
        if "Part Number" in built:
            panel_like["Interior/Box"] = built["Part Number"]
            panel_like["Cover"] = "Included"
        if "Box and Interior" in built:
            panel_like["Interior/Box"] = built["Box and Interior"]
        if "Cover" in built:
            panel_like["Cover"] = built["Cover"]
        if "Ground Bar Kit" in built:
            panel_like["Ground Bar Kit"] = built["Ground Bar Kit"]
        if "Message" in built:
            panel_like.setdefault("Notes", []).append(built["Message"])

        raw = dict(raw)
        raw["_allowSqdSpd"] = False

        panel_like = self._build_nq_branch_breakers(panel_like, raw, raw.get("detected_breakers", []))
        return panel_like

    def _build_main_lug(self, raw):
        panelAttrs = raw.copy()
        panelAttrs["typeOfMain"] = raw["typeOfMain"]
        panelAttrs.pop("mainBreakerAmperage", None)
#        panelAttrs.pop("intRating", None)
        panelAttrs.pop("tripFunction", None)
        panelAttrs.pop("poles", None)
        panelAttrs.pop("phasing", None)
        panelAttrs.pop("grounding", None)        
        panelAttrs["enclosure"] = str(panelAttrs.get("enclosure","")).upper()
        panelAttrs["trimStyle"] = str(panelAttrs.get("trimStyle","")).upper()
        panelAttrs["material"]  = str(panelAttrs.get("material","")).upper()
        panelAttrs["panelRatingType"] = str(panelAttrs.get("panelRatingType","")).upper()

        voltage = _safe_int(panelAttrs.get("voltage"), 0)
        phase = "1PHASE" if voltage == 120 else "3PHASE"
        lookup_voltage = 208 if (phase=="3PHASE" and voltage==240) else voltage
        amps   = _safe_int(panelAttrs.get("amperage"), 0)
        spaces = _safe_int(panelAttrs.get("spaces"), 0)
        panelAttrs["phase"] = phase

        print("[DBG][LUG] phase=", phase,
            " voltage=", voltage,
            " lookup_voltage=", lookup_voltage,
            " spaces=", spaces,
            " amps=", amps)

        # NF (and generally >400A) cannot have feed-thru lugs
        if _safe_int(panelAttrs.get("amperage"), 0) > 400:
            panelAttrs["feedThruLugs"] = False

        auto_spaces = self._estimate_iline_required_spaces(
            raw.get("detected_breakers", []),
            voltage=_safe_int(panelAttrs.get("voltage"), 0),
        )

        screened_spaces = max(spaces, auto_spaces)

        allow_nq, allow_nf, routing = _family_screen(raw, prefer_plug_on=bool(raw.get("allowPlugOn", True)))
        raw["_routing"] = routing

        nq = nqPanelboard(); nf = nfPanelboard(); il = iLinePanelboard()

        candidates = []
        cost_rank = {"NQ": 0, "NF": 1, "I-LINE": 2}
        series_rank = {"HCJ": 0, "HCP": 1, "HCR-U": 2}

        for fam, builder in (("NQ", nq), ("NF", nf), ("I-LINE", il)):
            if fam == "NQ" and not allow_nq:
                continue
            if fam == "NF" and not allow_nf:
                continue

            if fam in ("NQ", "NF"):
                table = getattr(builder, "allowedConfigurationsLug", {})
                for key, val in table.items():
                    a, s, p, v = key
                    volts = v if isinstance(v, (list, tuple)) else (v,)
                    if not (p == phase and lookup_voltage in volts and a >= amps and s >= spaces):
                        continue
                    diffA, diffS = a - amps, s - spaces
                    candidates.append((
                        cost_rank[fam],    # 0 for NQ, 1 for NF
                        0,
                        diffA,
                        diffS,
                        fam,
                        None,
                        a,
                        s,
                        panelAttrs["material"]
                    ))

            elif fam == "I-LINE":
                if amps <= 250:
                    series_order = ["HCJ", "HCP", "HCR-U"]
                elif amps <= 800:
                    series_order = ["HCP", "HCR-U"]
                else:
                    series_order = ["HCR-U"]

                for series in series_order:
                    if series == "HCJ":
                        cfg = builder.HCJ_configs
                    elif series == "HCP":
                        cfg = builder.HCP_configs
                    elif series == "HCR-U":
                        cfg = builder.HCRU_configs
                    else:
                        continue

                    for key, val in cfg.items():
                        panel_type = val[0]
                        if panel_type != series:
                            continue

                        type2, a2, s2, encl2, mat2 = key[:5]
                        if type2 != "MAIN LUG" or encl2 != panelAttrs["enclosure"]:
                            continue

                        if panelAttrs["material"] == "COPPER" and mat2 != "COPPER":
                            continue
                        if panelAttrs["material"] == "ALUMINUM":
                            if series == "HCJ" and mat2 != "ALUMINUM":
                                continue
                            if series == "HCP" and a2 <= 600 and mat2 != "ALUMINUM":
                                continue

                        if panelAttrs["enclosure"] != "NEMA3R":
                            trim2 = key[5]
                            if trim2 != panelAttrs["trimStyle"]:
                                continue

                        if a2 < amps or s2 < screened_spaces:
                            continue

                        diffA, diffS = a2 - amps, s2 - screened_spaces
                        candidates.append((
                            cost_rank[fam],
                            series_rank[panel_type],
                            diffA,
                            diffS,
                            fam,
                            panel_type,
                            a2,
                            s2,
                            mat2
                        ))

                    if any(c[4] == "I-LINE" and c[5] == series for c in candidates):
                        break

        print("[DBG][LUG] candidates=",
            [(c[4], c[6], c[7]) for c in candidates])  # (family, matched_amps, matched_spaces)

        # --- Last-chance NQ for 208V & big interiors (84+ circuits) ---
        if not candidates:
            if (phase == "3PHASE" and lookup_voltage == 208 and spaces >= 84):
                table = getattr(nq, "allowedConfigurationsLug", {})
                for (a, s, p, v), _val in table.items():
                    volts = v if isinstance(v, (list, tuple)) else (v,)
                    if p == phase and (208 in volts or 240 in volts) and a >= amps and s >= spaces:
                        candidates.append((0, 0, a - amps, s - spaces, "NQ", None, a, s, panelAttrs["material"]))
                        break

        # pick best in this order: cost_rank, series_priority, diffA, diffS
        if candidates:
            candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
            _, _, _, _, family, chosen_series, best_amps, best_spaces, matched_material = candidates[0]
        else:
            if not candidates:
                print("[DBG][LUG] No NQ/NF match → falling to I-LINE")
            family      = "I-LINE"
            best_amps   = amps
            best_spaces = spaces
            if amps > 800 or spaces > 99:
                chosen_series = "HCR-U"
            elif amps > 250:
                chosen_series = "HCP"
            else:
                chosen_series = "HCJ"
            if chosen_series == "HCR-U":
                matched_material = "COPPER"
            else:
                matched_material = panelAttrs["material"]
        if family == "I-LINE":
            panelAttrs["ilinePanelType"] = chosen_series
            panelAttrs["panelType"] = "I-LINE"
            panelAttrs["material"]  = matched_material

            # Convert classic pole spaces → I-LINE spaces: ×1.5 then cap
            requested = _safe_int(raw.get("spaces"), 0)
            target_spaces = min(108, int(math.ceil(requested * 1.5)))

            # Snap to a real interior for this series/config
            panelAttrs["spaces"] = self._iline_snap_spaces(
                il,
                chosen_series,
                "MAIN LUG",
                panelAttrs["enclosure"],
                panelAttrs["material"],
                panelAttrs.get("trimStyle",""),
                target_spaces
            )

        panelAttrs["amperage"] = best_amps

        if family != "I-LINE":
            panelAttrs["spaces"] = best_spaces

        if family == "NQ":
            panel = nq.generateNqPanelboardPartNumber(panelAttrs)
            if isinstance(panel, str):
                return {"error": panel}
            allow_sqd_spd = bool(raw.get("_allowSqdSpd", True))
            spd_present = allow_sqd_spd and (bool(raw.get("spd")) or bool(panel.get("SPD")))
            panel["spaces"] = (
                max(0, int(panelAttrs.get("spaces", 0)) - 12) if spd_present
                else panelAttrs.get("spaces", 0))
            return self._build_nq_branch_breakers(panel, raw, raw.get("detected_breakers", []))
        
        elif family == "NF":
            attrs_nf = dict(panelAttrs)
            attrs_nf.pop("spd", None)            
            panel = nf.generateNfPanelboardPartNumber(attrs_nf)
            if isinstance(panel, str):
                return {"error": panel}
            panel["spaces"] = panelAttrs.get("spaces", 0)
            return self._build_nf_branch_breakers(
                panel,
                raw,
                raw.get("detected_breakers", [])
            )
        else:
            panel = self._safe_generate_iline(panelAttrs)
            if isinstance(panel, str):
                return {"error": panel}
            panel["spaces"] = panelAttrs.get("spaces", 0)

            if panelAttrs.get("_resize_note"):
                panel.setdefault("Notes", []).append(panelAttrs["_resize_note"])

        panel = self._build_iline_branch_breakers(panel, raw, raw.get("detected_breakers", []))

        if int(panelAttrs.get("spaces", 0)) == 108 and int(panel.get("_side_deficit_spaces", 0) or 0) > 0:
            panel.setdefault("Notes", []).append(
                "Requested breaker counts exceed the largest I-LINE interior (108 spaces). User review required."
            )

        msg = _routing_bump_message(raw)
        if msg:
            panel.setdefault("Notes", []).append(msg)

        return panel

    def _build_main_breaker(self, raw):
        panelAttrs = raw.copy()
        panelAttrs["typeOfMain"] = "MAIN BREAKER"
        panelAttrs.pop("mainBreakerAmperage", None)
#        panelAttrs.pop("intRating", None)
        panelAttrs.pop("tripFunction", None)
        panelAttrs.pop("poles", None)
        panelAttrs.pop("phasing", None)
        panelAttrs.pop("grounding", None)
        panelAttrs["enclosure"]       = str(panelAttrs.get("enclosure","")).upper()
        panelAttrs["trimStyle"]       = str(panelAttrs.get("trimStyle","")).upper()
        panelAttrs["material"]        = str(panelAttrs.get("material","")).upper()
        panelAttrs["panelRatingType"] = str(panelAttrs.get("panelRatingType","")).upper()

        bus_amps     = _safe_int(panelAttrs.get("amperage"), 0)
        breaker_amps = _safe_int(raw.get("mainBreakerAmperage"), 0)
        raw_ir       = _safe_int(raw.get("intRating"), 0)
        spaces       = _safe_int(panelAttrs.get("spaces"), 0)

        auto_spaces = self._estimate_iline_required_spaces(
            raw.get("detected_breakers", []),
            voltage=_safe_int(panelAttrs.get("voltage"), 0),
        )
        original_spaces = spaces
        spaces = max(spaces, auto_spaces)
        panelAttrs["spaces"] = spaces

        if spaces > original_spaces:
            panelAttrs["_resize_note"] = (
                f"Panel spaces increased from {original_spaces} to {spaces} "
                f"based on detected I-LINE branch breakers."
            )

        if _safe_int(panelAttrs.get("amperage"), 0) > 400:
            panelAttrs["feedThruLugs"] = False

        voltage     = _safe_int(panelAttrs.get("voltage"), 0)
        phase       = "1PHASE" if voltage == 120 else "3PHASE"
        lookup_voltage = voltage
        breaker_amps = _safe_int(raw.get("mainBreakerAmperage"), 0)
#        if "intRating" not in raw:
#            return {"error": "Missing interrupting rating for main breaker"}
        raw_ir = _safe_int(raw.get("intRating"), 0)
        panelAttrs["phase"] = phase
        print("[DBG][MB] phase=", phase,
            " voltage=", voltage,
            " lookup_voltage=", lookup_voltage,
            " spaces=", spaces,
            " bus_amps=", bus_amps)

        allow_nq, allow_nf, routing = _family_screen(raw, prefer_plug_on=bool(raw.get("allowPlugOn", True)))
        raw["_routing"] = routing

        nq_builder = nqPanelboard()
        nf_builder = nfPanelboard()
        families = []
        if allow_nq:
            families.append(("NQ", nq_builder))
        if allow_nf:
            families.append(("NF", nf_builder))
        cost_rank = {"NQ": 0, "NF": 1}

        candidates = []
        for fam, builder in families:
            for (minA, maxA, spc, phs, voltSpec), params in builder.allowedConfigurationsBreaker.items():
                volts = voltSpec if isinstance(voltSpec, (list, tuple)) else (voltSpec,)
                if (
                    minA <= bus_amps <= maxA
                    and spc >= spaces
                    and phs == phase
                    and lookup_voltage in volts
                ):
                    candidates.append((cost_rank[fam], maxA, fam, params))

        print("[DBG][MB] candidates=", [(c[2], c[1], c[3][0]) for c in candidates])

        if not candidates:
            # Last-chance NQ for 208V & 84+ circuits before I-LINE
            if (phase == "3PHASE" and lookup_voltage == 208 and spaces >= 84):
                for (minA, maxA, spc, phs, voltSpec), params in nq_builder.allowedConfigurationsBreaker.items():
                    volts = voltSpec if isinstance(voltSpec, (list, tuple)) else (voltSpec,)
                    if (minA <= bus_amps <= maxA) and (spc >= spaces) and (phs == phase) and (208 in volts or 240 in volts):
                        candidates.append((0, maxA, "NQ", params))
                        break
            if not candidates:
                print("[DBG][MB] No NQ/NF match → falling to I-LINE with",
                    {"amps": _safe_int(panelAttrs.get("amperage"), 0),
                    "spaces": _safe_int(panelAttrs.get("spaces"), 0),
                    "enclosure": panelAttrs.get("enclosure"),
                    "material": panelAttrs.get("material"),
                    "trimStyle": panelAttrs.get("trimStyle")})
                _amps = _safe_int(panelAttrs.get("amperage"), 0)
                _spaces = _safe_int(panelAttrs.get("spaces"), 0)
                chosen_series = "HCR-U" if (_amps > 800 or _spaces > 99) else ("HCP" if _amps > 250 else "HCJ")

                probe = {
                    "panelType": "I-LINE",
                    "ilinePanelType": chosen_series,
                    "spaces": _spaces,
                    "Interrupting Rating": f"{_safe_int(raw.get('intRating'), 0)}k",
                }
                probe_out = self._build_iline_branch_breakers(dict(probe), raw, raw.get("detected_breakers", []))
                raw2 = dict(raw)

                # Convert classic pole spaces → I-LINE spaces: ×1.5 then cap
                requested = _safe_int(raw.get("spaces"), 0)
                target_spaces = min(108, int(math.ceil(requested * 1.5)))

                # Snap to a real interior for the chosen series/config
                il = iLinePanelboard()
                raw2["spaces"] = self._iline_snap_spaces(
                    il,
                    chosen_series,
                    "MAIN BREAKER",
                    panelAttrs["enclosure"],
                    panelAttrs["material"],
                    panelAttrs.get("trimStyle",""),
                    target_spaces
                )

                panel = self._build_iline_breaker(raw2)
                panel = self._build_iline_branch_breakers(panel, raw2, raw2.get("detected_breakers", []))

                if int(raw2["spaces"]) == 108 and int(panel.get("_side_deficit_spaces", 0) or 0) > 0:
                    panel.setdefault("Notes", []).append(
                        "Requested breaker counts exceed the largest I-LINE interior (108 spaces). User review required."
                    )

                msg = _routing_bump_message(raw)
                if msg:
                    panel.setdefault("Notes", []).append(msg)

                return panel

        _, best_maxA, best_fam, best_params = min(candidates, key=lambda x: (x[0], x[1]))

        panelAttrs["spaces"]          = spaces
        full_frame_code               = best_params[2][0]
        panelAttrs["mainBreakerType"] = full_frame_code
        all_kits        = list(best_params[3:-1])
        barrier_kit     = best_params[-1]

        panelAttrs["amperage"] = bus_amps

        attrs_for_family = dict(panelAttrs)
        if best_fam == "NF":
            attrs_for_family.pop("spd", None)

        if best_fam == "NQ":
            panel_output = nq_builder.generateNqPanelboardPartNumber(attrs_for_family)
            if isinstance(panel_output, str):
                return {"error": panel_output}

            allow_sqd_spd = bool(raw.get("_allowSqdSpd", True))
            spd_present = allow_sqd_spd and (bool(raw.get("spd")) or bool(panel_output.get("SPD")))
            panel_output["spaces"] = (
                max(0, int(panelAttrs.get("spaces", 0)) - 12) if spd_present
                else panelAttrs.get("spaces", 0))

        if best_fam == "NF":
            panel_output = nf_builder.generateNfPanelboardPartNumber(attrs_for_family)
            if isinstance(panel_output, str):
                return {"error": panel_output}
            panel_output["spaces"] = panelAttrs.get("spaces", 0)

        if set(best_params[2]) == {"LA","LH"} or full_frame_code == "LL":
            rating_key = "LL"
        elif full_frame_code in INTERRUPTING_RATINGS:
            rating_key = full_frame_code
        else:
            rating_key = full_frame_code[0]

        rating_voltage = 240 if phase == "1PHASE" or voltage == 208 else voltage
        snapped_ir     = snap_int_rating(rating_key, rating_voltage, raw_ir)

        mccb_builder = mccb()

        intr_letter = None
        if rating_key in ("H","J"):
            intr_letter = mccb_builder.getInterruptingLetterHJ(rating_voltage, snapped_ir)

        if rating_key == "LL":
            trip = "MAGNETIC"
        elif rating_key == "H" and intr_letter == "R":
            trip = "ELECTRONIC"
        elif rating_key in ("H","J"):
            trip = raw.get("tripFunction", "MAGNETIC")
        else:
            trip = raw.get("tripFunction", "ELECTRONIC")

        default_poles = 2 if phase == "1PHASE" and rating_key == "Q" else 3
        poles_count   = raw.get("poles", default_poles)

        breaker_attrs = {
            "frame":        rating_key[0],
            "termination":  "LUG",
            "poles":        poles_count,
            "amperage":     breaker_amps,
            "voltage":      rating_voltage,
            "intRating":    snapped_ir,
            "tripFunction": trip,
            "grounding":    raw.get("grounding", False),
        }

        breaker_part = mccb_builder.generateMccbPartNumber(breaker_attrs)

        if isinstance(breaker_part, str) and breaker_part.startswith("Invalid interrupting rating"):
            next_ir = snap_int_rating(rating_key, rating_voltage, breaker_attrs["intRating"] + 1)
            breaker_attrs["intRating"] = next_ir
            breaker_part = mccb_builder.generateMccbPartNumber(breaker_attrs)

        if isinstance(breaker_part, str) and "Allowed:" in breaker_part:
            import re
            m = re.search(r"Allowed: \[([0-9,\s]+)\]", breaker_part)
            if m:
                allowed_vals = sorted(int(x) for x in m.group(1).split(","))
                if breaker_part.startswith("Invalid amperage"):
                    for a in allowed_vals:
                        if a >= breaker_amps:
                            breaker_attrs["amperage"] = a
                            break
                else:
                    for ir in allowed_vals:
                        if ir >= breaker_attrs["intRating"]:
                            breaker_attrs["intRating"] = ir
                            break
                breaker_part = mccb_builder.generateMccbPartNumber(breaker_attrs)

        chosen_full_frame = full_frame_code
        if isinstance(breaker_part, str) and breaker_part.startswith("Invalid amperage"):
            for alt_full in best_params[2]:
                letter = alt_full[0]
                if letter == breaker_attrs["frame"]:
                    continue
                breaker_attrs["frame"] = letter
                chosen_full_frame = alt_full
                cand = mccb_builder.generateMccbPartNumber(breaker_attrs)
                if not (isinstance(cand, str) and cand.startswith("Invalid")):
                    breaker_part = cand
                    break

        all_kits    = list(best_params[3:-1])
        barrier_kit = best_params[-1]

        if chosen_full_frame[0] == "Q":
            panel_output["Main Breaker Kit"] = all_kits[1]
        else:
            panel_output["Main Breaker Kit"] = all_kits[0]

        if isinstance(breaker_part, str) and breaker_part.startswith("Invalid"):
            panel_output["Main Breaker Error"] = breaker_part
        else:
            panel_output["Main Breaker"]     = breaker_part 
        if panelAttrs.get("_resize_note"):
            panel_output.setdefault("Notes", []).append(panelAttrs["_resize_note"])

        # Clean up hidden metadata
        panel_output.pop("Allowed MCCB Frames", None)
        panel_output.pop("_hiddenMainBreakerKits",  None)

        if best_fam == "NQ":
            panel_output = self._build_nq_branch_breakers(
                panel_output,
                raw,
                raw.get("detected_breakers", [])
            )
        elif best_fam == "NF":
            panel_output = self._build_nf_branch_breakers(
                panel_output,
                raw,
                raw.get("detected_breakers", [])
            )
        elif best_fam == "I-LINE":
            print("ILINE BRANCH BREAKER BUILDER TRIGGERED")
            panel_output = self._build_iline_branch_breakers(
                panel_output,
                raw,
                raw.get("detected_breakers", [])
            )    
        return panel_output
    
    def _build_iline_breaker(self, raw):   
        panelAttrs = raw.copy()
        panelAttrs["typeOfMain"] = "MAIN BREAKER"
        for f in ("mainBreakerAmperage","intRating","tripFunction","poles","phasing","grounding"):
            panelAttrs.pop(f, None)

        bus_amps  = _safe_int(panelAttrs.get("amperage"), 0)
        # CAP AT 108
        spaces = min(108, _safe_int(panelAttrs.get("spaces"), 0))
        il = iLinePanelboard()
        series = panelAttrs.get("panelType") or panelAttrs.get("ilinePanelType") or "HCJ"
        spaces = self._iline_snap_spaces(
            il,
            series,
            "MAIN BREAKER",
            panelAttrs.get("enclosure",""),
            panelAttrs.get("material",""),
            panelAttrs.get("trimStyle",""),
            spaces
        )
        panelAttrs["spaces"] = spaces
        encl     = panelAttrs.get("enclosure", "").upper()
        mat      = panelAttrs.get("material",  "").upper()
        trim     = panelAttrs.get("trimStyle",  "").upper()
        breaker_amps = _safe_int(raw.get("mainBreakerAmperage"), 0)

        il       = iLinePanelboard()
        candidates = []

        for key, val in il.allowedConfigurationsBreaker.items():
            if   len(key) == 7:
                _, a2, s2, encl2, mat2, trim2, se = key
            elif len(key) == 6:
                _, a2, s2, encl2, mat2, trim2 = key
                se = False
            elif len(key) == 5:
                _, a2, s2, encl2, mat2      = key
                trim2, se = None, False
            else:
                continue

            panel_type = val[0]

            if panel_type == "HCP-SU":
                continue

            if encl2 != encl or mat2 != mat:
                continue
            if trim and trim2 != trim:
                continue

            if a2 < bus_amps or s2 < spaces:
                continue

            if panel_type == "HCJ":
                if bus_amps >= 250:
                    continue
            elif panel_type == "HCP":
                if not (250 < bus_amps <= 800):
                    continue
            elif panel_type == "HCR-U":
                if bus_amps <= 800:
                    continue
            else:
                continue

            diffA = a2 - bus_amps
            diffS = s2 - spaces
            candidates.append((diffA, diffS, a2, s2, panel_type))

        if not candidates:
            series, bestA, bestS, bestMat = self._nearest_iline_bucket(
                il,
                mode="MAIN BREAKER",
                amps=bus_amps,
                spaces=spaces,
                enclosure=encl,
                material=mat,
                trim=trim
            )
            panelAttrs["panelType"] = series
            panelAttrs["amperage"]  = bestA
            panelAttrs["spaces"]    = bestS
            panelAttrs["material"]  = bestMat
            result = self._safe_generate_iline(panelAttrs)
            return result if isinstance(result, dict) else {"error": result}

        # pick the tightest fit
        _, _, best_amps, best_spaces, best_series = min(candidates, key=lambda x: (x[0], x[1]))

        panelAttrs["amperage"]  = best_amps
        panelAttrs["spaces"]    = best_spaces
        panelAttrs["panelType"] = best_series

        result = self._safe_generate_iline(panelAttrs)

        if isinstance(result, dict) and best_series in ("HCJ", "HCP"):
            mb = {400:"LAP36400MB", 600:"MGP36600", 800:"MGP36800"}.get(best_amps)
            if mb:
                result["Main Breaker Note"] = f"Comes with a factory installed {mb}"

        if isinstance(result, dict) and best_series == "HCR-U":

            voltage = _safe_int(raw.get("voltage"), 0)
            rating_voltage = 240 if voltage == 208 else voltage

            frames = ["PG","PJ","PL","RGC","RJC","RLC"]
            mccb_builder = mccb()
            desired_ir = _safe_int(raw.get("intRating"), 0)

            for two_letter in frames:
                frame_letter = two_letter[0]

                ir = snap_int_rating(frame_letter, rating_voltage, desired_ir)

                breaker_attrs = {
                    "frame":        frame_letter,
                    "amperage":     breaker_amps,
                    "termination":  "ILINE",
                    "poles":        3,
                    "voltage":      rating_voltage,
                    "intRating":    ir,
                    "tripFunction": raw.get("tripFunction", "ELECTRONIC"),
                    "grounding":    raw.get("grounding", False),
                    "phasing":      "ABC",
                }

                part = mccb_builder.generateMccbPartNumber(breaker_attrs)

                if isinstance(part, str):
                    if part.startswith("Invalid amperage") and "Allowed:" in part:
                        import re
                        m = re.search(r"Allowed:\s*\[([0-9,\s]+)\]", part)
                        if m:
                            allowed = sorted(int(x) for x in m.group(1).split(","))
                            for a in allowed:
                                if a >= breaker_amps:
                                    breaker_attrs["amperage"] = a
                                    break
                            part = mccb_builder.generateMccbPartNumber(breaker_attrs)

                    if isinstance(part, str) and part.startswith("Invalid interrupting rating"):
                        bumped_ir = snap_int_rating(frame_letter, rating_voltage, breaker_attrs["intRating"] + 1)
                        breaker_attrs["intRating"] = bumped_ir
                        part = mccb_builder.generateMccbPartNumber(breaker_attrs)

                if not (isinstance(part, str) and part.startswith("Invalid")):
                    result["Main Breaker"] = part
                    break
            else:
                result["Main Breaker Error"] = "No valid MCCB frame found for HCR-U"

            result["Usable Spaces"] = max(0, best_spaces - 9)

        return result

    def _build_nq_branch_breakers(self, panel_result: dict, raw: dict, detected_breakers: list):
        def bump_to_next(sorted_vals, desired):
            for v in sorted_vals:
                if v >= desired:
                    return v
            return None

        voltage = _safe_int(raw.get("voltage"), 0)
        rating_voltage = 240 if voltage == 208 else voltage

        if raw.get("_notes"):
            panel_result.setdefault("Notes", []).extend(raw["_notes"])

        pr_ir = panel_result.get("Interrupting Rating", "") or raw.get("intRating", "")
        try:
            panel_k = int(str(pr_ir).rstrip("k"))
        except:
            panel_k = 0

        allow_plug  = raw["allowPlugOn"]
        rating_type = raw["panelRatingType"].upper()

        default_k = {
            "QO":     10, "QOB":   10,
            "QO-VH":  22, "QOB-VH":22,
            "QOH":    42, "QH":    65,
            "QHB":    65, "HOM":   10,
        }

        total_spaces = _safe_int(panel_result.get("spaces"), 0)

        summary = {}
        requested_used_spaces = 0
        built_used_spaces = 0

        def _base_to_frame_and_ir(base: str, rating_voltage: int):
            v = 240 if rating_voltage in (208, 240) else rating_voltage

            if base in ("QO-VH", "QOB-VH"):
                return ("Q", [22])
            if base in ("QO", "QOB"):
                return ("Q", INTERRUPTING_RATINGS["Q"].get(v, []))
            if base == "QOH":
                return ("LL", [42] if v == 240 else INTERRUPTING_RATINGS["LL"].get(v, []))
            if base in ("QH", "QHB"):
                return ("H", INTERRUPTING_RATINGS["H"].get(v, []))
            return (None, [])

        def _is_gfi_output(out: dict) -> bool:
            pn = str(out.get("Part Number", "")).upper()
            # Prefer explicit metadata if breakerSelector provides it
            meta = " ".join(str(out.get(k, "")).upper() for k in ("Special Features", "Features", "Description"))
            if "GFI" in meta:
                return True
            # Fallback heuristic
            return "GFI" in pn

        def choose_base_family(panel_k: int, rating_type: str, allow_plug: bool, poles: int) -> str:
            def supported_poles_for(base: str):
                pole_map = {
                    "QO":      {1,2,3},
                    "QOB":     {1,2,3},
                    "QO-VH":   {1,2,3},
                    "QOB-VH":  {1,2,3},
                    "QOH":     {1,2},
                    "QH":      {1,2,3},
                    "QHB":     {1,2,3},
                    "HOM":     {1,2},
                }
                return pole_map.get(base, {1,2,3})

            if rating_type == "SERIES_RATED":
                ladder = (["QO","QO-VH","QH","QHB"] if allow_plug
                        else ["QOB","QOB-VH","QOH","QH","QHB"])
                for base in ladder:
                    if poles in supported_poles_for(base):
                        return base
                return ladder[-1]

            # FULLY_RATED: treat 208V as 240V for IR capability
            desired_at_240 = panel_k if rating_voltage in (208, 240) else panel_k
            if desired_at_240 >= 65:
                base = "QH"
            elif desired_at_240 >= 42:
                base = "QOH"
            elif (rating_voltage in (208, 240)) and desired_at_240 >= 25:
                base = "QH"  # 25-41k requires H-family at 240V
            elif desired_at_240 >= 22:
                base = "QO-VH" if allow_plug else "QOB-VH"
            else:
                base = "QO" if allow_plug else "QOB"

            if poles not in supported_poles_for(base):
                for cand in {
                    "QOH":    ["QH","QHB"],
                    "QO":     ["QO-VH","QH","QHB"],
                    "QOB":    ["QOB-VH","QOH","QH","QHB"],
                    "QO-VH":  ["QH","QHB"],
                    "QOB-VH": ["QOH","QH","QHB"],
                }.get(base, []):
                    if poles in supported_poles_for(cand):
                        return cand
            return base

        for br in detected_breakers:
            poles    = _safe_int(br.get("poles"), 0)
            raw_amps = _safe_int(br.get("amperage"), 0)
            count    = _safe_int(br.get("count"), 1)
            features = str(br.get("specialFeatures", "")).upper()
            is_gfi_requested = ("GFI" in features)

            # Only NQ panelboards can attempt GFI.
            # If this is a LOADCENTER result (fast-path), do NOT build GFI — note only.
            if is_gfi_requested and str(panel_result.get("Panel Family", "")).upper() == "LOADCENTER":
                panel_result.setdefault("Notes", []).append(
                    f"GFI breaker requested ({raw_amps}A {poles}P x{count}) — not supported for loadcenters in automated output. User review required."
                )
                continue

            needed_spaces = poles * count
            requested_used_spaces += needed_spaces

            base = choose_base_family(panel_k, rating_type, allow_plug, poles)

            if base == "QOH" and poles == 3:
                base = "QH" if allow_plug else "QHB"

            raw_ir = panel_k if rating_type == "FULLY_RATED" else default_k.get(base, panel_k)

            tmp_sel = breakerSelector({
                "breakerType":     base,
                "poles":           poles,
                "amperage":        raw_amps,
                "interruptionRating": raw_ir,
                "specialFeatures": "GFI" if is_gfi_requested else "",
                "iline":           False,
            })

            valid_amps = sorted(tmp_sel.getAmperageOptionsByPoles().keys())
            amps = bump_to_next(valid_amps, raw_amps)

            if amps is None:
                max_cap = valid_amps[-1] if valid_amps else 0
                where = ("loadcenter" if panel_result.get("Panel Family") == "LOADCENTER" else "panel")
                summary.setdefault("Errors", []).append(
                    (f"Requested {raw_amps}A exceeds max {max_cap}A for {poles}P {base} "
                     f"branch in this {where}. Consider a feeder/subfeed kit or bumping the family.")
                )
                continue

            frame_letter, ir_ladder = _base_to_frame_and_ir(base, rating_voltage)

            desired_ir = raw_ir

            if ir_ladder:
                ir = next((k for k in sorted(ir_ladder) if k >= desired_ir), ir_ladder[-1])
            else:
                ir = desired_ir

            attrs = {
                "breakerType":        base,
                "poles":              poles,
                "amperage":           amps,
                "interruptionRating": ir,
                "specialFeatures":    ("GFI" if is_gfi_requested else ""),
                "iline":              False,
            }
            sel = breakerSelector(attrs)
            out = sel.selectBreaker()
            
            # If GFI was requested, NQ is allowed to attempt it.
            # If it doesn't actually build to a GFI breaker, leave a note and skip it.
            if is_gfi_requested:
                if not isinstance(out, dict) or ("Error" in out) or (not _is_gfi_output(out)):
                    err_txt = out.get("Error") if isinstance(out, dict) else None
                    msg = (
                        f"GFI breaker requested ({raw_amps}A {poles}P x{count}) — could not be built in NQ automated output. "
                        f"User review required."
                    )
                    if err_txt:
                        msg += f" ({err_txt})"
                    panel_result.setdefault("Notes", []).append(msg)
                    continue

            # 1) QOH pole mismatch → escalate to QH/QHB
            if isinstance(out, dict) and "Error" in out and base == "QOH" and "Invalid poles selection" in str(out["Error"]):
                fallback_base = "QH" if allow_plug else "QHB"
                attrs["breakerType"] = fallback_base
                out = breakerSelector(attrs).selectBreaker()

            # 2) Amp not valid → bump to next allowed for the chosen base/poles
            if isinstance(out, dict) and "Error" in out and "Invalid amperage selection" in str(out["Error"]):
                valid_amps = sorted(
                    breakerSelector({
                        "breakerType": attrs["breakerType"],
                        "poles":       attrs["poles"],
                        "amperage":    0,  # dummy to query the map
                        "interruptionRating": attrs["interruptionRating"],
                        "specialFeatures":    "",
                        "iline":       False,
                    }).getAmperageOptionsByPoles().keys()
                )
                bumped = bump_to_next(valid_amps, raw_amps)
                if bumped is not None:
                    attrs["amperage"] = bumped
                    out = breakerSelector(attrs).selectBreaker()

            # 3) Still failing? Put the message in Notes (not in the Branch Breakers list) and skip
            if isinstance(out, dict) and "Error" in out:
                panel_result.setdefault("Notes", []).append(
                    f"Branch {poles}P@{raw_amps}A could not be built: {out['Error']}"
                )
                continue

            pn = out["Part Number"]
            built_used_spaces += needed_spaces

            entry = summary.setdefault(pn, {
                "Part Number":         pn,
                "Interrupting Rating": out["Interrupting Rating"],
                "count":               0,
            })
            entry["count"] += count

            for k, v in out.items():
                if k not in ("Part Number", "Interrupting Rating"):
                    entry[k] = v

        total_spaces = _safe_int(panel_result.get("spaces"), 0)
        remaining_spaces = max(0, total_spaces - built_used_spaces)

        panel_result["Branch Breakers (NQ)"] = list(summary.values())
        panel_result["Spaces Remaining"] = remaining_spaces

        # Only add filler plates when there are blank spaces
        if remaining_spaces > 0:
            filler = blanks()
            if panel_result.get("Panel Family") == "LOADCENTER":
                series = (panel_result.get("Series") or "").upper()
                filler_panel = "QO" if series == "QO" else "HOM"
            else:
                filler_panel = "NQ"

            filler_info = filler.generateFillerPlatePartNumber({
                "panelType": filler_panel,
                "totalBlankSpaces": remaining_spaces
            })
            panel_result["Filler Plates"] = filler_info
        else:
            panel_result.pop("Filler Plates", None)

        # Overflow reporting (do not block build)
        if total_spaces > 0 and requested_used_spaces > total_spaces:
            panel_result.setdefault("Notes", []).append(
                "too many breakers detected. user review required."
            )
        panel_result.pop("spaces", None)

        return panel_result

    def _build_nf_branch_breakers(self, panel_result: dict, raw: dict, detected_breakers: list):
        def bump_to_next(sorted_vals, desired):
            for v in sorted_vals:
                if v >= desired:
                    return v
            return None

        if raw.get("_notes"):
            panel_result.setdefault("Notes", []).extend(raw["_notes"])

        valid_ir_options = [18, 35, 65]

        pr_ir = panel_result.get("Interrupting Rating", "") or raw.get("intRating", 0)
        try:
            panel_k = int(str(pr_ir).rstrip("k"))
        except:
            panel_k = 0

        rating_type = raw["panelRatingType"].upper()

        total_spaces = _safe_int(panel_result.get("spaces"), 0)
        used_spaces  = 0
        summary = {}

        for br in detected_breakers:
            poles = _safe_int(br.get("poles"), 0)
            amps  = _safe_int(br.get("amperage"), 0)
            count = _safe_int(br.get("count"), 1)
            features = str(br.get("specialFeatures", "")).upper()

            if "GFI" in features:
                panel_result.setdefault("Notes", []).append(
                    f"GFI breaker requested ({amps}A {poles}P x{count}) — not supported in automated output. User review required."
                )
                continue

            needed = poles * count

            allowed_amps = {
                "1":[15,20,25,30,35,40,45,50,60,70],
                "2":[15,20,25,30,35,40,45,50,60,70,80,90,100,110,125],
                "3":[15,20,25,30,35,40,45,50,60,70,80,90,100,110,125],
                "EDP":[15,20,30,40,50]
            }[str(poles)]

            bumped_amps = bump_to_next(allowed_amps, amps)
            if bumped_amps is None:
                max_cap = allowed_amps[-1] if allowed_amps else 0
                summary.setdefault("Errors", []).append(
                    f"Requested {amps}A exceeds max {max_cap}A for {poles}P NF branch breaker."
                )
                continue
            amps = bumped_amps

            # Force 18kA for SERIES_RATED; otherwise use the panel_k target
            desired_ir = 18 if rating_type == "SERIES_RATED" else panel_k
            ir = bump_to_next(valid_ir_options, desired_ir)

            selector = eFrameBreaker()

            attrs = {
                "poles":              str(poles),
                "amperage":           amps,
                "interruptingRating": ir,
            }
            out = selector.generateEBreakerPartNumber(attrs)
            if isinstance(out, dict) and "Error" in out:
                summary.setdefault("Errors", []).append(out["Error"])
                continue

            pn = out["Part Number"]
            entry = summary.setdefault(pn, {
                "Part Number":         pn,
                "Interrupting Rating": out["Interrupting Rating"],
                "count":               0,
            })
            entry["count"] += count

            used_spaces += needed

        panel_result["Branch Breakers (NF)"] = list(summary.values())
        panel_result["Spaces Remaining"] = max(0, total_spaces - used_spaces)

        # Only add filler plates when there are blank spaces
        if _safe_int(panel_result["Spaces Remaining"], 0) > 0:
            filler = blanks()
            filler_info = filler.generateFillerPlatePartNumber({
                "panelType": "NF",
                "totalBlankSpaces": panel_result["Spaces Remaining"]
            })
            panel_result["Filler Plates"] = filler_info
        else:
            panel_result.pop("Filler Plates", None)

        if used_spaces > total_spaces and total_spaces > 0:
            panel_result.setdefault("Notes", []).append(
                "too many breakers detected. user review required."
            )

        return panel_result

    def _build_iline_branch_breakers(self, panel_result: dict, raw: dict, detected_breakers: list):
        DEBUG_ILINE = False
        def d(*args, **kwargs):
            if DEBUG_ILINE:
                print("[ILINE]", *args, **kwargs)        
        
        if raw.get("_notes"):
            panel_result.setdefault("Notes", []).extend(raw["_notes"])
        
        def side_profiles_for(series: str):
            if series == "HCJ":
                prof = {"type": "narrow", "allows": {"Q", "H", "J"}}
                return {"left": prof.copy(), "right": prof.copy()}
            elif series == "HCP":
                return {
                    "left":  {"type": "narrow", "allows": {"H", "J"}},
                    "right": {"type": "wide",   "allows": {"H", "J", "L", "M", "P"}},
                }
            else:  # HCR-U
                return {
                    "left":  {"type": "narrow", "allows": {"H", "J"}},
                    "right": {"type": "wide",   "allows": {"H", "J", "L", "M", "P", "R"}},
                }

        def choose_side_for(frame: str, inches_needed: float, profiles: dict, remaining: dict, series: str) -> tuple[str, bool]:
            if series == "HCJ":
                candidates = [s for s in ("left", "right")
                            if frame in profiles[s]["allows"] and remaining[s] >= inches_needed]
                if not candidates:
                    return None, False
                return max(candidates, key=lambda s: remaining[s]), False

            # HCP / HCR-U: H/J prefer a narrow side first, else wide (+extension)
            if frame in {"H", "J"}:
                narrow_sides = [s for s in ("left", "right")
                                if profiles[s]["type"] == "narrow"
                                and frame in profiles[s]["allows"]
                                and remaining[s] >= inches_needed]
                if narrow_sides:
                    return max(narrow_sides, key=lambda s: remaining[s]), False

                wide_sides = [s for s in ("left", "right")
                            if profiles[s]["type"] == "wide"
                            and frame in profiles[s]["allows"]
                            and remaining[s] >= inches_needed]
                if wide_sides:
                    return max(wide_sides, key=lambda s: remaining[s]), True
                return None, False

            # L/M/P/R must go on wide side
            wide_sides = [s for s in ("left", "right")
                        if profiles[s]["type"] == "wide"
                        and frame in profiles[s]["allows"]
                        and remaining[s] >= inches_needed]
            if wide_sides:
                return max(wide_sides, key=lambda s: remaining[s]), False

            return None, False

        def _try_build_branch(mccb_builder, attrs, frame_letter, rating_voltage):
            import re

            def _attempt(a):
                out = mccb_builder.generateMccbPartNumber(a)
                if isinstance(out, str) and out.lower().startswith("invalid interrupting rating"):
                    a["intRating"] = snap_int_rating(frame_letter, rating_voltage, int(a["intRating"]) + 1)
                    out = mccb_builder.generateMccbPartNumber(a)
                if isinstance(out, str) and out.lower().startswith("invalid amperage") and "allowed:" in out.lower():
                    m = re.search(r"Allowed:\s*\[([0-9,\s]+)\]", out, re.IGNORECASE)
                    if m:
                        allowed = sorted(int(x) for x in m.group(1).split(","))
                        for aa in allowed:
                            if aa >= int(a["amperage"]):
                                a["amperage"] = aa
                                break
                        out = mccb_builder.generateMccbPartNumber(a)
                return out

            first = _attempt(attrs.copy())
            if not (isinstance(first, str) and first.startswith("Invalid")):
                return first

            attrs2 = attrs.copy()
            attrs2["intRating"] = snap_int_rating(frame_letter, rating_voltage, int(attrs2["intRating"]) + 1)
            second = _attempt(attrs2)
            return second

        mccb_builder = mccb()
        filler = blanks()
        summary = {}
        phase_load = {"A": 0.0, "B": 0.0, "C": 0.0}

        voltage = _safe_int(raw.get("voltage"), 0)
        rating_voltage = 240 if voltage == 208 else voltage

        pr_ir = panel_result.get("Interrupting Rating", "") or raw.get("intRating", 0)
        try:
            panel_k = int(str(pr_ir).rstrip("k"))
        except:
            panel_k = 0

        rating_type = raw.get("panelRatingType", "FULLY_RATED").upper()

        frame_rules = {
            "B":  { "2": 3, "3": 4.5,     "amps": [15,20,25,30,35,40,45,50,60,70,80,90,100,110,125] },
            "Q":  { "2": 3, "3": 4.5,     "amps": [70,80,90,100,110,125,150,175,200,225,250] },
            "H":  { "2": 4.5, "3": 4.5,   "amps_m":[15,20,25,30,35,40,45,50,60,70,80,90,100,110,125,150], "amps_e":[60,100,150] },
            "J":  { "2": 4.5, "3": 4.5,   "amps_m":[150,175,200,225,250], "amps_e":[250] },
            "LL": { "2": 6, "3": 6,       "amps":[125,150,175,200,225,250,300,350,400] },
            "L":  { "2": 6, "3": 6,       "amps": [250, 400, 600], "amps_e": [250, 400, 600] },
            "M":  { "2": 9, "3": 9,       "amps":[400,600] },
            "P":  { "3": 9,               "amps":[250,400,600,800,1000,1200] },
            "R":  { "3": 15,              "amps":[1000,1200] },
        }

        electronic_only = {"M","P","R"}
        magnetic_only   = {"B","Q","LL"}
        hybrid_frames   = {"H","J","L"}
        threepole_only  = {"P","R"}

        allowed_frames = {
            "HCJ": ["Q","H","J"],
            "HCP": ["Q","H","J","LL","L","M","P"],
            "HCP-SU":["Q","H","J","LL","L","M","P"],
            "HCR-U":["Q","H","J","LL","L","M","P","R"],
        }

        def _infer_series(pr: dict, raw: dict) -> str | None:
            s = (pr.get("ilinePanelType")
                or pr.get("panelType")
                or raw.get("panelType"))
            if s:
                return s

            interior = (pr.get("Interior") or "").upper()
            box      = (pr.get("Box") or "").upper()

            for token in (interior, box):
                if token.startswith("HCJ"):
                    return "HCJ"
                if token.startswith("HCP"):
                    return "HCP"
                if token.startswith("HCR"):
                    return "HCR-U"
            return None

        series = _infer_series(panel_result, raw)
        if series not in ("HCJ", "HCP", "HCR-U"):
            d("WARN: could not infer I-Line series; defaulting to HCJ rules")
            series = "HCJ"

        total_spaces = (
            panel_result.get("Usable Spaces")
            if panel_result.get("Usable Spaces") is not None else
            _safe_int(panel_result.get("spaces"), _safe_int(raw.get("spaces"), 0))
        )
        # CAP AT 108 (affects the *1.5 calculation below)
        total_spaces = min(108, int(total_spaces or 0))

        total_inches = float(total_spaces) * 1.5
        side_remaining = {"left": total_inches / 2.0, "right": total_inches / 2.0}

        profiles = side_profiles_for(series)
        series_allowed = allowed_frames.get(series, ["Q","H","J"])
        frame_rules = {f: r for f, r in frame_rules.items() if f in series_allowed}

        d("series:", series, "| profiles:", profiles)
        d("rating_voltage:", rating_voltage, "| panel_k:", panel_k, "| rating_type:", rating_type)
        d("total_spaces:", total_spaces, "→ total_inches:", total_inches, "| per side:", side_remaining)
        d("frames allowed in series:", list(frame_rules.keys()))

        blanks_pieces = {
            "HLN1BL": 0,  # narrow blank (1.5")
            "HLW1BL": 0,  # wide blank (1.5")
            "HLW4EBL": 0, # wide extension for H/J placed on wide side
        }

        side_deficit_inches = 0.0

        allow_sqd_spd = bool(raw.get("_allowSqdSpd", True))
        iline_spd_present = allow_sqd_spd and (bool(raw.get("spd")) or bool(panel_result.get("SPD")))

        if iline_spd_present:
            inches_needed = 13.5
            side, requires_ext = choose_side_for("J", inches_needed, profiles, side_remaining, series)
            if side is None:
                best_side = max(side_remaining, key=lambda s: side_remaining[s])
                missing = max(0.0, inches_needed - side_remaining[best_side])
                side_deficit_inches = max(side_deficit_inches, missing)
                side_remaining[best_side] = max(0.0, side_remaining[best_side] - inches_needed)
            else:
                side_remaining[side] = max(0.0, side_remaining[side] - inches_needed)
                if requires_ext:
                    blanks_pieces["HLW4EBL"] += 1

            panel_result.setdefault("Notes", []).append(
                "I-LINE cannot be service-entrance rated when an internal SPD is installed."
            )

        two_p_phasing_cycle = ["AB", "BC", "AC"]
        phasing_index = 0

        for br in detected_breakers:
            poles    = _safe_int(br.get("poles"), 0)
            raw_amps = _safe_int(br.get("amperage"), 0)
            count    = _safe_int(br.get("count"), 1)
            d("REQ breaker → poles:", poles, "amps:", raw_amps, "count:", count)

            features = str(br.get("specialFeatures", "")).upper()
            if "GFI" in features:
                panel_result.setdefault("Notes", []).append(
                    f"GFI breaker requested ({raw_amps}A {poles}P x{count}) — not supported in I-LINE automated output. User review required."
                )
                continue

            if poles == 3:
                phasing = "ABC"
            elif poles == 2:
                phasing = two_p_phasing_cycle[phasing_index % 3]
            else:
                summary.setdefault("Errors", []).append("I-Line breakers must be 2P or 3P only")
                continue

            placed_units = 0
            max_attempts = count * (len(["Q","H","J","LL","L","M","P","R"]) + 2)
            while placed_units < count:
                found = False
                final_part = None
                chosen_frame = None
                chosen_space_key = None
                chosen_trip = None
                chosen_amperage = None
                chosen_phase = None
                chosen_side = None
                needs_extension = False

                capacity_blocked_this_spec = False  # reset per attempt

                order = [f for f in ["Q","H","J","LL","L","M","P","R"] if f in frame_rules]
                d("frame search order:", order)

                for frame in order:
                    if frame == "Q" and rating_voltage > 240:
                        continue
                    if frame == "LL" and poles == 2:
                        continue
                    if frame in threepole_only and poles != 3:
                        continue

                    fr = frame_rules[frame]
                    key = str(poles)
                    if key not in fr:
                        continue

                    amps_list = fr.get("amps") or fr.get("amps_m", [])
                    amps = next((a for a in amps_list if a >= raw_amps), None)
                    if amps is None:
                        continue

                    inches_for_breaker = float(fr[key])
                    d("  trying frame:", frame, "| poles:", poles, "| amps cand:", amps,
                    "| inches_needed:", inches_for_breaker)

                    side, requires_ext = choose_side_for(frame, inches_for_breaker, profiles, side_remaining, series)
                    if side is None:
                        # Force to the side with the most remaining inches; allow negative and record deficit.
                        capacity_blocked_this_spec = True
                        best_side = max(side_remaining, key=lambda s: side_remaining[s])
                        need = inches_for_breaker - side_remaining[best_side]
                        if need > 0:
                            side_deficit_inches = max(side_deficit_inches, need)
                        side = best_side
                        # Keep requires_ext consistent with the chosen side type + H/J on wide sides
                        requires_ext = (profiles[side]["type"] == "wide" and frame in {"H","J"} and series in {"HCP","HCR-U"})

                    if rating_type == "SERIES_RATED":
                        valid_ir = sorted(INTERRUPTING_RATINGS.get(frame, {}).get(rating_voltage, []))
                        ir = valid_ir[0] if valid_ir else panel_k
                    else:
                        ir = snap_int_rating(frame, rating_voltage, panel_k)

                    mccb_frame = "L" if frame == "LL" else frame

                    # ---- MAGNETIC attempt (skip if electronic-only) ----
                    if frame in magnetic_only or frame not in electronic_only:
                        if poles == 2:
                            if frame in ("H","J"):
                                intr_letter = mccb_builder.getInterruptingLetterHJ(rating_voltage, ir)
                                raw_pairs = hj_allowed_pairs(frame, intr_letter, amps) if intr_letter else []
                                allowed_pairs = canonicalize_pairs(raw_pairs) if raw_pairs else ["AB","BC","AC"]
                            else:
                                allowed_pairs = ["AB","BC","AC"]

                            weight = float(amps)
                            try_phase = pick_balanced_pair(phase_load, allowed_pairs, weight)
                            if not try_phase:
                                pass
                            else:
                                d("    MAG attrs:", {
                                    "frame": mccb_frame, "termination": "ILINE",
                                    "poles": poles, "amperage": amps, "voltage": rating_voltage,
                                    "intRating": ir, "tripFunction": "MAGNETIC",
                                    "phasing": (try_phase if poles == 2 else phasing)
                                })
                                attrs = {
                                    "frame":        mccb_frame,
                                    "termination":  "ILINE",
                                    "poles":        poles,
                                    "amperage":     amps,
                                    "voltage":      rating_voltage,
                                    "intRating":    ir,
                                    "tripFunction": "MAGNETIC",
                                    "grounding":    False,
                                    "phasing":      (try_phase if poles == 2 else phasing),
                                }
                                out = _try_build_branch(mccb_builder, attrs, mccb_frame, rating_voltage)

                                if not (isinstance(out, str) and out.startswith("Invalid")):
                                    final_part = {"Part Number": out, "Interrupting Rating": attrs["intRating"]} if isinstance(out, str) \
                                                else {**out, "Interrupting Rating": attrs["intRating"]}
                                    chosen_frame = frame
                                    chosen_space_key = key
                                    chosen_trip = "MAGNETIC"
                                    chosen_amperage = amps
                                    chosen_phase = try_phase
                                    chosen_side = side
                                    needs_extension = (profiles[side]["type"] == "wide" and frame in {"H","J"} and series in {"HCP","HCR-U"})
                                    found = True

                    # ---- ELECTRONIC attempt (if not found yet and frame supports it) ----
                    if not found and (frame in hybrid_frames or frame in electronic_only):
                        e_list = fr.get("amps_e", fr.get("amps", []))
                        e_amps = next((a for a in e_list if a >= raw_amps), None)
                        if e_amps is not None:
                            if poles == 2:
                                if frame in ("H", "J"):
                                    intr_letter = mccb_builder.getInterruptingLetterHJ(rating_voltage, ir)
                                    raw_pairs = hj_allowed_pairs(frame, intr_letter, e_amps) if intr_letter else []
                                    allowed_pairs = canonicalize_pairs(raw_pairs) if raw_pairs else ["AB","BC","AC"]
                                else:
                                    allowed_pairs = ["AB","BC","AC"]
                                weight = float(e_amps)
                                try_phase = pick_balanced_pair(phase_load, allowed_pairs, weight) or "AB"
                            else:
                                try_phase = phasing
                            d("    ELEC attrs:", {
                                "frame": mccb_frame, "termination": "ILINE",
                                "poles": poles, "amperage": e_amps, "voltage": rating_voltage,
                                "intRating": ir, "tripFunction": "ELECTRONIC",
                                "phasing": (try_phase if poles == 2 else phasing)
                            })
                            attrs = {
                                "frame":        mccb_frame,
                                "termination":  "ILINE",
                                "poles":        poles,
                                "amperage":     e_amps,
                                "voltage":      rating_voltage,
                                "intRating":    ir,
                                "tripFunction": "ELECTRONIC",
                                "grounding":    False,
                                "phasing":      (try_phase if poles == 2 else phasing),
                            }
                            out = _try_build_branch(mccb_builder, attrs, mccb_frame, rating_voltage)

                            if not (isinstance(out, str) and out.startswith("Invalid")):
                                final_part = {"Part Number": out, "Interrupting Rating": attrs["intRating"]} if isinstance(out, str) \
                                            else {**out, "Interrupting Rating": attrs["intRating"]}
                                chosen_frame = frame
                                chosen_space_key = key
                                chosen_trip = "ELECTRONIC"
                                chosen_amperage = e_amps
                                chosen_phase = try_phase if poles == 2 else phasing
                                chosen_side = side
                                needs_extension = (profiles[side]["type"] == "wide" and frame in {"H","J"} and series in {"HCP","HCR-U"})
                                found = True

                # selection only; actual placement happens after the frame loop
                if found:
                    # Determine width and place the breaker — allow negative inches to track deficit.
                    inches = float(frame_rules[chosen_frame][chosen_space_key])
                    side_remaining[chosen_side] -= inches  # force place
                    d(
                        "    PLACE:", final_part["Part Number"],
                        "| frame:", chosen_frame,
                        "| trip:", chosen_trip,
                        "| phase:", (chosen_phase if poles == 2 else "ABC"),
                        "| side:", chosen_side,
                        "| inches:", inches,
                        "| extension:", ("YES (filler only, no space)" if needs_extension else "NO"),
                        "| side_remaining now:", side_remaining
                    )
                    if needs_extension:
                        blanks_pieces["HLW4EBL"] += 1
                    # Update phase balance post-placement
                    if poles == 2 and chosen_trip == "MAGNETIC":
                        a, b = pair_to_phases(chosen_phase)
                        phase_load[a] += float(chosen_amperage)
                        phase_load[b] += float(chosen_amperage)
                    pn = final_part["Part Number"]
                    entry = summary.setdefault(pn, {
                        "Part Number": pn,
                        "Interrupting Rating": final_part["Interrupting Rating"],
                        "count": 0
                    })
                    entry["count"] += 1
                    for k, v in final_part.items():
                        if k not in ("Part Number", "Interrupting Rating"):
                            entry[k] = v
                    placed_units += 1
                    continue

                # ---- NO-PROGRESS ESCAPE ----
                if not found:
                    reason = ("Insufficient I-LINE side capacity/allowance for required frames"
                            if capacity_blocked_this_spec else
                            "No compatible frame/IR/voltage combo for this breaker spec")

                    summary.setdefault("Errors", []).append(
                        (f"{reason}: {poles}P @ {raw_amps}A "
                        f"(series={series}, voltage={rating_voltage}V, IR target≈{panel_k}k). "
                        f"Placed {placed_units}/{count} for this spec.")
                    )
                    break

                # hard guard
                max_attempts -= 1
                if max_attempts <= 0:
                    summary.setdefault("Errors", []).append(
                        (f"Internal guard tripped while placing breakers; "
                        f"aborted remaining attempts for {poles}P @ {raw_amps}A.")
                    )
                    break

            if poles == 2 and placed_units > 0:
                phasing_index += placed_units

        for side in ("left", "right"):
            rem = max(0.0, side_remaining[side])
            pieces = int(math.ceil(rem / 1.5)) if rem > 0 else 0
            if pieces > 0:
                if profiles[side]["type"] == "narrow":
                    blanks_pieces["HLN1BL"] += pieces
                else:
                    blanks_pieces["HLW1BL"] += pieces
        d("BLANKS pieces:", blanks_pieces, "| spaces_remaining (calc):",
        max(0, int(round((side_remaining['left'] + side_remaining['right']) / 1.5))))

        panel_result["Branch Breakers (I-LINE)"] = list(summary.values())

        total_rem_inches = side_remaining["left"] + side_remaining["right"]
        panel_result["Spaces Remaining"] = max(0, int(round(float(total_rem_inches) / 1.5)))

        iline_blanks = filler.generateILineBlanks(
            widePieces=blanks_pieces["HLW1BL"],
            narrowPieces=blanks_pieces["HLN1BL"],
            extensionPieces=blanks_pieces["HLW4EBL"],
        )
        compact = self._compact_iline_blanks(iline_blanks)
        if compact:
            panel_result["I-LINE Blanks & Extensions"] = compact

        if side_deficit_inches > 0:
            needed_extra_spaces = int(math.ceil((2.0 * side_deficit_inches) / 1.5))
            panel_result.setdefault("Notes", []).append(
                f"Side-capacity bottleneck detected. Consider increasing by ~{needed_extra_spaces} spaces."
            )
            panel_result["_side_deficit_spaces"] = needed_extra_spaces
        # Uniform overflow flag
        if side_deficit_inches > 0 or raw.get("_overcapacity_spaces"):
            panel_result.setdefault("Notes", []).append(
                "too many breakers detected. user review required."
            )

        panel_result.pop("spaces", None)
        if not DEBUG_ILINE:
            panel_result.pop("Build Error", None)
        else:
            if "Build Error" in panel_result:
                panel_result.setdefault("Notes", []).append(
                    f"Panel auto-resized; original build failed for some options: {panel_result.pop('Build Error')}"
                )
        return panel_result

def snap_int_rating(frame: str, voltage: int, desired: int) -> int:
    """Enhanced snap function that ensures interrupting rating is properly bumped"""
    valid_sorted = sorted(INTERRUPTING_RATINGS.get(frame, {}).get(voltage, []))
    if not valid_sorted:
        return desired
    if desired in valid_sorted:
        return desired
    
    # Find the next higher rating (bumping behavior)
    for r in valid_sorted:
        if r > desired:
            return r
    
    # If requested rating is higher than all available, return the highest
    return valid_sorted[-1]

@register_engine("disconnect")
class DisconnectEngine(BaseEngine):
    def _bucket_voltage_for_family(self, family: str, v_actual: int) -> int:
        return 240 if family == "GENERAL_DUTY" else 600

    def _next_supported_amp(self, disc: Disconnect, family: str, fusible: bool, poles: int | None, v_bucket: int, enclosure: str, req_amp: int) -> int | None:
        poles_norm = None if (family == "GENERAL_DUTY" and not fusible) else poles

        cands = []
        for (sw, fu, pl, volt, amp, encl) in disc.disconnectLibrary.keys():
            if sw != family or fu != fusible:
                continue
            if pl is not None and poles_norm is not None and pl != poles_norm:
                continue
            if pl is None and poles_norm is not None and family != "GENERAL_DUTY":
                continue
            if pl is not None and poles_norm is None and not (family == "GENERAL_DUTY" and not fusible):
                continue

            if volt != v_bucket or encl != enclosure:
                continue
            if isinstance(amp, int) and amp >= req_amp:
                cands.append(amp)

        if not cands:
            return None
        return min(cands)

    def process(self) -> dict:
        defaults = ui_defaults().get("disconnects", {})
        raw = self.attrs.copy()

        seeded = {
            "switchType":     defaults.get("default_switch_type", "GENERAL_DUTY"),
            "enclosure":      defaults.get("default_enclosure",  "NEMA1"),
            "fusible":        defaults.get("default_fusible",     True),
            "groundRequired": defaults.get("default_ground_required", True),
            "solidNeutral":   defaults.get("default_solid_neutral",   True),
            "poles":          3,
        }
        for k in ("switchType","enclosure","fusible","groundRequired","solidNeutral",
                  "poles","voltage","amps","fuseAmperage"):
            if k in raw and raw[k] not in (None, ""):
                seeded[k] = raw[k]

        if isinstance(seeded.get("switchType"), str):
            seeded["switchType"] = seeded["switchType"].upper()
        if isinstance(seeded.get("enclosure"), str):
            seeded["enclosure"]  = seeded["enclosure"].upper()
        seeded["fusible"]        = bool(seeded.get("fusible", True))
        seeded["groundRequired"] = bool(seeded.get("groundRequired", True))
        seeded["solidNeutral"]   = bool(seeded.get("solidNeutral", True))

        missing = []
        def _to_int_or_none(v):
            if v in (None, ""): 
                return None
            try:
                return int(float(str(v)))
            except Exception:
                return None

        amps_req = _to_int_or_none(seeded.get("amps"))
        v_actual = _to_int_or_none(seeded.get("voltage"))

        if amps_req is None or amps_req <= 0:
            missing.append("amps")
        if v_actual is None or v_actual <= 0:
            missing.append("voltage")

        if missing:
            return {"_skipped": True, "reason": f"Missing required attributes: {', '.join(missing)}"}

        fusible = seeded["fusible"]
        try:
            poles = int(seeded.get("poles", 3))
        except Exception:
            poles = 3

        def gd_ok(v: int, a: int, is_fusible: bool) -> bool:
            if v > 240: return False
            return (a <= 800) if is_fusible else (a <= 600)

        def hd_ok(v: int, a: int) -> bool:
            return (v <= 600) and (a <= 1200)

        reason = None
        if gd_ok(v_actual, amps_req, fusible):
            family = "GENERAL_DUTY"
        elif hd_ok(v_actual, amps_req):
            family = "HEAVY_DUTY"
            reason = ("Escalated to Heavy Duty: Voltage > 240V"
                      if v_actual > 240 else "Escalated to Heavy Duty: Amperage exceeds GD limit")
        else:
            return {"error": f"No valid family for {amps_req}A @ {v_actual}V (fusible={fusible})."}

        solidNeutral = seeded["solidNeutral"]
        if family == "HEAVY_DUTY" and not fusible:
            solidNeutral = False

        enclosure = seeded["enclosure"]
        disc = Disconnect()
        tried = []

        def _attempt(fam: str, enc: str) -> tuple[dict | str, dict]:
            v_bucket = self._bucket_voltage_for_family(fam, v_actual)
            snapped = self._next_supported_amp(disc, fam, fusible, poles, v_bucket, enc, amps_req)
            attrs = {
                "switchType":     fam,
                "fusible":        fusible,
                "poles":          poles,
                "voltage":        v_bucket,
                "amps":           int(snapped if snapped is not None else amps_req),
                "enclosure":      enc,
                "groundRequired": True,
                "solidNeutral":   (False if (fam=="HEAVY_DUTY" and not fusible) else solidNeutral),
                "fuseAmperage":   seeded.get("fuseAmperage"),
            }
            out = disc.generateDisconnectPartNumber(attrs)
            meta = {"snapped_from": (amps_req if snapped and snapped != amps_req else None),
                    "snapped_to": snapped}
            return out, meta

        out, meta = _attempt(family, enclosure)
        tried.append({"family": family, "enclosure": enclosure, **meta})

        if isinstance(out, str) and out.lower().startswith("invalid") and family == "GENERAL_DUTY":
            family = "HEAVY_DUTY"
            reason = reason or "Escalated to Heavy Duty: no General Duty table match"
            out, meta = _attempt(family, enclosure)
            tried.append({"family": family, "enclosure": enclosure, **meta})

        def _was_field_missing(field_name: str) -> bool:
            return field_name not in raw or raw[field_name] in (None, "")

        if isinstance(out, str) and out.lower().startswith("invalid") and _was_field_missing("enclosure"):
            enc2 = "NEMA3R" if enclosure == "NEMA1" else "NEMA1"
            out2, meta2 = _attempt(family, enc2)
            tried.append({"family": family, "enclosure": enc2, **meta2})
            if not (isinstance(out2, str) and out2.lower().startswith("invalid")):
                out = out2
                meta = meta2

        # Final failure
        if isinstance(out, str) and out.lower().startswith("invalid"):
            return {"error": out, "_debug": {"attempts": tried}}

        # Success
        if isinstance(out, dict):
            notes = []

            defaults = ui_defaults().get("disconnects", {})
            default_enclosure = (defaults.get("default_enclosure") or "NEMA1").upper()
            default_neutral   = bool(defaults.get("default_solid_neutral", True))

            user_enclosure_supplied = ("enclosure" in raw and raw["enclosure"] not in (None, ""))
            user_enclosure_value    = (str(raw.get("enclosure")).upper() if user_enclosure_supplied else None)

            user_neutral_supplied   = ("solidNeutral" in raw and raw["solidNeutral"] not in (None, ""))
            user_neutral_value      = bool(raw.get("solidNeutral")) if user_neutral_supplied else None

            used_enclosure = str(seeded["enclosure"]).upper()
            used_neutral   = (False if (family == "HEAVY_DUTY" and not fusible) else bool(seeded["solidNeutral"]))

            if user_enclosure_supplied:
                if user_enclosure_value != default_enclosure and used_enclosure == user_enclosure_value:
                    notes.append(f"Enclosure overridden to {used_enclosure} based on detection.")

            if family == "HEAVY_DUTY" and not fusible:
                wanted_neutral = user_neutral_value if user_neutral_supplied else default_neutral
                if wanted_neutral and used_neutral is False:
                    notes.append("Neutral omitted: neutral not available on Heavy Duty non-fusible.")

            if reason:
                notes.append(reason)

            if isinstance(meta, dict) and meta.get("snapped_from") and meta.get("snapped_to"):
                notes.append(f"Amperage snapped {meta['snapped_from']} → {meta['snapped_to']}.")

            if notes:
                if isinstance(out.get("Notes"), list):
                    out["Notes"].extend(notes)
                elif isinstance(out.get("Notes"), str) and out["Notes"].strip():
                    out["Notes"] = out["Notes"] + " | " + " ".join(notes)
                else:
                    out["Notes"] = " ".join(notes)

            out.pop("_policy", None)
            return out

@register_engine("transformer")
class TransformerEngine(BaseEngine):
    def _normalize(self, s):
        return str(s).upper() if s is not None else None

    def _to_int(self, v):
        if v in (None, ""):
            return None
        try:
            return int(float(str(v)))
        except (TypeError, ValueError):
            return None

    def _to_float(self, v):
        if v in (None, ""):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _choose_type(self, seeded: dict) -> str:
        explicit = seeded.get("transformerType")
        if explicit:
            return explicit
        return "3PHASESTANDARD"

    def _infer_phase_from_volts(self, seeded: dict) -> str:
        pri = self._to_int(seeded.get("primaryVolts"))
        sec = self._to_int(seeded.get("secondaryVolts"))
        if sec in (120,) and pri in (240, 277, 208, 600):
            return "1PHASE"
        return "3PHASE"

    def _list_allowed_kva(self, rules: dict, key_tuple) -> list:
        row = rules.get(key_tuple, {})
        return sorted(row.keys(), key=lambda x: float(x))

    def _snap_kva(self, allowed: list, requested) -> tuple[float, float | None]:
        req = float(requested)
        for kv in allowed:
            if float(kv) >= req:
                return kv, (None if float(kv) == req else req)
        if allowed:
            return allowed[-1], req
        return req, None

    def _seed_inputs(self) -> dict:
        defaults = ui_defaults().get("transformers", {})
        raw = self.attrs.copy()

        alias = {
            "primary_volts": "primaryVolts",
            "secondary_volts": "secondaryVolts",
            "core_material": "coreMaterial",   # ALUMINUM/COPPER, or RESIN enclosure ("3R","SS","4X")
            "temperature_rating": "temperature",
        }
        for a, b in alias.items():
            if a in raw and b not in raw:
                raw[b] = raw[a]

        material_map = {"AL": "ALUMINUM", "ALUM": "ALUMINUM", "CU": "COPPER"}
        core_mat_raw = raw.get("coreMaterial") or defaults.get("winding_material", "ALUMINUM")
        core_mat = self._normalize(material_map.get(self._normalize(core_mat_raw), core_mat_raw))

        seeded = {
            "kva":              self._to_float(raw.get("kva")),
            "primaryVolts":     self._to_int(raw.get("primaryVolts")),
            "secondaryVolts":   self._to_int(raw.get("secondaryVolts")),
            "coreMaterial":     core_mat,
            "temperature":      self._to_int(raw.get("temperature")) or int(defaults.get("temperature_rating", 150)),
            "weathershield":    bool(raw.get("weathershield", defaults.get("weathershield", False))),
            "mounting":         self._normalize(raw.get("mounting", defaults.get("mounting", "FLOOR"))),
            "transformerType":  self._normalize(raw.get("transformerType")) if raw.get("transformerType") else None,
            "resin_enclosure":  self._normalize(raw.get("resin_enclosure", defaults.get("resin_enclosure", "3R"))),
        }

        if seeded["mounting"] not in ("WALL", "CEILING", "FLOOR", None):
            seeded["mounting"] = "FLOOR"

        missing = []
        if seeded["kva"] in (None, ""):            missing.append("kva")
        if seeded["primaryVolts"] in (None, ""):   missing.append("primaryVolts")
        if seeded["secondaryVolts"] in (None, ""): missing.append("secondaryVolts")
        if missing:
            return {"_skipped": True, "reason": f"Missing required attributes: {', '.join(missing)}"}

        seeded["phase"] = self._infer_phase_from_volts(seeded)
        return seeded

    def process(self) -> dict:
        seeded = self._seed_inputs()
        if seeded.get("_skipped"):
            return seeded
        if "error" in seeded:
            return seeded

        first_choice = self._choose_type(seeded)
        if seeded.get("transformerType"):
            type_candidates = [first_choice]
        else:
            fallback_chain = ["3PHASE115DEGREE", "3PHASE80DEGREE", "1PHASESTANDARD", "RESIN", "K13", "WATCHDOG"]
            type_candidates = [first_choice] + [t for t in fallback_chain if t != first_choice]

        tb = TransformerBuilder()
        attempts_meta = []

        catalog_kvas = [1,1.5,2,3,5,6,7,7.5,9,10,12,15,25,30,37.5,45,50,60,75,100,112.5,150,167,225,250,300,333,500,750]
        req = float(seeded["kva"])
        ups   = sorted([kv for kv in catalog_kvas if kv >= req], key=float)
        downs = sorted([kv for kv in catalog_kvas if kv <  req], key=float, reverse=True)
        test_order = [req] + [kv for kv in ups if kv != req] + downs

        for ttype in type_candidates:
            if ttype == "RESIN":
                core_candidates = [seeded["resin_enclosure"], "SS", "4X"]
            else:
                cm = seeded["coreMaterial"] or "ALUMINUM"
                cm = self._normalize(cm)
                if cm == "COPPER":
                    core_candidates = ["COPPER"]
                else:
                    core_candidates = ["ALUMINUM", "COPPER"]

            for core in core_candidates:
                attrs = {
                    "transformerType": ttype,
                    "kva":             req,
                    "primaryVolts":    int(seeded["primaryVolts"]),
                    "secondaryVolts":  int(seeded["secondaryVolts"]),
                    "coreMaterial":    core,
                    "temperature":     int(seeded["temperature"]) if seeded["temperature"] is not None else None,
                    "weathershield":   bool(seeded["weathershield"]),
                    "mounting":        seeded["mounting"],
                }

                notes = []
                for kv_try in test_order:
                    attrs["kva"] = kv_try
                    out = tb.generateTransformerPartNumber(attrs)
                    attempts_meta.append({
                        "type": ttype, "core": core, "kva": kv_try,
                        "result": ("OK" if isinstance(out, dict) else str(out)[:60])
                    })

                    if isinstance(out, dict):
                        result = dict(out)
                        if kv_try != req:
                            notes.append(f"kVA snapped {req:g} → {kv_try:g}")
                        if ttype != first_choice:
                            notes.append(f"Transformer type escalated to {ttype} for catalog fit.")
                        if ttype == "RESIN":
                            if core != seeded["resin_enclosure"]:
                                notes.append(f"RESIN enclosure overridden to {core} for catalog fit.")
                        else:
                            requested_cm = self._normalize(seeded["coreMaterial"])
                            if requested_cm != "COPPER" and core == "COPPER":
                                notes.append("Winding material upgraded to COPPER for catalog fit.")
                        if notes:
                            if isinstance(result.get("Notes"), list):
                                result["Notes"].extend(notes)
                            elif isinstance(result.get("Notes"), str) and result["Notes"].strip():
                                result["Notes"] = result["Notes"] + " | " + " ".join(notes)
                            else:
                                result["Notes"] = " | ".join(notes)
                        return result

        return {
            "error": "Invalid transformer configuration.",
            "_debug": {"attempts": attempts_meta[:10]}
        }
