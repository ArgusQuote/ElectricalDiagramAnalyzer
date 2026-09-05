# OcrLibrary/BreakerTableParserAPIv15.py
import sys, os, inspect
import re
import cv2
import copy

_THIS_FILE  = os.path.abspath(__file__)
_OCRLIB_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT  = os.path.dirname(_OCRLIB_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

API_VERSION = "API_15"
API_ORIGIN  = __file__

SNAP_MAP = {
    16: 18, 20: 18,
    28: 30, 32: 30,
    40: 42, 44: 42,
    52: 54, 56: 54,
    64: 66, 68: 66,
    70: 72, 74: 72,
    82: 84, 86: 84,
}
 
VALID_VOLTAGES = {120, 208, 240, 480, 600}
AMP_MIN = 100
AMP_MAX = 1200

_NAME_COUNTS = {}

def _norm_name(s):
    return str(s or "").strip().upper()

def _dedupe_name(raw_name: str | None) -> str:
    base = (str(raw_name or "").strip()) or "(unnamed)"
    key = _norm_name(base)
    cnt = _NAME_COUNTS.get(key, 0) + 1
    _NAME_COUNTS[key] = cnt
    return base if cnt == 1 else f"{base} ({cnt})"

def reset_name_deduper():
    _NAME_COUNTS.clear()

from OcrLibrary.BreakerTableAnalyzer12 import BreakerTableAnalyzer, ANALYZER_VERSION
from OcrLibrary.PanelHeaderParserV12   import PanelParser as PanelHeaderParser
from OcrLibrary.BreakerTableParser13   import BreakerTableParser, PARSER_VERSION
 
class BreakerTablePipeline:
    def __init__(self, *, debug: bool = True, reader=None):
        self.debug = bool(debug)
        self._shared_reader = reader
        self._analyzer = None
        self._header_parser = None

    def _to_int_or_none(self, v):
        if v is None:
            return None
        s = ''.join(ch for ch in str(v) if ch.isdigit())
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def _special_header_note(self, header_result: dict | None) -> str | None:
        if not isinstance(header_result, dict):
            return None

        note = str(header_result.get("panelNote") or "").strip()
        if note:
            return note

        sht = header_result.get("specialHeaderType")
        if isinstance(sht, dict):
            note = str(sht.get("note") or "").strip()
            if note:
                return note

        return None

    def _special_header_kind(self, header_result: dict | None) -> str | None:
        if not isinstance(header_result, dict):
            return None

        sht = header_result.get("specialHeaderType")
        if isinstance(sht, dict):
            kind = str(sht.get("kind") or "").strip()
            return kind or None

        return None

    def _mask_header_non_name(self, header_result: dict | None, *, detected_name):
        out = dict(header_result) if isinstance(header_result, dict) else {}
        out["name"] = detected_name

        attrs = out.get("attrs")
        if isinstance(attrs, dict):
            masked = {}
            for k in attrs.keys():
                if k == "detected_breakers":
                    masked[k] = []
                else:
                    masked[k] = "x"
            out["attrs"] = masked
        else:
            out["attrs"] = {}

        preserve_keys = {
            "name",
            "attrs",
            "panelNote",
            "specialHeaderType",
            "reviewOverlayPath",
            "winningBoxes",
            "boxImageShape",
        }

        for k in list(out.keys()):
            if k not in preserve_keys:
                out[k] = "x"

        return out

    def _mask_parser_non_name(self, parser_result: dict | None, *, detected_name):
        out = dict(parser_result) if isinstance(parser_result, dict) else {}
        out["name"] = detected_name
        out["spaces"] = "x"
        out["detected_breakers"] = []
        return out or {"name": detected_name, "spaces": "x", "detected_breakers": []}

    def _extract_panel_keys(self, analyzer_result: dict | None, header_result: dict | None, parser_result: dict | None):
        ar = analyzer_result or {}
        hdr = header_result or {}
        attrs = hdr.get("attrs") if isinstance(hdr.get("attrs"), dict) else {}
        prs = parser_result or {}

        ah = {}
        nh = ar.get("normalized_header")
        if isinstance(nh, dict):
            ah = {
                "name": nh.get("name"),
                "voltage": nh.get("voltage"),
                "bus_amps": nh.get("bus"),
                "main_amps": nh.get("main"),
            }

        def pick(d: dict, *keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None

        name = pick(hdr, "name") or ah.get("name")
        volts = pick(hdr, "volts", "voltage", "v") or pick(attrs, "volts", "voltage", "v") or ah.get("voltage")
        bus_amps = pick(hdr, "bus_amps", "busamps", "bus", "amperage") \
                   or pick(attrs, "bus_amps", "busamps", "bus", "amperage") \
                   or ah.get("bus_amps")
        main_amps = pick(hdr, "main_amps", "main", "main_rating", "main_breaker_amps", "mainBreakerAmperage") \
                    or pick(attrs, "main_amps", "main", "main_rating", "main_breaker_amps", "mainBreakerAmperage") \
                    or ah.get("main_amps")
        trim_style = pick(hdr, "trimStyle", "trim_style") or pick(attrs, "trimStyle", "trim_style")
        enclosure = pick(hdr, "enclosure") or pick(attrs, "enclosure")
        spaces = prs.get("spaces")
        return name, volts, bus_amps, main_amps, trim_style, enclosure, spaces

    def _parse_voltage(self, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v if v in VALID_VOLTAGES else None
        s = str(v)
        m_pair = re.search(r'(?<!\d)(120|208|240|277|480|600)\s*[Y/]\s*(120|208|240|277|480|600)(?!\d)', s, flags=re.I)
        if m_pair:
            hi = max(int(m_pair.group(1)), int(m_pair.group(2)))
            return hi if hi in VALID_VOLTAGES else None
        m_single = re.search(r'(?<!\d)(120|208|240|480|600)(?!\d)', s)
        return int(m_single.group(1)) if m_single and int(m_single.group(1)) in VALID_VOLTAGES else None

    def _is_valid_amp(self, val) -> bool:
        n = self._to_int_or_none(val)
        if n is None:
            return False
        if n < AMP_MIN or n > AMP_MAX:
            return False
        return (n % 10) in (0, 5)

    def _amp_over_max_message(self, bus_amps, main_amps):
        bus_i = self._to_int_or_none(bus_amps)
        main_i = self._to_int_or_none(main_amps)

        over = []
        if bus_i is not None and bus_i > AMP_MAX:
            over.append(f"bus amperage {bus_i}A exceeds max allowed system amperage ({AMP_MAX}A)")
        if main_i is not None and main_i > AMP_MAX:
            over.append(f"main amperage {main_i}A exceeds max allowed system amperage ({AMP_MAX}A)")

        if not over:
            return None

        return "; ".join(over)
    
    def _ensure_dir(self, p: str) -> str:
        os.makedirs(p, exist_ok=True)
        return p

    def _scale_box(self, box, src_w, src_h, dst_w, dst_h):
        # box = [x1,y1,x2,y2]
        if not box or src_w <= 0 or src_h <= 0:
            return None 
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except Exception:
            return None
        sx = float(dst_w) / float(src_w)
        sy = float(dst_h) / float(src_h)
        return [int(round(x1*sx)), int(round(y1*sy)), int(round(x2*sx)), int(round(y2*sy))]

    def _draw_box(self, vis, box, color_bgr, label=None, *, fill_alpha: float = 0.0, thickness: int = 2):
        if not box or len(box) != 4:
            return
        H, W = vis.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1:
            return

        # optional faint fill
        if fill_alpha and fill_alpha > 0.0:
            a = float(max(0.0, min(1.0, fill_alpha)))
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.addWeighted(overlay, a, vis, 1.0 - a, 0, dst=vis)

        # outline
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, int(thickness))

        if label:
            cv2.putText(
                vis, str(label),
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_bgr,
                1,
                cv2.LINE_AA,
            )

    def _attach_breaker_display_geometry(self, analyzer_result, parser_result):
        """
        Add normalized vertical positions to each detected breaker.

        These values are based on the exact analyzer gray image used to build
        the individual review overlay, so the UI can place left/right labels
        at the same vertical location as the breaker in the overlay.

        Existing breaker data is preserved. We only add:
            yTopRatio
            yBottomRatio
            yCenterRatio
        """
        if not isinstance(parser_result, dict):
            return

        ar = analyzer_result or {}
        gray = ar.get("gray")

        if gray is None or not hasattr(gray, "shape"):
            return

        image_h = int(gray.shape[0])

        if image_h <= 0:
            return

        breakers = parser_result.get("detected_breakers") or []

        if not isinstance(breakers, list):
            return

        for breaker in breakers:
            if not isinstance(breaker, dict):
                continue

            try:
                row_top = int(breaker.get("rowTop"))
                row_bottom = int(breaker.get("rowBottom"))
            except (TypeError, ValueError):
                continue

            if row_bottom <= row_top:
                continue

            center_y = (row_top + row_bottom) / 2.0

            breaker["yTopRatio"] = max(
                0.0,
                min(1.0, float(row_top) / float(image_h)),
            )

            breaker["yBottomRatio"] = max(
                0.0,
                min(1.0, float(row_bottom) / float(image_h)),
            )

            breaker["yCenterRatio"] = max(
                0.0,
                min(1.0, float(center_y) / float(image_h)),
            )

    def _build_review_overlay(self, *, image_path, analyzer_result, header_result, parser_result, dedup_name):
        """
        Writes ONE combined review overlay:
          - Magenta: winning header attrs
          - Green: trip columns (or combo columns if combined)
          - Blue: poles columns (separated)
        Returns absolute filepath or None.
        """
        ar = analyzer_result or {}
        gray = ar.get("gray")
        if gray is None or not hasattr(gray, "shape"):
            return None

        H, W = gray.shape[:2]
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Where to save (job_root/pdf_images/review_overlays/)
        norm_img = os.path.abspath(image_path).replace("\\", "/")

        rel_path = None
        out_path = None

        if "/pdf_images/" in norm_img:
            job_root = norm_img.split("/pdf_images/")[0]
            review_dir = self._ensure_dir(os.path.join(job_root, "pdf_images", "review_overlays"))
            safe_base = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(dedup_name or "panel")).strip("_") or "panel"
            fname = f"{safe_base}_review_overlay.png"
            out_path = os.path.join(review_dir, fname)
            rel_path = f"pdf_images/review_overlays/{fname}"
        else:
            # fallback if we can't infer job_root
            src_dir = os.path.dirname(os.path.abspath(image_path))
            review_dir = self._ensure_dir(os.path.join(src_dir, "review_overlays"))
            safe_base = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(dedup_name or "panel")).strip("_") or "panel"
            fname = f"{safe_base}_review_overlay.png"
            out_path = os.path.join(review_dir, fname)
            rel_path = f"review_overlays/{fname}"

        # --- Magenta header attrs ---
        # EXPECTATION: header parser returns:
        # header_result["winningBoxes"] = {"name":[x1,y1,x2,y2], "voltage":[...], ...}
        # header_result["boxImageShape"] = {"w": <int>, "h": <int>}  (shape used for those coords)
        hdr = header_result if isinstance(header_result, dict) else {}
        winning = hdr.get("winningBoxes") if isinstance(hdr.get("winningBoxes"), dict) else {}
        shape = hdr.get("boxImageShape") if isinstance(hdr.get("boxImageShape"), dict) else {}
        src_w = int(shape.get("w") or W)
        src_h = int(shape.get("h") or H)

        MAGENTA = (255, 0, 255)  # BGR
        for k, box in winning.items():
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            scaled = self._scale_box(box, src_w, src_h, W, H)
            self._draw_box(vis, scaled, MAGENTA, label=k, fill_alpha=0.0, thickness=3)

        # --- Breaker cell overlays ---
        #
        # Separated layout:
        #   good trip/amps cell -> GREEN
        #   good poles cell     -> BLUE
        #   unreadable/invalid  -> RED
        #
        # Combined layout:
        #   good combo cell     -> GREEN
        #   unreadable/invalid  -> RED
        #
        # Empty cells are intentionally absent from reviewCells and therefore
        # receive no overlay.

        prs = parser_result if isinstance(parser_result, dict) else {}
        review_cells = prs.get("reviewCells")
        if not isinstance(review_cells, list):
            review_cells = []

        GREEN = (0, 200, 0)    # BGR: valid amps or valid combined cell
        BLUE = (255, 0, 0)     # BGR: valid poles cell
        RED = (0, 0, 255)      # BGR: occupied but unreadable/invalid

        for cell in review_cells:
            if not isinstance(cell, dict):
                continue

            try:
                x1 = int(cell.get("xLeft"))
                x2 = int(cell.get("xRight"))
                y1 = int(cell.get("rowTop"))
                y2 = int(cell.get("rowBottom"))
            except (TypeError, ValueError):
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            role = str(cell.get("role") or "").strip().lower()
            status = str(cell.get("status") or "").strip().lower()

            if status == "issue":
                color = RED
                label = "CHECK"
                fill_alpha = 0.18
                thickness = 3

            elif status == "good" and role == "poles":
                color = BLUE
                label = "P"
                fill_alpha = 0.10
                thickness = 2

            elif status == "good" and role in {"trip", "combo"}:
                color = GREEN
                label = "A" if role == "trip" else "OK"
                fill_alpha = 0.10
                thickness = 2

            else:
                continue

            self._draw_box(
                vis,
                [x1, y1, x2, y2],
                color,
                label=label,
                fill_alpha=fill_alpha,
                thickness=thickness,
            )

        # Save one combined file for the UI
        safe_base = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(dedup_name or "panel")).strip("_") or "panel"
        out_path = os.path.join(review_dir, f"{safe_base}_review_overlay.png")
        try:
            ok = cv2.imwrite(out_path, vis)
            if not ok:
                if self.debug:
                    print(f"[WARN] cv2.imwrite returned False for: {out_path}")
                return None
            return rel_path
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to write review overlay: {e}")
            return None

    def _ensure_analyzer(self):
        if self._analyzer is None:
            self._analyzer = BreakerTableAnalyzer(debug=self.debug, reader=self._shared_reader)
        return self._analyzer

    def _ensure_header_parser(self):
        if self._header_parser is None:
            self._header_parser = PanelHeaderParser(debug=self.debug, reader=self._shared_reader)
        return self._header_parser

    def run(
        self,
        image_path: str,
        *,
        run_analyzer: bool = True,
        run_header:  bool = True,
        run_parser:  bool = True,
    ) -> dict:
        """
        Execute the pipeline in strict order:
          Analyzer → Header Parser → (header check) → ALT Table Parser

        Returns the same dict shape as the legacy parse_image().
        """
        img = os.path.abspath(os.path.expanduser(image_path))

        analyzer_result = None
        header_result   = None
        parser_result   = None

        # --- 1) Analyzer ---
        analyzer = self._ensure_analyzer()
        if run_analyzer:
            try:
                analyzer_result = analyzer.analyze(img)
            except Exception as e:
                analyzer_result = None
                if self.debug:
                    print(f"[WARN] Analyzer failed: {e}")

        # --- 2) Header Parser ---
        if run_header:
            try:
                header_parser = self._ensure_header_parser()
                if hasattr(analyzer, "reader"):
                    header_parser.reader = analyzer.reader
                if analyzer_result and isinstance(analyzer_result, dict):
                    hy   = analyzer_result.get("header_y")
                    gray = analyzer_result.get("gray")
                    if isinstance(hy, (int, float)) and hasattr(gray, "shape"):
                        H_ana = float(gray.shape[0])
                        header_ratio = max(0.0, min(1.0, float(hy) / H_ana))
                        header_result = header_parser.parse_panel(img, header_y_ratio=header_ratio)
                    else:
                        header_result = header_parser.parse_panel(img)
                else:
                    header_result = header_parser.parse_panel(img)
            except Exception as e:
                header_result = None
                if self.debug:
                    print(f"[WARN] Header parser failed: {e}")
        header_result_raw = copy.deepcopy(header_result) if isinstance(header_result, dict) else None

        # --- 2b) Apply de-duped display name as early as possible ---
        try:
            base_name, _v, _b, _m, _trim, _encl, _ = self._extract_panel_keys(analyzer_result, header_result, None)
        except Exception:
            base_name = None
        dedup_name = _dedupe_name(base_name)
        if isinstance(header_result, dict):
            header_result["name"] = dedup_name
        try:
            nh = (analyzer_result or {}).get("normalized_header")
            if isinstance(nh, dict):
                nh["name"] = dedup_name
        except Exception:
            pass

        # --- 3) Header validity check (but DO NOT skip the review overlay) ---
        panel_status = None
        header_invalid = False
        skip_bom_due_to_special_header = False

        special_kind = self._special_header_kind(header_result)
        if special_kind in {
            "switchboard",
            "wireway",
            "lighting_schedule",
            "conduit_schedule",
            "mechanical_schedule",
            "disconnect_schedule",
            "motor_schedule",
            "transformer_schedule",
            "generator_schedule",
            "labor_schedule",
            "equipment_schedule",
            "inverter",
        }:
            skip_bom_due_to_special_header = True
            special_note = (
                self._special_header_note(header_result)
                or f"{special_kind.replace('_', ' ')} detected"
            )
            panel_status = f"detected but skipped ({dedup_name})"
            if isinstance(header_result, dict):
                header_result["panelNote"] = special_note

        should_run_parser = (run_parser and not skip_bom_due_to_special_header)
        try:
            _dn, volts, bus_amps, main_amps, _trim_style, _enclosure, _spaces_unused = self._extract_panel_keys(
                analyzer_result, header_result, None
            )
            volts_i = self._parse_voltage(volts)
            volts_invalid = (volts_i is None) or (volts_i not in VALID_VOLTAGES)
            bus_invalid   = not self._is_valid_amp(bus_amps)                          # REQUIRED
            main_invalid  = (main_amps is not None) and (not self._is_valid_amp(main_amps))  # OPTIONAL
            amp_over_max_note = self._amp_over_max_message(bus_amps, main_amps)

            if volts_invalid or bus_invalid or main_invalid:
                special_note = self._special_header_note(header_result)

                if isinstance(header_result, dict):
                    if special_note:
                        header_result["panelNote"] = special_note
                    elif amp_over_max_note:
                        header_result["panelNote"] = amp_over_max_note

                # Important:
                # Do NOT set panel_status here.
                # Let the later structural/header validation decide the final status
                # after the breaker table parser runs.
                header_invalid = True

                if self.debug:
                    miss = []
                    if volts_invalid: miss.append(f"volts={volts!r}")
                    if bus_invalid:   miss.append(f"bus_amps={bus_amps!r}")
                    if main_invalid:  miss.append(f"main_amps={main_amps!r}")
                    print("[INFO] Header invalid; proceeding to TABLE PARSER anyway "
                          f"(name={dedup_name!r}; {', '.join(miss)})")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Header pre-parse validation failed: {e}")

        # ===>>> 3b) TABLE PARSER (ALT only) — PLACE IS HERE, always after header check
        if should_run_parser:
            try:
                # ALT only, no fallbacks
                parser = BreakerTableParser(debug=self.debug, reader=getattr(analyzer, "reader", None))
                if analyzer_result is not None:
                    parser_result = parser.parse_from_analyzer(analyzer_result)
                else:
                    parser_result = parser.parse_from_analyzer({})
            except Exception as e:
                parser_result = None
                if self.debug:
                    print(f"[WARN] Parser failed: {e}")
        # Ensure the table parser result advertises the deduped name for the UI
        # and expose normalized breaker positions for the review UI.

        if isinstance(parser_result, dict):
            parser_result["name"] = dedup_name

            self._attach_breaker_display_geometry(
                analyzer_result,
                parser_result,
            )

        review_overlay_path = None
        try:
            review_overlay_path = self._build_review_overlay(
                image_path=img,
                analyzer_result=analyzer_result,
                header_result=(header_result_raw or header_result),
                parser_result=parser_result,
                dedup_name=dedup_name,
            )
        except Exception as e:
            if self.debug:
                print(f"[WARN] Review overlay generation failed: {e}")

        # fallback preview path if overlay could not be written
        if not review_overlay_path:
            review_overlay_path = img

        # expose to UI
        if isinstance(parser_result, dict):
            parser_result["reviewOverlayPath"] = review_overlay_path
        if isinstance(header_result, dict):
            header_result["reviewOverlayPath"] = review_overlay_path

        if skip_bom_due_to_special_header:
            header_result = self._mask_header_non_name(header_result, detected_name=dedup_name)
            parser_result = self._mask_parser_non_name(parser_result, detected_name=dedup_name)

        # Do NOT mask recoverable header-invalid panels here.
        # Missing voltage/bus/main should keep any valid header attrs and parser breakers.

        # --- optional legacy prints (only if table parser ran) ---
        if self.debug and parser_result is not None:
            try:
                print("\n>>> FINAL (legacy-compatible prints) >>>")
                print(parser_result.get("spaces"))
                print(parser_result.get("detected_breakers"))
            except Exception:
                pass

        # --- 4) Panel validity & masking logic (spaces + header recheck) ---
        try:
            if not skip_bom_due_to_special_header:
                _dn2, volts, bus_amps, main_amps, trim_style, enclosure, spaces = self._extract_panel_keys(
                    analyzer_result, header_result, parser_result
                )

                volts_i = self._parse_voltage(volts)
                volts_invalid  = (volts_i is None) or (volts_i not in VALID_VOLTAGES)
                bus_invalid    = not self._is_valid_amp(bus_amps)
                main_invalid   = (main_amps is not None) and (not self._is_valid_amp(main_amps))
                amp_over_max_note = self._amp_over_max_message(bus_amps, main_amps)

                if isinstance(spaces, int):
                    spaces_norm = SNAP_MAP.get(spaces, spaces)
                    spaces_invalid = spaces_norm is None or spaces_norm <= 0
                else:
                    spaces_norm = None
                    spaces_invalid = True

                ar = analyzer_result or {}
                prs = parser_result or {}

                panel_size_unknown = (
                    spaces_invalid
                    and ar.get("panel_size") is None
                    and ar.get("footer_y") is None
                ) or prs.get("sizeDetectionStatus") == "unknown"

                header_fields_invalid = volts_invalid or bus_invalid or main_invalid

                # Structural breaker-table problems win first.
                # If spaces/panel size are unknown, suppress breakers even if voltage/bus/main
                # are also missing.
                if panel_size_unknown or spaces_invalid:
                    note = "Could not determine panel size/spaces."

                    panel_status = f"detected but skipped ({dedup_name})"

                    if isinstance(header_result, dict):
                        header_result["panelNote"] = note

                        attrs = header_result.get("attrs")
                        if isinstance(attrs, dict):
                            attrs["detected_breakers"] = []
                            attrs["breaker_data_suppressed"] = True
                            attrs["breaker_suppression_reason"] = note

                    if not isinstance(parser_result, dict):
                        parser_result = {}

                    parser_result["name"] = dedup_name
                    parser_result["spaces"] = None
                    parser_result["detected_breakers"] = []
                    parser_result["breakerCounts"] = {}
                    parser_result["gfiBreakerCounts"] = {}
                    parser_result["sizeDetectionStatus"] = "unknown"
                    parser_result["bodyParseStatus"] = "skipped"
                    parser_result["bodyParseNote"] = note


                # Recoverable header fields are missing/bad.
                # Preserve breakers.
                elif header_fields_invalid:
                    missing = []

                    if volts_invalid:
                        missing.append("voltage")
                    if bus_invalid:
                        missing.append("bus amps")
                    if main_invalid:
                        missing.append("main amps")

                    special_note = self._special_header_note(header_result)

                    if special_note:
                        note = special_note
                    elif amp_over_max_note:
                        note = amp_over_max_note
                    else:
                        note = "Missing/invalid: " + ", ".join(missing)

                    panel_status = f"detected but skipped ({dedup_name})"

                    if isinstance(header_result, dict):
                        header_result["panelNote"] = note
                        header_result["headerValidationStatus"] = "recoverable_header_issue"
                        header_result["headerValidationMissing"] = missing

                        attrs = header_result.get("attrs")
                        if isinstance(attrs, dict):
                            attrs["headerValidationStatus"] = "recoverable_header_issue"
                            attrs["headerValidationMissing"] = missing

                    if isinstance(parser_result, dict):
                        parser_result["bodyParseStatus"] = parser_result.get("bodyParseStatus") or "parsed"
                        parser_result["bodyParseNote"] = (
                            "Breaker table preserved despite recoverable header issue: " + note
                        )

        except Exception as e:
            if self.debug:
                print(f"[WARN] Panel validation/masking failed: {e}")

        # --- combined result ---
        return {
            "apiVersion": API_VERSION,
            "origin": API_ORIGIN,
            "stages": {
                "analyzer": run_analyzer,
                "parser": run_parser,
                "header": run_header,
            },
            "results": {
                "analyzer": analyzer_result,
                "parser": parser_result,
                "header": header_result,
            },
            "panelStatus": panel_status,  # None if OK; message if flagged
        }


def parse_image(
    image_path: str,
    *,
    run_analyzer: bool = True,
    run_parser: bool = True,
    run_header: bool = True,
    debug: bool = True
):
    pipe = BreakerTablePipeline(debug=debug)
    return pipe.run(
        image_path,
        run_analyzer=run_analyzer,
        run_header=run_header,
        run_parser=run_parser,
    )

if __name__ == "__main__":
    modA = sys.modules[BreakerTableAnalyzer.__module__]
    implA = inspect.getsourcefile(BreakerTableAnalyzer) or inspect.getfile(BreakerTableAnalyzer)
    print(">>> DEV Analyzer version:", getattr(modA, "ANALYZER_VERSION", "unknown"))
    print(">>> DEV Analyzer file:", os.path.abspath(implA))

    modP = sys.modules.get(BreakerTableParser.__module__)
    implP = inspect.getsourcefile(BreakerTableParser) or inspect.getfile(BreakerTableParser)
    print(">>> DEV Parser version:", PARSER_VERSION if 'PARSER_VERSION' in globals() else "unknown")
    print(">>> DEV Parser file:", os.path.abspath(implP))
    print(">>> DEV Parser module:", BreakerTableParser.__module__)

    # Require an image path; no hardcoded fallback
    if len(sys.argv) < 2:
        print("Usage: python BreakerTableParserAPIv4.py /path/to/image.png [--no-analyzer] [--no-parser] [--no-header]")
        sys.exit(1)

    img = sys.argv[1]
    print(">>> DEV Image:", img)

    # Simple flags: --no-analyzer --no-parser --no-header
    args = sys.argv[2:]
    run_analyzer = "--no-analyzer" not in args
    run_parser   = "--no-parser" not in args
    run_header   = "--no-header" not in args

    pipeline = BreakerTablePipeline(debug=True)
    result = pipeline.run(
        img,
        run_analyzer=run_analyzer,
        run_parser=run_parser,
        run_header=run_header,
    )

    print("\n>>> Stage Summary:")
    for key, val in result["results"].items():
        print(f"  {key}: {'OK' if val else 'None'}")
    if result.get("panelStatus"):
        print(f"  status: {result['panelStatus']}")
    else:
        print("  status: OK")
