# OcrLibrary/BreakerTableParser4.py
from __future__ import annotations
import os, re, json, cv2, numpy as np
from typing import Dict, Optional, List, Tuple

try:
    import easyocr
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

PARSER_VERSION = "V19_4_Parser"
PARSER_ORIGIN  = __file__

# --- utility mirrors from original ---
def _load_jsonc(path: str):
    try:
        import re, json
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
        return json.loads(s)
    except Exception:
        return None


class BreakerTableParser:
    """
        This class performs ONLY the second stage:
      - takes analyzer payload (gridless image, tp_cols, centers, header/footer)
      - crops 4 TP columns, saves crops next to gridless image
      - does OCR for TRIP/POLES with same logic (incl. dash/peripheral filters)
      - writes column_data.json (or top/bottom when chunked)
      - builds detected_breakers
      - writes per-crop debug overlays/json to <src_dir>/debug when debug=True
    """

    def __init__(self, debug: bool = False, config_path: str = "breaker_labels_config.jsonc", reader=None):
        self.debug = debug
        self._cfg = _load_jsonc(config_path) or {}
        self.reader = reader
        if self.reader is None and _HAS_OCR:
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)

    # ======= public =======
    def parse_from_analyzer(self, analyzer_payload: Dict) -> Dict:
        gray        = analyzer_payload["gray"]
        centers     = analyzer_payload["centers"]
        header_y    = analyzer_payload["header_y"]
        footer_y    = analyzer_payload["footer_y"]
        gridless    = analyzer_payload.get("gridless_gray")
        gridless_path = analyzer_payload.get("gridless_path")
        src_path    = analyzer_payload.get("src_path")
        debug_dir   = analyzer_payload.get("debug_dir")

        # Use ORIGINAL (pre-inpaint) gray for line detection; gridless for OCR
        work_gray = gridless if gridless is not None else gray
        orig_gray = analyzer_payload.get("orig_gray") or gray

        # Basic guards
        H, W = gray.shape[:2]
        if header_y is None or footer_y is None or footer_y <= header_y:
            if self.debug:
                print("[PARSER] Missing/invalid header/footer; nothing to OCR.")
            return {
                "rows": [],
                "detected_breakers": [],
                "row_count": len(centers or []),
                "spaces": analyzer_payload.get("spaces"),
                "tp_meta": {"combined": {"left": False, "right": False}, "columns_px": {}},
                "top_rows_dump": [],
                "top_rows_y_band": None,
                "header_map": {},
                "header_debug": {}
            }

        # Estimate two-row band height from row centers; fallback if needed
        med_gap = None
        if centers and len(centers) >= 2:
            gaps = np.diff(sorted(int(c) for c in centers))
            med_gap = float(np.median(gaps)) if gaps.size else None
        if not med_gap or med_gap <= 0:
            med_gap = max(18.0, 0.032 * H)  # conservative default

        # Build band: start just below header label row; cover ~2 rows, clamped to footer
        y1 = max(0, int(header_y) + 6)
        y2 = min(H - 1, int(header_y + 2.2 * med_gap))
        y2 = min(y2, int(footer_y) - 1)
        if y2 <= y1:
            y2 = min(int(header_y) + int(2.0 * med_gap), int(footer_y) - 1)
            y1 = max(0, min(y1, y2 - 3))

        # Prefer tight row crops for the first two rows to boost header OCR on large panels
        row_spans = self._compute_top_row_spans(centers, int(header_y), int(footer_y), H)
        dump, toprow_overlay = self._ocr_top_rows_precropped(work_gray, row_spans)

        # Decide which y-band to associate with these detections (union of spans)
        if row_spans:
            band_y1 = int(min(a for (a, _) in row_spans))
            band_y2 = int(max(b for (_, b) in row_spans))
        else:
            band_y1, band_y2 = int(max(0, int(header_y) + 6)), int(min(H - 1, int(footer_y) - 1))

        # Fallback to legacy band OCR if nothing useful came back
        if not dump:
            # Build the legacy band
            med_gap = med_gap if med_gap else max(18.0, 0.032 * H)
            y1 = max(0, int(header_y) + 6)
            y2 = min(H - 1, int(header_y + 2.2 * med_gap))
            y2 = min(y2, int(footer_y) - 1)
            if y2 <= y1:
                y2 = min(int(header_y) + int(2.0 * med_gap), int(footer_y) - 1)
                y1 = max(0, min(y1, y2 - 3))

            dump, toprow_overlay = self._ocr_top_rows_band(work_gray, y1, y2)
            band_y1, band_y2 = int(y1), int(y2)

        # ---- classify headers (now that we have 'dump' and the correct y-band) ----
        header_map, header_dbg = self._classify_top_rows_headers(
            top_rows_dump=dump,
            y_band=(band_y1, band_y2),
            page_width=W
        )

        # Persist debug artifacts
        base_dir = os.path.dirname(gridless_path or src_path or ".")
        base_name = os.path.splitext(os.path.basename(gridless_path or src_path or "page"))[0]
        if debug_dir and self.debug:
            os.makedirs(debug_dir, exist_ok=True)
            ov_path = os.path.join(debug_dir, f"{base_name}_toprows_overlay.png")
            try:
                cv2.imwrite(ov_path, toprow_overlay)
            except Exception as e:
                print(f"[PARSER] Failed to write overlay: {e}")
            json_path = os.path.join(debug_dir, f"{base_name}_toprows_dump.json")
            try:
                self._write_json(json_path, {"y_band":[int(band_y1), int(band_y2)], "items": dump})
            except Exception as e:
                print(f"[PARSER] Failed to write dump json: {e}")
            hdr_dbg_path = os.path.join(debug_dir, f"{base_name}_header_classify_debug.json")
            try:
                self._write_json(hdr_dbg_path, header_dbg)
            except Exception as e:
                print(f"[PARSER] Failed to write header debug json: {e}")
            hdr_ov_path = os.path.join(debug_dir, f"{base_name}_header_overlay.png")
            try:
                hdr_overlay = self._render_header_overlay(
                    gray=work_gray, y_band=(int(band_y1), int(band_y2)),
                    all_items=dump, header_map=header_map
                )
                cv2.imwrite(hdr_ov_path, hdr_overlay)
            except Exception as e:
                print(f"[PARSER] Failed to write header overlay: {e}")
            spans_dbg_path = os.path.join(debug_dir, f"{base_name}_toprows_spans.json")
            try:
                self._write_json(spans_dbg_path, {"row_spans": [[int(a), int(b)] for (a,b) in row_spans]})
            except Exception as e:
                print(f"[PARSER] Failed to write row spans: {e}")

        # === from header_map -> SNAP columns -> crop/OCR -> outputs ===
        rows_boxes = self._row_boxes_from_centers(work_gray.shape, centers, header_y, footer_y)
        n_rows = len(centers or [])

        # 1) Columns from header_map, snapped to grid (use ORIGINAL gray for v-lines)
        cols_snapped = self._columns_from_header_map(
            header_map=header_map,
            page_width=W,
            gray_for_lines=orig_gray,
            header_y=int(header_y),
            footer_y=int(footer_y)
        )

        # 2) Run the column strip scan (single or chunked like V19_4)
        base_dir  = os.path.dirname(gridless_path or src_path or ".")
        debug_dir = analyzer_payload.get("debug_dir") or os.path.join(base_dir, "debug")
        gridless_out_path = gridless_path or (src_path and os.path.splitext(src_path)[0] + "_gridless.png") or os.path.join(base_dir, "gridless.png")
        os.makedirs(debug_dir, exist_ok=True)

        all_finds = {}
        chunk_meta = None

        if n_rows <= 21:
            finds = self._scan_tp_column_strips(
                work_gray, cols_snapped, int(header_y), int(footer_y), gridless_out_path,
                save_debug=self.debug, debug_dir=debug_dir
            )
            try:
                col_json_path = os.path.join(os.path.dirname(gridless_out_path), "column_data.json")
                self._write_json(col_json_path, {
                    "l_trip":  finds.get("l_trip",  []),
                    "l_poles": finds.get("l_poles", []),
                    "r_poles": finds.get("r_poles", []),
                    "r_trip":  finds.get("r_trip",  []),
                })
            except Exception as e:
                if self.debug: print(f"[PARSER] failed to write column_data.json: {e}")
            all_finds = finds
        else:
            # Chunking: top 21 rows, bottom remainder (match V19_4)
            top_rows = rows_boxes[:21]
            bot_rows = rows_boxes[21:]

            Htot = work_gray.shape[0]
            gaps = np.diff([rb[0] for rb in rows_boxes] + [rows_boxes[-1][1]])
            med_gap = float(np.median(gaps)) if len(gaps) else 18.0
            pad = int(max(6, 0.20 * med_gap, 0.004 * Htot))

            y_top_min = max(0, min(r[0] for r in top_rows) - pad)
            y_top_max = min(Htot - 1, max(r[1] for r in top_rows) + pad)
            y_bot_min = max(0, min(r[0] for r in bot_rows) - pad)
            y_bot_max = min(Htot - 1, max(r[1] for r in bot_rows) + pad)

            base_name = os.path.splitext(os.path.basename(gridless_out_path))[0]
            top_gridless_path = os.path.join(base_dir, base_name.replace("_gridless", "_gridless_top") + ".png")
            bot_gridless_path = os.path.join(base_dir, base_name.replace("_gridless", "_gridless_bottom") + ".png")

            finds_top = self._scan_tp_column_strips(
                work_gray, cols_snapped, y_top_min, y_top_max, top_gridless_path,
                save_debug=self.debug, debug_dir=debug_dir, suppress_top_wipes=True
            )
            finds_bot = self._scan_tp_column_strips(
                work_gray, cols_snapped, y_bot_min, y_bot_max, bot_gridless_path,
                save_debug=self.debug, debug_dir=debug_dir, suppress_top_wipes=True
            )

            try:
                self._write_json(os.path.join(base_dir, "column_data_top.json"), {
                    "l_trip":  finds_top.get("l_trip",  []),
                    "l_poles": finds_top.get("l_poles", []),
                    "r_poles": finds_top.get("r_poles", []),
                    "r_trip":  finds_top.get("r_trip",  []),
                })
                self._write_json(os.path.join(base_dir, "column_data_bottom.json"), {
                    "l_trip":  finds_bot.get("l_trip",  []),
                    "l_poles": finds_bot.get("l_poles", []),
                    "r_poles": finds_bot.get("r_poles", []),
                    "r_trip":  finds_bot.get("r_trip",  []),
                })
            except Exception as e:
                if self.debug: print(f"[PARSER] failed to write chunked column_data: {e}")

            all_finds = {
                "l_trip":  (finds_top.get("l_trip",  []) + finds_bot.get("l_trip",  [])),
                "l_poles": (finds_top.get("l_poles", []) + finds_bot.get("l_poles", [])),
                "r_poles": (finds_top.get("r_poles", []) + finds_bot.get("r_poles", [])),
                "r_trip":  (finds_top.get("r_trip",  []) + finds_bot.get("r_trip",  [])),
            }
            chunk_meta = {
                "top":    {"rows": 21,        "y_band": [int(y_top_min), int(y_top_max)]},
                "bottom": {"rows": n_rows-21, "y_band": [int(y_bot_min), int(y_bot_max)]},
            }

        # Build detected_breakers (same as V19_4)
        def _to_int(s):
            try: return int(str(s).strip())
            except Exception: return None

        counts = {}
        L_A, L_P = all_finds.get("l_trip", []),  all_finds.get("l_poles", [])
        R_A, R_P = all_finds.get("r_trip", []),  all_finds.get("r_poles", [])
        for a_list, p_list in ((L_A, L_P), (R_A, R_P)):
            n = max(len(a_list), len(p_list))
            for i in range(n):
                amps  = _to_int(a_list[i]) if i < len(a_list) else None
                poles = _to_int(p_list[i]) if i < len(p_list) else None
                if amps is None or poles is None: continue
                key = (poles, amps)
                counts[key] = counts.get(key, 0) + 1

        detected_breakers = [
            {"poles": k[0], "amperage": k[1], "specialFeatures": "", "count": v}
            for k, v in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        ]

        if self.debug:
            ov_path = os.path.join(debug_dir, f"{base_name if gridless_out_path else 'page'}_page_overlay.png")
            self._save_page_overlay(work_gray, int(header_y), int(footer_y), cols_snapped, header_map, ov_path)

        tp_meta = {"combined": {"left": False, "right": False}, "columns_px": cols_snapped}
        if chunk_meta: tp_meta["chunks"] = chunk_meta

        return {
            "rows": [],
            "detected_breakers": detected_breakers,
            "row_count": len(centers or []),
            "spaces": analyzer_payload.get("spaces"),
            "tp_meta": tp_meta,
            "top_rows_dump": dump,
            "top_rows_y_band": [int(band_y1), int(band_y2)],
            "header_map": header_map,
            "header_debug": header_dbg
        }

    def _detect_vertical_lines(self, gray: np.ndarray, y1: int, y2: int) -> List[int]:
        """
        Return x positions (centers) of vertical gridlines in band [y1,y2].
        Robust to scan noise. Works on original (pre-inpaint) gray is ideal, but gridless works too.
        """
        H, W = gray.shape[:2]
        y1 = max(0, min(H-2, int(y1)))
        y2 = max(y1+1, min(H-1, int(y2)))
        band = gray[y1:y2, :]
        if band.size == 0:
            return []

        g  = cv2.GaussianBlur(band, (3,3), 0)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 21, 10)

        # Vertical kernel: tall and thin to preserve vertical rules.
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, int(0.06*(y2-y1)))))
        v  = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kv, iterations=1)

        # Column-wise projection -> peaks
        proj = v.sum(axis=0).astype(np.float32)
        if proj.max() <= 0:
            return []

        thr = 0.45 * proj.max()
        xs  = np.where(proj >= thr)[0].astype(int)
        if xs.size == 0:
            return []

        # Merge close hits
        peaks = [int(xs[0])]
        for x in xs[1:]:
            if x - peaks[-1] > 2:
                peaks.append(int(x))

        # Refine each peak to local centroid
        def centroid(x0: int) -> int:
            l = max(0, x0-3); r = min(W-1, x0+3)
            w = proj[l:r+1]; idx = np.arange(l, r+1, dtype=np.float32)
            s = w.sum()
            return int(round((idx*w).sum()/s)) if s>0 else x0

        return sorted(set(centroid(p) for p in peaks))


    def _bands_from_vlines(self, vlines: List[int], page_width: int) -> List[Tuple[int,int]]:
        """
        Convert sorted vertical line x-positions into x-bands (columns).
        Edges: [0] ... vlines ... [W-1]
        """
        W = int(page_width)
        if not vlines or len(vlines) < 1:
            return [(0, W-1)]
        edges = [0] + sorted(int(x) for x in vlines) + [W-1]
        bands = []
        pad = max(2, int(0.002 * W))
        for i in range(len(edges)-1):
            x1, x2 = edges[i], edges[i+1]
            a = max(0, x1 + pad); b = min(W-1, x2 - pad)
            if b <= a:
                m = max(0, min(W-2, (x1+x2)//2))
                a, b = max(0, m-1), min(W-1, m+1)
            bands.append((a, b))
        return bands


    def _pick_band_containing_x(self, bands: List[Tuple[int,int]], x: float) -> Optional[Tuple[int,int]]:
        for (a,b) in bands:
            if a <= x <= b:
                return (a,b)
        # fallback: nearest by center
        if bands:
            return min(bands, key=lambda ab: abs(0.5*(ab[0]+ab[1]) - x))
        return None

    def _columns_from_header_map(self, header_map: Dict[str, List[dict]], page_width: int,
                                gray_for_lines: np.ndarray, header_y: int, footer_y: int) -> Dict[str, Tuple[int,int]]:
        """
        Use OCR header picks to choose TRIP/POLES columns, then SNAP to nearest
        grid bands detected in [header_y, footer_y]. Handles overlap by splitting.
        """
        W = int(page_width)
        mid = W / 2.0

        # 1) detect vertical gridlines then make bands
        vlines = self._detect_vertical_lines(gray_for_lines, header_y, footer_y)
        bands  = self._bands_from_vlines(vlines, W)

        # Helper to choose left/right pick for each role
        def side_picks(role: str, left: bool) -> List[dict]:
            items = header_map.get(role, []) or []
            if not items:
                return []
            if left:
                return sorted([d for d in items if float(d.get("x_center", 0)) <  mid],
                            key=lambda d: d["x_center"])
            else:
                return sorted([d for d in items if float(d.get("x_center", 0)) >= mid],
                            key=lambda d: d["x_center"])

        L_poles = side_picks("poles", True)
        R_poles = side_picks("poles", False)
        L_trip  = side_picks("trip",  True)
        R_trip  = side_picks("trip",  False)

        cols: Dict[str, Tuple[int,int]] = {}

        def to_band(pick: dict) -> Optional[Tuple[int,int]]:
            x = float(pick.get("x_center", (pick.get("x1",0)+pick.get("x2",0))/2.0))
            return self._pick_band_containing_x(bands, x)

        # Primaries from POLES, neighbors: left side (TRIP left of POLES), right side (TRIP right)
        if L_poles:
            lP = to_band(L_poles[0])
            if lP:
                cols["L_POLES"] = lP
                idx = bands.index(lP) if lP in bands else None
                if idx is not None and idx-1 >= 0:
                    cols["L_TRIP"] = bands[idx-1]
                elif L_trip:
                    lt = to_band(L_trip[0])
                    if lt: cols["L_TRIP"] = lt

        if R_poles:
            rP = to_band(R_poles[0])
            if rP:
                cols["R_POLES"] = rP
                idx = bands.index(rP) if rP in bands else None
                if idx is not None and idx+1 < len(bands):
                    cols["R_TRIP"] = bands[idx+1]
                elif R_trip:
                    rt = to_band(R_trip[0])
                    if rt: cols["R_TRIP"] = rt

        # Helper to get header center or fall back to band center (NO accidental /2 on header centers)
        def _center_from_header_or_band(hlist, band):
            hc = (hlist or [{}])[0].get("x_center")
            return float(hc) if hc is not None else 0.5 * (band[0] + band[1])

        # If L_TRIP and L_POLES collide -> split locally (TRIP left of POLES)
        def _overlap(a,b):
            if not a or not b: return 0.0
            L1,R1 = a; L2,R2 = b
            inter = max(0, min(R1,R2) - max(L1,L2) + 1)
            union = (R1-L1+1) + (R2-L2+1) - inter
            return inter / max(1, union)

        if "L_TRIP" in cols and "L_POLES" in cols:
            if cols["L_TRIP"] == cols["L_POLES"] or _overlap(cols["L_TRIP"], cols["L_POLES"]) > 0.65:
                xa = _center_from_header_or_band(header_map.get("trip"),  cols["L_TRIP"])
                xb = _center_from_header_or_band(header_map.get("poles"), cols["L_POLES"])
                left_band, right_band = self._split_band_between_headers(
                    gray_for_lines, header_y, footer_y, cols["L_POLES"], xa, xb
                )
                cols["L_TRIP"], cols["L_POLES"] = left_band, right_band

        if "R_TRIP" in cols and "R_POLES" in cols:
            if cols["R_TRIP"] == cols["R_POLES"] or _overlap(cols["R_TRIP"], cols["R_POLES"]) > 0.65:
                xa = _center_from_header_or_band(header_map.get("poles"), cols["R_POLES"])
                xb = _center_from_header_or_band(header_map.get("trip"),  cols["R_TRIP"])
                left_band, right_band = self._split_band_between_headers(
                    gray_for_lines, header_y, footer_y, cols["R_TRIP"], xa, xb
                )
                # On the right side: POLES is left of TRIP
                cols["R_POLES"], cols["R_TRIP"] = left_band, right_band

        # Backfills from TRIP picks if needed
        if "L_POLES" not in cols and L_trip:
            lt = to_band(L_trip[0])
            if lt:
                idx = bands.index(lt) if lt in bands else None
                if idx is not None and idx+1 < len(bands):
                    cols["L_TRIP"] = lt
                    cols["L_POLES"] = bands[idx+1]
        if "R_POLES" not in cols and R_trip:
            rt = to_band(R_trip[0])
            if rt:
                idx = bands.index(rt) if rt in bands else None
                if idx is not None and idx-1 >= 0:
                    cols["R_TRIP"] = rt
                    cols["R_POLES"] = bands[idx-1]

        # Final safety
        fb = self._tp_fallback(W)
        for k, v in fb.items():
            cols.setdefault(k, v)

        return cols

    def _render_header_overlay(self, gray: np.ndarray, y_band: Tuple[int,int],
                            all_items: List[dict], header_map: Dict[str, List[dict]]) -> np.ndarray:
        """
        Color overlay:
        - Yellow: all OCR detections within band
        - Green : POLES picks
        - Red   : TRIP picks
        - Blue  : DESC picks
        - Black : CKT picks
        Draw chosen boxes using EXACT coords from header_map when available.
        Fallback to a Y-aware nearest match if coords are missing.
        """
        H, W = gray.shape[:2]
        y1, y2 = int(y_band[0]), int(y_band[1])

        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(overlay, (0, y1), (W - 1, y2), (60, 60, 60), 1)

        YELLOW = (0,255,255); GREEN=(0,200,0); RED=(0,0,255); BLUE=(255,0,0); BLACK=(0,0,0); WHITE=(255,255,255)

        # all detections
        for it in (all_items or []):
            x1b,y1b,x2b,y2b = int(it["x1"]), int(it["y1"]), int(it["x2"]), int(it["y2"])
            cv2.rectangle(overlay, (x1b,y1b), (x2b,y2b), YELLOW, 2)
            label = f'{str(it.get("text",""))[:22]} ({float(it.get("conf",0.0)):.2f})'
            cv2.putText(overlay, label, (x1b, max(12, y1b-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

        def draw_pick(c, color: Tuple[int,int,int], tag: str):
            # 1) prefer exact coords if present
            if "x1" in c and "x2" in c:
                x1b, x2b = int(c["x1"]), int(c["x2"])
                y1b, y2b = int(c["y1"]), int(c["y2"])
            else:
                # 2) Y-aware nearest OCR match (fallback)
                xc = float(c.get("x_center", c.get("x", 0.0)))
                cy1, cy2 = int(c.get("y1", y1)), int(c.get("y2", y2))
                match = None
                best_key = (1e9, 1e9)
                for it in all_items:
                    ix1,iy1,ix2,iy2 = int(it["x1"]), int(it["y1"]), int(it["x2"]), int(it["y2"])
                    yov = max(0, min(cy2, iy2) - max(cy1, iy1))
                    if yov <= 0: continue
                    ixc = 0.5*(ix1+ix2)
                    key = (-yov, abs(ixc - xc))
                    if key < best_key:
                        best_key = key; match = it
                if match is not None:
                    x1b, y1b, x2b, y2b = int(match["x1"]), int(match["y1"]), int(match["x2"]), int(match["y2"])
                else:
                    # 3) final minimal box around candidate center
                    w = 64
                    x1b = int(max(0, xc - w/2)); x2b = int(min(W-1, xc + w/2))
                    y1b = cy1; y2b = cy2

            cv2.rectangle(overlay, (x1b,y1b), (x2b,y2b), color, 3)
            # show role + raw if available
            raw = tag
            if "text_norm" in c:
                raw = f"{tag}: {str(c['text_norm'])[:18]}"
            cv2.putText(overlay, raw, (x1b, max(14, y1b-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 3, cv2.LINE_AA)
            cv2.putText(overlay, raw, (x1b, max(14, y1b-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        for c in header_map.get("poles", []):        draw_pick(c, GREEN, "POLES")
        for c in header_map.get("trip", []):         draw_pick(c, RED,   "TRIP")
        for c in header_map.get("description", []):  draw_pick(c, BLUE,  "DESC")
        for c in header_map.get("ckt", []):          draw_pick(c, BLACK, "CKT")

        return overlay

    def _cluster_header_row(self, items: List[dict], y_band: Tuple[int, int]) -> Tuple[float, float]:
        """
        1D k-means (k=2) on y-center to split the OCR band into header row (top)
        and everything below. Returns:
            (y_threshold, top_cluster_mean)
        Keep tokens with y_center <= y_threshold.
        """
        y1, y2 = int(y_band[0]), int(y_band[1])
        if not items:
            return (y1 + y2) / 2.0, float(y1 + 1)

        ys = [0.5 * (int(d.get("y1", y1)) + int(d.get("y2", y2))) for d in items]
        y_min, y_max = min(ys), max(ys)
        if y_max <= y_min:
            return y_min + 1.0, y_min

        # init two centroids toward top/bottom of the band
        c1 = y_min + 0.25 * (y_max - y_min)
        c2 = y_min + 0.75 * (y_max - y_min)

        for _ in range(12):
            g1, g2 = [], []
            for y in ys:
                (g1 if abs(y - c1) <= abs(y - c2) else g2).append(y)
            if not g1 or not g2:
                break
            new_c1 = sum(g1) / len(g1)
            new_c2 = sum(g2) / len(g2)
            if abs(new_c1 - c1) < 0.5 and abs(new_c2 - c2) < 0.5:
                c1, c2 = new_c1, new_c2
                break
            c1, c2 = new_c1, new_c2

        top_mean, bot_mean = (c1, c2) if c1 <= c2 else (c2, c1)
        y_threshold = 0.5 * (top_mean + bot_mean)
        return float(y_threshold), float(top_mean)

    def _classify_top_rows_headers(self, top_rows_dump: List[dict], y_band: Tuple[int, int], page_width: int):
        """
        Hard-coded header classifier for: CKT, DESCRIPTION, TRIP, POLES.
        - Prefilters to header row, splits glued tokens, scores by vocab similarity.
        - Picks up to 2 per type with global X conflict guard.
        - RETURNS x1/x2 so overlay can draw exact boxes.
        - Prints ONE tidy debug block to stdout.
        """
        import re, difflib

        VOCAB = {
            "ckt": ["CKT", "CCT"],
            "description": ["CIRCUIT DESCRIPTION", "DESCRIPTION", "LOAD DESCRIPTION", "DESIGNATION", "LOAD DESIGNATION", "NAME"],
            "trip": ["TRIP", "AMPS", "AMP", "BREAKER", "BKR", "SIZE"],
            "poles": ["POLES", "POLE", "P"],
        }
        TYPES = ("ckt", "description", "trip", "poles")
        ACCEPT_THR = 0.60
        X_TOL = 24
        STR_W, CONF_W = 0.90, 0.10

        def norm(s: str) -> str:
            s = (s or "").upper()
            s = re.sub(r"[\[\]():.,/_\-]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s.replace("TRLP","TRIP").replace("DESCRLPTION","DESCRIPTION")

        def sim(a: str, b: str) -> float:
            return difflib.SequenceMatcher(a=norm(a), b=norm(b)).ratio()

        def best_sim(token: str, vocab: list) -> float:
            return max((sim(token, v) for v in vocab), default=0.0)

        def alpha_ratio(s: str) -> float:
            letters = re.sub(r"[^A-Za-z]", "", s or "")
            return len(letters) / max(1, len(s or ""))

        # ---- build items (keep x1/x2) ----
        y1, y2 = int(y_band[0]), int(y_band[1])
        items = []
        for it in (top_rows_dump or []):
            x1 = int(it.get("x1", 0)); x2 = int(it.get("x2", 0))
            items.append({
                "text": it.get("text",""),
                "conf": float(it.get("conf",0.0)),
                "x": float(it.get("x_center", (x1 + x2)/2.0)),
                "x1": x1, "x2": x2,
                "y1": int(it.get("y1", y1)),
                "y2": int(it.get("y2", y2)),
            })
        for d in items:
            d["y_center"] = 0.5*(d["y1"]+d["y2"])

        # header-row keep
        y_thresh, _ = self._cluster_header_row(items, y_band) if items else (y1, y1)
        header_items = [d for d in items if d["y_center"] <= y_thresh + 1.0]

        # filter noise
        num_re = re.compile(r"^\d+(?:\.\d+)?\s*(A|AMP|AMPS)?$", re.I)
        phase = {"A","B","C"}
        header_items = [
            d for d in header_items
            if alpha_ratio(d["text"]) >= 0.40
            and norm(d["text"]) not in phase
            and not num_re.match(d["text"])
        ]
        header_items.sort(key=lambda d: d["x"])

        # ---- chunking: split glued tokens ----
        def split_by_words(d):
            txt = norm(d["text"])
            parts = [p for p in re.split(r"\s+", txt) if p]
            if len(parts) <= 1 or (d["x2"] - d["x1"]) <= 8:
                return [d]
            L = max(1, d["x2"] - d["x1"] + 1)
            total = sum(len(p) for p in parts)
            out = []
            x_left = d["x1"]
            for i, p in enumerate(parts):
                w = max(5, int(round(L * len(p) / max(1, total))))
                x_right = (d["x2"] if i == len(parts)-1 else (x_left + w - 1))
                xc = 0.5*(x_left + x_right)
                out.append({**d, "text": p, "x1": int(x_left), "x2": int(x_right), "x": float(xc)})
                x_left = x_right + 1
            return out

        ROLE_RX = {
            "poles": re.compile(r"\b(poles?|pol)\b", re.I),
            "trip":  re.compile(r"\b(amp(s)?|size|breaker|bkr)\b|\(A\)", re.I),
            "ckt":   re.compile(r"\b(ckt|cct|circuit)\b", re.I),
            "desc":  re.compile(r"\b(desc(ription)?|designation|name)\b", re.I),
        }

        def split_multi_role(d):
            txt = d["text"]
            hits = {k: bool(rx.search(txt)) for k,rx in ROLE_RX.items()}
            if hits["poles"] and hits["trip"]:
                m = ROLE_RX["poles"].search(txt)
                if m:
                    x_mid = int(0.5*(d["x1"]+d["x2"]))
                    return [
                        {**d, "text": txt[:m.end()].strip(),  "x2": x_mid, "x": 0.5*(d["x1"]+x_mid)},
                        {**d, "text": txt[m.end():].strip(), "x1": x_mid+1, "x": 0.5*(x_mid+1 + d["x2"])},
                    ]
            if hits["ckt"] and re.search(r"\b(remark|note|comment|desc)\w*", txt, re.I):
                return split_by_words(d)
            return [d]

        exploded = []
        for d in header_items:
            for e in split_by_words(d):
                exploded.extend(split_multi_role(e))

        header_items = [e for e in exploded if len(norm(e["text"])) >= 2]
        header_items.sort(key=lambda d: d["x"])

        # ---- score ----
        candidates = {t: [] for t in TYPES}
        readable_rows = []

        for it in header_items:
            txt, conf, x = it["text"], it["conf"], it["x"]
            scores = {t: (STR_W*best_sim(txt, VOCAB[t]) + CONF_W*max(0.0, conf)) for t in TYPES}
            for t in TYPES:
                if scores[t] > 0:
                    candidates[t].append({
                        "type": t, "text_norm": norm(txt),
                        "x": x, "x1": it["x1"], "x2": it["x2"],
                        "y1": it["y1"], "y2": it["y2"],
                        "conf": conf, "score": max(0.0, min(1.0, scores[t])),
                    })
            best_t = max(scores.items(), key=lambda kv: kv[1])[0]
            readable_rows.append({"text": txt, "conf": conf, "x": x, "scores": scores, "best": best_t})

        def dedupe(lst):
            lst = sorted(lst, key=lambda d: (-d["score"], d["x"]))
            keep = []
            for c in lst:
                if any(abs(c["x"] - k["x"]) <= X_TOL for k in keep): continue
                keep.append(c)
            return keep

        for t in TYPES:
            candidates[t] = dedupe(candidates[t])

        picks = {t: [] for t in TYPES}
        used_x = []
        def claim(x):
            if any(abs(x-ux) <= X_TOL for ux in used_x): return False
            used_x.append(x); return True

        for t in TYPES:
            for c in sorted(candidates[t], key=lambda d: -d["score"]):
                if len(picks[t]) >= 2: break
                if c["score"] < ACCEPT_THR: continue
                if claim(c["x"]): picks[t].append(c)
            for c in candidates[t]:
                if len(picks[t]) >= 2: break
                if any(c is pc for pc in picks[t]): continue
                if claim(c["x"]): picks[t].append(c)
            picks[t].sort(key=lambda d: d["x"])

        header_map = {
            t: [{
                "text_norm": c["text_norm"],
                "x_center": float(c["x"]),
                "x1": int(c["x1"]), "x2": int(c["x2"]),              # <-- keep exact box
                "y1": int(c["y1"]), "y2": int(c["y2"]),
                "conf": float(c["conf"]),
                "score": float(c["score"]),
            } for c in picks[t]] for t in TYPES
        }

        # ---- print once ----
        print("\n==== HEADER CLASSIFY (hard-coded) ====")
        print(f"Band: y=[{y1},{y2}]  page_w={page_width}  items={len(header_items)}")
        if not readable_rows:
            print("No header-like tokens found.")
        else:
            print("Detections (left→right):")
            for r in sorted(readable_rows, key=lambda d: d["x"]):
                s = r["scores"]
                print(f'  "{r["text"]}"  x={r["x"]:.1f}  conf={r["conf"]:.2f}  '
                    f'ckt={s["ckt"]:.2f}  desc={s["description"]:.2f}  '
                    f'trip={s["trip"]:.2f}  poles={s["poles"]:.2f}  -> {r["best"].upper()}')
        print("Picks:")
        for t in TYPES:
            picked = header_map[t]
            if not picked:
                print(f"  {t.upper()}: (none)")
            else:
                line = "  " + t.upper() + ": " + " | ".join(
                    [f'x={p["x_center"]:.1f}, score={p["score"]:.2f}, text="{p["text_norm"]}"' for p in picked]
                )
                print(line)
        print("======================================\n")

        return header_map, {"accept_threshold": ACCEPT_THR, "x_tol": X_TOL}

    def _compute_top_row_spans(self, centers: List[int], header_y: int, footer_y: int, H: int) -> List[Tuple[int,int]]:
        """
        Force the OCR spans to cover the top two table rows *right under the header_y*,
        not the first two detected row centers (which may start below the header).
        """
        if not centers:
            med_gap = max(18, int(round(0.032 * H)))
            y1 = max(0, int(header_y) + 4)
            y2 = min(H - 1, int(header_y) + int(2.2 * med_gap))
            return [(y1, min(y2, int(footer_y) - 1))]

        c = sorted(int(v) for v in centers)
        gaps = np.diff(c) if len(c) >= 2 else np.array([max(18, int(0.032 * H))])
        med_gap = int(float(np.median(gaps))) if gaps.size else max(18, int(0.032 * H))

        half = max(6, int(round(0.36 * med_gap)))

        # Row 1 → directly below header_y, Row 2 → next row below that
        row1_y1 = max(0, int(header_y) + 3)
        row1_y2 = min(H - 1, row1_y1 + int(0.9 * med_gap))
        row2_y1 = min(H - 1, row1_y2 + 1)
        row2_y2 = min(H - 1, row2_y1 + int(0.9 * med_gap))

        spans = [(row1_y1, row1_y2)]
        if row2_y2 - row2_y1 > 6:
            spans.append((row2_y1, row2_y2))

        # Clamp both within footer bounds
        clamped = [(max(y1, int(header_y) + 2), min(y2, int(footer_y) - 1)) for y1, y2 in spans]
        return clamped[:2]

    def _ocr_top_rows_precropped(self, gray: np.ndarray, row_spans: List[Tuple[int,int]]) -> Tuple[List[dict], np.ndarray]:
        """
        OCR the first two rows using tight crops per row, then merge results with absolute coords.
        Returns (items_dump, overlay_image). The overlay shows each row span and detections.
        """
        if self.reader is None:
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)

        H, W = gray.shape[:2]
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        def _pass(img, mag):
            try:
                return self.reader.readtext(
                    img, detail=1, paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]-/+.%: ",
                    mag_ratio=mag, contrast_ths=0.05, adjust_contrast=0.7,
                    text_threshold=0.40, low_text=0.25,
                )
            except Exception:
                return []

        items: List[dict] = []
        for (ry1, ry2) in row_spans:
            ry1 = int(max(0, min(H-2, ry1)))
            ry2 = int(max(ry1+1, min(H-1, ry2)))
            row_band = gray[ry1:ry2, :]

            # Light preproc per row
            g = cv2.GaussianBlur(row_band, (3, 3), 0)
            g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
            inv = cv2.bitwise_not(g)

            dets = _pass(g, 1.6) + _pass(g, 2.0) + _pass(inv, 1.6) + _pass(inv, 2.0)

            # Draw row window on overlay
            cv2.rectangle(overlay, (0, ry1), (W - 1, ry2), (60, 120, 255), 1)

            # Collect detections with absolute coords
            for box, txt, conf in dets:
                if not txt:
                    continue
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                x1 = max(0, min(xs))
                x2 = min(W - 1, max(xs))
                yb1 = ry1 + max(0, min(ys))
                yb2 = ry1 + min((ry2 - 1), max(ys))
                w = int(x2 - x1 + 1)
                h = int(yb2 - yb1 + 1)

                items.append({
                    "text": str(txt),
                    "conf": float(conf or 0.0),
                    "x1": int(x1), "y1": int(yb1), "x2": int(x2), "y2": int(yb2),
                    "w": w, "h": h,
                    "x_center": float((x1 + x2) / 2.0),
                })

                # Per-token overlay
                cv2.rectangle(overlay, (x1, yb1), (x2, yb2), (0, 200, 255), 2)
                label = f'{str(txt)[:24]} ({float(conf or 0.0):.2f})'
                cv2.putText(overlay, label, (x1, max(12, yb1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # Sort for readability
        items.sort(key=lambda d: (d["y1"], d["x1"]))
        return items, overlay

    def _ocr_top_rows_band(self, gray: np.ndarray, y1: int, y2: int) -> Tuple[List[dict], np.ndarray]:
        """
        OCR a horizontal band (bounded by analyzer's header/footer clamp) and
        return a dump of all detections with absolute coordinates. Also returns an
        RGB overlay image with boxes/labels drawn for debugging.
        """
        if self.reader is None:
            try:
                # Late-init reader if needed
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)

        H, W = gray.shape[:2]
        y1 = int(max(0, min(H - 2, y1)))
        y2 = int(max(y1 + 1, min(H - 1, y2)))

        band = gray[y1:y2, :]
        # Light preproc for OCR
        g = cv2.GaussianBlur(band, (3, 3), 0)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)

        def _pass(img, mag):
            try:
                return self.reader.readtext(
                    img, detail=1, paragraph=False,
                    # keep broad allowlist to capture both headers and values (we'll sift later)
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]-/+.%: ",
                    mag_ratio=mag, contrast_ths=0.05, adjust_contrast=0.7,
                    text_threshold=0.40, low_text=0.25,
                )
            except Exception:
                return []

        inv = cv2.bitwise_not(g)
        dets = _pass(g, 1.6) + _pass(g, 2.0) + _pass(inv, 1.6) + _pass(inv, 2.0)

        # Build dump (absolute coordinates)
        items: List[dict] = []
        for box, txt, conf in dets:
            if not txt:
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x1 = max(0, min(xs))
            x2 = min(W - 1, max(xs))
            yb1 = y1 + max(0, min(ys))
            yb2 = y1 + min((y2 - 1), max(ys))
            w = int(x2 - x1 + 1)
            h = int(yb2 - yb1 + 1)
            items.append({
                "text": str(txt),
                "conf": float(conf or 0.0),
                "x1": int(x1), "y1": int(yb1), "x2": int(x2), "y2": int(yb2),
                "w": w, "h": h,
                "x_center": float((x1 + x2) / 2.0),
            })

        # Sort by x for quick eyeballing/grouping
        items.sort(key=lambda d: (d["y1"], d["x1"]))

        # Build overlay
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(overlay, (0, y1), (W - 1, y2), (60, 60, 60), 1)
        for it in items:
            cv2.rectangle(overlay, (it["x1"], it["y1"]), (it["x2"], it["y2"]), (0, 200, 255), 2)
            label = f'{it["text"][:24]} ({it["conf"]:.2f})'
            cv2.putText(overlay, label, (it["x1"], max(12, it["y1"] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        return items, overlay

    def _split_band_between_headers(self, gray: np.ndarray, y1: int, y2: int,
                                    band: Tuple[int,int], xa: float, xb: float) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        """
        Split a wide band into two by finding the best vertical valley between xa and xb.
        Works on ORIGINAL gray (pre-inpaint). Returns (left_band, right_band).
        """
        H, W = gray.shape[:2]
        x1, x2 = max(0, int(band[0])), min(W-1, int(band[1]))
        if x2 - x1 < 12:
            m = (x1 + x2)//2
            return (x1, m), (m+1, x2)

        y1 = max(0, min(H-2, int(y1))); y2 = max(y1+1, min(H-1, int(y2)))
        roi = gray[y1:y2, x1:x2]
        g   = cv2.GaussianBlur(roi, (3,3), 0)
        bw1 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        kv  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, int(0.06*(y2-y1)))))
        v   = cv2.morphologyEx(bw1, cv2.MORPH_OPEN, kv, iterations=1)
        proj = v.sum(axis=0).astype(np.float32)

        # search window: between the two header centers, clamped to band
        xa0, xb0 = sorted([float(xa), float(xb)])
        xa = int(np.clip(xa0 - x1, 0, x2 - x1))
        xb = int(np.clip(xb0 - x1, 0, x2 - x1))
        if xb - xa < 6:
            xa, xb = 0, proj.size - 1

        segment = proj[xa:xb+1]
        if segment.size == 0:
            m = (x1 + x2)//2
            return (x1, m), (m+1, x2)

        # valley = argmin in smoothed projection
        k = max(5, ((xb - xa)//24)*2 + 1)  # odd
        smooth = cv2.GaussianBlur(segment.reshape(1,-1), (k,1), 0).reshape(-1)
        idx = int(np.argmin(smooth)) + xa
        cut = x1 + idx

        # enforce non-degenerate halves
        left  = (x1, max(x1+1, min(cut-1, x2-2)))
        right = (min(x2-1, max(cut+1, x1+2)), x2)
        if right[0] <= left[1]:
            m = (x1 + x2)//2
            left, right = (x1, m), (m+1, x2)
        return left, right

    # ======= internals (copied from V19_4 parsing half) =======
    def _row_boxes_from_centers(self, shape: Tuple[int,int], centers: List[int],
                                header_y_abs: Optional[int], footer_y_abs: Optional[int]) -> List[Tuple[int,int,int,int]]:
        H, W = shape
        centers = sorted(int(c) for c in centers)
        if not centers: return []
        borders = []
        if len(centers) == 1:
            g = 20; borders = [max(0, centers[0]-g//2), min(H-1, centers[0]+g//2)]
        else:
            borders.append(int(centers[0] - (centers[1]-centers[0])/2))
            for i in range(len(centers)-1): borders.append(int((centers[i]+centers[i+1])//2))
            borders.append(int(centers[-1] + (centers[-1]-centers[-2])/2))
        y_min = max(0, int(header_y_abs) + 6) if header_y_abs is not None else 0
        y_max = min(H-1, int(footer_y_abs) - 6) if footer_y_abs is not None else H-1
        rows=[]
        for i in range(len(borders)-1):
            y1 = max(y_min, borders[i] + 1); y2 = min(y_max, borders[i+1] - 1)
            if y2 - y1 >= 10: rows.append((y1, y2, 0, W))
        return rows

    def _tp_fallback(self, W: int) -> Dict[str, Tuple[int,int]]:
        fb = self._cfg.get("fallbacks", {
            "frac_l_trip":[0.260,0.300],
            "frac_l_poles":[0.305,0.335],
            "frac_r_poles":[0.655,0.690],
            "frac_r_trip":[0.695,0.735],
        })
        def frac_to_px(fr): return (int(W*fr[0]), int(W*fr[1]))
        return {"L_TRIP":  frac_to_px(tuple(fb.get("frac_l_trip",  (0.260,0.300)))),
                "L_POLES": frac_to_px(tuple(fb.get("frac_l_poles", (0.305,0.335)))),
                "R_POLES": frac_to_px(tuple(fb.get("frac_r_poles", (0.655,0.690)))),
                "R_TRIP":  frac_to_px(tuple(fb.get("frac_r_trip",  (0.695,0.735))))}

    # ==================== Weighted-vote helpers ====================
    def _get_scoring_cfg(self):
        # Configurable weights/thresholds with safe defaults
        cfg = self._cfg.get("poles_voting", {}) if hasattr(self, "_cfg") else {}
        return {
            "w_ocr": float(cfg.get("w_ocr", 2.0)),
            "w_tpl": float(cfg.get("w_tpl", 1.25)),   # can bump to 1.5 later if you want
            "w_ink": float(cfg.get("w_ink", 1.25)),
            "tpl_flat_eps": float(cfg.get("tpl_flat_eps", 0.10)),   # NCC spread below this -> abstain
            "ocr_min": float(cfg.get("ocr_min", 0.50)),             # clamp OCR low
            "ocr_max": float(cfg.get("ocr_max", 0.98)),             # clamp OCR high
            # override margins
            "delta_both": float(cfg.get("delta_both", 0.60)),       # both secondaries agree
            "delta_one": float(cfg.get("delta_one", 0.75)),         # one agrees + one abstains
            "delta_relax": float(cfg.get("delta_relax", 0.10)),     # subtract when OCR < 0.55
            # ink heuristics thresholds (initial safe values)
            "ar_1": float(cfg.get("ar_1", 2.3)),         # aspect ratio threshold favoring '1'
            "hx_1": float(cfg.get("hx_1", 0.35)),        # low horizontal (Sobel-x) for '1'
            "ink_big": float(cfg.get("ink_big", 0.12)),  # large ink mass for '2'/'3'
            "hx_big": float(cfg.get("hx_big", 0.45)),    # strong horizontal for '2'/'3'
            # 2 vs 3 separations
            "baseline_row_frac": float(cfg.get("baseline_row_frac", 0.33)),  # bottom third
            "baseline_peak_frac": float(cfg.get("baseline_peak_frac", 0.22)),# peak prominence for '2'
            "bimodal_valley_drop": float(cfg.get("bimodal_valley_drop", 0.18)), # valley between lobes for '3'
            "corner_eps": float(cfg.get("corner_eps", 0.015)),  # contour approx epsilon as frac of perimeter
            "corner_cut": int(cfg.get("corner_cut", 2)),        # >= this => "kinky" (2-ish)
        }

    def _clamp01(self, v: float) -> float:
        return 0.0 if v < 0 else (1.0 if v > 1 else v)

    def _normalize_probs(self, d: Dict[str, float]) -> Dict[str, float]:
        s = sum(max(0.0, float(v)) for v in d.values())
        if s <= 0: 
            # uniform fallback
            return {k: 1.0/len(d) for k in d.keys()}
        return {k: max(0.0, float(v))/s for k, v in d.items()}

    def _ocr_digit_probs(self, roi_gray: np.ndarray) -> Tuple[Dict[str,float], dict]:
        """
        OCR for '123' using EasyOCR. Returns P_ocr over {'1','2','3'}
        and meta with ('top','q','raw').
        """
        meta = {"top": None, "q": 0.0, "raw": []}
        P = {"1": 1/3, "2": 1/3, "3": 1/3}
        if self.reader is None:
            try:
                self.reader = easyocr.Reader(['en'], gpu=True)
            except Exception:
                self.reader = easyocr.Reader(['en'], gpu=False)
        try:
            up = cv2.resize(roi_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            det = self.reader.readtext(
                up, detail=1, paragraph=False, allowlist="123",
                mag_ratio=2.0, contrast_ths=0.05, adjust_contrast=0.7,
                text_threshold=0.30, low_text=0.15
            )
            # det: list of tuples (box, text, conf)
            candidates = []
            for _, txt, conf in det:
                if not txt: continue
                m = re.search(r"[123]", str(txt))
                if not m: continue
                c = m.group(0)
                candidates.append((c, float(conf or 0.0)))
            meta["raw"] = candidates
            if candidates:
                # take best
                best = max(candidates, key=lambda t: t[1])
                top, q = best[0], max(0.0, min(1.0, best[1]))
                meta["top"], meta["q"] = top, q
                cfg = self._get_scoring_cfg()
                q = max(cfg["ocr_min"], min(cfg["ocr_max"], q))
                # distribute
                rem = max(0.0, 1.0 - q); other = rem / 2.0
                P = {"1": other, "2": other, "3": other}
                P[top] = q
            else:
                meta["top"], meta["q"] = None, 0.0
        except Exception:
            pass
        return self._normalize_probs(P), meta

    def _template_digit_probs(self, bw_roi: np.ndarray) -> Tuple[Dict[str,float], dict]:
        """
        Use simple NCC with 1/2/3 templates at matched size.
        Returns P_tpl and meta: {'ncc':{'1':..}, 'abstain':bool}
        """
        meta = {"ncc": {"1":0.0,"2":0.0,"3":0.0}, "abstain": False}
        if bw_roi is None or bw_roi.size == 0:
            meta["abstain"] = True
            return {"1": 1/3, "2": 1/3, "3": 1/3}, meta

        h, w = bw_roi.shape[:2]
        h = max(16, min(120, h)); w = max(10, min(100, w))
        # build simple printed templates
        def mk_tpl(char):
            canvas = np.zeros((h, w), np.uint8)
            (tw, th), bl = cv2.getTextSize(char, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            x = max(0, (w - tw)//2); y = max(th, (h + th)//2 - bl)
            cv2.putText(canvas, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
            return canvas
        tpl = {"1": mk_tpl("1"), "2": mk_tpl("2"), "3": mk_tpl("3")}

        try:
            # resize roi to template size
            roi = cv2.resize(bw_roi, (w, h), interpolation=cv2.INTER_AREA)
            scores = {}
            for c, T in tpl.items():
                res = cv2.matchTemplate(roi, T, cv2.TM_CCOEFF_NORMED)
                s = float(res.max()) if res.size else -1.0
                scores[c] = s
            meta["ncc"] = {k: float(v) for k, v in scores.items()}
            # min-max normalize
            mn, mx = min(scores.values()), max(scores.values())
            cfg = self._get_scoring_cfg()
            if (mx - mn) < cfg["tpl_flat_eps"]:
                meta["abstain"] = True
                return {"1": 1/3, "2": 1/3, "3": 1/3}, meta
            P = {k: (scores[k] - mn) for k in scores.keys()}
            P = self._normalize_probs(P)
            return P, meta
        except Exception:
            meta["abstain"] = True
            return {"1": 1/3, "2": 1/3, "3": 1/3}, meta

    def _ink_digit_probs(self, roi_gray: np.ndarray) -> Tuple[Dict[str,float], dict]:
        """
        Hand-crafted features to separate 1 vs {2,3} and 2 vs 3.
        Returns P_ink and meta with features + abstain flag.
        """
        meta = {"abstain": False, "feat": {}}
        if roi_gray is None or roi_gray.size == 0:
            meta["abstain"] = True
            return {"1": 1/3, "2": 1/3, "3": 1/3}, meta

        # Clean & binarize
        g = cv2.medianBlur(roi_gray, 3)
        g = cv2.bilateralFilter(g, 5, 50, 50)
        thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 10)
        H, W = thr.shape[:2]
        area = float(H * W)
        ink = float((thr > 0).sum())
        ink_ratio = ink / max(1.0, area)

        # aspect ratio
        ys, xs = np.where(thr > 0)
        if xs.size == 0 or ys.size == 0:
            meta["abstain"] = True
            meta["feat"] = {"ink_ratio": 0.0}
            return {"1": 1/3, "2": 1/3, "3": 1/3}, meta
        h = float(ys.max() - ys.min() + 1)
        w = float(xs.max() - xs.min() + 1)
        ar = (h / max(1.0, w))

        # horizontal vs vertical gradient strengths
        sx = cv2.Sobel(thr, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(thr, cv2.CV_32F, 0, 1, ksize=3)
        sumx = float(np.abs(sx).sum()) + 1e-6
        sumy = float(np.abs(sy).sum()) + 1e-6
        hx = sumx / (sumx + sumy)  # 0..1 (more horizontal edges -> larger)

        # bottom baseline via horizontal projection
        proj = thr.sum(axis=1).astype(np.float32) / (255.0 * max(1, W))
        bottom_band = int(H * self._get_scoring_cfg()["baseline_row_frac"])
        bottom_slice = proj[max(0, H - bottom_band):H] if bottom_band > 0 else proj
        peak_bottom = float(bottom_slice.max()) if bottom_slice.size else 0.0

        # bimodality for '3' -> two lobes (two peaks with valley)
        top_half = proj[:H//2] if H >= 2 else proj
        bot_half = proj[H//2:] if H >= 2 else proj
        peak_top = float(top_half.max()) if top_half.size else 0.0
        peak_bot = float(bot_half.max()) if bot_half.size else 0.0
        valley_mid = float(proj[H//2]) if H > 2 else float(proj.min())
        drop = min(peak_top, peak_bot) - valley_mid  # positive if valley sits between lobes

        # corner count from contour
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        per = 0.0
        approx_corners = 0
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            per = float(cv2.arcLength(c, True))
            eps = self._get_scoring_cfg()["corner_eps"] * max(1.0, per)
            approx = cv2.approxPolyDP(c, eps, True)
            approx_corners = max(0, len(approx) - 1)

        cfg = self._get_scoring_cfg()
        # raw votes per class
        s1 = 0.0; s2 = 0.0; s3 = 0.0

        # 1-ish: tall & low horizontal content
        if ar > cfg["ar_1"] and hx < cfg["hx_1"]:
            s1 += 1.0
        if ink_ratio < 0.08:  # very thin
            s1 += 0.4

        # 2-ish: strong bottom baseline + some corners
        if peak_bottom > cfg["baseline_peak_frac"]:
            s2 += 0.8
        if approx_corners >= cfg["corner_cut"]:
            s2 += 0.5

        # 3-ish: two-lobe shape (top/bottom both strong) with valley
        if min(peak_top, peak_bot) > 0.20 and drop > cfg["bimodal_valley_drop"]:
            s3 += 0.9
        # smoothness: few corners and decent ink
        if approx_corners <= 1 and ink_ratio > cfg["ink_big"]:
            s3 += 0.3

        # fallback nudges using hx and ink
        if ink_ratio > cfg["ink_big"] and hx > cfg["hx_big"]:
            # more horizontal strokes -> less '1'
            s2 += 0.2; s3 += 0.2

        meta["feat"] = {
            "ink_ratio": round(ink_ratio, 4),
            "aspect_ratio": round(ar, 4),
            "horiz_strength": round(hx, 4),
            "bottom_peak": round(peak_bottom, 4),
            "peaks_top_bot": [round(peak_top,4), round(peak_bot,4)],
            "valley_drop": round(drop, 4),
            "corners": int(approx_corners),
        }

        total = s1 + s2 + s3
        # abstain if weak/ambiguous
        if total < 0.6:
            meta["abstain"] = True
            return {"1": 1/3, "2": 1/3, "3": 1/3}, meta

        P = {"1": s1, "2": s2, "3": s3}
        P = self._normalize_probs(P)
        return P, meta

    def _vote_poles_digit(self, P_ocr, meta_ocr, P_tpl, meta_tpl, P_ink, meta_ink) -> Tuple[str, dict]:
        """
        Combine sources with weights and apply conservative override rules.
        Returns (final_digit, debug)
        """
        cfg = self._get_scoring_cfg()
        w_ocr = cfg["w_ocr"]
        w_tpl = (0.0 if meta_tpl.get("abstain") else cfg["w_tpl"])
        w_ink = (0.0 if meta_ink.get("abstain") else cfg["w_ink"])

        classes = ["1","2","3"]
        S = {}
        for c in classes:
            S[c] = (w_ocr * float(P_ocr.get(c,0.0)) +
                    w_tpl * float(P_tpl.get(c,0.0)) +
                    w_ink * float(P_ink.get(c,0.0)))

        c_ocr = meta_ocr.get("top")
        if c_ocr not in classes:
            # if OCR didn't hit 1/2/3, infer c_ocr as argmax of P_ocr
            c_ocr = max(classes, key=lambda k: P_ocr.get(k,0.0))
        c_hat = max(classes, key=lambda k: S.get(k,0.0))
        margin = float(S.get(c_hat,0.0) - S.get(c_ocr,0.0))

        # Secondary agreements
        tpl_pref = max(classes, key=lambda k: P_tpl.get(k,0.0))
        ink_pref = max(classes, key=lambda k: P_ink.get(k,0.0))
        tpl_agrees = (w_tpl > 0 and tpl_pref == c_hat and c_hat != c_ocr)
        ink_agrees = (w_ink > 0 and ink_pref == c_hat and c_hat != c_ocr)
        n_agree = int(tpl_agrees) + int(ink_agrees)

        # Override gates
        delta = None
        if n_agree >= 2:
            delta = cfg["delta_both"]
        elif n_agree == 1:
            delta = cfg["delta_one"]

        ocr_q = float(meta_ocr.get("q") or 0.0)
        if delta is not None and ocr_q < 0.55:
            delta = max(0.0, delta - cfg["delta_relax"])

        override = False
        reason = "keep_ocr"
        final_digit = c_ocr

        if c_hat != c_ocr and delta is not None and margin >= delta:
            override = True
            final_digit = c_hat
            reason = f"override_{n_agree}agree_margin{margin:.2f}>=Δ{delta:.2f}"

        debug = {
            "ocr": {"top": c_ocr, "q": round(ocr_q, 3), "P": {k: round(P_ocr[k],3) for k in classes}, "raw": meta_ocr.get("raw", [])},
            "tpl": {"P": {k: round(P_tpl[k],3) for k in classes}, "abstain": bool(meta_tpl.get("abstain")), "ncc": meta_tpl.get("ncc")},
            "ink": {"P": {k: round(P_ink[k],3) for k in classes}, "abstain": bool(meta_ink.get("abstain")), "feat": meta_ink.get("feat", {})},
            "weights": {"ocr": w_ocr, "tpl": w_tpl, "ink": w_ink},
            "S": {k: round(S[k], 3) for k in classes},
            "c_hat": c_hat,
            "margin": round(margin, 3),
            "final": final_digit,
            "override": override,
            "reason": reason
        }
        return final_digit, debug

    # ---- everything below mirrors your V19_4 scan/ocr helpers (unchanged logic) ----
    def _scrub_hline_noise(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return img
        g = img
        H, W = g.shape[:2]
        g2 = cv2.GaussianBlur(g, (3,3), 0)
        g2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g2)
        outs = []
        outs.append(cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10))
        outs.append(cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12))
        _, o = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU); outs.append(o)
        h_max = max(2, int(0.020 * H))
        w_min = max(4, int(0.020 * W))
        w_max = max(w_min+1, int(0.40 * W))
        ar_min = 3.0
        out = g.copy()
        for bw in outs:
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)), 1)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if h <= h_max and w_min <= w <= w_max and (w / max(1, h)) >= ar_min:
                    x1 = max(0, x-1); y1 = max(0, y-1)
                    x2 = min(W-1, x + w + 1); y2 = min(H-1, y + h + 1)
                    out[y1:y2+1, x1:x2+1] = 255
        return out

    def _write_json(self, path: str, obj):
        import numpy as _np, json, os
        def _default(o):
            if isinstance(o, _np.ndarray):
                return {"_ndarray": True, "shape": list(o.shape), "dtype": str(o.dtype)}
            if isinstance(o, _np.generic):
                return o.item()
            raise TypeError(f"Unserializable type: {type(o).__name__}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)

    def _erase_top_labels(self, crop: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
        erased = []
        img = crop.copy()
        if self.reader is None or img.size == 0:
            return img, erased
        H, W = img.shape[:2]
        band_h = max(8, int(0.18 * H))
        roi = img[:band_h, :]
        try:
            dets = self.reader.readtext(
                roi, detail=1, paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                mag_ratio=1.6, contrast_ths=0.05, adjust_contrast=0.7,
                text_threshold=0.40, low_text=0.25
            )
        except Exception:
            dets = []
        KEYS = {"AMP", "AMPS", "TRIP", "POLE", "POLES"}
        for box, txt, _ in dets:
            s = re.sub(r"[^A-Z]", "", (txt or "").upper())
            if s in KEYS:
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                x1, x2 = max(0, min(xs)-2), min(W-1, max(xs)+2)
                y1, y2 = max(0, min(ys)-2), min(band_h-1, max(ys)+2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
                erased.append((x1, y1, x2, y2))
        return img, erased

    def _remove_dashes_strict(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
        """
        Aggressively remove short, flat, low-ink, mostly-horizontal blobs (panel dashes)
        from a column strip image (grayscale). Returns (clean_img, removed_boxes).
        Tunable via config["dash_killer"].
        """
        removed: List[Tuple[int,int,int,int]] = []
        if img is None or img.size == 0:
            return img, removed

        # thresholds (safe defaults; can tune in breaker_labels_config.jsonc)
        cfg = (self._cfg.get("dash_killer", {}) if hasattr(self, "_cfg") else {})
        h_frac_max   = float(cfg.get("h_frac_max", 0.18))   # dash height <= 18% of strip height
        ar_min       = float(cfg.get("ar_min", 3.5))        # width/height must be fairly wide
        ink_max      = float(cfg.get("ink_max", 0.12))      # low ink density
        horiz_min    = float(cfg.get("horiz_min", 0.75))    # strong horizontal edges
        require_k    = int(cfg.get("require_k_of", 2))      # any K of the conditions below => dash
        ksize_close  = int(cfg.get("ksize_close", 2))       # small close to merge tiny gaps

        g = cv2.GaussianBlur(img, (3,3), 0)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
        H, W = g.shape[:2]

        # try both normal & inverted binarization; union their detections
        def binarizations(m):
            outs = []
            outs.append(cv2.adaptiveThreshold(m, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10))
            outs.append(cv2.adaptiveThreshold(m, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12))
            _, o = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU); outs.append(o)
            return outs

        bws = binarizations(g) + binarizations(cv2.bitwise_not(g))

        mask = np.zeros_like(g, np.uint8)
        for bw in bws:
            if ksize_close > 0:
                bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_close,1)), 1)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if w <= 0 or h <= 0: 
                    continue

                # quick gates: very short vs strip height & reasonably wide
                cond_small_h = (h <= max(3, int(h_frac_max * H)))
                cond_flat_ar = ((w / float(max(1, h))) >= ar_min)

                roi = bw[y:y+h, x:x+w]
                ink = float((roi > 0).sum()) / float(max(1, roi.size))
                sx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                sy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
                hx = (np.abs(sx).sum() + 1e-6) / (np.abs(sx).sum() + np.abs(sy).sum() + 1e-6)
                cond_horiz_ink = (hx >= horiz_min and ink <= ink_max)

                # any K-of-3 => dash
                if (int(cond_small_h) + int(cond_flat_ar) + int(cond_horiz_ink)) >= require_k:
                    cv2.rectangle(mask, (x,y), (x+w-1, y+h-1), 255, -1)

        # expand just a touch so fragments go away
        if mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,2))
            mask = cv2.dilate(mask, kernel, 1)
            # record and remove
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = img.copy()
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                x1 = max(0, x); y1 = max(0, y); x2 = min(W-1, x+w-1); y2 = min(H-1, y+h-1)
                out[y1:y2+1, x1:x2+1] = 255
                removed.append((x1,y1,x2,y2))
            return out, removed

        return img, removed

    def _filter_peripheral_values(self, texts, boxes, crop_shape):
        cfg = self._cfg.get("peripheral_filter", {"x_margin_frac":0.0, "y_margin_frac":0.0})
        xmf = float(cfg.get("x_margin_frac", 0.0))
        ymf = float(cfg.get("y_margin_frac", 0.0))
        if xmf <= 0.0 and ymf <= 0.0:
            return texts, boxes
        Hc, Wc = crop_shape[:2]
        mx = int(round(Wc * xmf)); my = int(round(Hc * ymf))
        kept_texts, kept_boxes = [], []
        for i, (x1, y1, x2, y2, t) in enumerate(boxes):
            if x1 < mx or x2 > (Wc - mx) or y1 < my or y2 > (Hc - my):
                continue
            kept_boxes.append((x1, y1, x2, y2, t))
            kept_texts.append(texts[i] if i < len(texts) else t)
        return kept_texts, kept_boxes

    def _save_page_overlay(self, gray: np.ndarray, header_y: int, footer_y: int,
                        cols: Dict[str, Tuple[int,int]], header_map: Dict[str, List[dict]],
                        out_path: str):
        H, W = gray.shape
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # header/footer guides
        cv2.line(vis, (0, int(header_y)), (W-1, int(header_y)), (0,220,0), 2)
        cv2.putText(vis, "HEADER", (12, max(16, int(header_y)-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,0), 2, cv2.LINE_AA)
        cv2.line(vis, (0, int(footer_y)), (W-1, int(footer_y)), (0,0,255), 2)
        cv2.putText(vis, "FOOTER", (12, max(16, int(footer_y)-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

        # vertical lines in band
        vxs = self._detect_vertical_lines(gray, header_y, footer_y)
        for x in vxs:
            cv2.line(vis, (int(x), int(header_y)), (int(x), int(footer_y)), (180,180,180), 1)

        # chosen column bands
        for name, color in [
            ("L_TRIP",  (255,255,0)),  # yellow
            ("L_POLES", (0,255,0)),    # green
            ("R_POLES", (0,255,0)),    # green
            ("R_TRIP",  (255,255,0)),  # yellow
        ]:
            if name in cols and cols[name] is not None:
                x1, x2 = map(int, cols[name])
                cv2.rectangle(vis, (x1, int(header_y)), (x2, int(footer_y)), color, 2)
                cv2.putText(vis, name, (x1+4, int(header_y)+22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # header tokens (use your existing header boxes)
        for role, color in [("poles",(0,255,0)), ("trip",(0,0,255)), ("description",(255,0,0)), ("ckt",(0,0,0))]:
            for c in header_map.get(role, []):
                x1,y1,x2,y2 = int(c.get("x1",0)), int(c.get("y1",header_y)), int(c.get("x2",0)), int(c.get("y2",header_y+18))
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                tag = role.upper()
                cv2.putText(vis, tag, (x1, max(14, y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, vis)

    def _quick_blob_feats(self, bw_img: np.ndarray, gray_img: np.ndarray, x: int, y: int, w: int, h: int) -> dict:
        """
        Fast, cheap features for a candidate digit blob ROI. Works on already-binarized bw_img
        (white=255 digit ink) and preprocessed gray_img (CLAHE/blurred).
        Returns: dict with area, ar, ink_ratio, horiz_strength, bottom_peak.
        """
        x2, y2 = x + w, y + h
        H, W = bw_img.shape[:2]
        x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
        x2 = max(x+1, min(W, x2)); y2 = max(y+1, min(H, y2))

        roi_bw   = bw_img[y:y2, x:x2]
        roi_gray = gray_img[y:y2, x:x2]
        if roi_bw.size == 0:
            return {"area":0, "h":0, "w":0, "ar":0, "ink_ratio":0, "hx":0, "bottom_peak":0}

        H2, W2 = roi_bw.shape[:2]
        area = float(H2 * W2)
        ink  = float((roi_bw > 0).sum())
        ink_ratio = ink / max(1.0, area)

        # aspect ratio (height/width) based on tight mask
        ys, xs = np.where(roi_bw > 0)
        if xs.size and ys.size:
            h_t = float(ys.max() - ys.min() + 1)
            w_t = float(xs.max() - xs.min() + 1)
            ar  = (h_t / max(1.0, w_t))
        else:
            h_t, w_t, ar = float(H2), float(W2), float(H2 / max(1.0, W2))

        # horizontal vs vertical edge strength (more horizontal = dash-like, not digit '1')
        sx = cv2.Sobel(roi_bw, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(roi_bw, cv2.CV_32F, 0, 1, ksize=3)
        sumx = float(np.abs(sx).sum()) + 1e-6
        sumy = float(np.abs(sy).sum()) + 1e-6
        hx   = sumx / (sumx + sumy)  # 0..1

        # bottom baseline density proxy
        proj = (roi_bw.sum(axis=1).astype(np.float32) / (255.0 * max(1, W2))) if W2 > 0 else np.zeros((H2,), np.float32)
        bottom_band = max(1, int(round(0.33 * H2)))
        bottom_slice = proj[max(0, H2 - bottom_band):H2]
        bottom_peak  = float(bottom_slice.max()) if bottom_slice.size else 0.0

        return {
            "area": area, "h": float(H2), "w": float(W2), "ar": float(ar),
            "ink_ratio": float(ink_ratio), "hx": float(hx), "bottom_peak": float(bottom_peak),
        }


    def _calibrate_and_filter_blobs(self, blobs: List[Tuple[int,int,int,int,np.ndarray]], base_gray: np.ndarray) -> List[Tuple[int,int,int,int,np.ndarray]]:
        """
        Given merged candidate boxes [(x,y,w,h,bw_mask), ...] and the processed gray image,
        compute quick features, derive robust medians, and drop schniblets/outliers.
        Thresholds are pulled from config["blob_filter"] with sane defaults.
        """
        if not blobs:
            return blobs

        cfg = (self._cfg.get("blob_filter", {}) if hasattr(self, "_cfg") else {})
        # relative cutoffs vs median of TRUE-ish digits
        h_min_frac     = float(cfg.get("h_min_frac", 0.55))   # < 55% of median height → reject
        area_min_frac  = float(cfg.get("area_min_frac", 0.50))
        ink_min_frac   = float(cfg.get("ink_min_frac", 0.45))
        ar_low         = float(cfg.get("ar_low", 1.2))        # too squat (< 1.2) → likely fragments
        ar_high        = float(cfg.get("ar_high", 6.0))       # too skinny/tall (> 6) → often artifacts
        hx_max_soft    = float(cfg.get("hx_max_soft", 0.83))  # very horizontal → suspicious
        hx_ink_gate    = float(cfg.get("hx_ink_gate", 0.12))  # if super horizontal AND ink is low → drop
        max_outliers   = int(cfg.get("max_outliers", 999))    # leave generous

        feats = []
        for (x,y,w,h,bw) in blobs:
            feats.append(self._quick_blob_feats(bw, base_gray, x,y,w,h))

        # Build robust medians using the middle band (avoid tiny & giant extremes)
        def _median(vals): 
            return float(np.median(np.array(vals, dtype=np.float32))) if vals else 0.0

        heights = sorted([f["h"] for f in feats])
        areas   = sorted([f["area"] for f in feats])
        inks    = sorted([f["ink_ratio"] for f in feats])

        # Use interquartile slice for stability (25%..75%)
        def iqr_slice(v):
            if not v: return v
            n = len(v); a = int(0.25*n); b = max(a+1, int(0.75*n))
            return v[a:b]

        med_h    = _median(iqr_slice(heights)) or _median(heights)
        med_area = _median(iqr_slice(areas))   or _median(areas)
        med_ink  = _median(iqr_slice(inks))    or _median(inks)

        keep = []
        dropped = 0
        for (blob, f) in zip(blobs, feats):
            h_ok   = (f["h"]   >= max(1.0, h_min_frac   * med_h))
            area_ok= (f["area"]>= max(4.0, area_min_frac* med_area))
            ink_ok = (f["ink_ratio"] >= (ink_min_frac * med_ink))
            ar_ok  = (ar_low <= f["ar"] <= ar_high)

            # soft horizontal artifact veto (only if also low-ink)
            hx_bad = (f["hx"] >= hx_max_soft and f["ink_ratio"] <= hx_ink_gate)

            if h_ok and area_ok and ink_ok and ar_ok and not hx_bad:
                keep.append(blob)
            else:
                dropped += 1
                if dropped > max_outliers:
                    # stop dropping beyond cap; keep rest to avoid over-pruning
                    keep.append(blob)

        return keep

    # === main scan (identical outputs/paths) ===
    def _scan_tp_column_strips(
        self,
        gridless_gray: np.ndarray,
        cols: Dict[str, Tuple[int, int]],
        header_y: int,
        footer_y: int,
        gridless_path: str,
        save_debug: bool = False,
        debug_dir: Optional[str] = None,
        suppress_top_wipes: bool = False,
    ) -> dict:
        H, W = gridless_gray.shape
        y1 = max(0, int(header_y) + 6)
        footer_pad = max(1, int(0.003 * H))
        y2 = min(H - 1, int(footer_y) - footer_pad)
        pad_x = max(2, int(0.002 * W))

        crops_dir = os.path.dirname(gridless_path)
        if save_debug and debug_dir:
            os.path.isdir(debug_dir) or os.makedirs(debug_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(gridless_path))[0]

        def clamp_band(b):
            x1, x2 = int(b[0]), int(b[1])
            x1 = max(0, x1 - pad_x)
            x2 = min(W - 1, x2 + pad_x)
            if x2 <= x1:
                m = max(0, min(W - 2, (x1 + x2) // 2))
                x1, x2 = max(0, m - 1), min(W - 1, m + 1)
            return x1, x2

        def read_trip_strip(col_img: np.ndarray):
            if col_img is None or col_img.size == 0 or self.reader is None:
                return [], []
            sx = 2.0
            up = cv2.resize(col_img, None, fx=sx, fy=sx, interpolation=cv2.INTER_CUBIC)
            g  = cv2.GaussianBlur(up, (3,3), 0)
            g  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
            boxes = []
            out_texts = []

            def pass_once(im, mag):
                try:
                    return self.reader.readtext(
                        im, detail=1, paragraph=False,
                        allowlist="0123456789A ",
                        mag_ratio=mag, contrast_ths=0.05, adjust_contrast=0.7,
                        text_threshold=0.40, low_text=0.25,
                    )
                except Exception:
                    return []

            dets = []
            for im in (g, cv2.bitwise_not(g)):
                dets += pass_once(im, 1.6)
                dets += pass_once(im, 2.0)

            def y_center(box): return sum(p[1] for p in box) / 4.0 if box else 0.0
            dets = sorted(dets, key=lambda d: y_center(d[0]) if d else 0.0)

            last_y = -1e9
            for box, txt, _ in dets:
                yc = y_center(box)
                if abs(yc - last_y) < 10:
                    continue
                nums = re.findall(r"\d+", (txt or ""))
                if not nums:
                    continue
                keep = max(nums, key=lambda n: len(n))

                xs = [int(round(p[0] / sx)) for p in box]
                ys = [int(round(p[1] / sx)) for p in box]
                x1b, x2b = min(xs), max(xs)
                y1b, y2b = min(ys), max(ys)
                w = x2b - x1b + 1
                h = y2b - y1b + 1
                if h <= max(2, int(0.02 * col_img.shape[0])) and w >= int(3.0 * h):
                    continue

                out_texts.append(keep)
                boxes.append((x1b, y1b, x2b, y2b, keep))
                last_y = yc

            return out_texts, boxes

        def read_poles_with_boxes(col_img: np.ndarray):
            """
            Detect candidate digit blobs as before, but reject dash-like blobs up front.
            For each remaining ROI: compute OCR/template/ink probs and do weighted vote.
            Returns texts, boxes, votes (for compact debug JSON).
            """
            if col_img is None or col_img.size == 0:
                return [], [], []

            sx = 3.0
            up = cv2.resize(col_img, None, fx=sx, fy=sx, interpolation=cv2.INTER_CUBIC)
            base = cv2.GaussianBlur(up, (3,3), 0)
            base = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(base)

            def binarizations(g):
                outs = []
                outs.append(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10))
                outs.append(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12))
                _, o1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU); outs.append(o1)
                gin = cv2.bitwise_not(g)
                outs.append(cv2.adaptiveThreshold(gin, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10))
                _, o2 = cv2.threshold(gin, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU); outs.append(o2)
                return outs

            # ---- dash veto heuristic (fast) ----
            def is_dash(bw_img, x, y, w, h, H2, W2):
                """
                Reject short, flat, low-ink, mostly-horizontal blobs:
                - very small height relative to strip height
                - very large aspect ratio (w/h)
                - high horizontal edge energy, low ink mass
                Any 2 of 3 conditions = dash.
                """
                roi = bw_img[y:y+h, x:x+w]
                if roi.size == 0: 
                    return True
                # measures
                ar = w / max(1.0, float(h))
                ink = float((roi > 0).sum()) / max(1.0, float(roi.size))
                sx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                sy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
                hx = (np.abs(sx).sum() + 1e-6) / (np.abs(sx).sum() + np.abs(sy).sum() + 1e-6)

                cond_small_h = (h < max(10, int(0.12 * H2)))         # very short
                cond_flat_ar = (ar > 3.5)                             # very wide
                cond_horiz   = (hx > 0.75 and ink < 0.08)             # mostly horizontal + low ink

                hit_count = int(cond_small_h) + int(cond_flat_ar) + int(cond_horiz)
                return hit_count >= 2

            H2, W2 = base.shape[:2]
            all_boxes = []
            for bw in binarizations(base):
                bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), 1)
                bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1,2)), 1)
                cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    x,y,w,h = cv2.boundingRect(c)
                    # coarse size limits (keep potential digits only)
                    if h < 10 or h > int(0.6*H2): continue
                    if w < 2  or w > int(0.6*W2): continue
                    # NEW: drop dash-like blobs early
                    if is_dash(bw, x, y, w, h, H2, W2):
                        continue
                    yc = y + h/2.0
                    all_boxes.append((yc, x, y, w, h, bw))

            if not all_boxes:
                return [], [], []

            # Dedup by Y to keep one blob per row-ish
            all_boxes.sort(key=lambda t: t[0])
            merged = []; last_yc = -1e9; dedup_dist = max(10, int(0.035 * H2))
            for yc, x, y, w, h, bw in all_boxes:
                if abs(yc - last_yc) < dedup_dist: 
                    continue
                merged.append((x, y, w, h, bw))
                last_yc = yc

            # === Per-column calibration filter to kill schniblets ===
            merged = self._calibrate_and_filter_blobs(merged, base)
            if not merged:
                return [], [], []

            texts, boxes, votes = [], [], []
            for x, y, w, h, bw in merged:
                roi_bw   = bw[y:y+h, x:x+w]
                roi_gray = base[y:y+h, x:x+w]
 
                P_tpl, meta_tpl = self._template_digit_probs(roi_bw)
                P_ink, meta_ink = self._ink_digit_probs(roi_gray)
                P_ocr, meta_ocr = self._ocr_digit_probs(roi_gray)

                digit, dbg = self._vote_poles_digit(P_ocr, meta_ocr, P_tpl, meta_tpl, P_ink, meta_ink)

                # back map to original col_img coords
                x1 = int(round(x / sx)); y1b = int(round(y / sx))
                x2 = int(round((x + w) / sx)); y2b = int(round((y + h) / sx))

                texts.append(digit)
                boxes.append((x1, y1b, x2, y2b, digit))
                votes.append({"box": {"x1": x1, "y1": y1b, "x2": x2, "y2": y2b}, "vote": dbg})

            return texts, boxes, votes
 
        bands = {
            "l_trip":  cols.get("L_TRIP"),
            "l_poles": cols.get("L_POLES"),
            "r_poles": cols.get("R_POLES"),
            "r_trip":  cols.get("R_TRIP"),
        }

        finds = {k: [] for k in bands.keys()}

        cw_cfg = self._cfg.get("central_window", {})
        cw_enable = bool(cw_cfg.get("enable", True))
        cw_fx = float(cw_cfg.get("x_fraction", 0.60))
        cw_fy = float(cw_cfg.get("y_fraction", 1.00))
        cw_fx = min(max(cw_fx, 0.20), 1.0)
        cw_fy = min(max(cw_fy, 0.50), 1.0)

        for key, band in bands.items():
            if band is None:
                continue
            x1b, x2b = clamp_band(band)
            x_pad = 3 if key in ("l_trip", "r_trip") else 0
            if x_pad:
                x1b = max(0, x1b - x_pad)
                x2b = min(gridless_gray.shape[1], x2b + x_pad)

            raw_crop = gridless_gray[y1:y2, x1b:x2b]
            if raw_crop is None or raw_crop.size == 0:
                continue

            if cw_enable:
                Hc, Wc = raw_crop.shape[:2]
                keep_w = int(round(Wc * cw_fx))
                keep_h = int(round(Hc * cw_fy))
                xx1 = max(0, (Wc - keep_w) // 2); xx2 = min(Wc, xx1 + keep_w)
                yy1 = max(0, (Hc - keep_h) // 2); yy2 = min(Hc, yy1 + keep_h)
                crop = raw_crop[yy1:yy2, xx1:xx2]
            else:
                crop = raw_crop

            erased_boxes = []
            if not suppress_top_wipes:
                crop, erased_boxes = self._erase_top_labels(crop)
                tg_cfg = self._cfg.get("ocr_top_guard", {})
                if crop is not None and crop.size:
                    tg = max(
                        int(crop.shape[0] * float(tg_cfg.get("frac", 0.05))),
                        int(tg_cfg.get("min_px", 6))
                    )
                    if tg > 0:
                        crop[:tg, :] = 255
            dash_removed = []
            if key in ("l_poles", "r_poles") and crop is not None and crop.size:
                crop, dash_removed = self._remove_dashes_strict(crop)                        

            if crop is not None and crop.size:
                ew = self._cfg.get("edge_whiten", {})
                xm = max(int(crop.shape[1] * float(ew.get("x_margin_frac", 0.06))),
                        int(ew.get("min_px", 2)))
                crop[:, :xm]  = 255
                crop[:, -xm:] = 255

            crop = self._scrub_hline_noise(crop)

            crop_path = os.path.join(crops_dir, f"{base}_{key}.png")
            cv2.imwrite(crop_path, crop)

            if key in ("l_poles", "r_poles"):
                texts, boxes, vote_dbg = read_poles_with_boxes(crop)
            else:
                texts, boxes = read_trip_strip(crop)
                vote_dbg = []

            texts, boxes = self._filter_peripheral_values(texts, boxes, crop.shape)
            finds[key] = [t for t in texts if t]

            if save_debug and debug_dir:
                ov = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if len(crop.shape) == 2 else crop.copy()
                for (ex1,ey1,ex2,ey2) in erased_boxes:
                    cv2.rectangle(ov, (ex1,ey1), (ex2,ey2), (0,0,255), 2)  # red = header wipes
                for (dx1,dy1,dx2,dy2) in (dash_removed or []):
                    cv2.rectangle(ov, (dx1,dy1), (dx2,dy2), (130,0,200), 2)  # purple = dash kills
                for (bx1,by1,bx2,by2,txt) in boxes:
                    cv2.rectangle(ov, (bx1,by1), (bx2,by2), (0,255,255), 2)
                    cv2.putText(ov, str(txt), (bx1, max(12, by1-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                ov_path = os.path.join(debug_dir, f"{base}_{key}_overlay.png")
                cv2.imwrite(ov_path, ov)

                dump = {
                    "key": key,
                    "crop_path": crop_path,
                    "boxes": [{"x1":bx1,"y1":by1,"x2":bx2,"y2":by2,"text":str(t)} for (bx1,by1,bx2,by2,t) in boxes],
                    "erased": [{"x1":a,"y1":b,"x2":c,"y2":d} for (a,b,c,d) in erased_boxes],
                    "dash_removed": [{"x1":a,"y1":b,"x2":c,"y2":d} for (a,b,c,d) in (dash_removed or [])],  # NEW
                    "texts": finds[key],
                    "votes": vote_dbg
                }

                json_path = os.path.join(debug_dir, f"{base}_{key}_ocr.json")
                self._write_json(json_path, dump)

        return finds
