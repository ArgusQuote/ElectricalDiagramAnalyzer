#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import cv2
import pypdfium2 as pdfium  # Apache 2.0 - PDF rendering
import pikepdf              # MPL 2.0 - PDF manipulation
from PIL import Image
import json
import shutil

def _get_dbg_log_path() -> str | None:
    """
    Portable debug log path.
    Priority:
      1) EDA_DEBUG_LOG env var (lets you control it per machine)
      2) <repo_root>/.cursor/debug.log
      3) ~/.cursor/debug.log
    Returns None if we can't resolve a usable path.
    """
    # 1) Optional override
    env_p = os.getenv("EDA_DEBUG_LOG")
    if env_p:
        p = Path(env_p).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    # 2) Try repo-root relative to this file (adjust parents[] if needed)
    try:
        repo_root = Path(__file__).resolve().parents[2]  # tweak if file depth differs
        p = repo_root / ".cursor" / "debug.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    except Exception:
        pass

    # 3) Fallback: home directory
    try:
        p = Path.home() / ".cursor" / "debug.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    except Exception:
        return None

_DBG_LOG = _get_dbg_log_path()

def _dbg(hyp, loc, msg, data):
    if not _DBG_LOG:
        return
    try:
        with open(_DBG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"hypothesisId": hyp, "location": loc, "message": msg, "data": data},
                ensure_ascii=False
            ) + "\n")
    except Exception as e:
        # keep non-fatal, but don't spam too hard
        print(f"[DBG ERROR] {e}")

_dbg("INIT", "module_load", "PanelSearchToolV24 loaded (pypdfium2)", {"timestamp": "startup"})

class PanelBoardSearch:
    """
    Detect panel-board 'voids' on a detection bitmap rendered by pypdfium2
    (so coordinates match the PDF page), then:
      1) Write a vector-only PDF for each table via clip.
      2) Render a high-DPI PNG from the same clip.
    Always writes a magenta overlay per source page for QA.
      - Green boxes: detected panel/table regions (all detection methods)
      - Red boxes: raster images that are NOT inside those voids

    V24 Changes (from V23):
      - Added NMS deduplication on candidates before export to prevent
        duplicate detection of the same panel by multiple detection methods.
      - All detection method overlays now use green color consistently.
      - Fixed coordinate consistency bug: overlap checks now use separate
        unpadded_boxes list (vs void_boxes which contains padded coords).
        This ensures IoU comparisons are fair across detection methods.
      - Lowered NMS IoU threshold from 0.5 to 0.3 to match per-method overlap checks.
      - Added debug output showing candidates before NMS when verbose=True.

    V23 Changes (from V22):
      - Added border-region panel search to detect panels touching page borders
        that weren't detected as holes (e.g., panels merged with drawing frame).
      - Border-detected panels now use green color (consistent with other methods).

    V22 Changes:
      - Added _clear_border_ink() to prevent panels near page borders from
        merging with the border during binarization.
      - Added clear_border_ink parameter to control this behavior.
      - Added layout-based gap search to find panels that merge with the
        drawing frame (detects missing panels based on detected panel positions).
      - Added nested table search within large rejected holes.

    Public API:
        crops = PanelBoardSearch(...).readPdf("/path/to.pdf")
    """

    def __init__(
        self,
        output_dir: str,
        # --- detection (bitmap) ---
        dpi: int = 400,         # detection DPI (rendered with fitz for alignment)
        # --- PDF→PNG render ---
        render_dpi: int = 1200, # final PNG DPI
        aa_level: int = 8,      # fitz AA 0..8 (higher = smoother lines/text)
        render_colorspace: str = "gray",  # "gray" or "rgb"
        downsample_max_w: int | None = None,  # optional max width for PNGs

        # whitespace / void heuristics (same semantics as before)
        min_whitespace_area_fr: float = 0.01,
        margin_shave_px: int = 6,
        min_void_area_fr: float = 0.004,
        min_void_w_px: int = 90,
        min_void_h_px: int = 90,
        max_void_area_fr: float = 0.30,
        max_void_w_fr: float = 0.95,
        max_void_h_fr: float = 0.95,
        border_exclude_px: int = 4,
        void_w_fr_range: tuple | None = (0.10, 0.60),
        void_h_fr_range: tuple | None = (0.10, 0.60),

        # V22: border clearing to prevent panel-border merge
        clear_border_ink: bool = True,

        # misc
        pad: int = 6,
        debug: bool = False,
        verbose: bool = True,

        # one-box post-process (unchanged, operates on produced PNGs)
        enforce_one_box: bool = True,
        replace_multibox: bool = True,
        onebox_debug: bool = False,
        onebox_pad: int = 6,
        onebox_min_rel_area: float = 0.02,
        onebox_max_rel_area: float = 0.75,
        onebox_aspect_range: tuple = (0.4, 3.0),
        onebox_min_side_px: int = 80,
    ):
        self.output_dir = os.path.expanduser(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # subdirs
        self.magenta_dir = Path(self.output_dir) / "magenta_overlays"
        self.magenta_dir.mkdir(parents=True, exist_ok=True)
        self.vec_dir = Path(self.output_dir) / "cropped_tables_pdf"
        self.vec_dir.mkdir(parents=True, exist_ok=True)

        # where raster crops go
        self.raster_dir = Path(self.output_dir) / "raster_images"
        self.raster_dir.mkdir(parents=True, exist_ok=True)

        # knobs
        self.dpi = dpi
        self.render_dpi = render_dpi
        # aa_level kept for API compatibility (no-op with pypdfium2's Skia renderer)
        self.aa_level = int(max(0, min(8, aa_level)))

        self.render_colorspace = render_colorspace.lower().strip()
        self.downsample_max_w = downsample_max_w

        self.min_whitespace_area_fr = min_whitespace_area_fr
        self.margin_shave_px = margin_shave_px
        self.min_void_area_fr = min_void_area_fr
        self.min_void_w_px = min_void_w_px
        self.min_void_h_px = min_void_h_px
        self.max_void_area_fr = max_void_area_fr
        self.max_void_w_fr = max_void_w_fr
        self.max_void_h_fr = max_void_h_fr
        self.border_exclude_px = border_exclude_px
        self.void_w_fr_range = void_w_fr_range
        self.void_h_fr_range = void_h_fr_range

        # V22: border clearing
        self.clear_border_ink = clear_border_ink

        self.pad = pad
        self.debug = debug
        self.verbose = verbose

        # one-box
        self.enforce_one_box = enforce_one_box
        self.replace_multibox = replace_multibox
        self.onebox_debug = onebox_debug
        self.onebox_pad = onebox_pad
        self.onebox_min_rel_area = onebox_min_rel_area
        self.onebox_max_rel_area = onebox_max_rel_area
        self.onebox_aspect_range = onebox_aspect_range
        self.onebox_min_side_px = onebox_min_side_px

    # ----------------- Public API -----------------
    def readPdf(self, pdf_path: str) -> list[str]:
        """Detect panel-board voids on each PDF page, export vector-PDF clips and hi-DPI PNG crops, and return the list of PNG paths."""
        pdf_path_str = os.path.expanduser(pdf_path)
        if not Path(pdf_path_str).is_file():
            raise FileNotFoundError(pdf_path_str)

        doc = pdfium.PdfDocument(pdf_path_str)
        base = Path(pdf_path_str).stem
        all_pngs: list[str] = []

        if self.verbose:
            print(f"[INFO] Detecting with pypdfium2 @ {self.dpi} DPI")
            print(f"[INFO] Pages: {len(doc)}")

        det_scale = self.dpi / 72.0
        rend_scale = self.render_dpi / 72.0

        # track void/table boxes per page in detection pixel coords
        page_void_boxes: dict[int, list[tuple[int, int, int, int]]] = {}

        for pidx in range(len(doc)):
            page = doc[pidx]
            page_width_pt = page.get_width()
            page_height_pt = page.get_height()

            # list of panel/table void boxes for this page in detection pixels
            void_boxes: list[tuple[int, int, int, int]] = []

            # 1) Detection bitmap
            bitmap = page.render(scale=det_scale)
            det_bgr = self._bitmap_to_bgr(bitmap)
            H, W = det_bgr.shape[:2]
            page_area = H * W

            gray = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2GRAY)
            # Gentle morphology preserves thin frames; adjust if needed
            # V22: pass clear_border flag to prevent panel-border merge
            ink = self._binarize_ink(gray, close_px=0, dilate_px=1,
                                     clear_border=self.clear_border_ink)
            whitespace = cv2.bitwise_not(ink)
            whitespace = self._shave_margin(whitespace, self.margin_shave_px)

            num, labels, stats, _ = self._components(whitespace)
            keep_ids = [
                cid for cid in range(1, num)
                if stats[cid][4] / page_area >= max(0.006, self.min_whitespace_area_fr)
            ]

            sel_ws = self._selected_ws_mask(labels, keep_ids)
            contours, hier = cv2.findContours(sel_ws, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # base overlay image
            overlay_img = det_bgr.copy()
            saved_idx = 0
            candidates: list[tuple[float, float, float, float]] = []  # (x0, y0, x1, y1) in PDF points
            # Track UNPADDED boxes for overlap comparison (separate from void_boxes which is padded for export)
            unpadded_boxes: list[tuple[int, int, int, int]] = []

            def _box_passes(x, y, w, h, border_allowed=False):
                area = w * h
                if area / page_area < max(0.0025, self.min_void_area_fr):
                    return False
                if w < max(60, self.min_void_w_px) or h < max(60, self.min_void_h_px):
                    return False
                if (w / W) >= self.max_void_w_fr or (h / H) >= self.max_void_h_fr:
                    return False
                if (area / page_area) >= self.max_void_area_fr:
                    return False
                # Skip border exclusion check when border_allowed=True
                # (used for border-region panel search where panels touch page edges)
                if not border_allowed:
                    if (x <= self.border_exclude_px or y <= self.border_exclude_px or
                        (x + w) >= (W - self.border_exclude_px) or
                        (y + h) >= (H - self.border_exclude_px)):
                        return False
                if self.void_w_fr_range is not None:
                    lo, hi = self.void_w_fr_range
                    wf = w / W
                    if (lo is not None and wf < lo) or (hi is not None and wf > hi):
                        return False
                if self.void_h_fr_range is not None:
                    lo, hi = self.void_h_fr_range
                    hf = h / H
                    if (lo is not None and hf < lo) or (hi is not None and hf > hi):
                        return False
                return True

            # ---- Primary: holes inside whitespace comps ----
            large_rejected_holes = []  # Track large holes rejected due to max_area
            if hier is not None and len(hier) > 0:
                hier = hier[0]
                for ci, cnt in enumerate(contours):
                    if hier[ci][3] == -1:
                        continue  # only holes are voids
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Track large holes that fail due to area (potential nested tables)
                    area = w * h
                    if (area / page_area) >= self.max_void_area_fr and area / page_area < 0.95:
                        large_rejected_holes.append((x, y, w, h))
                    if not _box_passes(x, y, w, h):
                        continue

                    # GREEN for void/table regions
                    cv2.drawContours(overlay_img, [cnt], -1, (0, 255, 0), 8)

                    # Store UNPADDED box for overlap comparison
                    unpadded_boxes.append((x, y, x + w, y + h))
                    # #region agent log
                    _dbg("A", "primary:258", "PRIMARY detected", {"page": pidx+1, "box": [x, y, x+w, y+h], "unpadded_count": len(unpadded_boxes)})
                    # #endregion

                    # pad & map detection bbox → PDF clip
                    y0 = max(0, y - self.pad); x0 = max(0, x - self.pad)
                    y1 = min(H, y + h + self.pad); x1 = min(W, x + w + self.pad)

                    # remember this void box (padded, in detection pixel coords)
                    void_boxes.append((x0, y0, x1, y1))

                    x0f, y0f, x1f, y1f = x0 / W, y0 / H, x1 / W, y1 / H
                    clip = (
                        page_width_pt * x0f,
                        page_height_pt * y0f,
                        page_width_pt * x1f,
                        page_height_pt * y1f,
                    )
                    candidates.append(clip)

            # ---- Secondary: search for nested tables within large rejected holes ----
            for (lx, ly, lw, lh) in large_rejected_holes:
                # Crop the region and search for enclosed rectangles within
                crop_x0 = max(0, lx)
                crop_y0 = max(0, ly)
                crop_x1 = min(W, lx + lw)
                crop_y1 = min(H, ly + lh)
                cropped_region = det_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
                
                if cropped_region.size == 0:
                    continue
                
                # Use edge detection + contours to find enclosed rectangular regions
                nested_proposals = self._find_nested_tables(cropped_region)
                
                for (nx0, ny0, nx1, ny1) in nested_proposals:
                    # Convert back to full-page coordinates
                    abs_x0 = crop_x0 + nx0
                    abs_y0 = crop_y0 + ny0
                    abs_x1 = crop_x0 + nx1
                    abs_y1 = crop_y0 + ny1
                    nw, nh = abs_x1 - abs_x0, abs_y1 - abs_y0
                    
                    # Check if this nested box passes filters
                    if not _box_passes(abs_x0, abs_y0, nw, nh):
                        continue
                    
                    # Check overlap with already-detected candidates (using unpadded coords for fair comparison)
                    # Use BOTH IoU and containment ratio
                    overlaps = False
                    # #region agent log
                    max_iou_nested = 0
                    max_contain_nested = 0
                    # #endregion
                    new_box = (abs_x0, abs_y0, abs_x1, abs_y1)
                    for existing in unpadded_boxes:
                        iou_val = self._iou(new_box, existing)
                        contain_val = self._containment_ratio(new_box, existing)
                        # #region agent log
                        max_iou_nested = max(max_iou_nested, iou_val)
                        max_contain_nested = max(max_contain_nested, contain_val)
                        # #endregion
                        if iou_val > 0.3 or contain_val > 0.5:
                            overlaps = True
                            break
                    # #region agent log
                    _dbg("B", "nested:305", "NESTED overlap check", {"page": pidx+1, "new_box": list(new_box), "max_iou": max_iou_nested, "max_contain": max_contain_nested, "overlaps": overlaps, "existing_count": len(unpadded_boxes)})
                    # #endregion
                    if overlaps:
                        continue
                    
                    # Content validation: verify region actually contains table structure
                    nested_crop = det_bgr[abs_y0:abs_y1, abs_x0:abs_x1]
                    if nested_crop.size == 0:
                        continue
                    nested_mask = self._horizontal_line_mask(nested_crop)
                    nested_metrics = self._extract_horizontal_line_metrics(nested_mask)
                    if not self._is_valid_table_spacing(nested_metrics, nested_mask.shape[1], min_repeats=4):
                        if self.verbose:
                            print(f"[INFO] Nested candidate at ({abs_x0},{abs_y0}) rejected: no table structure")
                        continue
                    
                    # GREEN for nested detections (consistent with other methods)
                    cv2.rectangle(overlay_img, (abs_x0, abs_y0), (abs_x1, abs_y1), (0, 255, 0), 8)
                    
                    # Store unpadded for future overlap checks
                    unpadded_boxes.append((abs_x0, abs_y0, abs_x1, abs_y1))
                    
                    # Add to void boxes and candidates (padded)
                    padded_x0 = max(0, abs_x0 - self.pad)
                    padded_y0 = max(0, abs_y0 - self.pad)
                    padded_x1 = min(W, abs_x1 + self.pad)
                    padded_y1 = min(H, abs_y1 + self.pad)
                    
                    void_boxes.append((padded_x0, padded_y0, padded_x1, padded_y1))
                    
                    x0f, y0f = padded_x0 / W, padded_y0 / H
                    x1f, y1f = padded_x1 / W, padded_y1 / H
                    clip = (
                        page_width_pt * x0f,
                        page_height_pt * y0f,
                        page_width_pt * x1f,
                        page_height_pt * y1f,
                    )
                    candidates.append(clip)

            # ---- Tertiary: Layout-based gap search using detected panel positions ----
            if len(candidates) >= 2 and len(large_rejected_holes) > 0:
                # Use detected panel positions to find expected but missing panels
                detected_positions = [(vb[0], vb[1], vb[2]-vb[0], vb[3]-vb[1]) for vb in void_boxes]
                gap_candidates = self._find_layout_gaps(det_bgr, detected_positions, W, H)
                
                for (gx, gy, gw, gh) in gap_candidates:
                    if not _box_passes(gx, gy, gw, gh):
                        continue
                    
                    # Check overlap with already-detected (using unpadded coords for fair comparison)
                    # Use BOTH IoU and containment ratio
                    overlaps = False
                    new_box = (gx, gy, gx+gw, gy+gh)
                    for existing in unpadded_boxes:
                        iou_val = self._iou(new_box, existing)
                        contain_val = self._containment_ratio(new_box, existing)
                        if iou_val > 0.3 or contain_val > 0.5:
                            overlaps = True
                            break
                    if overlaps:
                        continue
                    
                    # Content validation: verify region actually contains table structure
                    # Gap search guesses positions - we must validate before accepting
                    gap_crop = det_bgr[gy:gy+gh, gx:gx+gw]
                    if gap_crop.size == 0:
                        continue
                    gap_mask = self._horizontal_line_mask(gap_crop)
                    gap_metrics = self._extract_horizontal_line_metrics(gap_mask)
                    if not self._is_valid_table_spacing(gap_metrics, gap_mask.shape[1], min_repeats=4):
                        if self.verbose:
                            print(f"[INFO] Gap candidate at ({gx},{gy}) rejected: no table structure")
                        continue
                    
                    # GREEN for gap-detected panels (consistent with other methods)
                    cv2.rectangle(overlay_img, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 8)
                    
                    # Store unpadded for future overlap checks
                    unpadded_boxes.append((gx, gy, gx+gw, gy+gh))
                    
                    padded_x0 = max(0, gx - self.pad)
                    padded_y0 = max(0, gy - self.pad)
                    padded_x1 = min(W, gx + gw + self.pad)
                    padded_y1 = min(H, gy + gh + self.pad)
                    
                    void_boxes.append((padded_x0, padded_y0, padded_x1, padded_y1))
                    
                    x0f, y0f = padded_x0 / W, padded_y0 / H
                    x1f, y1f = padded_x1 / W, padded_y1 / H
                    clip = (
                        page_width_pt * x0f,
                        page_height_pt * y0f,
                        page_width_pt * x1f,
                        page_height_pt * y1f,
                    )
                    candidates.append(clip)

            # ---- Quaternary: Border-region panel search ----
            # Search for panels touching page borders that weren't detected as holes
            if len(void_boxes) >= 1:  # Have detected panels to use as size reference
                detected_positions = [(vb[0], vb[1], vb[2]-vb[0], vb[3]-vb[1]) for vb in void_boxes]
                border_candidates = self._find_border_region_panels(det_bgr, detected_positions, W, H)
                
                for (bx, by, bw, bh) in border_candidates:
                    # Use relaxed border check (border_allowed=True)
                    if not _box_passes(bx, by, bw, bh, border_allowed=True):
                        continue
                    
                    # Check overlap with already-detected (using unpadded coords for fair comparison)
                    # Use BOTH IoU and containment ratio - reject if new box is mostly inside existing
                    overlaps = False
                    # #region agent log
                    max_iou_border = 0
                    max_contain_border = 0
                    all_ious_border = []
                    # #endregion
                    new_box = (bx, by, bx+bw, by+bh)
                    for existing in unpadded_boxes:
                        iou_val = self._iou(new_box, existing)
                        contain_val = self._containment_ratio(new_box, existing)
                        # #region agent log
                        max_iou_border = max(max_iou_border, iou_val)
                        max_contain_border = max(max_contain_border, contain_val)
                        if iou_val > 0.05 or contain_val > 0.3: all_ious_border.append({"existing": list(existing), "iou": round(iou_val, 3), "containment": round(contain_val, 3)})
                        # #endregion
                        # Reject if IoU > 0.3 OR if new box is >50% contained in existing
                        if iou_val > 0.3 or contain_val > 0.5:
                            overlaps = True
                            break
                    # #region agent log
                    _dbg("C", "border:430", "BORDER overlap check", {"page": pidx+1, "new_box": list(new_box), "max_iou": round(max_iou_border, 3), "max_contain": round(max_contain_border, 3), "overlaps": overlaps, "existing_count": len(unpadded_boxes), "ious": all_ious_border})
                    # #endregion
                    if overlaps:
                        continue
                    
                    # Content validation: verify region actually contains table structure
                    border_crop = det_bgr[by:by+bh, bx:bx+bw]
                    if border_crop.size == 0:
                        continue
                    border_mask = self._horizontal_line_mask(border_crop)
                    border_metrics = self._extract_horizontal_line_metrics(border_mask)
                    if not self._is_valid_table_spacing(border_metrics, border_mask.shape[1], min_repeats=4):
                        if self.verbose:
                            print(f"[INFO] Border candidate at ({bx},{by}) rejected: no table structure")
                        continue
                    
                    # GREEN for border-detected panels (same as primary detection)
                    cv2.rectangle(overlay_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 8)
                    
                    # Store unpadded for future overlap checks
                    unpadded_boxes.append((bx, by, bx+bw, by+bh))
                    
                    padded_x0 = max(0, bx - self.pad)
                    padded_y0 = max(0, by - self.pad)
                    padded_x1 = min(W, bx + bw + self.pad)
                    padded_y1 = min(H, by + bh + self.pad)
                    
                    void_boxes.append((padded_x0, padded_y0, padded_x1, padded_y1))
                    
                    x0f, y0f = padded_x0 / W, padded_y0 / H
                    x1f, y1f = padded_x1 / W, padded_y1 / H
                    clip = (
                        page_width_pt * x0f,
                        page_height_pt * y0f,
                        page_width_pt * x1f,
                        page_height_pt * y1f,
                    )
                    candidates.append(clip)
                    
                    if self.verbose:
                        print(f"[INFO] Border search found panel at ({bx},{by},{bw}x{bh})")

            # ---- Fallback: grid proposals if nothing passed ----
            if not candidates:
                for (x0, y0, x1, y1) in self._grid_proposals_fallback(det_bgr):
                    w, h = x1 - x0, y1 - y0
                    if not _box_passes(x0, y0, w, h):
                        continue
                    # ORANGE fallback boxes (debug: distinguishes from primary)
                    cv2.rectangle(overlay_img, (x0, y0), (x1, y1), (0, 165, 255), 8)

                    # remember unpadded for overlap checks and padded for export
                    unpadded_boxes.append((x0, y0, x1, y1))
                    void_boxes.append((x0, y0, x1, y1))  # fallback doesn't add extra padding

                    x0f, y0f, x1f, y1f = x0 / W, y0 / H, x1 / W, y1 / H
                    clip = (
                        page_width_pt * x0f,
                        page_height_pt * y0f,
                        page_width_pt * x1f,
                        page_height_pt * y1f,
                    )
                    candidates.append(clip)

            # ---- V24: Deduplicate candidates using NMS before export ----
            # This prevents the same panel from being exported multiple times
            # when detected by different methods with slightly different coordinates
            # NOTE: Using iou_thr=0.3 to match the overlap checks in detection methods
            if len(candidates) > 1:
                # Candidates are already tuples (x0, y0, x1, y1)
                rect_tuples = list(candidates)
                
                # #region agent log
                _dbg("D", "nms:490", "NMS input", {"page": pidx+1, "count": len(rect_tuples), "boxes": [[round(r[0],1), round(r[1],1), round(r[2],1), round(r[3],1)] for r in rect_tuples]})
                # #endregion
                
                if self.verbose:
                    print(f"[DEBUG] Page {pidx+1}: {len(rect_tuples)} candidates before NMS:")
                    for i, rt in enumerate(rect_tuples):
                        print(f"        [{i}] ({rt[0]:.1f}, {rt[1]:.1f}, {rt[2]:.1f}, {rt[3]:.1f})")
                
                deduped_tuples = self._nms_keep_larger(rect_tuples, iou_thr=0.3)
                # #region agent log
                _dbg("D", "nms:500", "NMS output", {"page": pidx+1, "count": len(deduped_tuples), "boxes": [[round(r[0],1), round(r[1],1), round(r[2],1), round(r[3],1)] for r in deduped_tuples]})
                # #endregion
                # Already tuples, just update candidates
                candidates = deduped_tuples
                if self.verbose and len(deduped_tuples) < len(rect_tuples):
                    print(f"[INFO] Page {pidx+1}: deduplicated {len(rect_tuples)} -> {len(deduped_tuples)} candidates")

            # ---- Export vector PDF + hi-DPI PNG for each candidate ----
            # #region agent log
            _dbg("D", "export:515", "FINAL candidates for export", {"page": pidx+1, "count": len(candidates), "boxes": [[round(c[0],1), round(c[1],1), round(c[2],1), round(c[3],1)] for c in candidates]})
            # #endregion
            for clip in candidates:
                saved_idx += 1
                clip_x0, clip_y0, clip_x1, clip_y1 = clip
                clip_width = clip_x1 - clip_x0
                clip_height = clip_y1 - clip_y0

                # Vector crop PDF using pikepdf (lossless vectors)
                pdf_path_out = self.vec_dir / f"{base}_page{pidx+1:03d}_panel{saved_idx:02d}.pdf"
                try:
                    with pikepdf.open(pdf_path_str) as src_pdf:
                        dst_pdf = pikepdf.new()
                        src_page = src_pdf.pages[pidx]
                        # Set MediaBox and CropBox to clip region
                        # PDF coordinates: origin at bottom-left, y increases upward
                        src_page.mediabox = pikepdf.Array([clip_x0, page_height_pt - clip_y1, clip_x1, page_height_pt - clip_y0])
                        src_page.cropbox = pikepdf.Array([clip_x0, page_height_pt - clip_y1, clip_x1, page_height_pt - clip_y0])
                        dst_pdf.pages.append(src_page)
                        dst_pdf.save(str(pdf_path_out))
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Failed to create vector PDF: {e}")

                # High-DPI PNG from the same clip using pypdfium2
                # Render full page at render_dpi, then crop
                render_bitmap = page.render(scale=rend_scale)
                render_pil = render_bitmap.to_pil()
                if self.render_colorspace == "gray":
                    render_pil = render_pil.convert("L")
                
                # Calculate clip coordinates in render pixels
                rx0 = int(clip_x0 * rend_scale)
                ry0 = int(clip_y0 * rend_scale)
                rx1 = int(clip_x1 * rend_scale)
                ry1 = int(clip_y1 * rend_scale)
                
                # Crop the rendered image
                cropped_pil = render_pil.crop((rx0, ry0, rx1, ry1))
                png = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR if cropped_pil.mode == "RGB" else cv2.COLOR_GRAY2BGR)
                
                if self.downsample_max_w and png.shape[1] > self.downsample_max_w:
                    scale_factor = self.downsample_max_w / float(png.shape[1])
                    new_wh = (self.downsample_max_w, max(1, int(round(png.shape[0] * scale_factor))))
                    png = cv2.resize(png, new_wh, interpolation=cv2.INTER_AREA)
                png_path = Path(self.output_dir) / f"{base}_page{pidx+1:03d}_panel{saved_idx:02d}.png"
                cv2.imwrite(str(png_path), png)
                all_pngs.append(str(png_path))

            # Per-page overlay (voids only for now, rasters added later)
            ov_path = self.magenta_dir / f"{base}_page{pidx+1:03d}_void_perimeters.png"
            cv2.imwrite(str(ov_path), overlay_img)

            # store void boxes for this page
            page_void_boxes[pidx] = void_boxes

            if self.verbose:
                print(f"[INFO] Page {pidx+1}: saved {saved_idx} crop(s)")

        doc.close()

        # --- post-pass: FIRST run horizontal-line spacing / validity ---
        if all_pngs:
            valid_pngs, invalid_pngs = self._save_horizontal_line_masks(all_pngs)
            all_pngs = valid_pngs  # only keep valid tables for next step

        # --- then enforce one table per PNG (lossless) on the remaining valid ones ---
        if self.enforce_one_box and all_pngs:
            all_pngs = self._enforce_one_box_on_paths(all_pngs)

        # final raster pass — crop rasters + augment overlays,
        # but DO NOT mark rasters that overlap detected panel voids
        self._extract_rasters_and_augment_magenta(pdf_path_str, page_void_boxes)

        if self.verbose:
            print(f"[DONE] Outputs → {self.output_dir}")
            print(f"      Vector PDFs → {self.vec_dir}")
            print(f"      Overlays    → {self.magenta_dir}")
            print(f"      Raster PNGs → {self.raster_dir}")

        return all_pngs

    def _find_layout_gaps(self, img_bgr, detected_positions, page_w, page_h):
        """
        Find tables in gaps based on detected panel positions.
        If we detect panels at certain X positions in one row and certain Y positions,
        we expect panels at those X positions in other Y rows too.
        """
        if len(detected_positions) < 2:
            return []
        
        # Extract unique X positions (left edges) and Y positions (top edges)
        x_positions = sorted(set(p[0] for p in detected_positions))
        y_positions = sorted(set(p[1] for p in detected_positions))
        
        # Get typical panel dimensions from detected panels
        widths = [p[2] for p in detected_positions]
        heights = [p[3] for p in detected_positions]
        median_w = sorted(widths)[len(widths)//2]
        median_h = sorted(heights)[len(heights)//2]
        
        # Size bounds for filtering (50%-150% of median, same as border detection)
        min_w = int(median_w * 0.5)
        max_w = int(median_w * 1.5)
        min_h = int(median_h * 0.5)
        max_h = int(median_h * 1.5)
        
        gap_candidates = []
        
        # For each X position, check if there are panels at all Y positions
        for x in x_positions:
            for y in y_positions:
                # Check if we already have a panel near this (x, y)
                already_detected = False
                for (dx, dy, dw, dh) in detected_positions:
                    if abs(dx - x) < 100 and abs(dy - y) < 100:
                        already_detected = True
                        break
                
                if not already_detected:
                    # Search for a table at this expected position
                    search_x0 = max(0, x - 50)
                    search_y0 = max(0, y - 50)
                    search_x1 = min(page_w, x + median_w + 50)
                    search_y1 = min(page_h, y + median_h + 50)
                    
                    crop = img_bgr[search_y0:search_y1, search_x0:search_x1]
                    if crop.size == 0:
                        continue
                    
                    # Use find_table_boxes to search for table structure
                    boxes = self._find_table_boxes(
                        crop,
                        min_rel_area=0.3,  # Must fill most of search area
                        max_rel_area=0.98,
                        aspect_range=(0.3, 3.5),
                        min_side_px=80,
                    )
                    
                    for (bx0, by0, bx1, by1) in boxes:
                        # Convert to full page coordinates
                        abs_x = search_x0 + bx0
                        abs_y = search_y0 + by0
                        abs_w = bx1 - bx0
                        abs_h = by1 - by0
                        
                        # Filter by expected panel size (reject oversized boxes)
                        if abs_w < min_w or abs_w > max_w:
                            continue
                        if abs_h < min_h or abs_h > max_h:
                            continue
                        
                        gap_candidates.append((abs_x, abs_y, abs_w, abs_h))
        
        return gap_candidates

    def _find_border_region_panels(self, img_bgr, detected_positions, W, H):
        """
        Search for panels in border regions that weren't detected as holes.
        Panels touching the drawing frame/border often merge with page edges
        and fail hole detection. This method searches border strips using
        line detection and validates with horizontal line spacing.
        
        Args:
            img_bgr: Full page image in BGR format
            detected_positions: List of (x, y, w, h) for already-detected panels
            W, H: Page dimensions in pixels
            
        Returns:
            List of (x, y, w, h) candidate boxes in border regions
        """
        if len(detected_positions) < 1:
            return []
        
        # Get typical panel dimensions from detected panels
        widths = [p[2] for p in detected_positions]
        heights = [p[3] for p in detected_positions]
        median_w = sorted(widths)[len(widths) // 2]
        median_h = sorted(heights)[len(heights) // 2]
        
        # Size tolerance: accept panels within 50% of median size
        min_w = int(median_w * 0.5)
        max_w = int(median_w * 1.5)
        min_h = int(median_h * 0.5)
        max_h = int(median_h * 1.5)
        
        # Define border strip width (25% of page dimension, but at least panel-sized)
        border_strip_w = max(int(W * 0.25), max_w + 50)
        border_strip_h = max(int(H * 0.25), max_h + 50)
        
        # Define border regions: left, right, bottom, top strips
        border_regions = [
            ("left", 0, 0, border_strip_w, H),
            ("right", W - border_strip_w, 0, border_strip_w, H),
            ("bottom", 0, H - border_strip_h, W, border_strip_h),
            ("top", 0, 0, W, border_strip_h),
        ]
        
        candidates = []
        
        for region_name, rx, ry, rw, rh in border_regions:
            
            # Crop the border region
            crop_x0 = max(0, rx)
            crop_y0 = max(0, ry)
            crop_x1 = min(W, rx + rw)
            crop_y1 = min(H, ry + rh)
            
            if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                continue
            
            crop = img_bgr[crop_y0:crop_y1, crop_x0:crop_x1]
            if crop.size == 0:
                continue
            
            # Search for table-like boxes in this border region
            boxes = self._find_table_boxes(
                crop,
                min_rel_area=0.02,  # Lower threshold since we're in a strip
                max_rel_area=0.95,
                aspect_range=(0.3, 3.5),
                min_side_px=80,
            )
            
            for (bx0, by0, bx1, by1) in boxes:
                # Convert to full page coordinates
                abs_x = crop_x0 + bx0
                abs_y = crop_y0 + by0
                abs_w = bx1 - bx0
                abs_h = by1 - by0
                
                # Filter by expected panel size (based on detected panels)
                if abs_w < min_w or abs_w > max_w:
                    continue
                if abs_h < min_h or abs_h > max_h:
                    continue
                
                # Check if this overlaps with already detected panels
                overlaps = False
                for (dx, dy, dw, dh) in detected_positions:
                    if self._iou((abs_x, abs_y, abs_x + abs_w, abs_y + abs_h),
                                 (dx, dy, dx + dw, dy + dh)) > 0.3:
                        overlaps = True
                        break
                
                if not overlaps:
                    candidates.append((abs_x, abs_y, abs_w, abs_h))
        
        # Remove duplicates using NMS
        if len(candidates) > 1:
            # Convert to (x0, y0, x1, y1) format for NMS
            boxes_xyxy = [(x, y, x + w, y + h) for (x, y, w, h) in candidates]
            boxes_xyxy = self._nms_keep_larger(boxes_xyxy, iou_thr=0.3)
            candidates = [(x0, y0, x1 - x0, y1 - y0) for (x0, y0, x1, y1) in boxes_xyxy]
        
        return candidates

    def _find_nested_tables(self, img_bgr):
        """
        Find table-like rectangular regions within a cropped image.
        Uses hole detection within whitespace to find enclosed panels.
        Also uses rectangle detection as a fallback for border-connected tables.
        """
        H, W = img_bgr.shape[:2]
        if H < 100 or W < 100:
            return []
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        crop_area = H * W
        proposals = []
        
        # Method 1: Hole detection (works for fully enclosed panels)
        ink = self._binarize_ink(gray, close_px=0, dilate_px=1, clear_border=True)
        whitespace = cv2.bitwise_not(ink)
        whitespace = self._shave_margin(whitespace, 4)
        
        num, labels, stats, _ = self._components(whitespace)
        keep_ids = [
            cid for cid in range(1, num)
            if stats[cid][4] / crop_area >= 0.005
        ]
        
        if keep_ids:
            sel_ws = self._selected_ws_mask(labels, keep_ids)
            contours, hier = cv2.findContours(sel_ws, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            if hier is not None and len(hier) > 0:
                hier_arr = hier[0]
                for ci, cnt in enumerate(contours):
                    if hier_arr[ci][3] == -1:
                        continue  # only holes
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    if area / crop_area < 0.02 or area / crop_area > 0.6:
                        continue
                    if w < 80 or h < 80:
                        continue
                    aspect = w / float(h)
                    if aspect < 0.3 or aspect > 3.5:
                        continue
                    proposals.append((x, y, x + w, y + h))
        
        # Method 2: Rectangle detection (finds border-connected tables)
        # Use adaptive threshold to find table lines
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 10)
        
        # Detect horizontal and vertical lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 20, 30), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 20, 30)))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        
        # Combine lines
        table_mask = cv2.bitwise_or(h_lines, v_lines)
        table_mask = cv2.dilate(table_mask, None, iterations=2)
        
        # Find contours of table-like regions
        rect_contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in rect_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            cnt_area = cv2.contourArea(cnt)
            if area / crop_area < 0.02 or area / crop_area > 0.6:
                continue
            if w < 80 or h < 80:
                continue
            aspect = w / float(h)
            if aspect < 0.3 or aspect > 3.5:
                continue
            # Check contour area vs bounding box (should be rectangular)
            if cnt_area / area < 0.3:  # Too sparse, not a solid table outline
                continue
            proposals.append((x, y, x + w, y + h))
        
        # Sort by area (largest first) and apply NMS
        proposals.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        proposals = self._nms_keep_larger(proposals, iou_thr=0.3)
        
        return proposals

    def _grid_proposals_fallback(self, img_bgr):
        """
        Return a list of (x0,y0,x1,y1) covering big grid-like regions even if
        the 'holes inside whitespace' logic yields nothing.
        """
        H, W = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)
        edges = cv2.Canny(gray, 60, 180, L2gradient=True)

        min_len = int(max(W * 0.08, H * 0.08, 30))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90,
                                minLineLength=min_len, maxLineGap=4)

        # accumulate vertical & horizontal edges
        acc = np.zeros_like(gray, dtype=np.uint8)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0, :]:
                cv2.line(acc, (x1, y1), (x2, y2), 255, 2)

        # close gaps → contours
        acc = cv2.dilate(acc, None, iterations=1)
        acc = cv2.erode(acc, None, iterations=1)
        cnts, _ = cv2.findContours(acc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        props = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # prefer table-like rectangles (reasonably big, not micro)
            if w < 120 or h < 90:
                continue
            props.append((x, y, x + w, y + h))
        # largest first
        props.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return props

    # ---------------- helpers ----------------
    @staticmethod
    def _bitmap_to_bgr(bitmap, grayscale: bool = False) -> np.ndarray:
        """Convert pypdfium2 bitmap to BGR numpy array."""
        pil_img = bitmap.to_pil()
        if grayscale:
            pil_img = pil_img.convert("L")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _rect_to_pix_rect(rect: tuple[float, float, float, float],
                          zoom: float,
                          rotation: int = 0,
                          page_width: float = 0) -> tuple[float, float, float, float]:
        """
        Map a rectangle from PDF coordinate space to pixmap coordinate space.
        Works for rotation=0 and rotation=270.
        
        Args:
            rect: (x0, y0, x1, y1) in PDF points
            zoom: scale factor (dpi / 72)
            rotation: page rotation in degrees
            page_width: unrotated page width (needed for rotation=270)
        
        Returns:
            (x0, y0, x1, y1) in pixel coordinates
        """
        x0, y0, x1, y1 = rect
        rot = rotation % 360

        # Normal pages
        if rot == 0:
            return (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)

        # 270° rotation explicitly
        if rot == 270:
            xs = [x0, x1, x1, x0]
            ys = [y0, y0, y1, y1]

            # (x, y) -> (y, w0 - x)
            xps = [y for x, y in zip(xs, ys)]
            yps = [page_width - x for x, y in zip(xs, ys)]

            minx, maxx = min(xps), max(xps)
            miny, maxy = min(yps), max(yps)

            return (minx * zoom, miny * zoom, maxx * zoom, maxy * zoom)

        # Fallback: treat other rotations like 0°
        return (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)

    @staticmethod
    def _clear_border_ink(ink_mask: np.ndarray) -> np.ndarray:
        """
        Remove ink regions connected to the image border via flood-fill.
        This prevents page borders from merging with panel outlines.

        Args:
            ink_mask: Binary mask where 255 = ink (dark regions)

        Returns:
            Cleared mask with border-connected ink removed
        """
        h, w = ink_mask.shape
        cleared = ink_mask.copy()
        # flood_mask must be 2 pixels larger than image in each dimension
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Flood-fill from all border pixels that contain ink
        # Top edge
        for x in range(w):
            if cleared[0, x] > 0:
                cv2.floodFill(cleared, flood_mask, (x, 0), 0)
        # Bottom edge
        for x in range(w):
            if cleared[h - 1, x] > 0:
                cv2.floodFill(cleared, flood_mask, (x, h - 1), 0)
        # Left edge
        for y in range(h):
            if cleared[y, 0] > 0:
                cv2.floodFill(cleared, flood_mask, (0, y), 0)
        # Right edge
        for y in range(h):
            if cleared[y, w - 1] > 0:
                cv2.floodFill(cleared, flood_mask, (w - 1, y), 0)

        return cleared

    @staticmethod
    def _binarize_ink(gray: np.ndarray, close_px: int = 3, dilate_px: int = 3,
                      clear_border: bool = True) -> np.ndarray:
        """
        Binarize grayscale image to extract ink (dark) regions.

        Args:
            gray: Grayscale input image
            close_px: Morphological close kernel radius (0 to disable)
            dilate_px: Dilation kernel radius (0 to disable)
            clear_border: If True, remove ink connected to image border
                          (prevents panel-border merge issue)

        Returns:
            Binary mask where 255 = ink
        """
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # V22: Clear border-connected ink BEFORE morphology
        # This prevents page borders from merging with nearby panel outlines
        if clear_border:
            bw_inv = PanelBoardSearch._clear_border_ink(bw_inv)

        if close_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * close_px + 1, 2 * close_px + 1))
            bw_inv = cv2.morphologyEx(bw_inv, cv2.MORPH_CLOSE, k, iterations=1)
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_px + 1, 2 * dilate_px + 1))
            bw_inv = cv2.dilate(bw_inv, k, iterations=1)
        return bw_inv

    @staticmethod
    def _shave_margin(mask: np.ndarray, px: int) -> np.ndarray:
        """Zero out the border of *mask* by *px* pixels on all sides to suppress edge artifacts."""
        if px <= 0:
            return mask
        m = mask.copy()
        h, w = m.shape
        m[:px, :] = 0; m[-px:, :] = 0; m[:, :px] = 0; m[:, -px:] = 0
        return m

    @staticmethod
    def _components(mask: np.ndarray):
        """Run 8-connected component analysis on a binary mask; return (num, labels, stats, centroids)."""
        lab = (mask > 0).astype(np.uint8)
        return cv2.connectedComponentsWithStats(lab, connectivity=8)

    @staticmethod
    def _selected_ws_mask(labels: np.ndarray, keep_ids: list[int]) -> np.ndarray:
        """Build a binary mask containing only the whitespace components whose IDs are in *keep_ids*."""
        m = np.zeros_like(labels, dtype=np.uint8)
        for cid in keep_ids:
            m[labels == cid] = 255
        return m

    # One box logic
    def _enforce_one_box_on_paths(self, image_paths: list[str]) -> list[str]:
        final_paths = []
        for p in image_paths:
            try:
                img = cv2.imread(p)
                if img is None:
                    if self.verbose:
                        print(f"[WARN] Could not read {p}")
                    continue

                boxes = self._find_table_boxes(
                    img,
                    min_rel_area=self.onebox_min_rel_area,
                    max_rel_area=self.onebox_max_rel_area,
                    aspect_range=self.onebox_aspect_range,
                    min_side_px=self.onebox_min_side_px,
                )

                base = Path(p).stem
                out_dir = Path(self.output_dir)
                H, W = img.shape[:2]

                # If 0 or 1 table box -> keep as-is
                if len(boxes) <= 1:
                    final_paths.append(p)
                    if self.onebox_debug:
                        dbg = img.copy()
                        for (x1, y1, x2, y2) in boxes:
                            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.imwrite(str(out_dir / f"{base}__debug_single_box.png"), dbg)
                    continue

                # Split into multiple
                saved = []
                for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
                    x1p, y1p = max(0, x1 - self.onebox_pad), max(0, y1 - self.onebox_pad)
                    x2p, y2p = min(W, x2 + self.onebox_pad), min(H, y2 + self.onebox_pad)
                    crop = img[y1p:y2p, x1p:x2p]
                    out_path = out_dir / f"{base}__tbl{i:02d}.png"
                    cv2.imwrite(str(out_path), crop)
                    saved.append(str(out_path))

                if self.onebox_debug:
                    dbg = img.copy()
                    for (x1, y1, x2, y2) in boxes:
                        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imwrite(str(out_dir / f"{base}__debug_boxes.png"), dbg)

                if self.replace_multibox:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

                final_paths.extend(saved)

            except Exception as e:
                if self.verbose:
                    print(f"[WARN] one-box step failed for {p}: {e}")
                final_paths.append(p)

        return final_paths

    def _find_table_boxes(self, img_bgr,
                        min_rel_area=0.02,
                        max_rel_area=0.75,
                        aspect_range=(0.4, 3.0),
                        min_side_px=80):
        """Find table-like rectangular regions via adaptive threshold + morphological line extraction, then NMS."""
        H, W = img_bgr.shape[:2]
        page_area = H * W
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10
        )
        hk = max(W // 80, 5)
        vk = max(H // 80, 5)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
        lh = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kh)
        lv = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kv)
        lines = cv2.bitwise_or(lh, lv)
        lines = cv2.dilate(lines, None, iterations=1)
        lines = cv2.erode(lines, None, iterations=1)
        cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < min_side_px or h < min_side_px:
                continue
            area = w * h
            rel = area / page_area
            if rel < min_rel_area or rel > max_rel_area:
                continue
            aspect = w / float(h)
            if aspect < aspect_range[0] or aspect > aspect_range[1]:
                continue
            boxes.append((x, y, x + w, y + h))

        if not boxes:
            return []
        boxes = self._nms_keep_larger(boxes, iou_thr=0.5)
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    @staticmethod
    def _nms_keep_larger(boxes, iou_thr=0.5):
        """Non-maximum suppression: keep the larger box when two overlap above *iou_thr*."""
        if len(boxes) <= 1:
            return boxes
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        keep = []
        for box in boxes:
            # #region agent log
            ious_for_box = [(round(PanelBoardSearch._iou(box, k), 3), list(k)) for k in keep]
            max_iou = max([iou for iou, _ in ious_for_box], default=0)
            _dbg("E", "nms_fn:1105", "NMS checking box", {"box": list(box), "vs_kept": len(keep), "max_iou": max_iou, "threshold": iou_thr, "will_keep": max_iou <= iou_thr})
            # #endregion
            if all(PanelBoardSearch._iou(box, k) <= iou_thr for k in keep):
                keep.append(box)
        return keep

    @staticmethod
    def _iou(a, b):
        """Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        denom = (area_a + area_b - inter)
        return inter / float(denom) if denom > 0 else 0.0

    @staticmethod
    def _containment_ratio(new_box, existing_box):
        """
        Calculate what fraction of new_box is contained within existing_box.
        Returns value 0-1 where 1 means new_box is fully inside existing_box.
        """
        ax1, ay1, ax2, ay2 = new_box
        bx1, by1, bx2, by2 = existing_box
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_new = (ax2 - ax1) * (ay2 - ay1)
        return inter / float(area_new) if area_new > 0 else 0.0

    # ---------------- Horizontal-line debug masks ----------------
    def _save_horizontal_line_masks(self, image_paths: list[str]) -> tuple[list[str], list[str]]:
        debug_dir = Path(self.output_dir) / "debug_horizontal_lines"
        if self.debug:
            debug_dir.mkdir(parents=True, exist_ok=True)

        invalid_dir = Path(self.output_dir) / "invalid_tables"
        invalid_dir.mkdir(parents=True, exist_ok=True)

        valid_paths: list[str] = []
        invalid_paths: list[str] = []

        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                if self.verbose:
                    print(f"[WARN] Could not read crop for horiz debug: {p}")
                valid_paths.append(p)
                continue

            mask = self._horizontal_line_mask(img)
            metrics = self._extract_horizontal_line_metrics(mask)

            valid_table = self._is_valid_table_spacing(
                metrics,
                mask_width=mask.shape[1],
            )

            if not valid_table:
                new_path = invalid_dir / Path(p).name
                try:
                    shutil.move(p, new_path)
                    invalid_paths.append(str(new_path))
                    if self.verbose:
                        print(f"[INFO] Moved invalid table → {new_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Failed to move invalid table {p}: {e}")
                    invalid_paths.append(p)
            else:
                valid_paths.append(p)

            if not self.debug:
                continue

            mask_path = debug_dir / f"{Path(p).stem}__horiz_mask.png"
            cv2.imwrite(str(mask_path), mask)

            payload = {
                "lines": metrics,
                "valid_table": bool(valid_table),
            }
            json_path = debug_dir / f"{Path(p).stem}__horiz_lines.json"
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to write horiz lines JSON for {p}: {e}")

        return valid_paths, invalid_paths

    @staticmethod
    def _horizontal_line_mask(img_bgr: np.ndarray) -> np.ndarray:
        """Extract a binary mask of horizontal lines from *img_bgr* using adaptive threshold + morphological opening."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )
        H, W = gray.shape
        hk = max(W // 80, 5)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kh)
        _, horiz_bin = cv2.threshold(horiz, 0, 255, cv2.THRESH_BINARY)
        return horiz_bin

    @staticmethod
    def _extract_horizontal_line_metrics(mask: np.ndarray) -> list[list[int]]:
        """Extract row-center spacing and line-length metrics from a horizontal-line mask; returns [[idx, dy, length], ...]."""
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        h, w = mask.shape[:2]
        row_has_line = (mask > 0).any(axis=1)

        spans = []
        in_span = False
        start_y = 0

        for y, has in enumerate(row_has_line):
            if has and not in_span:
                in_span = True
                start_y = y
            elif not has and in_span:
                in_span = False
                end_y = y - 1
                spans.append((start_y, end_y))

        if in_span:
            spans.append((start_y, h - 1))

        centers: list[int] = []
        lengths: list[int] = []
        for (s, e) in spans:
            cy = (s + e) // 2
            span_mask = (mask[s:e+1, :] > 0)
            col_has_line = span_mask.any(axis=0)
            length = int(col_has_line.sum())
            centers.append(cy)
            lengths.append(length)

        metrics: list[list[int]] = []
        prev_center = None

        for idx, (cy, length) in enumerate(zip(centers, lengths), start=1):
            if prev_center is None:
                dy = 0
            else:
                dy = int(cy - prev_center)
            metrics.append([idx, dy, int(length)])
            prev_center = cy

        return metrics

    @staticmethod
    def _is_valid_table_spacing(
        metrics: list[list[int]],
        mask_width: int,
        spacing_tol_px: int = 4,
        min_repeats: int = 5,
        min_len_frac: float = 0.5,
    ) -> bool:
        """Return True if *metrics* shows at least *min_repeats* lines with consistent spacing (a table-like pattern)."""
        min_len_px = int(mask_width * min_len_frac)

        filtered = [
            (idx, dy, length)
            for (idx, dy, length) in metrics
            if dy > 0 and length >= min_len_px
        ]

        if len(filtered) < min_repeats:
            return False

        spacings = sorted(dy for (_, dy, _) in filtered)
        n = len(spacings)

        for i in range(n):
            base = spacings[i]
            count = 1
            j = i + 1
            while j < n and abs(spacings[j] - base) <= spacing_tol_px:
                count += 1
                j += 1
            if count >= min_repeats:
                return True

        return False

    # ----- Raster cropping + overlay augmentation -----
    def _extract_rasters_and_augment_magenta(
        self,
        pdf_path: str,
        page_void_boxes: dict[int, list[tuple[int, int, int, int]]],
    ):
        """
        Second pass over the PDF:
          - Extract embedded raster images using pikepdf
          - draw RED boxes for rasters directly on the existing magenta
            overlay PNGs (one per page),
            BUT skip rasters whose bbox overlaps a detected panel/table
            void region (green) with IoU > 0.5.
        
        Note: pypdfium2 doesn't have direct image extraction APIs like PyMuPDF,
        so we use pikepdf for XObject enumeration.
        """
        try:
            pdf = pikepdf.open(pdf_path)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Could not reopen PDF for raster pass: {pdf_path} ({e})")
            return

        base = Path(pdf_path).stem
        zoom = self.dpi / 72.0

        # Also open with pypdfium2 for rendering
        try:
            pdfium_doc = pdfium.PdfDocument(pdf_path)
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Could not open PDF with pypdfium2 for raster pass: {e}")
            pdf.close()
            return

        for pidx in range(len(pdf.pages)):
            page = pdf.pages[pidx]
            pdfium_page = pdfium_doc[pidx]
            
            # Render at detection DPI so coords match overlays
            bitmap = pdfium_page.render(scale=zoom)
            page_img = self._bitmap_to_bgr(bitmap)
            H, W = page_img.shape[:2]

            # void/panel boxes for this page in detection pixel coords
            void_boxes = page_void_boxes.get(pidx, [])

            ov_path = self.magenta_dir / f"{base}_page{pidx+1:03d}_void_perimeters.png"
            overlay = cv2.imread(str(ov_path), cv2.IMREAD_COLOR)
            if overlay is None or overlay.shape[:2] != (H, W):
                overlay = page_img.copy()

            raster_idx = 0

            # Extract images from page Resources using pikepdf
            try:
                if "/Resources" in page and "/XObject" in page["/Resources"]:
                    xobjects = page["/Resources"]["/XObject"]
                    for name, xobj_ref in xobjects.items():
                        try:
                            xobj = xobj_ref
                            if xobj.get("/Subtype") == "/Image":
                                # This is an image XObject
                                # Note: Getting precise placement requires parsing content stream
                                # For simplicity, we skip detailed bbox extraction here
                                # The main panel detection already handles most cases
                                pass
                        except Exception:
                            continue
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] raster pass {base} page {pidx+1}: XObject extraction failed: {e}")

            # Note: Full raster extraction with precise bounding boxes requires
            # parsing the PDF content stream, which is complex. For now, we rely
            # on the panel detection to identify table regions. Raster overlay
            # marking is disabled in the pypdfium2 version.
            
            if self.verbose and raster_idx > 0:
                print(f"[INFO] Raster pass page {pidx+1}: "
                      f"{raster_idx} raster(s), overlay updated")

        pdfium_doc.close()
        pdf.close()
