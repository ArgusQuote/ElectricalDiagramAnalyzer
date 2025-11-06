#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PanelBoardSearch
----------------
Find panel-board tables as 'voids' (holes) inside large whitespace regions
and save crops. Overlays are saved only if debug=True — EXCEPT the magenta
perimeter overlay, which is always saved to output_dir/magenta_overlays.

NEW (debug JSON with DPI):
  When debug=True, writes a JSON file that records, for EVERY saved crop:
    - which page it came from
    - table index on that page
    - crop path + size
    - bbox (px and fractional)
    - rasterization metadata, including the *effective DPI* used
  Path:
    <output_dir>/search_debug/<basename>_crops_debug.json

Public API:
    crops = PanelBoardSearch(...).readPdf("/path/to.pdf")
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageFile
from pdf2image import convert_from_path

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PanelBoardSearch:
    def __init__(
        self,
        output_dir: str,
        dpi: int = 400,
        # Whitespace selection (green regions)
        min_whitespace_area_fr: float = 0.01,
        margin_shave_px: int = 6,
        # Void (table) filters to avoid tiny letters/specks
        min_void_area_fr: float = 0.004,
        min_void_w_px: int = 90,
        min_void_h_px: int = 90,
        # Upper bounds to avoid whole-page crops
        max_void_area_fr: float = 0.30,
        max_void_w_fr: float = 0.95,
        max_void_h_fr: float = 0.95,
        border_exclude_px: int = 4,
        # width/height fraction gates to avoid thin strips (fractions of page)
        void_w_fr_range: tuple | None = (0.10, 0.60),
        void_h_fr_range: tuple | None = (0.10, 0.60),
        # Cropping + debug
        pad: int = 6,
        debug: bool = False,
        verbose: bool = True,
        save_masked_shape_crop: bool = False,
        # PDF conversion safety
        max_megapixels: int = 170_000_000,
        dpi_fallbacks: tuple = (350, 320, 300, 260, 220, 200),
        # --- one-box enforcement (post-process) ---
        enforce_one_box: bool = True,
        replace_multibox: bool = True,   # delete original multi-box crop after splitting
        onebox_debug: bool = False,
        onebox_pad: int = 6,
        onebox_min_rel_area: float = 0.02,
        onebox_max_rel_area: float = 0.75,
        onebox_aspect_range: tuple = (0.4, 3.0),
        onebox_min_side_px: int = 80,
    ):
        self.output_dir = os.path.expanduser(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Always-on magenta overlay directory
        self.magenta_dir = Path(self.output_dir) / "magenta_overlays"
        self.magenta_dir.mkdir(parents=True, exist_ok=True)

        # NEW: search-debug directory + in-memory log (only used if debug=True)
        self.search_debug_dir = Path(self.output_dir) / "search_debug"
        self.debug = bool(debug)
        if self.debug:
            self.search_debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_crops_log: List[Dict[str, Any]] = []

        self.verbose = bool(verbose)
        self.dpi = int(dpi)

        # WS / void selection params
        self.min_whitespace_area_fr = float(min_whitespace_area_fr)
        self.margin_shave_px = int(margin_shave_px)

        self.min_void_area_fr = float(min_void_area_fr)
        self.min_void_w_px = int(min_void_w_px)
        self.min_void_h_px = int(min_void_h_px)

        self.max_void_area_fr = float(max_void_area_fr)
        self.max_void_w_fr = float(max_void_w_fr)
        self.max_void_h_fr = float(max_void_h_fr)
        self.border_exclude_px = int(border_exclude_px)

        self.void_w_fr_range = void_w_fr_range
        self.void_h_fr_range = void_h_fr_range

        self.pad = int(pad)
        self.save_masked_shape_crop = bool(save_masked_shape_crop)

        # pdf2image conversion controls
        self.max_megapixels = int(max_megapixels)
        self.dpi_fallbacks = tuple(dpi_fallbacks)

        # one-box
        self.enforce_one_box = bool(enforce_one_box)
        self.replace_multibox = bool(replace_multibox)
        self.onebox_debug = bool(onebox_debug)
        self.onebox_pad = int(onebox_pad)
        self.onebox_min_rel_area = float(onebox_min_rel_area)
        self.onebox_max_rel_area = float(onebox_max_rel_area)
        self.onebox_aspect_range = tuple(onebox_aspect_range)
        self.onebox_min_side_px = int(onebox_min_side_px)

    # ----------------- Public API -----------------
    def readPdf(self, pdf_path: str) -> List[str]:
        pdf_path = os.path.expanduser(pdf_path)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if self.verbose:
            print(f"[INFO] Converting PDF → images @ {self.dpi} DPI")

        # Convert; now returns list of (PIL.Image, meta)
        page_tuples = self._convert_with_size_cap(pdf_path, self.dpi)
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        all_crops: List[str] = []

        if self.verbose:
            print(f"[INFO] Pages: {len(page_tuples)}")

        for page_idx, (pil_page, page_meta) in enumerate(page_tuples, 1):
            try:
                img = self._pil_to_cv(pil_page)
                page_crops = self._process_page(img, base, page_idx, page_meta)
                if self.enforce_one_box and page_crops:
                    onebox_results = self._enforce_one_box_on_paths(page_crops)
                    all_crops.extend(onebox_results)
                else:
                    all_crops.extend(page_crops)

                if self.verbose:
                    print(f"[INFO] Page {page_idx}: saved {len(page_crops)} crop(s)")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Page {page_idx} failed: {e}")

        if self.verbose:
            print(f"[DONE] Total crops (after 1-box step): {len(all_crops)} → {self.output_dir}")

        # ---- Write debug JSON with DPI per crop ----
        if self.debug:
            dbg_path = self.search_debug_dir / f"{base}_crops_debug.json"
            try:
                with open(dbg_path, "w", encoding="utf-8") as f:
                    json.dump(self._debug_crops_log, f, indent=2)
                if self.verbose:
                    print(f"[DEBUG] Wrote crop debug JSON → {dbg_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed writing debug JSON: {e}")

        return all_crops

    # ----------------- Internals -----------------
    @staticmethod
    def _pil_to_cv(img_pil) -> np.ndarray:
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _convert_with_size_cap(self, pdf_path: str, dpi: int) -> List[Tuple[Any, Dict[str, Any]]]:
        """
        Returns a list of tuples: (PIL.Image, meta)
        meta = {
          "mode": "direct" | "size_cap" | "fallback",
          "requested_dpi": <int>,
          "effective_dpi": [dx, dy] or <int>,
          "size_cap_px": <int or None>,
          "fallback_dpi": <int or None>
        }
        """
        def _wrap(pages, mode, requested_dpi, size_cap_px=None, fallback_dpi=None):
            out = []
            for im in pages:
                eff = im.info.get("dpi", requested_dpi)  # PIL often stores (xdpi, ydpi)
                out.append((im, {
                    "mode": mode,
                    "requested_dpi": requested_dpi,
                    "effective_dpi": eff,
                    "size_cap_px": size_cap_px,
                    "fallback_dpi": fallback_dpi
                }))
            return out

        try:
            pages = convert_from_path(pdf_path, dpi=dpi)
            return _wrap(pages, mode="direct", requested_dpi=dpi)
        except Exception as e1:
            if self.verbose:
                print(f"[convert] initial {dpi} DPI failed: {type(e1).__name__}: {e1}")

        longest = int((self.max_megapixels * 1.3) ** 0.5)
        try:
            if self.verbose:
                print(f"[convert] retry {dpi} DPI with size cap = {longest}px")
            pages = convert_from_path(pdf_path, dpi=dpi, size=longest)
            return _wrap(pages, mode="size_cap", requested_dpi=dpi, size_cap_px=longest)
        except Exception as e2:
            if self.verbose:
                print(f"[convert] size-capped {dpi} DPI failed: {type(e2).__name__}: {e2}")

        for step in self.dpi_fallbacks:
            try:
                if self.verbose:
                    print(f"[convert] trying fallback DPI {step}")
                pages = convert_from_path(pdf_path, dpi=step)
                return _wrap(pages, mode="fallback", requested_dpi=dpi, fallback_dpi=step)
            except Exception as e3:
                if self.verbose:
                    print(f"[convert] {step} DPI failed: {type(e3).__name__}: {e3}")

        if self.verbose:
            print("[convert] disabling PIL MAX_IMAGE_PIXELS and retrying original DPI")
        Image.MAX_IMAGE_PIXELS = None
        pages = convert_from_path(pdf_path, dpi=dpi)
        return _wrap(pages, mode="direct", requested_dpi=dpi)

    @staticmethod
    def _binarize_ink(gray: np.ndarray, close_px: int = 3, dilate_px: int = 3) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if close_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * close_px + 1, 2 * close_px + 1))
            bw_inv = cv2.morphologyEx(bw_inv, cv2.MORPH_CLOSE, k, iterations=1)
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_px + 1, 2 * dilate_px + 1))
            bw_inv = cv2.dilate(bw_inv, k, iterations=1)
        return bw_inv

    @staticmethod
    def _shave_margin(mask: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return mask
        m = mask.copy()
        h, w = m.shape
        m[:px, :] = 0; m[-px:, :] = 0; m[:, :px] = 0; m[:, -px:] = 0
        return m

    @staticmethod
    def _components(mask: np.ndarray):
        lab = (mask > 0).astype(np.uint8)
        return cv2.connectedComponentsWithStats(lab, connectivity=8)

    @staticmethod
    def _selected_ws_mask(labels: np.ndarray, keep_ids: List[int]) -> np.ndarray:
        m = np.zeros_like(labels, dtype=np.uint8)
        for cid in keep_ids:
            m[labels == cid] = 255
        return m

    def _draw_green_overlay(self, bgr, labels, stats, keep_ids, alpha=0.45):
        overlay = bgr.copy()
        tint = np.array([0, 255, 0], dtype=np.float32)
        H, W = bgr.shape[:2]
        for comp_id in keep_ids:
            x, y, w, h, area = stats[comp_id]
            region = (labels == comp_id)
            base = overlay[region].astype(np.float32)
            overlay[region] = np.clip((1.0 - alpha) * base + alpha * tint, 0, 255).astype(np.uint8)
            cv2.rectangle(overlay, (max(0, x-1), max(0, y-1)), (min(W-1, x+w), min(H-1, y+h)), (0, 200, 0), 5)
        return overlay

    # ------------- FULL per-page processing -------------
    def _process_page(self, img_bgr: np.ndarray, base_name: str, page_idx: int, page_meta: Dict[str, Any]) -> List[str]:
        H, W = img_bgr.shape[:2]
        page_area = H * W
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1) INK / WHITESPACE
        ink = self._binarize_ink(gray, close_px=3, dilate_px=3)   # 255 ink
        whitespace = cv2.bitwise_not(ink)                         # 255 bg
        whitespace = self._shave_margin(whitespace, self.margin_shave_px)

        num, labels, stats, _ = self._components(whitespace)
        keep_ids: List[int] = []
        for cid in range(1, num):
            x, y, w, h, area = stats[cid]
            if area / page_area >= self.min_whitespace_area_fr:
                keep_ids.append(cid)

        sel_ws_mask = self._selected_ws_mask(labels, keep_ids)

        # 2) FIND VOIDS (holes) INSIDE SELECTED WHITESPACE
        contours, hier = cv2.findContours(sel_ws_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Debug green overlay
        if self.debug:
            overlay_green = self._draw_green_overlay(img_bgr, labels, stats, keep_ids, alpha=0.45)

        # Always-on magenta perimeter overlay (start from original page)
        magenta_canvas = img_bgr.copy()

        saved_paths: List[str] = []
        saved_idx = 0

        if hier is not None and len(hier) > 0:
            hier = hier[0]
            for idx, cnt in enumerate(contours):
                parent = hier[idx][3]
                if parent == -1:
                    continue  # only holes are voids

                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h

                # ---- FILTERS ----
                too_small = (
                    area / float(page_area) < self.min_void_area_fr
                    or w < self.min_void_w_px
                    or h < self.min_void_h_px
                )
                too_wide  = (w / float(W)) >= self.max_void_w_fr
                too_tall  = (h / float(H)) >= self.max_void_h_fr
                too_big   = (area / float(page_area)) >= self.max_void_area_fr
                touches_border = (
                    x <= self.border_exclude_px or
                    y <= self.border_exclude_px or
                    (x + w) >= (W - self.border_exclude_px) or
                    (y + h) >= (H - self.border_exclude_px)
                )

                w_fr = w / float(W)
                h_fr = h / float(H)
                w_out_of_range = (
                    (self.void_w_fr_range is not None) and
                    (
                        (self.void_w_fr_range[0] is not None and w_fr < self.void_w_fr_range[0]) or
                        (self.void_w_fr_range[1] is not None and w_fr > self.void_w_fr_range[1])
                    )
                )
                h_out_of_range = (
                    (self.void_h_fr_range is not None) and
                    (
                        (self.void_h_fr_range[0] is not None and h_fr < self.void_h_fr_range[0]) or
                        (self.void_h_fr_range[1] is not None and h_fr > self.void_h_fr_range[1])
                    )
                )

                if (too_small or too_big or too_wide or too_tall or touches_border
                    or w_out_of_range or h_out_of_range):
                    continue

                # ALWAYS draw magenta perimeter for accepted voids
                cv2.drawContours(magenta_canvas, [cnt], -1, (255, 0, 255), 5)

                # Save rectangular crop
                y0 = max(0, y - self.pad)
                x0 = max(0, x - self.pad)
                y1 = min(H, y + h + self.pad)
                x1 = min(W, x + w + self.pad)
                rect_crop = img_bgr[y0:y1, x0:x1]
                saved_idx += 1
                rect_path = Path(self.output_dir) / f"{base_name}_page{page_idx:03d}_table{saved_idx:02d}_rect.png"
                cv2.imwrite(str(rect_path), rect_crop)
                saved_paths.append(str(rect_path))

                if self.save_masked_shape_crop:
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    exact = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)[y0:y1, x0:x1]
                    mask_path = Path(self.output_dir) / f"{base_name}_page{page_idx:03d}_table{saved_idx:02d}_mask.png"
                    cv2.imwrite(str(mask_path), exact)
                    saved_paths.append(str(mask_path))

                # ---- DEBUG LOG ENTRY (with DPI) ----
                if self.debug:
                    eff = page_meta.get("effective_dpi", self.dpi)
                    if isinstance(eff, (list, tuple)) and len(eff) >= 2:
                        dpi_x, dpi_y = eff[0], eff[1]
                    else:
                        dpi_x = dpi_y = eff
                    self._debug_crops_log.append({
                        "page": page_idx,
                        "table_index": saved_idx,
                        "crop_path": str(rect_path),
                        "crop_size": {"w": int(rect_crop.shape[1]), "h": int(rect_crop.shape[0])},
                        "bbox_on_page": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                        "bbox_frac": {
                            "w_fr": round(w_fr, 6),
                            "h_fr": round(h_fr, 6),
                            "area_fr": round((w * h) / float(H * W), 6)
                        },
                        "rasterization": {
                            "mode": page_meta.get("mode"),
                            "requested_dpi": int(page_meta.get("requested_dpi", self.dpi)),
                            "effective_dpi_x": int(dpi_x) if dpi_x else None,
                            "effective_dpi_y": int(dpi_y) if dpi_y else None,
                            "size_cap_px": page_meta.get("size_cap_px"),
                            "fallback_dpi": page_meta.get("fallback_dpi"),
                        }
                    })

        # SAVE magenta perimeter overlay for this page (always)
        magenta_path = self.magenta_dir / f"{base_name}_page{page_idx:03d}_void_perimeters.png"
        cv2.imwrite(str(magenta_path), magenta_canvas)

        # Optional debug artifacts
        if self.debug:
            prefix = Path(self.output_dir) / f"{base_name}_page{page_idx:03d}"
            cv2.imwrite(str(prefix.with_suffix("")) + "_whitespace_mask.png", whitespace)
            cv2.imwrite(str(prefix.with_suffix("")) + "_green_overlay.png", overlay_green)

        return saved_paths

    # ------------------ one-box post-processing ------------------
    def _enforce_one_box_on_paths(self, image_paths: List[str]) -> List[str]:
        final_paths: List[str] = []
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

                if len(boxes) <= 1:
                    final_paths.append(p)
                    if self.onebox_debug:
                        dbg = img.copy()
                        for (x1, y1, x2, y2) in boxes:
                            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.imwrite(str(out_dir / f"{base}__debug_single_box.png"), dbg)
                    continue

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

    # ---------- One-box detector ----------
    def _find_table_boxes(self, img_bgr,
                          min_rel_area=0.02,
                          max_rel_area=0.75,
                          aspect_range=(0.4, 3.0),
                          min_side_px=80):
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
        if len(boxes) <= 1:
            return boxes
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        keep = []
        for box in boxes:
            if all(PanelBoardSearch._iou(box, k) <= iou_thr for k in keep):
                keep.append(box)
        return keep

    @staticmethod
    def _iou(a, b):
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
