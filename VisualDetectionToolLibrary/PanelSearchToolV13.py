#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PanelBoardSearch
----------------
Find panel-board tables as 'voids' (holes) inside large whitespace regions
and save crops. Overlays are saved only if debug=True — EXCEPT the magenta
perimeter overlay, which is always saved to output_dir/magenta_overlays.

NEW (normalized output):
  - Always outputs normalized crops only (width ~2000px by default)
  - Writes full-resolution crops only if debug=True
  - Writes per-crop JSON with page, bbox, scale, and DPI info
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
        pad: int = 6,
        debug: bool = False,
        verbose: bool = True,
        save_masked_shape_crop: bool = False,
        max_megapixels: int = 170_000_000,
        dpi_fallbacks: tuple = (350, 320, 300, 260, 220, 200),
        enforce_one_box: bool = True,
        norm_fixed_width_px: int = 1489,
        norm_fixed_height_px: int = 1184,
        norm_minify_only: bool = True,
    ):
        self.output_dir = os.path.expanduser(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.magenta_dir = Path(self.output_dir) / "magenta_overlays"
        self.magenta_dir.mkdir(parents=True, exist_ok=True)

        self.search_debug_dir = Path(self.output_dir) / "search_debug"
        self.debug = bool(debug)
        if self.debug:
            self.search_debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_crops_log: List[Dict[str, Any]] = []

        self.verbose = bool(verbose)
        self.dpi = int(dpi)

        # thresholds
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

        # pdf2image fallback handling
        self.max_megapixels = int(max_megapixels)
        self.dpi_fallbacks = tuple(dpi_fallbacks)

        # --- Normalization options ---
        self.normalize_crops = True
        self.norm_fixed_size = (norm_fixed_width_px, norm_fixed_height_px)
        self.norm_minify_only = bool(norm_minify_only)

        # one-box flags
        self.enforce_one_box = bool(enforce_one_box)

    # ----------------- Public API -----------------
    def readPdf(self, pdf_path: str) -> List[str]:
        pdf_path = os.path.expanduser(pdf_path)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if self.verbose:
            print(f"[INFO] Converting PDF → images @ {self.dpi} DPI")

        page_tuples = self._convert_with_size_cap(pdf_path, self.dpi)
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        all_crops: List[str] = []

        if self.verbose:
            print(f"[INFO] Pages: {len(page_tuples)}")

        for page_idx, (pil_page, page_meta) in enumerate(page_tuples, 1):
            try:
                img = self._pil_to_cv(pil_page)
                page_crops = self._process_page(img, base, page_idx, page_meta)
                all_crops.extend(page_crops)
                if self.verbose:
                    print(f"[INFO] Page {page_idx}: saved {len(page_crops)} crop(s)")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Page {page_idx} failed: {e}")

        if self.verbose:
            print(f"[DONE] Total crops: {len(all_crops)} → {self.output_dir}")

        if self.debug:
            dbg_path = self.search_debug_dir / f"{base}_crops_debug.json"
            with open(dbg_path, "w", encoding="utf-8") as f:
                json.dump(self._debug_crops_log, f, indent=2)
            if self.verbose:
                print(f"[DEBUG] Wrote crop debug JSON → {dbg_path}")

        return all_crops

    # ----------------- Helpers -----------------
    @staticmethod
    def _pil_to_cv(img_pil) -> np.ndarray:
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _convert_with_size_cap(self, pdf_path: str, dpi: int):
        def _wrap(pages, mode, requested_dpi, size_cap_px=None, fallback_dpi=None):
            out = []
            for im in pages:
                eff = im.info.get("dpi", requested_dpi)
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
            return _wrap(pages, "direct", dpi)
        except Exception as e1:
            if self.verbose:
                print(f"[convert] direct {dpi} failed: {e1}")

        longest = int((self.max_megapixels * 1.3) ** 0.5)
        try:
            pages = convert_from_path(pdf_path, dpi=dpi, size=longest)
            return _wrap(pages, "size_cap", dpi, size_cap_px=longest)
        except Exception as e2:
            if self.verbose:
                print(f"[convert] size_cap {dpi} failed: {e2}")

        for step in self.dpi_fallbacks:
            try:
                pages = convert_from_path(pdf_path, dpi=step)
                return _wrap(pages, "fallback", dpi, fallback_dpi=step)
            except Exception as e3:
                if self.verbose:
                    print(f"[convert] fallback {step} failed: {e3}")

        Image.MAX_IMAGE_PIXELS = None
        pages = convert_from_path(pdf_path, dpi=dpi)
        return _wrap(pages, "direct", dpi)

    @staticmethod
    def _binarize_ink(gray: np.ndarray, close_px=3, dilate_px=3):
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
    def _shave_margin(mask: np.ndarray, px: int):
        m = mask.copy()
        h, w = m.shape
        m[:px, :] = 0; m[-px:, :] = 0; m[:, :px] = 0; m[:, -px:] = 0
        return m

    @staticmethod
    def _components(mask: np.ndarray):
        lab = (mask > 0).astype(np.uint8)
        return cv2.connectedComponentsWithStats(lab, connectivity=8)

    @staticmethod
    def _selected_ws_mask(labels: np.ndarray, keep_ids: List[int]):
        m = np.zeros_like(labels, dtype=np.uint8)
        for cid in keep_ids:
            m[labels == cid] = 255
        return m

    def _draw_green_overlay(self, bgr, labels, stats, keep_ids, alpha=0.45):
        overlay = bgr.copy()
        tint = np.array([0, 255, 0], dtype=np.float32)
        for comp_id in keep_ids:
            region = (labels == comp_id)
            base = overlay[region].astype(np.float32)
            overlay[region] = np.clip((1.0 - alpha) * base + alpha * tint, 0, 255).astype(np.uint8)
        return overlay

    # ----------------- Page processor -----------------
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

                # Save rectangular crop (original, unnormalized in-memory)
                y0 = max(0, y - self.pad)
                x0 = max(0, x - self.pad)
                y1 = min(H, y + h + self.pad)
                x1 = min(W, x + w + self.pad)
                rect_crop = img_bgr[y0:y1, x0:x1]

                saved_idx += 1

                # --- Normalize crop: scale-to-fit then pad to exact size ---
                # Expect self.norm_fixed_size to be provided via __init__ (W,H); fallback to (1489,1184) if missing
                target_w, target_h = getattr(self, "norm_fixed_size", (1489, 1184))
                h_c, w_c = rect_crop.shape[:2]
                # scale to fit within target while preserving aspect
                scale = min(target_w / float(w_c), target_h / float(h_c))
                new_w = max(1, int(round(w_c * scale)))
                new_h = max(1, int(round(h_c * scale)))
                resized = cv2.resize(rect_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # letterbox pad to exact (target_w, target_h) with white background
                rect_crop_norm = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
                off_x = (target_w - new_w) // 2
                off_y = (target_h - new_h) // 2
                rect_crop_norm[off_y:off_y+new_h, off_x:off_x+new_w] = resized

                # Filenames: normalized is the main output; unnormalized only if debug=True
                rect_base = f"{base_name}_page{page_idx:03d}_table{saved_idx:02d}_rect"
                rect_path = Path(self.output_dir) / f"{rect_base}.png"
                cv2.imwrite(str(rect_path), rect_crop_norm)
                saved_paths.append(str(rect_path))

                if self.debug:
                    rect_path_unnorm = Path(self.output_dir) / f"{rect_base}_unnormalized.png"
                    cv2.imwrite(str(rect_path_unnorm), rect_crop)

                if self.save_masked_shape_crop:
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    exact = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)[y0:y1, x0:x1]
                    mask_path = Path(self.output_dir) / f"{base_name}_page{page_idx:03d}_table{saved_idx:02d}_mask.png"
                    cv2.imwrite(str(mask_path), exact)
                    # (mask crops are auxiliary; we don't add them to saved_paths)

                # ---- OPTIONAL: if you keep a debug JSON log elsewhere, you can append DPI etc. here ----
                if self.debug and hasattr(self, "_debug_crops_log"):
                    eff = page_meta.get("effective_dpi", self.dpi)
                    if isinstance(eff, (list, tuple)) and len(eff) >= 2:
                        dpi_x, dpi_y = eff[0], eff[1]
                    else:
                        dpi_x = dpi_y = eff
                    self._debug_crops_log.append({
                        "page": page_idx,
                        "table_index": saved_idx,
                        "crop_path": str(rect_path),
                        "crop_size": {"w": int(rect_crop_norm.shape[1]), "h": int(rect_crop_norm.shape[0])},
                        "bbox_on_page": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                        "bbox_frac": {
                            "w_fr": round(w / float(W), 6),
                            "h_fr": round(h / float(H), 6),
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
