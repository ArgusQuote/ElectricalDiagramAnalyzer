#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import fitz  # PyMuPDF


class PanelBoardSearch:
    """
    Find panel-board tables as 'voids' (holes) inside large whitespace regions.
    Save:
      • Magenta overlays per page (as before, from raster page)
      • Panel crops as **high-DPI clipped renders from the ORIGINAL PDF** (no intermediate PDFs)

    Public API:
        crops = PanelBoardSearch(...).readPdf("/path/to.pdf")  # returns list of PNG paths
    """

    def __init__(
        self,
        output_dir: str,
        dpi: int = 400,                         # detection DPI (for page raster + magenta overlays)
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
        # width/height % gates to avoid thin strips
        void_w_fr_range: tuple | None = (0.10, 0.60),
        void_h_fr_range: tuple | None = (0.10, 0.60),
        # Cropping + debug
        pad: int = 6,
        debug: bool = False,
        verbose: bool = True,
        save_masked_shape_crop: bool = False,
        # PDF conversion safety (for detection rasterization only)
        max_megapixels: int = 170_000_000,
        dpi_fallbacks: tuple = (350, 320, 300, 260, 220, 200),
        # One-box enforcement (kept for compatibility, not used in this hi-DPI pipeline)
        enforce_one_box: bool = False,
        replace_multibox: bool = True,
        onebox_debug: bool = False,
        onebox_pad: int = 6,
        onebox_min_rel_area: float = 0.02,
        onebox_max_rel_area: float = 0.75,
        onebox_aspect_range: tuple = (0.4, 3.0),
        onebox_min_side_px: int = 80,
        # ---------- NEW: Hi-fidelity render knobs ----------
        render_dpi: int = 1200,                # DPI used to render panel crops from the source PDF
        aa_level: int = 8,                     # 0..8 (PyMuPDF anti-aliasing)
        render_colorspace: str = "gray",       # "gray" or "rgb"
        downsample_max_w: int | None = None,   # optional: area-downsample to this width (px) with BOX
    ):
        self.output_dir = os.path.expanduser(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # Always-on magenta overlay directory
        self.magenta_dir = Path(self.output_dir) / "magenta_overlays"
        self.magenta_dir.mkdir(parents=True, exist_ok=True)

        # Detection params (work on rasterized page)
        self.dpi = dpi
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

        self.pad = pad
        self.debug = debug
        self.verbose = verbose
        self.save_masked_shape_crop = save_masked_shape_crop

        self.max_megapixels = max_megapixels
        self.dpi_fallbacks = dpi_fallbacks

        # (kept for compatibility)
        self.enforce_one_box = enforce_one_box
        self.replace_multibox = replace_multibox
        self.onebox_debug = onebox_debug
        self.onebox_pad = onebox_pad
        self.onebox_min_rel_area = onebox_min_rel_area
        self.onebox_max_rel_area = onebox_max_rel_area
        self.onebox_aspect_range = onebox_aspect_range
        self.onebox_min_side_px = onebox_min_side_px

        # Hi-fidelity render knobs
        self.render_dpi = render_dpi
        self.aa_level = max(0, min(int(aa_level), 8))
        self.render_colorspace = render_colorspace.lower().strip()
        self.downsample_max_w = downsample_max_w

        # where we store normalized panel boxes per page (for hi-DPI render)
        # {page_idx_1based: [ {"bbox_frac":[x0f,y0f,x1f,y1f]}, ... ]}
        self.panel_boxes_by_page: dict[int, list[dict]] = {}

    # ----------------- Public API -----------------
    def readPdf(self, pdf_path: str) -> list[str]:
        """
        1) Rasterize pages (at self.dpi) only for detection + debug overlays
        2) Detect panel voids and store normalized boxes
        3) Render each panel crop directly from the SOURCE PDF at high DPI

        Returns list of final high-DPI PNG paths.
        """
        pdf_path = os.path.expanduser(pdf_path)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if self.verbose:
            print(f"[INFO] Converting PDF → images (for detection) @ {self.dpi} DPI")

        # reset boxes for this run
        self.panel_boxes_by_page = {}

        pages = self._convert_with_size_cap(pdf_path, self.dpi)
        base = os.path.splitext(os.path.basename(pdf_path))[0]

        if self.verbose:
            print(f"[INFO] Pages: {len(pages)}")

        # Pass 1: DETECTION (and magenta overlays)
        for page_idx, pil_page in enumerate(pages, 1):
            try:
                img = self._pil_to_cv(pil_page)
                _ = self._process_page(img, base, page_idx)  # only for overlays + box list
                if self.verbose:
                    n = len(self.panel_boxes_by_page.get(page_idx, []))
                    print(f"[INFO] Page {page_idx}: found {n} panel candidate(s)")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Page {page_idx} detection failed: {e}")

        # Pass 2: HI-DPI RENDER straight from source PDF using recorded boxes
        rendered_paths = self._render_panels_from_pdf(pdf_path, base)

        if self.verbose:
            print(f"[DONE] Rendered {len(rendered_paths)} panel crop(s) → {self.output_dir}")

        return rendered_paths

    # ----------------- Hi-DPI panel rendering from source PDF -----------------
    def _render_panels_from_pdf(self, pdf_path: str, base_name: str) -> list[str]:
        out_paths: list[str] = []
        if not self.panel_boxes_by_page:
            if self.verbose:
                print("[INFO] No panel boxes recorded; skipping hi-DPI render.")
            return out_paths

        # Set global AA for the session
        try:
            fitz.TOOLS.set_aa_level(self.aa_level)
        except Exception:
            pass

        # Colorspace
        cs = fitz.csGRAY if self.render_colorspace == "gray" else fitz.csRGB

        doc = fitz.open(pdf_path)
        if self.verbose:
            print(f"[INFO] Rendering panels @ {self.render_dpi} DPI, AA={self.aa_level}, colorspace={self.render_colorspace}")

        for page_num in sorted(self.panel_boxes_by_page.keys()):
            page = doc[page_num - 1]
            rect_page = page.rect

            for idx, entry in enumerate(self.panel_boxes_by_page[page_num], start=1):
                x0f, y0f, x1f, y1f = entry["bbox_frac"]
                # fractions → page coords
                x0 = rect_page.x0 + rect_page.width * x0f
                y0 = rect_page.y0 + rect_page.height * y0f
                x1 = rect_page.x0 + rect_page.width * x1f
                y1 = rect_page.y0 + rect_page.height * y1f
                clip_rect = fitz.Rect(x0, y0, x1, y1)

                # render this region directly from vector
                pix = page.get_pixmap(
                    dpi=self.render_dpi,
                    clip=clip_rect,
                    alpha=False,
                    colorspace=cs,
                )
                out_png = Path(self.output_dir) / f"{base_name}_page{page_num:03d}_panel{idx:02d}.png"
                pix.save(str(out_png))

                # Optional area-downsample for very large renders (preserves hairlines)
                if self.downsample_max_w is not None:
                    try:
                        from PIL import Image as _PILImage
                        im = _PILImage.open(out_png)
                        if im.width > self.downsample_max_w:
                            target_h = int(round(im.height * (self.downsample_max_w / float(im.width))))
                            im = im.resize((self.downsample_max_w, target_h), resample=_PILImage.BOX)
                            im.save(out_png)
                    except Exception as _:
                        pass

                out_paths.append(str(out_png))

        doc.close()
        return out_paths

    # ----------------- Internals (detection stage) -----------------
    @staticmethod
    def _pil_to_cv(img_pil: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _convert_with_size_cap(self, pdf_path: str, dpi: int):
        try:
            return convert_from_path(pdf_path, dpi=dpi)
        except Exception as e1:
            if self.verbose:
                print(f"[convert] initial {dpi} DPI failed: {type(e1).__name__}: {e1}")
        longest = int((self.max_megapixels * 1.3) ** 0.5)
        try:
            if self.verbose:
                print(f"[convert] retry {dpi} DPI with size cap = {longest}px")
            return convert_from_path(pdf_path, dpi=dpi, size=longest)
        except Exception as e2:
            if self.verbose:
                print(f"[convert] size-capped {dpi} DPI failed: {type(e2).__name__}: {e2}")
        for step in self.dpi_fallbacks:
            try:
                if self.verbose:
                    print(f"[convert] trying fallback DPI {step}")
                return convert_from_path(pdf_path, dpi=step)
            except Exception as e3:
                if self.verbose:
                    print(f"[convert] {step} DPI failed: {type(e3).__name__}: {e3}")
        if self.verbose:
            print("[convert] disabling PIL MAX_IMAGE_PIXELS and retrying original DPI")
        Image.MAX_IMAGE_PIXELS = None
        return convert_from_path(pdf_path, dpi=dpi)

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
    def _selected_ws_mask(labels: np.ndarray, keep_ids: list[int]) -> np.ndarray:
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

    # ------------- per-page detection (records boxes; no low-DPI crops) -------------
    def _process_page(self, img_bgr: np.ndarray, base_name: str, page_idx: int) -> list[str]:
        H, W = img_bgr.shape[:2]
        page_area = H * W
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1) INK / WHITESPACE
        ink = self._binarize_ink(gray, close_px=3, dilate_px=3)     # 255 ink
        whitespace = cv2.bitwise_not(ink)                           # 255 bg
        whitespace = self._shave_margin(whitespace, self.margin_shave_px)

        num, labels, stats, _ = self._components(whitespace)
        keep_ids = []
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

        # Always-on magenta perimeter overlay
        magenta_canvas = img_bgr.copy()

        saved_paths = []   # kept for API compatibility (we now return paths after hi-DPI render)
        saved_idx = 0

        # ensure page key exists
        if page_idx not in self.panel_boxes_by_page:
            self.panel_boxes_by_page[page_idx] = []

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

                # width/height fraction gates (avoid thin strips)
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

                # record normalized bbox (pad is applied here to match prior behavior)
                y0 = max(0, y - self.pad)
                x0 = max(0, x - self.pad)
                y1 = min(H, y + h + self.pad)
                x1 = min(W, x + w + self.pad)

                self.panel_boxes_by_page[page_idx].append({
                    "bbox_frac": [
                        x0 / float(W),
                        y0 / float(H),
                        x1 / float(W),
                        y1 / float(H),
                    ]
                })

                saved_idx += 1

        # SAVE magenta overlay for this page (always)
        magenta_path = self.magenta_dir / f"{base_name}_page{page_idx:03d}_void_perimeters.png"
        cv2.imwrite(str(magenta_path), magenta_canvas)

        # Optional debug artifacts
        if self.debug:
            prefix = Path(self.output_dir) / f"{base_name}_page{page_idx:03d}"
            cv2.imwrite(str(prefix.with_suffix("")) + "_whitespace_mask.png", whitespace)
            cv2.imwrite(str(prefix.with_suffix("")) + "_green_overlay.png", overlay_green)

        return saved_paths  # empty; final PNGs are emitted in _render_panels_from_pdf()

    # ---------- One-box detector (unused in hi-DPI path, kept for compat) ----------
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
