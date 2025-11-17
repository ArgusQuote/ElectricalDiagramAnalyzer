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
    Always saves per-page magenta perimeter overlays.
    NEW FLOW:
      • First write vector-only PDFs for each detected panel to output_dir/copped_tables_pdf
      • Then rasterize those PDFs to PNGs in output_dir (no extra folder)
    Public API:
        pngs = PanelBoardSearch(...).readPdf("/path/to.pdf")
    """

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
        # width/height % gates
        void_w_fr_range: tuple | None = (0.10, 0.60),
        void_h_fr_range: tuple | None = (0.10, 0.60),
        # Debug + conversion
        pad: int = 6,
        debug: bool = False,
        verbose: bool = True,
        save_masked_shape_crop: bool = False,  # (kept; not used in new flow)
        # PDF conversion safety
        max_megapixels: int = 170_000_000,
        dpi_fallbacks: tuple = (350, 320, 300, 260, 220, 200),
        # One-box post-processing (still used when creating PNGs directly from page, but here we rasterize PDFs)
        enforce_one_box: bool = False,
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

        # per-page overlay dir (unchanged)
        self.magenta_dir = Path(self.output_dir) / "magenta_overlays"
        self.magenta_dir.mkdir(parents=True, exist_ok=True)

        # per-panel vector PDFs dir (name per your request)
        self.cropped_pdf_dir = Path(self.output_dir) / "copped_tables_pdf"
        self.cropped_pdf_dir.mkdir(parents=True, exist_ok=True)

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

        self.enforce_one_box = enforce_one_box
        self.replace_multibox = replace_multibox
        self.onebox_debug = onebox_debug
        self.onebox_pad = onebox_pad
        self.onebox_min_rel_area = onebox_min_rel_area
        self.onebox_max_rel_area = onebox_max_rel_area
        self.onebox_aspect_range = onebox_aspect_range
        self.onebox_min_side_px = onebox_min_side_px

        # Will collect normalized panel boxes per page during page processing
        # {page_idx (1-based): [{"bbox_frac":[x0f,y0f,x1f,y1f]}]}
        self.panel_boxes_by_page: dict[int, list[dict]] = {}

    # ----------------- Public API -----------------
    def readPdf(self, pdf_path: str) -> list[str]:
        """
        Detect panels and overlays; then:
          (1) write per-panel vector PDFs to output_dir/copped_tables_pdf/
          (2) rasterize those PDFs to PNGs in output_dir/
        Returns: list[str] of created PNG paths.
        """
        pdf_path = os.path.expanduser(pdf_path)
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # reset per-run collection
        self.panel_boxes_by_page = {}

        if self.verbose:
            print(f"[INFO] Converting PDF → images @ {self.dpi} DPI")

        pages = self._convert_with_size_cap(pdf_path, self.dpi)
        base = os.path.splitext(os.path.basename(pdf_path))[0]

        if self.verbose:
            print(f"[INFO] Pages: {len(pages)}")

        # 1) Process each page to discover panel regions + write magenta overlays
        for page_idx, pil_page in enumerate(pages, 1):
            try:
                img = self._pil_to_cv(pil_page)
                self._process_page(img, base, page_idx)  # no longer writes rect PNGs
                if self.verbose:
                    n = len(self.panel_boxes_by_page.get(page_idx, []))
                    print(f"[INFO] Page {page_idx}: found {n} panel region(s)")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Page {page_idx} failed: {e}")

        # 2) Create vector PDFs for each detected panel
        pdf_crops = self._write_panel_pdfs(source_pdf=pdf_path, base_name=base)

        # 3) Rasterize those PDFs to PNGs in output_dir
        pngs = self._pdf_crops_to_pngs(pdf_crops)

        if self.verbose:
            print(f"[DONE] Wrote {len(pdf_crops)} panel PDF(s) → {self.cropped_pdf_dir}")
            print(f"[DONE] Wrote {len(pngs)} PNG(s)            → {self.output_dir}")

        return pngs

    # --------------- create per-panel vector PDFs ---------------
    def _write_panel_pdfs(self, source_pdf: str, base_name: str) -> list[Path]:
        """Create one vector-only PDF per panel into self.cropped_pdf_dir."""
        out_paths: list[Path] = []
        if not self.panel_boxes_by_page:
            if self.verbose:
                print("[INFO] No panel boxes recorded; skipping vector PDF crops.")
            return out_paths

        doc = fitz.open(source_pdf)

        for page_num in sorted(self.panel_boxes_by_page.keys()):
            page_idx0 = page_num - 1
            if page_idx0 < 0 or page_idx0 >= len(doc):
                if self.verbose:
                    print(f"[WARN] Page {page_num} out of range in source PDF")
                continue

            page = doc[page_idx0]
            rect_page = page.rect

            for i, entry in enumerate(self.panel_boxes_by_page[page_num], start=1):
                x0f, y0f, x1f, y1f = entry["bbox_frac"]

                # Map normalized fractions -> PDF coordinates (points)
                x0 = rect_page.x0 + rect_page.width * x0f
                y0 = rect_page.y0 + rect_page.height * y0f
                x1 = rect_page.x0 + rect_page.width * x1f
                y1 = rect_page.y0 + rect_page.height * y1f
                clip_rect = fitz.Rect(x0, y0, x1, y1)

                out_doc = fitz.open()
                new_page = out_doc.new_page(width=clip_rect.width, height=clip_rect.height)
                new_page.show_pdf_page(new_page.rect, doc, page_idx0, clip=clip_rect)

                out_path = self.cropped_pdf_dir / f"{base_name}_page{page_num:03d}_panel{i:02d}.pdf"
                out_doc.save(str(out_path))
                out_doc.close()
                out_paths.append(out_path)

        doc.close()
        return out_paths

    # --------------- rasterize panel PDFs to PNGs ---------------
    def _pdf_crops_to_pngs(self, pdf_paths: list[Path]) -> list[str]:
        """Convert each per-panel PDF to a single PNG placed in output_dir."""
        png_paths: list[str] = []
        for p in pdf_paths:
            try:
                # each crop PDF is a single page
                images = convert_from_path(str(p), dpi=self.dpi)
                if not images:
                    continue
                img = images[0]
                out_png = Path(self.output_dir) / (p.stem + ".png")
                img.save(str(out_png))
                png_paths.append(str(out_png))
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] PNG conversion failed for {p.name}: {e}")
        return png_paths

    # ----------------- Internals -----------------
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

    # ------------- FULL per-page processing -------------
    def _process_page(self, img_bgr: np.ndarray, base_name: str, page_idx: int) -> None:
        """Detect panel regions and write magenta overlay. No PNG crops here."""
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

        if self.debug:
            overlay_green = self._draw_green_overlay(img_bgr, labels, stats, keep_ids, alpha=0.45)

        # Always-on magenta perimeter overlay
        magenta_canvas = img_bgr.copy()

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

                # record normalized bbox for this panel for later PDF vector export
                x0 = max(0, x - self.pad); y0 = max(0, y - self.pad)
                x1 = min(W, x + w + self.pad); y1 = min(H, y + h + self.pad)
                self.panel_boxes_by_page[page_idx].append({
                    "bbox_frac": [
                        x0 / float(W),
                        y0 / float(H),
                        x1 / float(W),
                        y1 / float(H),
                    ]
                })

        # SAVE magenta overlay for this page (always)
        magenta_path = self.magenta_dir / f"{base_name}_page{page_idx:03d}_void_perimeters.png"
        cv2.imwrite(str(magenta_path), magenta_canvas)

        if self.debug:
            prefix = Path(self.output_dir) / f"{base_name}_page{page_idx:03d}"
            cv2.imwrite(str(prefix.with_suffix("")) + "_whitespace_mask.png", whitespace)
            cv2.imwrite(str(prefix.with_suffix("")) + "_green_overlay.png", overlay_green)

    # ---------- Legacy one-box helpers kept (unused by new flow) ----------
    def _enforce_one_box_on_paths(self, image_paths: list[str]) -> list[str]:
        return image_paths  # not used in new PDF→PNG flow; kept for compatibility

    def _find_table_boxes(self, img_bgr,
                          min_rel_area=0.02,
                          max_rel_area=0.75,
                          aspect_range=(0.4, 3.0),
                          min_side_px=80):
        return []

    @staticmethod
    def _nms_keep_larger(boxes, iou_thr=0.5):
        return boxes

    @staticmethod
    def _iou(a, b):
        return 0.0
