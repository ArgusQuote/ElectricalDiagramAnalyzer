#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import cv2
import fitz  # PyMuPDF


class PanelBoardSearch:
    """
    Detect panel-board 'voids' on a detection bitmap rendered by PyMuPDF
    (so coordinates match the PDF page), then:
      1) Write a vector-only PDF for each table via clip.
      2) Render a high-DPI PNG from the same clip.
    Always writes a magenta overlay per source page for QA.

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

        # knobs
        self.dpi = dpi
        self.render_dpi = render_dpi
        self.aa_level = int(max(0, min(8, aa_level)))
        try:
            fitz.TOOLS.set_aa_level(self.aa_level)
        except Exception:
            pass

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
        pdf_path = os.path.expanduser(pdf_path)
        if not Path(pdf_path).is_file():
            raise FileNotFoundError(pdf_path)

        doc = fitz.open(pdf_path)
        base = Path(pdf_path).stem
        all_pngs: list[str] = []

        if self.verbose:
            print(f"[INFO] Detecting with fitz @ {self.dpi} DPI")
            print(f"[INFO] Pages: {len(doc)}")

        det_mat  = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        rend_mat = fitz.Matrix(self.render_dpi / 72.0, self.render_dpi / 72.0)
        cs       = fitz.csGRAY if self.render_colorspace == "gray" else fitz.csRGB

        # Turn AA OFF for detection so lines are crisp (prevents “hole” fill-in)
        try:
            fitz.TOOLS.set_aa_level(0)
        except Exception:
            pass

        for pidx in range(len(doc)):
            page = doc[pidx]

            # 1) Detection bitmap (AA=0)
            pix = page.get_pixmap(matrix=det_mat, alpha=False)
            det_bgr = self._pix_to_bgr(pix)
            H, W = det_bgr.shape[:2]
            page_area = H * W

            gray = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2GRAY)
            # Gentle morphology preserves thin frames; adjust if needed
            ink = self._binarize_ink(gray, close_px=0, dilate_px=1)
            whitespace = cv2.bitwise_not(ink)
            whitespace = self._shave_margin(whitespace, self.margin_shave_px)

            num, labels, stats, _ = self._components(whitespace)
            keep_ids = [
                cid for cid in range(1, num)
                if stats[cid][4] / page_area >= max(0.006, self.min_whitespace_area_fr)
            ]

            sel_ws = self._selected_ws_mask(labels, keep_ids)
            contours, hier = cv2.findContours(sel_ws, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            magenta = det_bgr.copy()
            saved_idx = 0
            candidates: list[fitz.Rect] = []

            def _box_passes(x, y, w, h):
                area = w * h
                if area / page_area < max(0.0025, self.min_void_area_fr):
                    return False
                if w < max(60, self.min_void_w_px) or h < max(60, self.min_void_h_px):
                    return False
                if (w / W) >= self.max_void_w_fr or (h / H) >= self.max_void_h_fr:
                    return False
                if (area / page_area) >= self.max_void_area_fr:
                    return False
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
            if hier is not None and len(hier) > 0:
                hier = hier[0]
                for ci, cnt in enumerate(contours):
                    if hier[ci][3] == -1:
                        continue  # only holes are voids
                    x, y, w, h = cv2.boundingRect(cnt)
                    if not _box_passes(x, y, w, h):
                        continue

                    cv2.drawContours(magenta, [cnt], -1, (255, 0, 255), 5)

                    # pad & map detection bbox → PDF clip
                    y0 = max(0, y - self.pad); x0 = max(0, x - self.pad)
                    y1 = min(H, y + h + self.pad); x1 = min(W, x + w + self.pad)
                    x0f, y0f, x1f, y1f = x0 / W, y0 / H, x1 / W, y1 / H
                    pr = page.rect
                    clip = fitz.Rect(
                        pr.x0 + pr.width  * x0f,
                        pr.y0 + pr.height * y0f,
                        pr.x0 + pr.width  * x1f,
                        pr.y0 + pr.height * y1f,
                    )
                    candidates.append(clip)

            # ---- Fallback: grid proposals if nothing passed ----
            if not candidates:
                for (x0, y0, x1, y1) in self._grid_proposals_fallback(det_bgr):
                    w, h = x1 - x0, y1 - y0
                    if not _box_passes(x0, y0, w, h):
                        continue
                    cv2.rectangle(magenta, (x0, y0), (x1, y1), (255, 0, 255), 5)
                    x0f, y0f, x1f, y1f = x0 / W, y0 / H, x1 / W, y1 / H
                    pr = page.rect
                    clip = fitz.Rect(
                        pr.x0 + pr.width  * x0f,
                        pr.y0 + pr.height * y0f,
                        pr.x0 + pr.width  * x1f,
                        pr.y0 + pr.height * y1f,
                    )
                    candidates.append(clip)

            # ---- Export vector PDF + hi-DPI PNG for each candidate ----
            for clip in candidates:
                saved_idx += 1

                # Vector crop PDF (lossless vectors)
                out_pdf = fitz.open()
                new_page = out_pdf.new_page(width=clip.width, height=clip.height)
                new_page.show_pdf_page(new_page.rect, doc, pidx, clip=clip)
                pdf_path = self.vec_dir / f"{base}_page{pidx+1:03d}_panel{saved_idx:02d}.pdf"
                out_pdf.save(str(pdf_path))
                out_pdf.close()

                # High-DPI PNG from the same clip (lossless PNG write)
                pixc = page.get_pixmap(matrix=rend_mat, clip=clip, alpha=False, colorspace=cs)
                png = self._pix_to_bgr(pixc)
                if self.downsample_max_w and png.shape[1] > self.downsample_max_w:
                    scale = self.downsample_max_w / float(png.shape[1])
                    new_wh = (self.downsample_max_w, max(1, int(round(png.shape[0] * scale))))
                    png = cv2.resize(png, new_wh, interpolation=cv2.INTER_AREA)
                png_path = Path(self.output_dir) / f"{base}_page{pidx+1:03d}_panel{saved_idx:02d}.png"
                cv2.imwrite(str(png_path), png)
                all_pngs.append(str(png_path))

            # Per-page overlay
            ov_path = self.magenta_dir / f"{base}_page{pidx+1:03d}_void_perimeters.png"
            cv2.imwrite(str(ov_path), magenta)

            if self.verbose:
                print(f"[INFO] Page {pidx+1}: saved {saved_idx} crop(s)")

        doc.close()

        # Restore configured AA for any later fitz use
        try:
            fitz.TOOLS.set_aa_level(self.aa_level)
        except Exception:
            pass

        # --- post-pass: enforce one table per PNG (lossless) ---
        if self.enforce_one_box and all_pngs:
            all_pngs = self._enforce_one_box_on_paths(all_pngs)

        # NEW: debug horizontal-line masks for each final crop
        if self.debug and all_pngs:
            self._save_horizontal_line_masks(all_pngs)

        if self.verbose:
            print(f"[DONE] Outputs → {self.output_dir}")
            print(f"      Vector PDFs → {self.vec_dir}")
            print(f"      Overlays    → {self.magenta_dir}")

        return all_pngs

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

    # ---------------- helpers (unchanged semantics) ----------------
    @staticmethod
    def _pix_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
        H, W, C = pix.height, pix.width, pix.n
        buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(H, W, C)
        if C == 1:
            return cv2.cvtColor(buf, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

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

    # ---------------- Horizontal-line debug masks ----------------
    def _save_horizontal_line_masks(self, image_paths: list[str]) -> None:
        """
        For each final crop PNG, create a binarized mask that shows only
        horizontal lines and save it into a debug folder.

        Output: <output_dir>/debug_horizontal_lines/<crop_stem>__horiz_mask.png
        """
        debug_dir = Path(self.output_dir) / "debug_horizontal_lines"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                if self.verbose:
                    print(f"[WARN] Could not read crop for horiz debug: {p}")
                continue

            mask = self._horizontal_line_mask(img)
            out_path = debug_dir / f"{Path(p).stem}__horiz_mask.png"
            cv2.imwrite(str(out_path), mask)

    @staticmethod
    def _horizontal_line_mask(img_bgr: np.ndarray) -> np.ndarray:
        """
        Return a binarized mask (0/255) highlighting horizontal table lines only.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Same style as _find_table_boxes: text/lines become white (255)
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )

        H, W = gray.shape
        # Horizontal structuring element: wide, 1 pixel tall
        hk = max(W // 80, 5)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))

        # Morphological open: keeps horizontal structures, removes non-horizontal noise
        horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kh)

        # Ensure strict binary 0/255 output
        _, horiz_bin = cv2.threshold(horiz, 0, 255, cv2.THRESH_BINARY)
        return horiz_bin