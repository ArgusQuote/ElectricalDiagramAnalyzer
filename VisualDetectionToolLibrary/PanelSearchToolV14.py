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

        det_mat   = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        rend_mat  = fitz.Matrix(self.render_dpi / 72.0, self.render_dpi / 72.0)
        cs        = fitz.csGRAY if self.render_colorspace == "gray" else fitz.csRGB

        # --- IMPORTANT: turn AA OFF for detection so lines are crisp ---
        try:
            fitz.TOOLS.set_aa_level(0)
        except Exception:
            pass

        for pidx in range(len(doc)):
            page = doc[pidx]

            # 1) detection bitmap (AA=0 so outlines are sharp)
            pix = page.get_pixmap(matrix=det_mat, alpha=False)
            det_bgr = self._pix_to_bgr(pix)
            H, W = det_bgr.shape[:2]
            page_area = H * W

            gray = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2GRAY)
            # slightly gentler morphology than before; preserves thin frames
            ink = self._binarize_ink(gray, close_px=0, dilate_px=1)
            whitespace = cv2.bitwise_not(ink)
            whitespace = self._shave_margin(whitespace, self.margin_shave_px)

            num, labels, stats, _ = self._components(whitespace)
            keep_ids = [cid for cid in range(1, num)
                        if stats[cid][4] / page_area >= max(0.006, self.min_whitespace_area_fr)]  # loosened

            sel_ws = self._selected_ws_mask(labels, keep_ids)
            contours, hier = cv2.findContours(sel_ws, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            magenta = det_bgr.copy()
            saved_idx = 0
            candidates: list[fitz.Rect] = []

            def _box_passes(x, y, w, h):
                area = w * h
                if area / page_area < max(0.0025, self.min_void_area_fr):  # loosened
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

            # ---- primary: holes inside whitespace comps ----
            if hier is not None and len(hier) > 0:
                hier = hier[0]
                for ci, cnt in enumerate(contours):
                    if hier[ci][3] == -1:
                        continue  # holes only
                    x, y, w, h = cv2.boundingRect(cnt)
                    if not _box_passes(x, y, w, h):
                        continue
                    cv2.drawContours(magenta, [cnt], -1, (255, 0, 255), 5)

                    # pad and convert to PDF clip
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

            # ---- fallback: if nothing passed, propose by grid lines (no 'holes' assumption) ----
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

            # ---- export vector PDF + hi-DPI PNG for every candidate ----
            for clip in candidates:
                saved_idx += 1

                # vector crop PDF
                out_pdf = fitz.open()
                new_page = out_pdf.new_page(width=clip.width, height=clip.height)
                new_page.show_pdf_page(new_page.rect, doc, pidx, clip=clip)
                pdf_path = self.vec_dir / f"{base}_page{pidx+1:03d}_panel{saved_idx:02d}.pdf"
                out_pdf.save(str(pdf_path))
                out_pdf.close()

                # high-DPI PNG
                pixc = page.get_pixmap(matrix=rend_mat, clip=clip, alpha=False, colorspace=cs)
                png = self._pix_to_bgr(pixc)
                if self.downsample_max_w and png.shape[1] > self.downsample_max_w:
                    scale = self.downsample_max_w / float(png.shape[1])
                    new_wh = (self.downsample_max_w, max(1, int(round(png.shape[0] * scale))))
                    png = cv2.resize(png, new_wh, interpolation=cv2.INTER_AREA)
                png_path = Path(self.output_dir) / f"{base}_page{pidx+1:03d}_panel{saved_idx:02d}.png"
                cv2.imwrite(str(png_path), png)
                all_pngs.append(str(png_path))

            # overlay per page
            ov_path = self.magenta_dir / f"{base}_page{pidx+1:03d}_void_perimeters.png"
            cv2.imwrite(str(ov_path), magenta)

            if self.verbose:
                print(f"[INFO] Page {pidx+1}: saved {saved_idx} crop(s)")

        doc.close()

        # restore AA for anything else that uses fitz later (optional)
        try:
            fitz.TOOLS.set_aa_level(self.aa_level)
        except Exception:
            pass

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

    # (Optional) one-box logic can be copied back in if you still want it.
