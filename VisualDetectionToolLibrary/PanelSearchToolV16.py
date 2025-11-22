#!/usr/bin/env python3
import os
from pathlib import Path
import json
import numpy as np
import cv2
import fitz  # PyMuPDF


class PanelBoardSearch:
    """
    Detect panel-board 'voids' on a detection bitmap rendered by PyMuPDF
    (so coordinates match the PDF page), then:
      1) Write a vector-only PDF for each table via clip.
      2) Render a high-DPI PNG from the same clip.
      3) (POST) Run the same horizontal-line check you validated in your simple script
         on every produced PNG. Writes per-image overlays + JSON; deletes PNGs that fail.

    Public API:
        crops = PanelBoardSearch(...).readPdf("/path/to.pdf")
    """

    def __init__(
        self,
        output_dir: str,
        # --- detection (bitmap) ---
        dpi: int = 400,
        # --- PDF→PNG render ---
        render_dpi: int = 1200,
        aa_level: int = 8,
        render_colorspace: str = "gray",  # "gray" or "rgb"
        downsample_max_w: int | None = None,

        # whitespace / void heuristics
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

        # ------- line-check params (your working settings) -------
        linecheck_enable: bool = True,
        line_run_len: int = 5,                 # you said this worked
        line_tolerance: int = 8,               # you said this worked
        line_min_width_frac: float = 0.30,
        line_hkernel_frac: float = 0.035,
        line_max_thickness_px: int = 6,
        line_y_merge_tol_px: int = 4,
        line_out_subdir: str = "line_check",
    ):
        self.output_dir = os.path.expanduser(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # subdirs
        self.magenta_dir = Path(self.output_dir) / "magenta_overlays"
        self.magenta_dir.mkdir(parents=True, exist_ok=True)
        self.vec_dir = Path(self.output_dir) / "cropped_tables_pdf"
        self.vec_dir.mkdir(parents=True, exist_ok=True)
        self.line_dir = Path(self.output_dir) / line_out_subdir
        self.line_dir.mkdir(parents=True, exist_ok=True)

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

        # line-check
        self.linecheck_enable = linecheck_enable
        self.line_run_len = line_run_len
        self.line_tolerance = line_tolerance
        self.line_min_width_frac = line_min_width_frac
        self.line_hkernel_frac = line_hkernel_frac
        self.line_max_thickness_px = line_max_thickness_px
        self.line_y_merge_tol_px = line_y_merge_tol_px

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

        # Turn AA OFF for detection so lines are crisp
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

        # --- post: split multi-table crops (lossless) ---
        if self.enforce_one_box and all_pngs:
            all_pngs = self._enforce_one_box_on_paths(all_pngs)

        # --- final: run line checker on all PNGs; delete failures; return kept only ---
        if self.linecheck_enable and all_pngs:
            all_pngs = self._post_linecheck_delete_failures(all_pngs)

        if self.verbose:
            print(f"[DONE] Outputs → {self.output_dir}")
            print(f"      Vector PDFs → {self.vec_dir}")
            print(f"      Overlays    → {self.magenta_dir}")

        return all_pngs

    # ---------------- line-check (simple script port) ----------------
    def _post_linecheck_delete_failures(self, image_paths: list[str]) -> list[str]:
        kept: list[str] = []
        for ipath in image_paths:
            try:
                img = cv2.imread(ipath, cv2.IMREAD_COLOR)
                if img is None:
                    if self.verbose:
                        print(f"[lines] unreadable: {ipath}")
                    continue

                lines = self._detect_horizontal_lines(
                    img_bgr=img,
                    horiz_kernel_frac=self.line_hkernel_frac,
                    min_width_frac=self.line_min_width_frac,
                    max_thickness_px=self.line_max_thickness_px,
                    y_merge_tol_px=self.line_y_merge_tol_px,
                )
                lines = self._add_gaps(lines)

                subdir = self.line_dir / Path(ipath).stem
                subdir.mkdir(parents=True, exist_ok=True)

                overlay = self._overlay_lines(img, lines, thickness=2)
                cv2.imwrite(str(subdir / "overlay_horizontal_lines.png"), overlay)

                with open(subdir / "lines.json", "w", encoding="utf-8") as f:
                    json.dump(lines, f, indent=2)

                next_gaps = [ln.get("next_gap", None) for ln in lines]
                with open(subdir / "line_measurements.json", "w", encoding="utf-8") as f:
                    json.dump({"next_gaps": next_gaps}, f, indent=2)

                ok = self._uniform_run_found(next_gaps, run_len=self.line_run_len, tol=self.line_tolerance)

                summary = {
                    "input_image": str(Path(ipath).resolve()),
                    "overlay_image": str((subdir / "overlay_horizontal_lines.png").resolve()),
                    "lines_json": str((subdir / "lines.json").resolve()),
                    "measurements_json": str((subdir / "line_measurements.json").resolve()),
                    "run_len": self.line_run_len,
                    "tolerance": self.line_tolerance,
                    "uniform_run_found": bool(ok),
                    "num_lines": len(lines),
                }
                with open(subdir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

                if ok:
                    kept.append(ipath)
                    print(f"[lines] {Path(ipath).name}: PASS | lines={len(lines)}")
                else:
                    # delete failing PNG
                    try:
                        os.remove(ipath)
                        print(f"[lines] {Path(ipath).name}: FAIL → deleted")
                    except Exception as de:
                        print(f"[lines] {Path(ipath).name}: FAIL (delete error: {de})")

            except Exception as e:
                print(f"[lines] ERROR {ipath}: {e}")
        return kept

    @staticmethod
    def _detect_horizontal_lines(
        img_bgr: np.ndarray,
        horiz_kernel_frac: float = 0.035,
        min_width_frac: float = 0.20,
        max_thickness_px: int = 6,
        y_merge_tol_px: int = 4,
    ):
        """
        Adaptive + multi-scale horizontal line extraction.
        Returns [{y_center,y_top,y_bottom,x_left,x_right,width,height}, ...]
        """
        H, W = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 10
        )

        fracs = [
            max(0.02, horiz_kernel_frac * 0.7),
            horiz_kernel_frac,
            min(0.08, horiz_kernel_frac * 1.8),
        ]
        masks = []
        for f in fracs:
            klen = max(5, int(round(W * f)))
            kh = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
            m = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kh, iterations=1)
            m = cv2.dilate(m, None, iterations=1)
            m = cv2.erode(m, None, iterations=1)
            masks.append(m)

        horiz = masks[0]
        for m in masks[1:]:
            horiz = cv2.bitwise_or(horiz, m)

        cnts, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_width_px = int(round(W * min_width_frac))
        raws = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < min_width_px or h < 1 or h > max_thickness_px:
                continue
            y_mid = y + h // 2
            row = horiz[y_mid, :]
            xs = np.where(row > 0)[0]
            if xs.size == 0:
                continue
            x_left = int(xs.min())
            x_right = int(xs.max())
            if (x_right - x_left + 1) < min_width_px:
                continue
            raws.append((y, y + h - 1, x_left, x_right))

        if not raws:
            return []

        raws.sort(key=lambda r: (r[0] + r[1]) / 2.0)
        merged, cur = [], [raws[0]]
        for r in raws[1:]:
            y0, y1, _, _ = r
            py0, py1, _, _ = cur[-1]
            if abs(((y0 + y1) / 2.0) - ((py0 + py1) / 2.0)) <= y_merge_tol_px:
                cur.append(r)
            else:
                merged.append(cur); cur = [r]
        merged.append(cur)

        lines = []
        for grp in merged:
            y_t = min(g[0] for g in grp)
            y_b = max(g[1] for g in grp)
            x_l = min(g[2] for g in grp)
            x_r = max(g[3] for g in grp)
            y_c = (y_t + y_b) // 2
            lines.append({
                "y_center": int(y_c),
                "y_top":    int(y_t),
                "y_bottom": int(y_b),
                "x_left":   int(x_l),
                "x_right":  int(x_r),
                "width":    int(x_r - x_l + 1),
                "height":   int(max(1, y_b - y_t + 1)),
            })
        lines.sort(key=lambda d: d["y_center"])
        return lines

    @staticmethod
    def _add_gaps(lines):
        n = len(lines)
        for i in range(n):
            prev_gap = None if i == 0 else int(lines[i]["y_center"] - lines[i-1]["y_center"])
            next_gap = None if i == n - 1 else int(lines[i+1]["y_center"] - lines[i]["y_center"])
            lines[i]["prev_gap"] = prev_gap
            lines[i]["next_gap"] = next_gap
        return lines

    @staticmethod
    def _overlay_lines(img_bgr: np.ndarray, lines, thickness: int = 2) -> np.ndarray:
        out = img_bgr.copy()
        for ln in lines:
            y = (ln["y_top"] + ln["y_bottom"]) // 2
            cv2.line(out, (ln["x_left"], y), (ln["x_right"], y), (0, 0, 255), thickness)
        return out

    @staticmethod
    def _uniform_run_found(next_gaps, run_len=20, tol=8):
        i = 0
        N = len(next_gaps)
        while i < N:
            while i < N and next_gaps[i] is None:
                i += 1
            if i >= N:
                break
            total = 0.0
            count = 0
            lo = hi = None
            length = 0
            j = i
            while j < N:
                g = next_gaps[j]
                if g is None:
                    break
                if count == 0:
                    total = float(g); count = 1; length = 1
                    lo = g - tol; hi = g + tol
                else:
                    mean = total / count
                    if abs(g - mean) <= tol and lo <= g <= hi:
                        total += g; count += 1; length += 1
                        lo = max(lo, int(mean - tol))
                        hi = min(hi, int(mean + tol))
                    else:
                        break
                if length >= run_len:
                    return True
                j += 1
            i = j + 1
        return False

    # ---------------- other helpers (unchanged) ----------------
    def _grid_proposals_fallback(self, img_bgr):
        H, W = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)
        edges = cv2.Canny(gray, 60, 180, L2gradient=True)

        min_len = int(max(W * 0.08, H * 0.08, 30))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90,
                                minLineLength=min_len, maxLineGap=4)

        acc = np.zeros_like(gray, dtype=np.uint8)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0, :]:
                cv2.line(acc, (x1, y1), (x2, y2), 255, 2)

        acc = cv2.dilate(acc, None, iterations=1)
        acc = cv2.erode(acc, None, iterations=1)
        cnts, _ = cv2.findContours(acc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        props = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 120 or h < 90:
                continue
            props.append((x, y, x + w, y + h))
        props.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return props

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
