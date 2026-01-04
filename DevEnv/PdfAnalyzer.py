#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import fitz  # PyMuPDF
import cv2
import numpy as np

# ================== CONFIG ==================
SOURCE_DIR = Path("~/Documents/pdfToScan").expanduser().resolve()
OUTPUT_DIR = Path("~/Documents/output_buckets6").expanduser().resolve()
OVERLAY_DIR = OUTPUT_DIR / "overlays"

MAX_PAGES_OVERLAY = 1    # how many pages per PDF to render with overlays
ZOOM = 2.0               # zoom factor for rendering overlays
# ============================================


def analyze_pdf(pdf_path: Path):
    """
    Returns (has_raster, has_vector) for the given PDF.

    has_raster: True if any embedded bitmap images exist.
    has_vector: True if any vector drawings or text objects exist.
    """
    has_raster = False
    has_vector = False

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Could not open {pdf_path}: {e}")
        return False, False

    try:
        for page_index, page in enumerate(doc, start=1):
            # --- Raster detection: embedded images ---
            try:
                images = page.get_images(full=True)
            except Exception as e:
                print(f"[WARN] {pdf_path.name} page {page_index}: get_images() failed: {e}")
                images = []

            if images:
                has_raster = True

            # --- Vector detection: drawings ---
            try:
                drawings = page.get_drawings()
            except Exception as e:
                print(f"[WARN] {pdf_path.name} page {page_index}: get_drawings() failed: {e}")
                drawings = []

            if drawings:
                has_vector = True

            # --- Vector detection: text objects ---
            try:
                raw_text = page.get_text("raw")
            except Exception as e:
                print(f"[WARN] {pdf_path.name} page {page_index}: get_text('raw') failed: {e}")
                raw_text = ""

            if raw_text and raw_text.strip():
                has_vector = True

            if has_raster and has_vector:
                break
    finally:
        doc.close()

    return has_raster, has_vector


def _page_to_bgr_image(page: fitz.Page, zoom: float = 2.0):
    """
    Render a page to a BGR OpenCV image and return (img, matrix).

    The matrix is the same one used to render the pixmap, so you can
    transform PDF coordinates (Rects, word boxes, etc.) to pixel coords:

        img_rect = rect * matrix
    """
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:  # 1-channel -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img, mat


def _rect_to_pix_rect(page: fitz.Page, rect: fitz.Rect, mat: fitz.Matrix, zoom: float) -> fitz.Rect:
    """
    Map a rectangle from PDF coordinate space to pixmap coordinate space.

    - For rotation == 0: just rect * mat (your original behavior).
    - For rotation == 270: rotate the rect from unrotated mediabox space
      into the rotated page space, then scale by zoom.
    """
    rot = page.rotation % 360

    # Normal pages: original behavior
    if rot == 0:
        return rect * mat

    # Handle your problematic case: rotation == 270
    if rot == 270:
        mbox = page.mediabox
        w0, h0 = mbox.width, mbox.height  # unrotated width / height

        # corners of the rect in unrotated space
        xs = [rect.x0, rect.x1, rect.x1, rect.x0]
        ys = [rect.y0, rect.y0, rect.y1, rect.y1]

        # Rotate 270° clockwise (equivalently 90° CCW) in (x right, y down):
        # (x, y) -> (y, w0 - x)
        xps = [y for x, y in zip(xs, ys)]
        yps = [w0 - x for x, y in zip(xs, ys)]

        minx, maxx = min(xps), max(xps)
        miny, maxy = min(yps), max(yps)

        # Now scale into pixel space using zoom
        return fitz.Rect(minx * zoom, miny * zoom, maxx * zoom, maxy * zoom)

    # Fallback: treat other rotations like 0°
    return rect * mat


def make_debug_overlays_for_pdf(
    pdf_path: Path,
    out_dir: Path,
    max_pages: int = 1,
    zoom: float = 2.0,
):
    """
    For the given PDF, render up to `max_pages` pages and draw:
      - BLUE rectangles around raster images
      - GREEN rectangles around text blocks (vector-ish tables / labels)

    Output PNGs into out_dir with names like: <stem>_p01.png
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[WARN] Could not open {pdf_path} for overlay: {e}")
        return

    out_dir.mkdir(exist_ok=True, parents=True)

    try:
        num_pages = min(len(doc), max_pages)
        for page_index in range(num_pages):
            page = doc[page_index]

            img, mat = _page_to_bgr_image(page, zoom=zoom)
            h, w = img.shape[:2]

            # ----- RASTER IMAGES (BLUE: (255, 0, 0)) -----
            try:
                images = page.get_images(full=True)
            except Exception as e:
                print(f"[WARN] {pdf_path.name} page {page_index+1}: "
                      f"get_images() failed in overlay: {e}")
                images = []

            for img_info in images:
                xref = img_info[0]
                try:
                    rects = page.get_image_rects(xref)
                except Exception as e:
                    print(f"[WARN] {pdf_path.name} page {page_index+1}: "
                          f"get_image_rects({xref}) failed: {e}")
                    continue

                for r in rects:
                    rect_obj = fitz.Rect(r)
                    r_pix = _rect_to_pix_rect(page, rect_obj, mat, zoom)
                    x0 = int(r_pix.x0)
                    y0 = int(r_pix.y0)
                    x1 = int(r_pix.x1)
                    y1 = int(r_pix.y1)

                    cv2.rectangle(
                        img,
                        (max(0, x0), max(0, y0)),
                        (min(w - 1, x1), min(h - 1, y1)),
                        (255, 0, 0),  # BLUE for raster
                        2,
                    )

            # ----- TEXT BLOCKS (GREEN: (0, 255, 0)) -----
            # Use blocks instead of get_drawings() to avoid huge full-page boxes
            try:
                blocks = page.get_text("blocks")
            except Exception as e:
                print(f"[WARN] {pdf_path.name} page {page_index+1}: "
                      f"get_text('blocks') failed in overlay: {e}")
                blocks = []

            for b in blocks:
                if len(b) < 4:
                    continue
                x0, y0, x1, y1 = b[0:4]

                rect_obj = fitz.Rect(x0, y0, x1, y1)
                r_pix = _rect_to_pix_rect(page, rect_obj, mat, zoom)
                X0 = int(r_pix.x0)
                Y0 = int(r_pix.y0)
                X1 = int(r_pix.x1)
                Y1 = int(r_pix.y1)

                cv2.rectangle(
                    img,
                    (max(0, X0), max(0, Y0)),
                    (min(w - 1, X1), min(h - 1, Y1)),
                    (0, 255, 0),  # GREEN for vector-ish text blocks
                    1,
                )

            out_path = out_dir / f"{pdf_path.stem}_p{page_index+1:02d}.png"
            cv2.imwrite(str(out_path), img)
            print(f"[OVERLAY] {pdf_path.name} page {page_index+1} -> {out_path}")
    finally:
        doc.close()


def process_pdfs(source_dir: Path, overlay_dir: Path):
    """
    Scan source_dir for PDFs, analyze them, and produce overlays only.
    Does NOT move or modify PDFs or create classification folders.
    """
    overlay_dir.mkdir(exist_ok=True, parents=True)

    stats = {
        "raster_only": 0,
        "vector_only": 0,
        "mixed": 0,
        "skipped": 0,
    }

    for pdf_path in sorted(source_dir.glob("*.pdf")):
        has_raster, has_vector = analyze_pdf(pdf_path)

        if not has_raster and not has_vector:
            print(f"[SKIP] {pdf_path.name}: no raster or vector detected (maybe encrypted/odd).")
            stats["skipped"] += 1
            continue

        if has_raster and has_vector:
            key = "mixed"
            classification = "MIXED (raster + vector)"
        elif has_raster:
            key = "raster_only"
            classification = "RASTER ONLY"
        else:
            key = "vector_only"
            classification = "VECTOR ONLY"

        stats[key] += 1
        print(f"[CLASSIFY] {pdf_path.name} -> {classification}")

        try:
            make_debug_overlays_for_pdf(
                pdf_path,
                overlay_dir,
                max_pages=MAX_PAGES_OVERLAY,
                zoom=ZOOM,
            )
        except Exception as e:
            print(f"[WARN] Failed to make overlay for {pdf_path.name}: {e}")

    print("\nSummary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source directory does not exist or is not a directory: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    process_pdfs(SOURCE_DIR, OVERLAY_DIR)
