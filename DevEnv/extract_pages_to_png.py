#!/usr/bin/env python3
"""
One-off script to extract specific pages from PDFs and render them as
annotation-quality PNGs matching the existing images in TableAnnotations/images/.

Settings match render_pdfs_for_annotation.py: 200 DPI, no dimension cap, RGB.
"""

import uuid
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

OUTPUT_DIR = Path("/home/marco/Documents/TableAnnotations/images")
DPI = 200
SCALE = DPI / 72.0

JOBS = [
    ("/home/marco/Documents/new panels/derek2.pdf", 9),
    ("/home/marco/Documents/new panels/derekfirst.pdf", 7),
]


def render_page(pdf_path: str, page_num: int) -> None:
    pdf_path = Path(pdf_path)
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[page_num - 1]
    bitmap = page.render(scale=SCALE)
    img = bitmap.to_pil()

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    short_id = uuid.uuid4().hex[:8]
    name = f"{short_id}-{pdf_path.stem}_page{page_num:03d}.png"
    img.save(str(OUTPUT_DIR / name), "PNG")
    print(f"Saved: {name} ({img.size[0]}x{img.size[1]})")
    doc.close()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path, page_num in JOBS:
        render_page(pdf_path, page_num)
    print("Done.")
