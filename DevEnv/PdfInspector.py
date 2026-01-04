#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import fitz  # PyMuPDF

SOURCE_DIR = Path("~/Documents/pdfToScan").expanduser().resolve()

def inspect_pdfs(source_dir: Path):
    if not source_dir.is_dir():
        raise SystemExit(f"Source dir not found: {source_dir}")

    for pdf_path in sorted(source_dir.glob("*.pdf")):
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"\n=== {pdf_path.name} ===")
            print(f"  [ERROR] cannot open: {e}")
            continue

        page = doc[0]  # assume 1st page is representative
        print(f"\n=== {pdf_path.name} ===")
        print(f"  pages        : {len(doc)}")
        print(f"  rotation     : {page.rotation}")          # 0 / 90 / 180 / 270
        print(f"  rect (w x h) : {page.rect.width:.2f} x {page.rect.height:.2f}")
        print(f"  mediabox     : {page.mediabox}")
        print(f"  cropbox      : {page.cropbox}")

        # rendered size with your usual zoom
        ZOOM = 2.0
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
        print(f"  pix size     : {pix.width} x {pix.height}")

        # content stats
        try:
            images = page.get_images(full=True)
        except Exception:
            images = []
        print(f"  images       : {len(images)}")

        try:
            blocks = page.get_text("blocks")
        except Exception:
            blocks = []
        print(f"  text blocks  : {len(blocks)}")

        doc.close()


if __name__ == "__main__":
    inspect_pdfs(SOURCE_DIR)
