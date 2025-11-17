#!/usr/bin/env python3
import os
import sys
import math
from pathlib import Path

import fitz  # PyMuPDF

# ---------- USER CONFIG ----------
PDF_PATH = Path(
    "~/Documents/Diagrams/PdfOuput/ELECTRICAL SET (Mark Up)_electrical_filtered.pdf"
).expanduser()

OUTPUT_DIR = Path(
    "~/Documents/Diagrams/PdfVectorTests"
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"[INFO] Opening PDF: {PDF_PATH}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    doc = fitz.open(str(PDF_PATH))

    # Track global longest line: (length_sq, page_index, (x1,y1,x2,y2))
    longest = None

    for page_index in range(len(doc)):
        page = doc[page_index]
        drawings = page.get_drawings()
        print(f"[PAGE {page_index+1}] drawings: {len(drawings)}")

        for draw in drawings:
            items = draw.get("items", [])
            for draw_op in items:
                if not draw_op:
                    continue

                op = draw_op[0]
                if op != "l":
                    continue

                pts = draw_op[1]

                # pts can be ((x1,y1),(x2,y2)) OR (x1,y1,x2,y2)
                if (
                    isinstance(pts, (tuple, list))
                    and len(pts) == 2
                    and all(isinstance(p, (tuple, list)) and len(p) == 2 for p in pts)
                ):
                    (x1, y1), (x2, y2) = pts
                elif (
                    isinstance(pts, (tuple, list))
                    and len(pts) == 4
                ):
                    x1, y1, x2, y2 = pts
                else:
                    # unexpected format; skip
                    continue

                dx = x2 - x1
                dy = y2 - y1
                length_sq = dx * dx + dy * dy

                if longest is None or length_sq > longest[0]:
                    longest = (length_sq, page_index, (x1, y1, x2, y2))

    if longest is None:
        print("[WARNING] No vector lines detected in this PDF.")
        doc.close()
        return

    length_sq, page_index, (x1, y1, x2, y2) = longest
    length = math.sqrt(length_sq)
    print(
        f"[RESULT] Longest line on page {page_index+1}: "
        f"({x1:.2f}, {y1:.2f}) → ({x2:.2f}, {y2:.2f}), length ≈ {length:.2f} pts"
    )

    # ---- Build output PDF with same pages, plus highlighted longest line ----
    out_doc = fitz.open()
    for i in range(len(doc)):
        src_page = doc[i]
        rect = src_page.rect
        new_page = out_doc.new_page(width=rect.width, height=rect.height)

        # Copy original page as vector
        new_page.show_pdf_page(new_page.rect, doc, i)

        # If this is the page with the longest line, draw the highlight
        if i == page_index:
            new_page.draw_line(
                (x1, y1),
                (x2, y2),
                color=(1, 0, 0),  # red
                width=2.0,
            )

    out_name = PDF_PATH.stem + "_largest_line.pdf"
    out_path = OUTPUT_DIR / out_name
    out_doc.save(str(out_path))
    out_doc.close()
    doc.close()

    print(f"[DONE] Highlighted longest line written to: {out_path}")


if __name__ == "__main__":
    main()
