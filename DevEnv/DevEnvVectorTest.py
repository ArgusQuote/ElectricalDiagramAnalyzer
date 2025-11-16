#!/usr/bin/env python3
import os
from pathlib import Path
import math
import fitz  # PyMuPDF


# ========= USER CONFIG =========
PDF_PATH = Path(
    "~/Documents/Diagrams/PdfOuput/Last5_LetterLandscape_VECTOR.pdf"
).expanduser()

OUTPUT_DIR = Path(
    "~/Documents/Diagrams/PdfVectorTests"
).expanduser()
# ===============================


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Opening PDF: {PDF_PATH}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    doc = fitz.open(str(PDF_PATH))

    max_len2 = 0.0
    max_info = None  # (page_index, (x1,y1),(x2,y2))

    # ---- PASS 1: find longest vector line ----
    for page_index in range(len(doc)):
        page = doc[page_index]
        drawings = page.get_drawings()

        for obj in drawings:
            items = obj.get("items") or []
            for item in items:
                cmd = item[0]
                pts = item[1]

                # Only literal line commands
                if cmd != "l":
                    continue

                #_pts must be ((x1,y1),(x2,y2))
                if not (isinstance(pts, (tuple, list)) and len(pts) == 2):
                    continue

                p1, p2 = pts

                if not (
                    isinstance(p1, (tuple, list)) and len(p1) == 2 and
                    isinstance(p2, (tuple, list)) and len(p2) == 2
                ):
                    continue

                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])

                dx, dy = x2 - x1, y2 - y1
                length2 = dx * dx + dy * dy

                if length2 > max_len2:
                    max_len2 = length2
                    max_info = (page_index, (x1, y1), (x2, y2))

    if not max_info:
        print("[WARNING] No vector lines detected in this PDF.")
        return

    page_idx, (lx1, ly1), (lx2, ly2) = max_info
    length = round(math.sqrt(max_len2), 2)

    print("\n==================== RESULT ====================")
    print(f"Longest vector found on page {page_idx+1}")
    print(f"Coordinates: ({lx1:.2f}, {ly1:.2f}) → ({lx2:.2f}, {ly2:.2f})")
    print(f"Length: {length}px equivalent (PDF units)\n")

    # ---- PASS 2: overlay highlight vector on that one page ----
    page = doc[page_idx]

    # Draw as vector overlay
    shape = page.new_shape()
    shape.draw_line((lx1, ly1), (lx2, ly2))
    shape.finish(color=(1, 0, 0), width=5)  # red, moderately thick
    shape.commit()

    out_path = OUTPUT_DIR / f"{PDF_PATH.stem}_largestvec.pdf"
    doc.save(str(out_path))

    print(f"[DONE] Saved annotated PDF → {out_path}\n")


if __name__ == "__main__":
    main()
