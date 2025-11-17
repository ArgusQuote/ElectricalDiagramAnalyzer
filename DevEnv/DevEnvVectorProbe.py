#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from VisualDetectionToolLibrary.PanelSearchToolV13 import PanelBoardSearch

# =============== USER CONFIG =================
PDF_PATH = Path(
    "~/Documents/Diagrams/generic3.pdf"
).expanduser()

OUTPUT_DIR = Path(
    "~/Documents/Diagrams/CaseStudy_PanelVectorCrops2"
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# =============================================


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(PDF_PATH)

    print(f"[INFO] Source PDF: {PDF_PATH}")
    print(f"[INFO] Output dir : {OUTPUT_DIR}")

    # 1) Run PanelBoardSearch to get panel_boxes_by_page (fractions)
    finder = PanelBoardSearch(
        output_dir=str(OUTPUT_DIR),
        debug=False,
        verbose=True,
    )
    finder.readPdf(str(PDF_PATH))  # populates finder.panel_boxes_by_page

    if not finder.panel_boxes_by_page:
        print("[WARN] No panels recorded in panel_boxes_by_page.")
        return

    # 2) Open original PDF for vector cropping
    doc = fitz.open(str(PDF_PATH))
    base_name = PDF_PATH.stem

    print("\n==============================")
    print(" Cropping panels as vector PDFs")
    print("==============================\n")

    total_panels = 0

    for page_num, panels in sorted(finder.panel_boxes_by_page.items()):
        page_index0 = page_num - 1
        if page_index0 < 0 or page_index0 >= len(doc):
            print(f"[WARN] page_num {page_num} out of range for doc (len={len(doc)})")
            continue

        page = doc[page_index0]
        rect_page = page.rect  # PDF coordinate system (points)

        print(f"[PAGE {page_num}] panels: {len(panels)}")

        for idx, entry in enumerate(panels, start=1):
            x0f, y0f, x1f, y1f = entry["bbox_frac"]

            # Map fractions -> PDF coords (same basis as vectors & text)
            x0 = rect_page.x0 + rect_page.width * x0f
            y0 = rect_page.y0 + rect_page.height * y0f
            x1 = rect_page.x0 + rect_page.width * x1f
            y1 = rect_page.y0 + rect_page.height * y1f

            panel_rect = fitz.Rect(x0, y0, x1, y1)

            # 3) Create a new PDF that contains ONLY this panel region
            out_doc = fitz.open()
            new_page = out_doc.new_page(
                width=panel_rect.width,
                height=panel_rect.height,
            )

            # Copy original page content into new_page, clipped to panel_rect
            new_page.show_pdf_page(
                new_page.rect,   # fill whole new page
                doc,
                page_index0,
                clip=panel_rect,  # <== this is the key: keep only that region
            )

            out_path = (
                OUTPUT_DIR
                / f"{base_name}_page{page_num:03d}_panel{idx:02d}_VECTOR.pdf"
            )
            out_doc.save(str(out_path))
            out_doc.close()

            total_panels += 1
            print(f"  → panel {idx}: saved vector crop → {out_path}")

    doc.close()

    print(f"\n[DONE] Wrote {total_panels} vector panel PDF(s) to {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
