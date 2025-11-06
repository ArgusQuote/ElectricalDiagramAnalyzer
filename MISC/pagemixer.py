#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rearrange PDF pages without re-rasterizing (preserves original quality).

Features:
  - Define input/output paths and page order as variables at the top.
  - Supports page ranges (e.g. "1-3,5,7-") and repeats.
  - Fully lossless: pages are copied directly, no recompression.
Requires:
  pip install PyMuPDF
"""

import fitz  # PyMuPDF
import os
from typing import List


# ------------------ USER VARIABLES ------------------

SRC_PDF  = os.path.expanduser("~/Documents/Diagrams/PdfOuput/ELECTRICAL SET (Mark Up)_electrical_filtered.pdf")
OUT_PDF  = os.path.expanduser("~/Documents/Diagrams/OutputMixed/reordered.pdf")

# Page order specification:
# "3,1,2,5-7,4"   → reorder as 3,1,2,5,6,7,4
# "1-5"           → keep pages 1 through 5
# "1,3,3,2"       → duplicates allowed
# "8-"            → from page 8 to the end
PAGE_ORDER = "2,1,3,4,5"

# ----------------------------------------------------


def parse_page_spec(spec: str, total: int) -> List[int]:
    """Convert a page order spec string to a 0-based list of indices."""
    def parse_one(num: str) -> int:
        n = int(num)
        if n < 0:
            raise ValueError("Negative pages not supported here (use 1-based).")
        if n < 1 or n > total:
            raise ValueError(f"Page {n} out of range 1..{total}")
        return n - 1

    result: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 1 or parts[1] == "":
                start = parse_one(parts[0])
                end = total - 1
            else:
                start = parse_one(parts[0])
                end = parse_one(parts[1])
            step = 1 if start <= end else -1
            result.extend(list(range(start, end + step, step)))
        else:
            result.append(parse_one(token))
    return result


def rearrange_pdf(src_pdf: str, out_pdf: str, page_order: str):
    """Copy pages from src_pdf to out_pdf in the specified order."""
    doc = fitz.open(src_pdf)
    n = len(doc)
    order = parse_page_spec(page_order, n)

    out = fitz.open()
    for idx in order:
        out.insert_pdf(doc, from_page=idx, to_page=idx)
    out.save(out_pdf)
    out.close()
    doc.close()
    print(f"[OK] Wrote reordered PDF: {out_pdf}")
    print(f"  Original pages: {n}, new page count: {len(order)}")


def main():
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    rearrange_pdf(SRC_PDF, OUT_PDF, PAGE_ORDER)


if __name__ == "__main__":
    main()
