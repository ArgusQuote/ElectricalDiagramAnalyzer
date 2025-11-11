#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess, shutil
from pathlib import Path
import fitz  # only used to count pages


# ====== PATHS (EDIT HERE) ====================================================
SRC_PDF  = Path("~/Documents/Diagrams/ELECTRICAL SET (Mark Up).pdf").expanduser()
OUT_DIR  = Path("~/Documents/Diagrams/PdfOuput").expanduser()
OUT_FILE = OUT_DIR / "Last5_LetterLandscape_VECTOR.pdf"
# ==============================================================================


def export_last5_letter_landscape(src_pdf: Path, out_pdf: Path) -> Path:
    """
    Vector-preserving: takes the last 5 pages of src_pdf,
    fits each on a US Letter (11×8.5in) landscape page, and writes out_pdf.
    No clipping. Uses Ghostscript (same backend as 'Print to PDF').
    """
    src_pdf = src_pdf.expanduser().resolve()
    out_pdf = out_pdf.expanduser().resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if not src_pdf.exists():
        raise FileNotFoundError(src_pdf)

    # determine total page count
    with fitz.open(src_pdf) as doc:
        total_pages = len(doc)
    first_page = max(1, total_pages - 4)
    last_page  = total_pages

    # locate ghostscript binary
    gs = shutil.which("gs")
    if not gs:
        raise RuntimeError("Ghostscript not found. Install with: sudo apt-get install -y ghostscript")

    # build command
    args = [
        gs, "-q", "-dNOPAUSE", "-dBATCH",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.7",
        "-dFIXEDMEDIA", "-dPDFFitPage",
        "-dAutoRotatePages=/None",
        "-dUseCropBox",
        "-dDEVICEWIDTHPOINTS=792", "-dDEVICEHEIGHTPOINTS=612",  # Letter landscape
        f"-dFirstPage={first_page}", f"-dLastPage={last_page}",
        "-sOutputFile=" + str(out_pdf),
        str(src_pdf),
    ]

    # run ghostscript
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ghostscript failed:\n{result.stderr.strip()}")

    print(f"[OK] Saved → {out_pdf}  ({last_page - first_page + 1} pages)")
    return out_pdf


# ====== EXECUTE ==============================================================
if __name__ == "__main__":
    export_last5_letter_landscape(SRC_PDF, OUT_FILE)
