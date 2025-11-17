#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF

# =============== USER CONFIG =================
PDF_PATH = Path(
    "~/Documents/Diagrams/generic3.pdf"
).expanduser()

OUTPUT_DIR = Path(
    "~/Documents/Diagrams/PdfCodeDump"
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DUMP_PATH = OUTPUT_DIR / "pdf_object_dump.txt"
# =============================================


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"[INFO] Opening PDF: {PDF_PATH}")
    print(f"[INFO] Dump file  : {DUMP_PATH}")

    doc = fitz.open(str(PDF_PATH))

    with open(DUMP_PATH, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"PDF object dump for: {PDF_PATH}\n")
        f.write("=" * 80 + "\n\n")

        xref_count = doc.xref_length()
        f.write(f"Total xref objects: {xref_count}\n\n")

        # xref indices start at 1 in PyMuPDF
        for xref in range(1, xref_count):
            try:
                src = doc.xref_object(xref, compressed=False)
            except Exception as e:
                src = f"<< error reading object {xref}: {e} >>"

            f.write(f"====================== object {xref} ======================\n")
            f.write(src)
            f.write("\n\n")

    doc.close()
    print("[DONE] PDF objects written to:")
    print(f"       {DUMP_PATH}")


if __name__ == "__main__":
    main()
