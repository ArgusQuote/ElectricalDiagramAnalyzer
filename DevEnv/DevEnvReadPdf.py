#!/usr/bin/env python3
import fitz  # PyMuPDF
from pathlib import Path
import textwrap

# ------------------------------------------------------------
# SET YOUR PDF INPUT HERE  (just like the rest of your project)
# ------------------------------------------------------------
PDF_PATH = Path("~/Documents/Diagrams/PdfOuput/Last5_LetterLandscape_VECTOR.pdf").expanduser()

def inspect_pdf(pdf_path: Path):
    pdf_path = pdf_path.expanduser()
    if not pdf_path.is_file():
        print(f"[ERROR] File not found: {pdf_path}")
        return

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"[ERROR] Could not open PDF: {e}")
        return

    print("=" * 80)
    print(f"Inspecting PDF: {pdf_path}")
    print(f"Total pages: {len(doc)}")
    print("=" * 80)

    for page_index, page in enumerate(doc, start=1):
        txt = page.get_text("text") or ""
        txt_clean = txt.strip()

        has_text = bool(txt_clean)
        print(f"Page {page_index:03d}: has_text={has_text}, text_length={len(txt_clean)}")

        if has_text:
            snippet = txt_clean[:350]
            snippet = snippet.replace("\r", " ").replace("\n", " ")
            snippet = " ".join(snippet.split())  # collapse whitespace

            wrapped = textwrap.fill(snippet, width=80)
            print("  Text snippet:")
            print("  " + wrapped.replace("\n", "\n  "))
        else:
            print("  (no extractable text)")

        print("-" * 80)

    doc.close()


if __name__ == "__main__":
    inspect_pdf(PDF_PATH)
