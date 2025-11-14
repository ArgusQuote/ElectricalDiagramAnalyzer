#!/usr/bin/env python3
import json
from pathlib import Path

from pdf2image import convert_from_path
import easyocr
import fitz


# ---------------- USER SETTINGS ----------------
PDF_PATH = Path("~/Documents/Diagrams/CaseStudy_PanelVectorCrops/Last5_LetterLandscape_VECTOR_page003_panel01_VECTOR.pdf").expanduser()
OUTPUT_JSON = Path("~/Documents/Diagrams/easyocrdump.json").expanduser()
DPI = 400   # good balance for engineering drawings
LANGS = ['en']  # can add: ['en', 'es', 'fr'] if needed
# ------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int):
    """Render each page of PDF as an image using PyMuPDF @ specified DPI."""
    doc = fitz.open(pdf_path)
    images = []

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        images.append((i + 1, pix))
    doc.close()
    return images


def run_easyocr_on_pdf(pdf_path: str, dpi: int, langs: list[str]):
    """Run EasyOCR on each rendered PDF page."""
    reader = easyocr.Reader(langs)
    pages = pdf_to_images(pdf_path, dpi)

    results = []

    for page_num, pix in pages:
        img = pix.tobytes("png")  # convert PyMuPDF pixmap → PNG bytes
        detections = reader.readtext(img)

        results.append({
            "page": page_num,
            "results": [
                {
                    "bbox": d[0],
                    "text": d[1],
                    "confidence": d[2]
                }
                for d in detections
            ]
        })

    return results


print(f"[INFO] Running OCR on: {PDF_PATH}")
data = run_easyocr_on_pdf(str(PDF_PATH), DPI, LANGS)

# ---- FIX JSON SERIALIZATION (convert numpy types → python) ----
def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_python(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy scalar to python
        return obj.item()
    return obj

cleaned = _to_python(data)

with open(OUTPUT_JSON, "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"\n[DONE] OCR results saved to: {OUTPUT_JSON}\n")
