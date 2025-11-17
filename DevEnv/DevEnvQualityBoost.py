#!/usr/bin/env python3
from pathlib import Path
import fitz  # PyMuPDF

# ---- set this to the folder that contains the cropped panel PDFs ----
CROPS_DIR = Path("~/Documents/Diagrams/CaseStudyPdfTextSearch3/copped_tables_pdf").expanduser()
PNG_OUT   = Path("~/Documents/Diagrams/PanelPNGsHiRes").expanduser()
PNG_OUT.mkdir(parents=True, exist_ok=True)

print(f"[INFO] CROPS_DIR: {CROPS_DIR}  (exists={CROPS_DIR.exists()})")

# Recursively find PDFs (case-insensitive)
pdfs = sorted(p for p in CROPS_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
print(f"[INFO] Found {len(pdfs)} PDF(s) under {CROPS_DIR}")
for p in pdfs:
    print("   -", p)

if not pdfs:
    # Help you spot typos like 'copped_tables_pdf' vs 'cropped_tables_pdf'
    parent = CROPS_DIR.parent
    print(f"[HINT] Siblings of target dir in {parent}:")
    for x in sorted(parent.glob("*")):
        print("   •", x.name, "(dir)" if x.is_dir() else "")
    raise SystemExit("[EXIT] No PDFs found. Check folder name and location.")

# Optional: stronger anti-aliasing for text/lines
try:
    fitz.TOOLS.set_aa_level(8)  # 0..8
except Exception:
    pass

for pdf_path in pdfs:
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=900, alpha=False, colorspace=fitz.csGRAY)
        out_png = PNG_OUT / f"{pdf_path.stem}_p{i:02d}_900dpi.png"
        pix.save(str(out_png))
        print("[WRITE]", out_png)
    doc.close()

print(f"[DONE] Wrote PNGs to: {PNG_OUT}")
