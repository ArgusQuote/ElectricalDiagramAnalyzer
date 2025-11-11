from pdf2image import convert_from_path
import cv2, numpy as np
from pathlib import Path

pdfA = Path("~/Documents/Diagrams/PdfOuput/ELECTRICAL SET (Mark Up)_electrical_filtered.pdf").expanduser()
pdfB = Path("~/Documents/Diagrams/Last5pgs.pdf").expanduser()
dpi  = 400

pagesA = convert_from_path(str(pdfA), dpi=dpi)
pagesB = convert_from_path(str(pdfB), dpi=dpi)

for i, (a,b) in enumerate(zip(pagesA, pagesB), 1):
    imgA = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2GRAY)
    imgB = cv2.cvtColor(np.array(b), cv2.COLOR_RGB2GRAY)
    if imgA.shape != imgB.shape:
        print(f"Page {i}: different sizes {imgA.shape} vs {imgB.shape}")
        continue
    diff = cv2.absdiff(imgA, imgB)
    nonzero = np.count_nonzero(diff)
    print(f"Page {i}: {nonzero} differing pixels ({nonzero / diff.size:.4%})")
    if nonzero:
        cv2.imwrite(f"diff_page{i:03}.png", diff)