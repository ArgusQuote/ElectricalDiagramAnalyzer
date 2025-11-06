import sys 
import os 
from pathlib import Path
# Path setup 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(script_dir) 
if project_root not in sys.path: sys.path.append(project_root) 

from PageFilter.PageFilter import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV12 import PanelBoardSearch

# ---- Inputs/Outputs ----
INPUT_PDF = Path("~/Documents/Diagrams/ELECTRICAL SET (Mark Up).pdf").expanduser()
FILTER_OUT_DIR = Path("~/Documents/Diagrams/PdfOuput").expanduser()
FINDER_OUT_DIR = Path("~/Documents/Diagrams/PanelSearchOuput").expanduser()
FILTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
FINDER_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1) Run PageFilter ----
FILTER = PageFilter(
    output_dir=str(FILTER_OUT_DIR),
    dpi=400,                 # a bit higher for crisper OCR crop & line segmentation
    longest_cap_px=12000,    # allow larger raster if pages are huge
    proc_scale=0.75,         # keep more detail for footprint geometry
    use_ocr=True,
    ocr_gpu=False,
    verbose=True,
    debug=True,
    # Relaxed ranges
    rect_w_fr_range=(0.10, 0.75),
    rect_h_fr_range=(0.10, 0.85),
    min_rectangularity=0.65,  # slightly relaxed
    min_rect_count=1,         # 1 big table box is enough to KEEP
    # A bit more permissive area cut
    min_whitespace_area_fr=0.004,
)
kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(INPUT_PDF))
print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")

# Choose PDF for finder: filtered if we kept something, else original
pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept_pages) > 0) else INPUT_PDF
print(f"[PanelFinder] using PDF: {pdf_for_finder}")

# ---- 2) Run PanelBoardSearch on chosen PDF ----
FINDER = PanelBoardSearch(
    output_dir=str(FINDER_OUT_DIR),
    dpi=400,
    min_void_area_fr=0.004,
    min_void_w_px=90,
    min_void_h_px=90,
    # ↓ tighten these three to kill giant/near-page blobs
    max_void_area_fr=0.30,          # was 0.30
    void_w_fr_range=(0.20, 0.60),   # was (0.10, 0.60)
    void_h_fr_range=(0.20, 0.55),   # was (0.10, 0.60)
    min_whitespace_area_fr=0.01,
    margin_shave_px=6,
    pad=6,
    debug=True,
    verbose=True,
    save_masked_shape_crop=False,
    replace_multibox=True,
)

crops = FINDER.readPdf(str(pdf_for_finder))
print(f"[PanelFinder] wrote {len(crops)} crop(s) to {FINDER_OUT_DIR}")
