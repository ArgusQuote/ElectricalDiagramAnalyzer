import sys 
import os 
from pathlib import Path
# Path setup 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(script_dir) 
if project_root not in sys.path: sys.path.append(project_root) 

from PageFilter.PageFilterV2 import PageFilter

# ---- Inputs/Outputs ----
INPUT_PDF = Path("~/Documents/Diagrams/ELECTRICAL SET (Mark Up).pdf").expanduser()
FILTER_OUT_DIR = Path("~/Documents/Diagrams/PdfOuput").expanduser()
FILTER_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1) Run PageFilter ----
FILTER = PageFilter(
    output_dir=str(FILTER_OUT_DIR),
    dpi=400,                 # a bit higher for crisper OCR crop & line segmentation
    longest_cap_px=9000,
    proc_scale=0.5,
    use_ocr=True,
    ocr_gpu=False,
    verbose=True,
    debug=False,                      # set True to write JSON log at output_dir/filter_debug/
    rect_w_fr_range=(0.20, 0.60),
    rect_h_fr_range=(0.20, 0.60),
    min_rectangularity=0.70,
    min_rect_count=2,
    # A bit more permissive area cut
    min_whitespace_area_fr=0.004,
    use_ghostscript_letter=True,       # turn GS letter step on/off
    letter_orientation="landscape",    # "portrait" or "landscape"
    gs_use_cropbox=True,               # True: fit what's inside CropBox; False: use MediaBox
    gs_compat="1.7"                    # PDF compatibility level      
)
kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(INPUT_PDF))
print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")
