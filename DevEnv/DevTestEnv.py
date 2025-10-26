import sys 
import os 
from pathlib import Path
# Path setup 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(script_dir) 
if project_root not in sys.path: sys.path.append(project_root) 

from PageFilter.PageFilter import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV11 import PanelBoardSearch
from OcrLibrary.BreakerTableAnalyzer3 import BreakerTableAnalyzer

# ---- Inputs/Outputs ----
INPUT_PDF = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/generic2.pdf").expanduser()
FILTER_OUT_DIR = Path("~/ElectricalDiagramAnalyzer/DevEnv/PdfOuput").expanduser()
FINDER_OUT_DIR = Path("~/ElectricalDiagramAnalyzer/DevEnv/PanelSearchOuput").expanduser()
FILTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
FINDER_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 1) Run PageFilter ----
FILTER = PageFilter(
    output_dir=str(FILTER_OUT_DIR),   # filtered PDF will be written here
    dpi=350,                          # raster DPI used only for undecided pages
    longest_cap_px=9000,
    proc_scale=0.5,
    use_ocr=True,
    ocr_gpu=False,
    verbose=True,
    debug=False,                      # set True to write JSON log at output_dir/filter_debug/
    rect_w_fr_range=(0.10, 0.55),
    rect_h_fr_range=(0.10, 0.60),
    min_rectangularity=0.70,
    min_rect_count=2,
)
kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(INPUT_PDF))
print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")

# Choose PDF for finder: filtered if we kept something, else original
pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept_pages) > 0) else INPUT_PDF
print(f"[PanelFinder] using PDF: {pdf_for_finder}")

FILTER.readPdf('~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/generic2.pdf')

# ---- 2) Run PanelBoardSearch on chosen PDF ----
FINDER = PanelBoardSearch(
    output_dir=str(FINDER_OUT_DIR),
    dpi=400,
    min_whitespace_area_fr=0.01,
    margin_shave_px=6,
    min_void_area_fr=0.004,
    min_void_w_px=90,
    min_void_h_px=90,
    max_void_area_fr=0.50,
    pad=6,
    debug=False,
    verbose=True,
    save_masked_shape_crop=False,
    replace_multibox=True,
)

crops = FINDER.readPdf(str(pdf_for_finder))
print(f"[PanelFinder] wrote {len(crops)} crop(s) to {FINDER_OUT_DIR}")