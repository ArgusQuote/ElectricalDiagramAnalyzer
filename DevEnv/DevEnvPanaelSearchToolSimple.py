import sys 
import os 
from pathlib import Path
# Path setup 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(script_dir) 
if project_root not in sys.path: sys.path.append(project_root) 

from VisualDetectionToolLibrary.PanelSearchToolV12 import PanelBoardSearch

# ---- Inputs/Outputs ----
INPUT_PDF = Path("~/Documents/Diagrams/Last4pgs.pdf").expanduser()
FINDER_OUT_DIR = Path("~/Documents/Diagrams/CaseStudy5").expanduser()
FINDER_OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    debug=False,
    verbose=True,
    save_masked_shape_crop=False,
    replace_multibox=True,
)

crops = FINDER.readPdf(str(INPUT_PDF))
print(f"[PanelFinder] wrote {len(crops)} crop(s) to {FINDER_OUT_DIR}")
