#!/usr/bin/env python3
import sys, os
from pathlib import Path

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from VisualDetectionToolLibrary.PanelSearchToolV21 import PanelBoardSearch

# Inputs/Outputs
INPUT_PDF = Path("~/Documents/pdfToScan/A.pdf").expanduser()
OUT_DIR   = Path("~/Documents/pdfScanTest5").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

finder = PanelBoardSearch(
    output_dir=OUT_DIR,
    dpi=400,
    render_dpi=1400,
    aa_level=8,
    render_colorspace="gray",
    min_void_area_fr=0.004,
    min_void_w_px=90,
    min_void_h_px=90,
    max_void_area_fr=0.30,
    void_w_fr_range=(0.20, 0.60),
    void_h_fr_range=(0.15, 0.55),
    min_whitespace_area_fr=0.01,
    margin_shave_px=6,
    pad=6,
    verbose=True,
    debug=False
    # one-box settings (defaults are fine, but you can loosen slightly if needed):
    # onebox_min_rel_area=0.02, onebox_max_rel_area=0.75,
    # onebox_aspect_range=(0.4, 3.0), onebox_min_side_px=80,
)


pngs = finder.readPdf(str(INPUT_PDF))
print(f"\nWrote {len(pngs)} PNGs to {OUT_DIR}")
print(f"Vector crops → {OUT_DIR/'cropped_tables_pdf'}")
