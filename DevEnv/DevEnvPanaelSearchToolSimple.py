#!/usr/bin/env python3
import sys, os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from VisualDetectionToolLibrary.PanelSearchToolV14 import PanelBoardSearch

INPUT_PDF = Path("~/Documents/Diagrams/makayla2.pdf").expanduser()
OUT_DIR   = Path("~/Documents/Diagrams/CaseStudy_VectorCrop_Run").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

finder = PanelBoardSearch(
    output_dir=OUT_DIR,
    dpi=400,
    render_dpi=1400,
    aa_level=8,                # used for the final render; detection forces AA=0
    render_colorspace="gray",

    min_void_area_fr=0.004,
    min_void_w_px=90,
    min_void_h_px=90,
    max_void_area_fr=0.30,
    void_w_fr_range=(0.20, 0.60),
    void_h_fr_range=(0.20, 0.55),
    min_whitespace_area_fr=0.01,
    margin_shave_px=6,
    pad=6,
    verbose=True,
)

pngs = finder.readPdf(str(INPUT_PDF))
print(f"\nWrote {len(pngs)} PNGs to {OUT_DIR}")
print(f"Vector crops → {OUT_DIR/'cropped_tables_pdf'}")
