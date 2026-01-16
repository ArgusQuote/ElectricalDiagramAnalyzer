#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PageFilterDevEnv.py
A minimal dev harness for PageFilter (labels + E-sheet + footprints), modeled after your reference.

Usage examples:
  # Single file (default paths)
  python PageFilterDevEnv.py

  # Specific file
  python PageFilterDevEnv.py --pdf ~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/generic2.pdf

  # Batch all PDFs in a folder (non-recursive)
  python PageFilterDevEnv.py --pdf_dir ~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf

  # Enable debug JSON logs and bump DPI
  python PageFilterDevEnv.py --debug --dpi 400
"""

import sys
import os
from pathlib import Path
import time
import argparse

# ----- Path setup (same pattern as your reference) -----
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports from your project
from PageFilter.PageFilterV3 import PageFilter

# ----- Defaults (same locations you used) -----
DEFAULT_INPUT_PDF = Path("~/Documents/Diagrams/ELECTRICAL SET (Mark Up).pdf").expanduser()
DEFAULT_OUT_DIR  = Path("~/Documents/Diagrams/PdfOuput").expanduser()


# Construct PageFilter with dev-friendly knobs (mirrors your reference)
FILTER = PageFilter(
    output_dir=str(DEFAULT_OUT_DIR),
    dpi=400,                          # raster DPI used only for undecided pages
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

kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(DEFAULT_INPUT_PDF))
