#!/usr/bin/env python3
"""
Batch PDF Scanner - Processes all PDFs in pdfToScan folder.
Creates a unique TestScanN folder for each PDF document.
"""
import sys, os
from pathlib import Path

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch

# Configuration
PDF_INPUT_DIR = Path("~/Documents/pdfToScan").expanduser()
OUTPUT_BASE_DIR = Path("~/Documents").expanduser()

def get_next_testscan_index(base_dir: Path) -> int:
    """Find the next available TestScan index starting from 0."""
    index = 0
    while (base_dir / f"TestScan{index}").exists():
        index += 1
    return index

def process_all_pdfs():
    """Process all PDF files in the input directory."""
    # Find all PDF files
    pdf_files = sorted(PDF_INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_INPUT_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF(s) to process:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    print()
    
    # Get starting index for TestScan folders
    start_index = get_next_testscan_index(OUTPUT_BASE_DIR)
    
    # Process each PDF
    for i, pdf_path in enumerate(pdf_files):
        folder_index = start_index + i
        out_dir = OUTPUT_BASE_DIR / f"TestScan{folder_index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing [{i+1}/{len(pdf_files)}]: {pdf_path.name}")
        print(f"Output folder: TestScan{folder_index}")
        print(f"{'='*60}")
        
        finder = PanelBoardSearch(
            output_dir=out_dir,
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
            debug=False,
        )
        
        pngs = finder.readPdf(str(pdf_path))
        print(f"\nWrote {len(pngs)} PNGs to {out_dir}")
        print(f"Vector crops → {out_dir/'cropped_tables_pdf'}")
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Processed {len(pdf_files)} PDF(s)")
    print(f"Output folders: TestScan{start_index} - TestScan{start_index + len(pdf_files) - 1}")
    print(f"{'='*60}")

if __name__ == "__main__":
    process_all_pdfs()
