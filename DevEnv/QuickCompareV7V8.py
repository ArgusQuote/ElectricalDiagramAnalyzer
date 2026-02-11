#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick V7 vs V8 Header Parser Comparison - Single Image Test
Tests both parsers on the same image and shows differences.
"""
import os
import sys
import json
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from OcrLibrary.PanelHeaderParserV7 import PanelParser as ParserV7
from OcrLibrary.PanelHeaderParserV8 import PanelParser as ParserV8

# Force CPU mode for consistent testing
import easyocr
_shared_reader = None

def get_shared_reader():
    global _shared_reader
    if _shared_reader is None:
        _shared_reader = easyocr.Reader(['en'], gpu=False)
    return _shared_reader


def compare_on_image(image_path: str, header_y_ratio: float = None) -> dict:
    """Run both V7 and V8 parsers on an image and compare results."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}")
    
    # Get shared reader (CPU mode)
    reader = get_shared_reader()
    
    # Run V7
    print("\n--- V7 Parser ---")
    p7 = ParserV7(debug=False)
    p7.reader = reader  # Use shared reader
    try:
        r7 = p7.parse_panel(image_path, header_y_ratio=header_y_ratio)
    except Exception as e:
        print(f"V7 ERROR: {e}")
        r7 = {"name": None, "attrs": {}}
    
    # Run V8
    print("\n--- V8 Parser ---")
    p8 = ParserV8(debug=False)
    p8.reader = reader  # Use shared reader
    try:
        r8 = p8.parse_panel(image_path, header_y_ratio=header_y_ratio)
    except Exception as e:
        print(f"V8 ERROR: {e}")
        r8 = {"name": None, "attrs": {}}
    
    # Extract values
    def get_attrs(r):
        return {
            "name": r.get("name"),
            "voltage": (r.get("attrs") or {}).get("voltage"),
            "amperage": (r.get("attrs") or {}).get("amperage"),
            "mainBreakerAmperage": (r.get("attrs") or {}).get("mainBreakerAmperage"),
            "intRating": (r.get("attrs") or {}).get("intRating"),
        }
    
    v7_attrs = get_attrs(r7)
    v8_attrs = get_attrs(r8)
    
    # Print comparison
    print("\n--- COMPARISON ---")
    fields = ["name", "voltage", "amperage", "mainBreakerAmperage", "intRating"]
    differences = []
    
    for f in fields:
        v7_val = v7_attrs.get(f)
        v8_val = v8_attrs.get(f)
        match = "✓" if v7_val == v8_val else "✗"
        print(f"  {f:22s}: V7={str(v7_val):12s}  V8={str(v8_val):12s}  {match}")
        if v7_val != v8_val:
            differences.append((f, v7_val, v8_val))
    
    return {"v7": v7_attrs, "v8": v8_attrs, "differences": differences}


def main():
    # Find test images
    test_dirs = [
        Path("~/Documents/DevEnv2_FullAnalysis").expanduser(),
    ]
    
    test_images = []
    for d in test_dirs:
        if d.is_dir():
            for subdir in d.iterdir():
                if subdir.is_dir():
                    panel_dir = subdir / "PanelSearchOutput"
                    if panel_dir.is_dir():
                        for f in panel_dir.glob("*.png"):
                            if "overlay" not in f.name.lower() and "debug" not in str(f).lower():
                                test_images.append(str(f))
    
    # Also check pdfToScan folder for any direct images
    pdf_scan_dir = Path("~/Documents/pdfToScan").expanduser()
    if pdf_scan_dir.is_dir():
        for f in pdf_scan_dir.glob("*.png"):
            test_images.append(str(f))
    
    if not test_images:
        print("No test images found. Please run DevTestEnvV2.py first to generate crop images.")
        return
    
    print(f"Found {len(test_images)} test image(s)")
    
    all_diffs = []
    for img in test_images[:5]:  # Test first 5 images
        result = compare_on_image(img, header_y_ratio=0.25)
        if result["differences"]:
            all_diffs.append({"image": img, "diffs": result["differences"]})
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Images tested: {min(5, len(test_images))}")
    print(f"Images with differences: {len(all_diffs)}")
    
    if all_diffs:
        print("\nDifferences found:")
        for item in all_diffs:
            print(f"\n  {Path(item['image']).name}:")
            for f, v7, v8 in item["diffs"]:
                print(f"    {f}: V7={v7} → V8={v8}")


if __name__ == "__main__":
    main()
