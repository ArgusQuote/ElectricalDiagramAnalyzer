#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Table Detection Comparison Script

Compares fine-tuned vs pretrained Table Transformer models on the same PDF.
Outputs are saved to separate folders for easy comparison.

Usage:
    python DevEnv/DevMLTableDetection.py
"""

import os
import sys
from pathlib import Path

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- IMPORTS ----------
from MLTableDetection import TableDetectorML

# ---------- IO PATHS ----------
# Change this to your PDF path
INPUT_PDF = Path("~/Documents/SinglePdf").expanduser()
OUTPUT_BASE = Path("~/Documents/ML_Test").expanduser()

# Separate output directories for comparison
OUTPUT_FINETUNED = OUTPUT_BASE / "finetuned"
OUTPUT_PRETRAINED = OUTPUT_BASE / "pretrained"

# Fine-tuned model path
FINETUNED_MODEL = Path("~/Documents/TableAnnotations/models/final").expanduser()


def find_pdf_in_dir(dir_path: Path) -> Path:
    """Find the first PDF in a directory, or return the path if it's a file."""
    if dir_path.is_file() and dir_path.suffix.lower() == '.pdf':
        return dir_path
    
    if dir_path.is_dir():
        pdfs = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF"))
        if pdfs:
            return pdfs[0]
    
    raise FileNotFoundError(f"No PDF found in: {dir_path}")


def run_detection(pdf_path: Path, output_dir: Path, model_path: str | None, 
                  model_name: str, conf_threshold: float = 0.5) -> list[str]:
    """
    Run table detection with specified model.
    
    Args:
        pdf_path: Path to input PDF.
        output_dir: Directory for output files.
        model_path: Path to model or None for pretrained.
        model_name: Display name for logging.
        conf_threshold: Confidence threshold for detections.
    
    Returns:
        List of generated PNG paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print(f"Running: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Model: {model_path or 'microsoft/table-transformer-detection'}")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  Output: {output_dir}")
    
    detector = TableDetectorML(
        output_dir=str(output_dir),
        model_path=model_path,
        conf_threshold=conf_threshold,
        enforce_one_box=False,
        verbose=True,
    )
    
    crops = detector.readPdf(str(pdf_path))
    
    print(f"\n[{model_name}] Tables detected: {len(crops)}")
    return crops


def print_comparison(finetuned_crops: list[str], pretrained_crops: list[str]):
    """Print side-by-side comparison of results."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<25} {'Tables Detected':<20}")
    print("-" * 45)
    print(f"{'Fine-tuned':<25} {len(finetuned_crops):<20}")
    print(f"{'Pretrained':<25} {len(pretrained_crops):<20}")
    
    diff = len(finetuned_crops) - len(pretrained_crops)
    if diff > 0:
        print(f"\nFine-tuned detected {diff} MORE table(s)")
    elif diff < 0:
        print(f"\nPretrained detected {abs(diff)} MORE table(s)")
    else:
        print("\nBoth models detected the same number of tables")
    
    print("\n" + "-" * 70)
    print("OUTPUT LOCATIONS:")
    print("-" * 70)
    print(f"\nFine-tuned outputs:")
    print(f"  - PNG crops: {OUTPUT_FINETUNED}")
    print(f"  - PDF crops: {OUTPUT_FINETUNED / 'cropped_tables_pdf'}")
    print(f"  - Overlays:  {OUTPUT_FINETUNED / 'magenta_overlays'}")
    
    print(f"\nPretrained outputs:")
    print(f"  - PNG crops: {OUTPUT_PRETRAINED}")
    print(f"  - PDF crops: {OUTPUT_PRETRAINED / 'cropped_tables_pdf'}")
    print(f"  - Overlays:  {OUTPUT_PRETRAINED / 'magenta_overlays'}")
    
    print("\n" + "-" * 70)
    print("TIP: Compare the overlay images in 'magenta_overlays' folders")
    print("     to see which regions each model detected.")
    print("-" * 70)


def main():
    print("\n" + "=" * 70)
    print("ML Table Detection - Threshold Comparison")
    print("Testing different confidence thresholds")
    print("=" * 70)
    
    # Find the PDF
    try:
        pdf_path = find_pdf_in_dir(INPUT_PDF)
        print(f"\n[INPUT] PDF: {pdf_path}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease update INPUT_PDF in this script to point to your PDF.")
        return
    
    # Thresholds to test
    # Fine-tuned: higher thresholds to reduce over-detection
    FINETUNED_THRESHOLDS = [0.7, 0.8, 0.9]
    # Pretrained: lower thresholds to see if it detects anything
    PRETRAINED_THRESHOLDS = [0.3, 0.2, 0.1]
    
    results = []
    
    # Check if fine-tuned model exists
    if FINETUNED_MODEL.exists():
        print(f"\n[Fine-tuned model] Testing thresholds: {FINETUNED_THRESHOLDS}")
        for thresh in FINETUNED_THRESHOLDS:
            output_dir = OUTPUT_BASE / f"finetuned_conf{thresh}"
            crops = run_detection(
                pdf_path, output_dir,
                model_path=str(FINETUNED_MODEL),
                model_name=f"Fine-tuned (conf={thresh})",
                conf_threshold=thresh
            )
            results.append(("Fine-tuned", thresh, len(crops), output_dir))
    else:
        print(f"\n[WARNING] Fine-tuned model not found at: {FINETUNED_MODEL}")
    
    # Run pretrained with lower thresholds
    print(f"\n[Pretrained model] Testing thresholds: {PRETRAINED_THRESHOLDS}")
    for thresh in PRETRAINED_THRESHOLDS:
        output_dir = OUTPUT_BASE / f"pretrained_conf{thresh}"
        crops = run_detection(
            pdf_path, output_dir,
            model_path=None,
            model_name=f"Pretrained (conf={thresh})",
            conf_threshold=thresh
        )
        results.append(("Pretrained", thresh, len(crops), output_dir))
    
    # Print summary
    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<15} {'Threshold':<12} {'Tables':<10} {'Output Folder'}")
    print("-" * 70)
    for model, thresh, count, output_dir in results:
        print(f"{model:<15} {thresh:<12} {count:<10} {output_dir.name}")
    
    print("\n" + "-" * 70)
    print(f"All outputs in: {OUTPUT_BASE}")
    print("-" * 70)
    print("\nTIP: Check the 'magenta_overlays' folder in each output to see detections.")
    
    print("\n[Done]")


if __name__ == "__main__":
    main()
