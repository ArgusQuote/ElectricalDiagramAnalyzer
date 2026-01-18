"""
MLTableDetection - Machine learning tools for table detection.

Uses Table Transformer (MIT License) as the primary detector.
Pretrained on 1M+ table images - works zero-shot or with minimal fine-tuning.

LICENSE: All components are commercial-friendly (MIT, Apache 2.0, BSD)

This module provides:
- TableDetectorML: ML-based table detector (drop-in replacement for PanelBoardSearch)
- render_pdfs_for_annotation.py: Render PDFs to images for annotation
- train_table_transformer.py: Fine-tuning script for Table Transformer
- evaluate_model.py: Evaluation and comparison scripts

Quick Start (Zero-Shot - No Training Needed):
    from MLTableDetection import TableDetectorML
    
    detector = TableDetectorML(output_dir="./output")
    images = detector.readPdf("document.pdf")
"""

from MLTableDetection.TableDetectorML import TableDetectorML, PanelBoardSearch

__all__ = ["TableDetectorML", "PanelBoardSearch"]
