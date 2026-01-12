#!/usr/bin/env python3
"""
Table detection using Microsoft Table Transformer (TATR).

Processes all PDFs in input folder, detects tables using ML,
and outputs cropped tables + annotated debug images.

License: MIT (TATR model is MIT licensed)
"""

import sys
import os
from pathlib import Path
import json

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection


class TableTransformerDetector:
    """
    Detect tables in PDF documents using Microsoft's Table Transformer.
    
    Responsibilities:
    - Load and manage the TATR model
    - Convert PDF pages to images
    - Run table detection inference
    - Crop and save detected tables
    - Generate debug visualizations
    
    Dependencies: transformers, torch, PyMuPDF, OpenCV, PIL
    """
    
    def __init__(
        self,
        output_dir: str,
        detection_threshold: float = 0.5,
        render_dpi: int = 200,
        crop_dpi: int = 300,
        padding: int = 10,
        verbose: bool = True,
        debug: bool = True,
    ):
        """
        Initialize the Table Transformer detector.
        
        Args:
            output_dir: Directory to save outputs
            detection_threshold: Confidence threshold for detections (0.0-1.0)
            render_dpi: DPI for rendering pages for detection
            crop_dpi: DPI for final cropped table images
            padding: Pixels to add around detected tables
            verbose: Print progress information
            debug: Save annotated debug images
        """
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.detection_threshold = detection_threshold
        self.render_dpi = render_dpi
        self.crop_dpi = crop_dpi
        self.padding = padding
        self.verbose = verbose
        self.debug = debug
        
        # Create subdirectories
        self.tables_dir = self.output_dir / "detected_tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        if self.debug:
            self.debug_dir = self.output_dir / "debug_overlays"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        if self.verbose:
            print("[INFO] Loading Microsoft Table Transformer model...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose:
            print(f"[INFO] Using device: {self.device}")
        
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        self.model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        ).to(self.device)
        self.model.eval()
        
        if self.verbose:
            print("[INFO] Model loaded successfully")
    
    def process_folder(self, input_folder: str) -> dict:
        """
        Process all PDFs in a folder.
        
        Args:
            input_folder: Path to folder containing PDFs
            
        Returns:
            Dict with processing results per file
        """
        input_path = Path(input_folder).expanduser()
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input folder not found: {input_path}")
        
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"[WARN] No PDF files found in {input_path}")
            return {}
        
        if self.verbose:
            print(f"[INFO] Found {len(pdf_files)} PDF files to process")
        
        results = {}
        total_tables = 0
        
        for pdf_path in pdf_files:
            try:
                file_results = self.process_pdf(str(pdf_path))
                results[pdf_path.name] = file_results
                total_tables += file_results["total_tables"]
            except Exception as e:
                print(f"[ERROR] Failed to process {pdf_path.name}: {e}")
                results[pdf_path.name] = {"error": str(e)}
        
        if self.verbose:
            print(f"\n[DONE] Processed {len(pdf_files)} PDFs, found {total_tables} tables total")
            print(f"       Tables saved to: {self.tables_dir}")
            if self.debug:
                print(f"       Debug images saved to: {self.debug_dir}")
        
        # Save summary JSON
        summary_path = self.output_dir / "detection_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def process_pdf(self, pdf_path: str) -> dict:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict with detection results
        """
        pdf_path = Path(pdf_path).expanduser()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        base_name = pdf_path.stem
        
        if self.verbose:
            print(f"\n[INFO] Processing: {pdf_path.name}")
        
        doc = fitz.open(str(pdf_path))
        
        results = {
            "file": str(pdf_path),
            "pages": len(doc),
            "total_tables": 0,
            "detections": []
        }
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_results = self._process_page(doc, page, page_idx, base_name)
            results["detections"].append(page_results)
            results["total_tables"] += len(page_results["tables"])
        
        doc.close()
        
        if self.verbose:
            print(f"[INFO] {pdf_path.name}: Found {results['total_tables']} table(s) across {results['pages']} page(s)")
        
        return results
    
    def _process_page(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_idx: int,
        base_name: str
    ) -> dict:
        """
        Process a single page for table detection.
        
        Args:
            doc: The fitz Document
            page: The fitz Page object
            page_idx: Page index (0-based)
            base_name: Base name for output files
            
        Returns:
            Dict with page detection results
        """
        # Render page at detection DPI
        det_mat = fitz.Matrix(self.render_dpi / 72.0, self.render_dpi / 72.0)
        pix = page.get_pixmap(matrix=det_mat, alpha=False)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # Run detection
        detections = self._detect_tables(pil_image)
        
        page_results = {
            "page": page_idx + 1,
            "image_size": [pil_image.width, pil_image.height],
            "tables": []
        }
        
        # Convert PIL to OpenCV for cropping and debug
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        debug_image = cv_image.copy() if self.debug else None
        
        H, W = cv_image.shape[:2]
        
        # Process each detection
        for det_idx, detection in enumerate(detections):
            box = detection["box"]
            score = detection["score"]
            label = detection["label"]
            
            # Add padding
            x0 = max(0, int(box[0]) - self.padding)
            y0 = max(0, int(box[1]) - self.padding)
            x1 = min(W, int(box[2]) + self.padding)
            y1 = min(H, int(box[3]) + self.padding)
            
            # Skip if box is too small
            if (x1 - x0) < 50 or (y1 - y0) < 50:
                continue
            
            # Crop and save table at higher DPI
            table_crop = self._crop_table_high_dpi(
                doc, page, box, page_idx, base_name, det_idx + 1
            )
            
            table_info = {
                "index": det_idx + 1,
                "label": label,
                "confidence": round(score, 4),
                "box": [x0, y0, x1, y1],
                "output_file": table_crop
            }
            page_results["tables"].append(table_info)
            
            # Draw on debug image
            if self.debug and debug_image is not None:
                color = (0, 255, 0)  # Green
                cv2.rectangle(debug_image, (x0, y0), (x1, y1), color, 3)
                
                # Label with confidence
                label_text = f"{label}: {score:.2f}"
                font_scale = 0.8
                thickness = 2
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                cv2.rectangle(debug_image, (x0, y0 - th - 10), (x0 + tw + 10, y0), color, -1)
                cv2.putText(debug_image, label_text, (x0 + 5, y0 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Save debug image
        if self.debug and debug_image is not None:
            debug_path = self.debug_dir / f"{base_name}_page{page_idx + 1:03d}_detections.png"
            cv2.imwrite(str(debug_path), debug_image)
        
        return page_results
    
    def _detect_tables(self, image: Image.Image) -> list[dict]:
        """
        Run table detection on a PIL image.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            List of detection dicts with box, score, label
        """
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([[image.height, image.width]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.detection_threshold,
            target_sizes=target_sizes
        )[0]
        
        detections = []
        for score, label_id, box in zip(
            results["scores"].cpu(),
            results["labels"].cpu(),
            results["boxes"].cpu()
        ):
            label_name = self.model.config.id2label[label_id.item()]
            detections.append({
                "box": box.tolist(),
                "score": score.item(),
                "label": label_name
            })
        
        return detections
    
    def _crop_table_high_dpi(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        detection_box: list,
        page_idx: int,
        base_name: str,
        table_idx: int
    ) -> str:
        """
        Crop a detected table at high DPI for quality output.
        
        Args:
            doc: The fitz Document
            page: The fitz Page
            detection_box: [x0, y0, x1, y1] in detection image coordinates
            page_idx: Page index
            base_name: Base name for output file
            table_idx: Table index on this page
            
        Returns:
            Path to saved cropped image
        """
        # Convert detection coordinates back to PDF coordinates
        det_scale = self.render_dpi / 72.0
        pdf_box = [
            detection_box[0] / det_scale,
            detection_box[1] / det_scale,
            detection_box[2] / det_scale,
            detection_box[3] / det_scale
        ]
        
        # Add padding in PDF coordinates
        pad_pdf = self.padding / det_scale
        pr = page.rect
        clip = fitz.Rect(
            max(pr.x0, pdf_box[0] - pad_pdf),
            max(pr.y0, pdf_box[1] - pad_pdf),
            min(pr.x1, pdf_box[2] + pad_pdf),
            min(pr.y1, pdf_box[3] + pad_pdf)
        )
        
        # Render at crop DPI
        crop_mat = fitz.Matrix(self.crop_dpi / 72.0, self.crop_dpi / 72.0)
        pix = page.get_pixmap(matrix=crop_mat, clip=clip, alpha=False)
        
        # Save
        output_path = self.tables_dir / f"{base_name}_page{page_idx + 1:03d}_table{table_idx:02d}.png"
        pix.save(str(output_path))
        
        return str(output_path)


def main():
    """Main entry point."""
    # Configuration
    INPUT_FOLDER = Path("~/Documents/pdfToScan").expanduser()
    OUTPUT_DIR = Path("~/Documents/pdfScanTest9").expanduser()
    
    # Create detector
    detector = TableTransformerDetector(
        output_dir=OUTPUT_DIR,
        detection_threshold=0.5,  # Lower = more detections, Higher = fewer but more confident
        render_dpi=200,           # DPI for detection (200 is good balance of speed/accuracy)
        crop_dpi=300,             # DPI for output crops (higher = better quality)
        padding=15,               # Pixels padding around detected tables
        verbose=True,
        debug=True,               # Save annotated images showing detections
    )
    
    # Process all PDFs
    results = detector.process_folder(str(INPUT_FOLDER))
    
    # Print summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    for filename, file_results in results.items():
        if "error" in file_results:
            print(f"  {filename}: ERROR - {file_results['error']}")
        else:
            print(f"  {filename}: {file_results['total_tables']} table(s) found")
    print("="*60)


if __name__ == "__main__":
    main()
