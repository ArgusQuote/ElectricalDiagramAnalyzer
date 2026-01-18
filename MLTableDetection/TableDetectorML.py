#!/usr/bin/env python3
"""
ML-based table detector using Table Transformer (TATR) or Detectron2.

Table Transformer is MIT licensed and pretrained on 1M+ table images,
making it ideal for table detection with minimal or no fine-tuning.

Detectron2 (Apache 2.0) is available as a fallback for custom training.

Usage:
    from MLTableDetection.TableDetectorML import TableDetectorML
    
    # Zero-shot (no training needed - uses pretrained TATR)
    detector = TableDetectorML(output_dir="/path/to/output")
    
    # With custom model
    detector = TableDetectorML(
        output_dir="/path/to/output",
        model_path="/path/to/fine-tuned/model"
    )
    
    panel_images = detector.readPdf("/path/to/electrical.pdf")
"""

import os
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import cv2
import pypdfium2 as pdfium
import pikepdf
from PIL import Image


class TableDetectorML:
    """
    ML-based table detector using Table Transformer (TATR).
    
    Table Transformer is pretrained on PubTables-1M dataset (1M+ table images)
    and works well zero-shot or with minimal fine-tuning.
    
    License: MIT (commercial-friendly)
    
    Provides the same `readPdf()` API as PanelBoardSearch for compatibility
    with the existing pipeline.
    """
    
    # Available backends
    BACKEND_TATR = "table-transformer"
    BACKEND_DETECTRON2 = "detectron2"
    
    # Default Hugging Face model for Table Transformer
    DEFAULT_TATR_MODEL = "microsoft/table-transformer-detection"
    
    def __init__(
        self,
        output_dir: str,
        model_path: Optional[str] = None,
        backend: Literal["table-transformer", "detectron2"] = "table-transformer",
        # Detection settings
        dpi: int = 400,
        render_dpi: int = 1200,
        render_colorspace: str = "gray",
        downsample_max_w: Optional[int] = None,
        # ML model settings
        conf_threshold: float = 0.5,  # TATR is more confident, use higher threshold
        # Post-processing
        pad: int = 6,
        enforce_one_box: bool = True,
        min_area_fraction: float = 0.004,
        max_area_fraction: float = 0.30,
        # Behavior
        device: Optional[str] = None,  # "cuda", "cpu", or None for auto
        debug: bool = False,
        verbose: bool = True,
        # API compatibility - accept but ignore legacy params
        **kwargs,
    ):
        """
        Initialize the ML table detector.
        
        Args:
            output_dir: Directory to save output files.
            model_path: Path to model. For TATR, can be HuggingFace model ID
                        or local path. If None, uses pretrained TATR.
            backend: Detection backend ("table-transformer" or "detectron2").
            dpi: Detection DPI (for rendering PDF pages).
            render_dpi: Output PNG DPI (higher quality for OCR).
            render_colorspace: Output colorspace ("gray" or "rgb").
            downsample_max_w: Max width for output PNGs (None for no limit).
            conf_threshold: Minimum confidence for detections.
            pad: Padding around detected regions (pixels at detection DPI).
            enforce_one_box: Split crops containing multiple tables.
            min_area_fraction: Minimum detection area as fraction of page.
            max_area_fraction: Maximum detection area as fraction of page.
            device: Device for inference ("cuda", "cpu", or None for auto).
            debug: Save debug visualizations.
            verbose: Print progress information.
        """
        self.output_dir = os.path.expanduser(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Output subdirectories (matching PanelBoardSearch structure)
        self.magenta_dir = Path(self.output_dir) / "magenta_overlays"
        self.magenta_dir.mkdir(parents=True, exist_ok=True)
        self.vec_dir = Path(self.output_dir) / "cropped_tables_pdf"
        self.vec_dir.mkdir(parents=True, exist_ok=True)
        self.raster_dir = Path(self.output_dir) / "raster_images"
        self.raster_dir.mkdir(parents=True, exist_ok=True)
        
        # Settings
        self.backend = backend
        self.dpi = dpi
        self.render_dpi = render_dpi
        self.render_colorspace = render_colorspace.lower().strip()
        self.downsample_max_w = downsample_max_w
        self.conf_threshold = conf_threshold
        self.pad = pad
        self.enforce_one_box = enforce_one_box
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.debug = debug
        self.verbose = verbose
        
        # Device selection
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Model path
        self.model_path = model_path
        if model_path is None and backend == self.BACKEND_TATR:
            self.model_path = self.DEFAULT_TATR_MODEL
        
        # Load model based on backend
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the detection model based on selected backend."""
        if self.backend == self.BACKEND_TATR:
            self._load_tatr_model()
        elif self.backend == self.BACKEND_DETECTRON2:
            self._load_detectron2_model()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _load_tatr_model(self):
        """Load Table Transformer model from Hugging Face."""
        try:
            from transformers import TableTransformerForObjectDetection, AutoImageProcessor
            import torch
            
            if self.verbose:
                print(f"[INFO] Loading Table Transformer: {self.model_path}")
                print(f"[INFO] Device: {self.device}")
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_path)
            self.model = TableTransformerForObjectDetection.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            if self.verbose:
                print(f"[INFO] Model loaded successfully (MIT License)")
                
        except ImportError:
            print("[ERROR] transformers not installed. Run: pip install transformers")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load Table Transformer: {e}")
            raise
    
    def _load_detectron2_model(self):
        """Load Detectron2 model (Apache 2.0 license)."""
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2 import model_zoo
            
            if self.verbose:
                print(f"[INFO] Loading Detectron2 model")
            
            cfg = get_cfg()
            
            if self.model_path and Path(self.model_path).exists():
                # Load custom config/weights
                cfg.merge_from_file(self.model_path)
            else:
                # Use pretrained Faster R-CNN
                cfg.merge_from_file(model_zoo.get_config_file(
                    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                ))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                )
            
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_threshold
            cfg.MODEL.DEVICE = self.device
            
            self.model = DefaultPredictor(cfg)
            
            if self.verbose:
                print(f"[INFO] Detectron2 model loaded (Apache 2.0 License)")
                
        except ImportError:
            print("[ERROR] detectron2 not installed. See: https://detectron2.readthedocs.io/")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load Detectron2: {e}")
            raise
    
    def _detect_tables_tatr(self, image: Image.Image) -> list[dict]:
        """
        Run Table Transformer detection on an image.
        
        Args:
            image: PIL Image to detect tables in.
        
        Returns:
            List of detection dicts with 'bbox', 'confidence', 'label'.
        """
        import torch
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.conf_threshold,
            target_sizes=target_sizes
        )[0]
        
        detections = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy()
        ):
            # TATR labels: 0=table, 1=table rotated
            # We want both
            detections.append({
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "confidence": float(score),
                "label": self.model.config.id2label.get(int(label), "table"),
            })
        
        return detections
    
    def _detect_tables_detectron2(self, image: np.ndarray) -> list[dict]:
        """
        Run Detectron2 detection on an image.
        
        Args:
            image: BGR numpy array.
        
        Returns:
            List of detection dicts with 'bbox', 'confidence', 'label'.
        """
        outputs = self.model(image)
        instances = outputs["instances"].to("cpu")
        
        detections = []
        for i in range(len(instances)):
            box = instances.pred_boxes[i].tensor.numpy()[0]
            score = float(instances.scores[i])
            label = int(instances.pred_classes[i])
            
            detections.append({
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "confidence": score,
                "label": f"class_{label}",
            })
        
        return detections
    
    def readPdf(self, pdf_path: str) -> list[str]:
        """
        Detect tables in a PDF and extract high-quality image crops.
        
        This is the main entry point, matching the PanelBoardSearch API.
        
        Args:
            pdf_path: Path to the PDF file.
        
        Returns:
            List of paths to the generated PNG files.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")
        
        pdf_path_str = os.path.expanduser(pdf_path)
        if not Path(pdf_path_str).is_file():
            raise FileNotFoundError(pdf_path_str)
        
        doc = pdfium.PdfDocument(pdf_path_str)
        base = Path(pdf_path_str).stem
        all_pngs: list[str] = []
        
        if self.verbose:
            backend_name = "Table Transformer" if self.backend == self.BACKEND_TATR else "Detectron2"
            print(f"[INFO] ML Detection with {backend_name} @ {self.dpi} DPI")
            print(f"[INFO] Pages: {len(doc)}")
        
        det_scale = self.dpi / 72.0
        rend_scale = self.render_dpi / 72.0
        
        for pidx in range(len(doc)):
            page = doc[pidx]
            page_width_pt = page.get_width()
            page_height_pt = page.get_height()
            
            # Render page at detection DPI
            bitmap = page.render(scale=det_scale)
            det_pil = bitmap.to_pil()
            
            # Convert to RGB for detection
            if det_pil.mode == 'RGBA':
                background = Image.new('RGB', det_pil.size, (255, 255, 255))
                background.paste(det_pil, mask=det_pil.split()[3])
                det_pil = background
            elif det_pil.mode != 'RGB':
                det_pil = det_pil.convert('RGB')
            
            det_bgr = cv2.cvtColor(np.array(det_pil), cv2.COLOR_RGB2BGR)
            H, W = det_bgr.shape[:2]
            page_area = H * W
            
            # Run detection based on backend
            if self.backend == self.BACKEND_TATR:
                detections = self._detect_tables_tatr(det_pil)
            else:
                detections = self._detect_tables_detectron2(det_bgr)
            
            # Create overlay image for visualization
            overlay_img = det_bgr.copy()
            
            # Process detections
            candidates = []  # (x0, y0, x1, y1) in PDF points
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = det["confidence"]
                label = det["label"]
                
                # Calculate area
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                # Filter by area
                if area / page_area < self.min_area_fraction:
                    if self.verbose:
                        print(f"  [SKIP] Detection too small: {w}x{h}")
                    continue
                if area / page_area > self.max_area_fraction:
                    if self.verbose:
                        print(f"  [SKIP] Detection too large: {w}x{h}")
                    continue
                
                # Draw on overlay (green for detections)
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 8)
                cv2.putText(
                    overlay_img,
                    f"{label}: {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                
                # Add padding
                px0 = max(0, x1 - self.pad)
                py0 = max(0, y1 - self.pad)
                px1 = min(W, x2 + self.pad)
                py1 = min(H, y2 + self.pad)
                
                # Convert to PDF coordinates
                x0f, y0f = px0 / W, py0 / H
                x1f, y1f = px1 / W, py1 / H
                clip = (
                    page_width_pt * x0f,
                    page_height_pt * y0f,
                    page_width_pt * x1f,
                    page_height_pt * y1f,
                )
                candidates.append(clip)
            
            # Export crops
            saved_idx = 0
            for clip in candidates:
                saved_idx += 1
                clip_x0, clip_y0, clip_x1, clip_y1 = clip
                
                # Vector crop PDF using pikepdf
                pdf_path_out = self.vec_dir / f"{base}_page{pidx+1:03d}_panel{saved_idx:02d}.pdf"
                try:
                    with pikepdf.open(pdf_path_str) as src_pdf:
                        dst_pdf = pikepdf.new()
                        src_page = src_pdf.pages[pidx]
                        # PDF coordinates: origin at bottom-left
                        src_page.mediabox = pikepdf.Array([
                            clip_x0,
                            page_height_pt - clip_y1,
                            clip_x1,
                            page_height_pt - clip_y0
                        ])
                        src_page.cropbox = pikepdf.Array([
                            clip_x0,
                            page_height_pt - clip_y1,
                            clip_x1,
                            page_height_pt - clip_y0
                        ])
                        dst_pdf.pages.append(src_page)
                        dst_pdf.save(str(pdf_path_out))
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Failed to create vector PDF: {e}")
                
                # High-DPI PNG from the same clip
                render_bitmap = page.render(scale=rend_scale)
                render_pil = render_bitmap.to_pil()
                if self.render_colorspace == "gray":
                    render_pil = render_pil.convert("L")
                
                # Calculate clip coordinates in render pixels
                rx0 = int(clip_x0 * rend_scale)
                ry0 = int(clip_y0 * rend_scale)
                rx1 = int(clip_x1 * rend_scale)
                ry1 = int(clip_y1 * rend_scale)
                
                # Crop the rendered image
                cropped_pil = render_pil.crop((rx0, ry0, rx1, ry1))
                if cropped_pil.mode == "RGB":
                    png = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                else:
                    png = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_GRAY2BGR)
                
                # Optional downsampling
                if self.downsample_max_w and png.shape[1] > self.downsample_max_w:
                    scale_factor = self.downsample_max_w / float(png.shape[1])
                    new_wh = (
                        self.downsample_max_w,
                        max(1, int(round(png.shape[0] * scale_factor)))
                    )
                    png = cv2.resize(png, new_wh, interpolation=cv2.INTER_AREA)
                
                png_path = Path(self.output_dir) / f"{base}_page{pidx+1:03d}_panel{saved_idx:02d}.png"
                cv2.imwrite(str(png_path), png)
                all_pngs.append(str(png_path))
            
            # Save overlay image
            ov_path = self.magenta_dir / f"{base}_page{pidx+1:03d}_void_perimeters.png"
            cv2.imwrite(str(ov_path), overlay_img)
            
            if self.verbose:
                print(f"[INFO] Page {pidx+1}: saved {saved_idx} crop(s)")
        
        doc.close()
        
        # Post-processing: enforce one table per PNG
        if self.enforce_one_box and all_pngs:
            all_pngs = self._enforce_one_box_on_paths(all_pngs)
        
        if self.verbose:
            print(f"[DONE] Outputs → {self.output_dir}")
            print(f"      Vector PDFs → {self.vec_dir}")
            print(f"      Overlays    → {self.magenta_dir}")
        
        return all_pngs
    
    def _enforce_one_box_on_paths(self, image_paths: list[str]) -> list[str]:
        """
        Ensure each output PNG contains only one table.
        
        If a crop contains multiple tables, split it into separate files.
        """
        final_paths = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Convert to PIL for TATR
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Run detection on the cropped image
            if self.backend == self.BACKEND_TATR:
                detections = self._detect_tables_tatr(img_pil)
            else:
                detections = self._detect_tables_detectron2(img)
            
            if len(detections) <= 1:
                # Single or no detection - keep as is
                final_paths.append(img_path)
            else:
                # Multiple detections - split into separate files
                base_path = Path(img_path)
                stem = base_path.stem
                suffix = base_path.suffix
                parent = base_path.parent
                
                if self.verbose:
                    print(f"[INFO] Splitting {base_path.name}: {len(detections)} tables")
                
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    
                    # Add padding
                    h, w = img.shape[:2]
                    px1 = max(0, x1 - self.pad)
                    py1 = max(0, y1 - self.pad)
                    px2 = min(w, x2 + self.pad)
                    py2 = min(h, y2 + self.pad)
                    
                    crop = img[py1:py2, px1:px2]
                    new_path = parent / f"{stem}_split{i+1:02d}{suffix}"
                    cv2.imwrite(str(new_path), crop)
                    final_paths.append(str(new_path))
                
                # Remove original multi-table image
                Path(img_path).unlink(missing_ok=True)
        
        return final_paths


# Alias for API compatibility with existing code
PanelBoardSearch = TableDetectorML
