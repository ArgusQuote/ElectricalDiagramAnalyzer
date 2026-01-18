#!/usr/bin/env python3
"""
Evaluate a trained YOLOv8 table detection model.

This script runs evaluation on the validation set and generates
detailed metrics, visualizations, and comparison with ground truth.

Usage:
    python evaluate_model.py --model best.pt --data data.yaml
    python evaluate_model.py --model best.pt --images /path/to/test/images
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import json


def evaluate_on_dataset(
    model_path: str,
    data_yaml: str,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Evaluate model on the validation set defined in data.yaml.
    
    Args:
        model_path: Path to trained model weights (.pt file).
        data_yaml: Path to data.yaml configuration.
        output_dir: Directory to save evaluation results.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for mAP calculation.
        verbose: Print evaluation progress.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        raise
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data config not found: {data_yaml}")
    
    # Set up output directory
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = model_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[INFO] Loading model: {model_path}")
    
    model = YOLO(str(model_path))
    
    if verbose:
        print(f"[INFO] Evaluating on dataset: {data_yaml}")
    
    # Run validation
    metrics = model.val(
        data=str(data_yaml),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=verbose,
        save_json=True,
        project=str(output_dir),
        name="val_results",
        exist_ok=True,
    )
    
    # Extract key metrics
    results = {
        'model': str(model_path),
        'dataset': str(data_yaml),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'metrics': {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        },
        'per_class': {},
    }
    
    # Per-class metrics if available
    if hasattr(metrics.box, 'ap_class_index'):
        class_names = model.names
        for i, class_idx in enumerate(metrics.box.ap_class_index):
            class_name = class_names[class_idx]
            results['per_class'][class_name] = {
                'AP50': float(metrics.box.ap50[i]),
                'AP': float(metrics.box.ap[i]),
            }
    
    # Save results
    results_file = output_dir / "val_results" / "metrics.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Model: {model_path.name}")
        print(f"mAP@0.5:      {results['metrics']['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {results['metrics']['mAP50-95']:.4f}")
        print(f"Precision:    {results['metrics']['precision']:.4f}")
        print(f"Recall:       {results['metrics']['recall']:.4f}")
        
        if results['per_class']:
            print(f"\nPer-class AP@0.5:")
            for class_name, class_metrics in results['per_class'].items():
                print(f"  {class_name}: {class_metrics['AP50']:.4f}")
        
        print(f"\nResults saved to: {results_file}")
    
    return results


def run_inference_on_images(
    model_path: str,
    images_dir: str,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.25,
    save_visualizations: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Run inference on a directory of images and save results.
    
    Args:
        model_path: Path to trained model weights.
        images_dir: Directory containing test images.
        output_dir: Directory to save results.
        conf_threshold: Confidence threshold for detections.
        save_visualizations: Save images with drawn bounding boxes.
        verbose: Print progress.
    
    Returns:
        List of detection results per image.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        raise
    
    model_path = Path(model_path)
    images_dir = Path(images_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Set up output directory
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = images_dir.parent / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"[WARN] No images found in {images_dir}")
        return []
    
    if verbose:
        print(f"[INFO] Loading model: {model_path}")
        print(f"[INFO] Found {len(image_files)} images")
    
    model = YOLO(str(model_path))
    
    all_results = []
    
    for img_path in sorted(image_files):
        if verbose:
            print(f"  Processing: {img_path.name}")
        
        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        for result in results:
            detections = []
            
            for box in result.boxes:
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    'bbox_normalized': box.xyxyn[0].tolist(),
                }
                detections.append(detection)
            
            image_result = {
                'image': str(img_path),
                'image_name': img_path.name,
                'detections': detections,
                'num_detections': len(detections),
            }
            all_results.append(image_result)
            
            # Save visualization
            if save_visualizations:
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f"{img_path.stem}_detected.jpg"
                result.save(str(vis_path))
    
    # Save all results to JSON
    results_file = output_dir / "detections.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary statistics
    total_detections = sum(r['num_detections'] for r in all_results)
    images_with_detections = sum(1 for r in all_results if r['num_detections'] > 0)
    
    if verbose:
        print(f"\n{'='*50}")
        print("INFERENCE SUMMARY")
        print(f"{'='*50}")
        print(f"Total images:           {len(all_results)}")
        print(f"Images with detections: {images_with_detections}")
        print(f"Total detections:       {total_detections}")
        print(f"Avg detections/image:   {total_detections/len(all_results):.2f}")
        print(f"\nResults saved to: {results_file}")
        if save_visualizations:
            print(f"Visualizations in: {output_dir / 'visualizations'}")
    
    return all_results


def compare_with_heuristic(
    model_path: str,
    pdf_path: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Compare ML model detections with the heuristic PanelBoardSearch.
    
    Args:
        model_path: Path to trained model weights.
        pdf_path: Path to a test PDF file.
        output_dir: Directory to save comparison results.
        verbose: Print progress.
    
    Returns:
        Dictionary with comparison metrics.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from ultralytics import YOLO
        from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch
        import pypdfium2 as pdfium
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        raise
    
    model_path = Path(model_path)
    pdf_path = Path(pdf_path)
    
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[INFO] Comparing ML vs Heuristic on: {pdf_path.name}")
    
    # Run heuristic detector
    heuristic_dir = output_dir / "heuristic"
    heuristic_detector = PanelBoardSearch(
        output_dir=str(heuristic_dir),
        dpi=400,
        verbose=False,
    )
    heuristic_results = heuristic_detector.readPdf(str(pdf_path))
    
    # Run ML detector (on rendered pages)
    model = YOLO(str(model_path))
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = 400 / 72.0  # Match heuristic DPI
    
    ml_results = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        bitmap = page.render(scale=scale)
        img_array = np.array(bitmap.to_pil())
        
        # Convert to BGR for YOLO
        if len(img_array.shape) == 2:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        results = model(img_bgr, verbose=False)
        for r in results:
            for box in r.boxes:
                ml_results.append({
                    'page': page_idx + 1,
                    'bbox': box.xyxy[0].tolist(),
                    'confidence': float(box.conf[0]),
                    'class': model.names[int(box.cls[0])],
                })
    
    doc.close()
    
    comparison = {
        'pdf': str(pdf_path),
        'heuristic': {
            'num_detections': len(heuristic_results),
            'output_files': heuristic_results,
        },
        'ml_model': {
            'num_detections': len(ml_results),
            'detections': ml_results,
        },
    }
    
    # Save comparison
    comparison_file = output_dir / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    if verbose:
        print(f"\n{'='*50}")
        print("COMPARISON RESULTS")
        print(f"{'='*50}")
        print(f"Heuristic detections: {len(heuristic_results)}")
        print(f"ML model detections:  {len(ml_results)}")
        print(f"\nResults saved to: {comparison_file}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained table detection model"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained model weights (.pt file)"
    )
    parser.add_argument(
        "--data", "-d",
        help="Path to data.yaml for validation set evaluation"
    )
    parser.add_argument(
        "--images", "-i",
        help="Directory of images for inference"
    )
    parser.add_argument(
        "--pdf",
        help="PDF file for comparison with heuristic detector"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for results"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for mAP (default: 0.5)"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Don't save visualization images"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.data:
        evaluate_on_dataset(
            model_path=args.model,
            data_yaml=args.data,
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            verbose=verbose,
        )
    elif args.images:
        run_inference_on_images(
            model_path=args.model,
            images_dir=args.images,
            output_dir=args.output,
            conf_threshold=args.conf,
            save_visualizations=not args.no_visualizations,
            verbose=verbose,
        )
    elif args.pdf:
        compare_with_heuristic(
            model_path=args.model,
            pdf_path=args.pdf,
            output_dir=args.output,
            verbose=verbose,
        )
    else:
        print("[ERROR] Must provide --data, --images, or --pdf")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
