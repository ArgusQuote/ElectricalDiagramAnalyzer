#!/usr/bin/env python3
"""
Render PDF pages to images for annotation with Label Studio.

This script takes a directory of PDFs and renders each page as a PNG image
suitable for object detection annotation.

Usage:
    python render_pdfs_for_annotation.py --input /path/to/pdfs --output /path/to/images
    python render_pdfs_for_annotation.py --input /path/to/single.pdf --output /path/to/images
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from PIL import Image
import numpy as np


def render_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 200,
    max_dimension: Optional[int] = 2000,
    verbose: bool = True,
) -> list[str]:
    """
    Render all pages of a PDF to PNG images.
    
    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save output images.
        dpi: Resolution for rendering (200-300 recommended for annotation).
        max_dimension: Optional max width/height to cap image size.
        verbose: Print progress information.
    
    Returns:
        List of paths to the generated PNG files.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    doc = pdfium.PdfDocument(str(pdf_path))
    base_name = pdf_path.stem
    output_paths = []
    
    scale = dpi / 72.0
    
    if verbose:
        print(f"[INFO] Rendering {pdf_path.name} ({len(doc)} pages) at {dpi} DPI")
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        
        # Render to bitmap
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        
        # Optionally resize if image is too large
        if max_dimension:
            width, height = pil_image.size
            if width > max_dimension or height > max_dimension:
                ratio = min(max_dimension / width, max_dimension / height)
                new_size = (int(width * ratio), int(height * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                if verbose:
                    print(f"  [RESIZE] Page {page_idx + 1}: {width}x{height} -> {new_size[0]}x{new_size[1]}")
        
        # Convert to RGB if necessary (remove alpha channel)
        if pil_image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Generate output filename
        output_name = f"{base_name}_page{page_idx + 1:03d}.png"
        output_path = output_dir / output_name
        
        pil_image.save(str(output_path), 'PNG')
        output_paths.append(str(output_path))
        
        if verbose:
            print(f"  [SAVED] {output_name} ({pil_image.size[0]}x{pil_image.size[1]})")
    
    doc.close()
    return output_paths


def render_directory(
    input_dir: str,
    output_dir: str,
    dpi: int = 200,
    max_dimension: Optional[int] = 2000,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """
    Render all PDFs in a directory to images.
    
    Args:
        input_dir: Directory containing PDF files.
        output_dir: Directory to save output images.
        dpi: Resolution for rendering.
        max_dimension: Optional max width/height.
        verbose: Print progress information.
    
    Returns:
        Dictionary mapping PDF filenames to lists of output image paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    
    if not pdf_files:
        print(f"[WARN] No PDF files found in {input_dir}")
        return {}
    
    if verbose:
        print(f"[INFO] Found {len(pdf_files)} PDF files")
    
    results = {}
    for pdf_path in sorted(pdf_files):
        try:
            images = render_pdf_to_images(
                pdf_path=str(pdf_path),
                output_dir=str(output_dir),
                dpi=dpi,
                max_dimension=max_dimension,
                verbose=verbose,
            )
            results[pdf_path.name] = images
        except Exception as e:
            print(f"[ERROR] Failed to render {pdf_path.name}: {e}")
            results[pdf_path.name] = []
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Render PDF pages to images for ML annotation"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for rendered images"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering DPI (default: 200)"
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=2000,
        help="Max image dimension in pixels (default: 2000, use 0 for no limit)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    max_dim = args.max_dimension if args.max_dimension > 0 else None
    verbose = not args.quiet
    
    if input_path.is_file():
        # Single PDF
        images = render_pdf_to_images(
            pdf_path=str(input_path),
            output_dir=args.output,
            dpi=args.dpi,
            max_dimension=max_dim,
            verbose=verbose,
        )
        print(f"\n[DONE] Rendered {len(images)} pages")
    elif input_path.is_dir():
        # Directory of PDFs
        results = render_directory(
            input_dir=str(input_path),
            output_dir=args.output,
            dpi=args.dpi,
            max_dimension=max_dim,
            verbose=verbose,
        )
        total_images = sum(len(imgs) for imgs in results.values())
        print(f"\n[DONE] Rendered {total_images} pages from {len(results)} PDFs")
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
