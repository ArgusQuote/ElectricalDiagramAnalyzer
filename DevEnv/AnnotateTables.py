#!/usr/bin/env python3
"""
Simple Table Annotation Tool (Matplotlib version)

Draw bounding boxes on images to create training data.
Works in any environment with matplotlib.

Usage:
    python DevEnv/AnnotateTables.py
"""

import os
import sys
import json
from pathlib import Path

# Force interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RectangleSelector
from PIL import Image

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- CONFIG ----------
# All annotation data stored in ~/Documents/TableAnnotations
TABLE_ANNOTATIONS_ROOT = Path("~/Documents/TableAnnotations").expanduser()
IMAGES_DIR = TABLE_ANNOTATIONS_ROOT / "images"
ANNOTATIONS_FILE = TABLE_ANNOTATIONS_ROOT / "annotations/annotations_coco.json"
MODELS_DIR = TABLE_ANNOTATIONS_ROOT / "models"


class TableAnnotator:
    """Matplotlib-based annotation tool."""
    
    def __init__(self, images_dir: Path, output_file: Path):
        self.images_dir = Path(images_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        self.image_files = sorted(
            list(self.images_dir.glob("*.png")) + 
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.jpeg"))
        )
        
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {images_dir}")
        
        # Load existing annotations or create new
        self.annotations = self._load_or_create_coco()
        
        # State
        self.current_idx = 0
        self.current_boxes = []
        self.rect_patches = []
        
        # Setup figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        plt.subplots_adjust(bottom=0.15)
        
        # Buttons
        ax_prev = plt.axes([0.2, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.35, 0.02, 0.1, 0.05])
        ax_undo = plt.axes([0.50, 0.02, 0.1, 0.05])
        ax_save = plt.axes([0.65, 0.02, 0.1, 0.05])
        
        self.btn_prev = Button(ax_prev, '← Prev')
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_save = Button(ax_save, 'Save')
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_undo.on_clicked(self._on_undo)
        self.btn_save.on_clicked(self._on_save)
        
        # Rectangle selector
        self.selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False
        )
        
    def _load_or_create_coco(self) -> dict:
        """Load existing COCO annotations or create empty structure."""
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                return json.load(f)
        
        return {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "table", "supercategory": ""}]
        }
    
    def _get_image_id(self, filename: str) -> int:
        """Get or create image ID for filename."""
        for img in self.annotations["images"]:
            if img["file_name"] == filename:
                return img["id"]
        return len(self.annotations["images"]) + 1
    
    def _get_boxes_for_image(self, image_id: int) -> list:
        """Get existing boxes for an image."""
        boxes = []
        for ann in self.annotations["annotations"]:
            if ann["image_id"] == image_id:
                boxes.append({"id": ann["id"], "bbox": ann["bbox"]})
        return boxes
    
    def _save_annotations(self):
        """Save annotations to COCO JSON file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"[SAVED] {self.output_file}")
    
    def _update_coco_for_current_image(self):
        """Update COCO structure with current image's boxes."""
        img_path = self.image_files[self.current_idx]
        filename = img_path.name
        
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
        
        image_id = self._get_image_id(filename)
        
        # Update or add image entry
        img_entry = None
        for img in self.annotations["images"]:
            if img["id"] == image_id:
                img_entry = img
                break
        
        if img_entry is None:
            img_entry = {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height
            }
            self.annotations["images"].append(img_entry)
        else:
            img_entry["width"] = width
            img_entry["height"] = height
        
        # Remove old annotations for this image
        self.annotations["annotations"] = [
            ann for ann in self.annotations["annotations"]
            if ann["image_id"] != image_id
        ]
        
        # Add current boxes
        for box in self.current_boxes:
            x, y, w, h = box["bbox"]
            ann_id = max([0] + [a["id"] for a in self.annotations["annotations"]]) + 1
            self.annotations["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(w * h),
                "iscrowd": 0
            })
    
    def _draw_boxes(self):
        """Draw all current boxes on the image."""
        # Remove old patches
        for patch in self.rect_patches:
            patch.remove()
        self.rect_patches = []
        
        # Draw new boxes
        for i, box in enumerate(self.current_boxes):
            x, y, w, h = box["bbox"]
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.3
            )
            self.ax.add_patch(rect)
            self.rect_patches.append(rect)
            
            # Label
            text = self.ax.text(x, y - 5, f'table_{i+1}', 
                               color='lime', fontsize=10, fontweight='bold')
            self.rect_patches.append(text)
        
        self.fig.canvas.draw_idle()
    
    def _load_image(self):
        """Load and display current image."""
        img_path = self.image_files[self.current_idx]
        img = Image.open(img_path)
        
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_title(
            f"[{self.current_idx + 1}/{len(self.image_files)}] {img_path.name}\n"
            f"Draw boxes around tables, then click 'Save'",
            fontsize=12
        )
        self.ax.axis('off')
        
        # Load existing boxes
        image_id = self._get_image_id(img_path.name)
        self.current_boxes = self._get_boxes_for_image(image_id)
        self.rect_patches = []
        
        # Re-create selector for new axes
        self.selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False
        )
        
        self._draw_boxes()
        
        print(f"\n[Image {self.current_idx + 1}/{len(self.image_files)}] {img_path.name}")
        print(f"  Existing boxes: {len(self.current_boxes)}")
    
    def _on_select(self, eclick, erelease):
        """Handle rectangle selection."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return
        
        # Ensure proper order
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        w = x2 - x1
        h = y2 - y1
        
        # Only add if box is large enough
        if w > 20 and h > 20:
            self.current_boxes.append({
                "id": len(self.current_boxes) + 1,
                "bbox": [x1, y1, w, h]
            })
            print(f"  Added box {len(self.current_boxes)}: [{int(x1)}, {int(y1)}, {int(w)}, {int(h)}]")
            self._draw_boxes()
    
    def _on_prev(self, event):
        """Go to previous image."""
        self._update_coco_for_current_image()
        self.current_idx = (self.current_idx - 1) % len(self.image_files)
        self._load_image()
    
    def _on_next(self, event):
        """Go to next image."""
        self._update_coco_for_current_image()
        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        self._load_image()
    
    def _on_undo(self, event):
        """Undo last box."""
        if self.current_boxes:
            removed = self.current_boxes.pop()
            print(f"  Removed box: {[int(x) for x in removed['bbox']]}")
            self._draw_boxes()
    
    def _on_save(self, event):
        """Save annotations."""
        self._update_coco_for_current_image()
        self._save_annotations()
    
    def run(self):
        """Main annotation loop."""
        print("\n" + "=" * 60)
        print("TABLE ANNOTATION TOOL")
        print("=" * 60)
        print(f"Images: {len(self.image_files)} found")
        print("\nInstructions:")
        print("  1. Click and drag to draw a box around each table")
        print("  2. Click 'Save' to save your annotations")
        print("  3. Use 'Prev'/'Next' to navigate images")
        print("  4. Click 'Undo' to remove the last box")
        print("  5. Close the window when done")
        print("=" * 60)
        
        self._load_image()
        plt.show()
        
        # Save on close
        self._update_coco_for_current_image()
        self._save_annotations()
        print("\n[DONE] Annotations saved!")


def main():
    print("\n" + "=" * 60)
    print("SIMPLE TABLE ANNOTATION TOOL (Matplotlib)")
    print("=" * 60)
    
    if not IMAGES_DIR.exists():
        print(f"\n[ERROR] Images directory not found: {IMAGES_DIR}")
        print("\nFirst, render your PDFs to images:")
        print("  python MLTableDetection/render_pdfs_for_annotation.py \\")
        print("      --input /path/to/pdfs \\")
        print(f"      --output {IMAGES_DIR}")
        return 1
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        annotator = TableAnnotator(IMAGES_DIR, ANNOTATIONS_FILE)
        annotator.run()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
