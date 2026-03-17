# Annotation Guide for Panel Schedule Detection

Standards for annotating panel schedule tables in electrical PDF drawings to produce consistent, high-quality training data.

## Label Schema

| Label | Description | Required |
|---|---|---|
| `panel_schedule` | Rectangular panel schedule table (header + circuit rows + footer) | Yes |
| `motor_schedule` | Motor schedule table (if present on page) | Optional |
| `riser_diagram` | Riser diagram (if present on page) | Optional |

Focus on `panel_schedule` first. Only add other labels if the user specifically requests multi-class detection.

---

## Bounding Box Conventions

### What to Include

Draw the bounding box around the **entire panel schedule table**, including:

- **Panel header**: The row containing panel name, location, voltage, phases, wires, supply info (e.g., "BRANCH PANEL: M.1", "VOLTS: 208Y/120V")
- **Column headers**: "CKT", "CIRCUIT DESCRIPTION", "TRIP", "POLE", "A", "B", "C", etc.
- **All circuit rows**: Every numbered circuit row (odd on left, even on right)
- **Footer section**: Load classification table, total connected load, total estimated demand, notes

### What to Exclude

Do NOT extend the bounding box to include:

- Title block (sheet number, project info, drawn/designed/checked fields)
- Drawing border or frame lines
- Notes that appear outside the table boundary
- Revision blocks
- Adjacent tables (each table gets its own bounding box)

### Edge Cases

**Tables touching page borders**: Include the full table. The bounding box may touch or slightly exceed the visible content area. This is acceptable.

**Partially visible tables**: If a table is cut off by the page edge and less than 50% is visible, do NOT annotate it. If 50% or more is visible, annotate the visible portion.

**Nested tables within a larger frame**: Some drawings place multiple panel schedules inside a single drawing frame with a shared border. Annotate each panel schedule as a separate bounding box. Do not draw one box around the entire frame.

**Tables with irregular spacing**: Some schedules have non-uniform row heights or merged cells. Still draw a single tight bounding box around the full table extent.

---

## Bounding Box Tightness

- The box should be **tight** -- touching the outermost ink of the table on all four sides
- Allow 2-5 pixels of margin beyond the table border lines for safety
- Do NOT add large padding; the detection pipeline adds its own padding during inference

### Good vs Bad Examples

**Good**: Box tightly encloses the table from the top of the header to the bottom of the footer, left edge of the leftmost column to right edge of the rightmost column.

**Bad**: Box includes large whitespace margins, or includes adjacent text/tables, or cuts off part of the footer.

---

## Multi-Table Pages

Electrical drawings often contain 2-4 panel schedules per page.

- Annotate **each table separately** with its own bounding box
- Bounding boxes should NOT overlap
- If two tables share a border line, split the boundary at the midpoint of the shared line
- Verify all tables on the page are annotated -- missed tables create false negatives in training

---

## Annotation Tool Setup: Label Studio

### Installation

Use the existing setup script:

```bash
bash MLTableDetection/setup_label_studio.sh
```

This installs Label Studio and creates a project with the label config in `label_config.xml`.

### Manual Setup (if script is unavailable)

```bash
pip install label-studio
label-studio start --port 8080
```

Create a new project with this labeling config:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="panel_schedule" background="green"/>
    <Label value="motor_schedule" background="blue"/>
    <Label value="riser_diagram" background="orange"/>
  </RectangleLabels>
</View>
```

### Import Images

1. Open the Label Studio project
2. Go to Settings > Cloud Storage (or use local file import)
3. Import all rendered PNGs from `MLTableDetection/data/images/`
4. Images appear as tasks in the annotation queue

### Annotation Workflow

1. Open each task (image)
2. Select the `panel_schedule` label
3. Draw a rectangle around each panel schedule table
4. Repeat for all tables on the page
5. Submit the annotation

### Keyboard Shortcuts

- `1` -- Select `panel_schedule` label
- `2` -- Select `motor_schedule` label
- `3` -- Select `riser_diagram` label
- `Ctrl+Enter` -- Submit annotation

---

## Export to COCO Format

### From Label Studio UI

1. Go to the project page
2. Click "Export"
3. Select "COCO" format
4. Download the JSON file
5. Save as `MLTableDetection/data/annotations/annotations.json`

### From Label Studio API

```bash
# Get project ID (usually 1 for the first project)
curl -H "Authorization: Token YOUR_TOKEN" \
  http://localhost:8080/api/projects/1/export?exportType=COCO \
  -o MLTableDetection/data/annotations/annotations.json
```

### COCO Format Structure

The exported JSON follows this structure:

```json
{
  "images": [
    {"id": 0, "file_name": "page_001.png", "width": 3300, "height": 2550}
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": 123456,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "panel_schedule"}
  ]
}
```

The `bbox` field uses absolute pixel coordinates in `[x, y, width, height]` format (top-left corner + dimensions). This is the COCO standard and is directly compatible with Table Transformer and RF-DETR training scripts.

---

## Validation Checklist

Before using annotations for training, verify:

- [ ] Every panel schedule on every page has a bounding box
- [ ] No bounding boxes overlap with each other
- [ ] No bounding boxes include title blocks or revision tables
- [ ] Boxes are tight (2-5px margin, not large whitespace)
- [ ] COCO JSON parses correctly and image file names match actual files
- [ ] Category IDs are consistent (0 = panel_schedule)
- [ ] At least 50 annotations for reliable fine-tuning (100+ preferred)

### Quick Validation Script

```python
import json
from pathlib import Path

ann_path = "MLTableDetection/data/annotations/annotations.json"
img_dir = Path("MLTableDetection/data/images")

with open(ann_path) as f:
    coco = json.load(f)

# Check image files exist
missing = []
for img in coco["images"]:
    if not (img_dir / img["file_name"]).exists():
        missing.append(img["file_name"])

# Check annotation quality
zero_area = [a for a in coco["annotations"] if a["area"] <= 0]
negative_bbox = [a for a in coco["annotations"]
                 if any(v < 0 for v in a["bbox"])]

print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print(f"Categories: {[c['name'] for c in coco['categories']]}")
print(f"Missing image files: {len(missing)}")
print(f"Zero-area annotations: {len(zero_area)}")
print(f"Negative bbox values: {len(negative_bbox)}")

if missing or zero_area or negative_bbox:
    print("ISSUES FOUND -- fix before training")
else:
    print("Annotations look good")
```
