#!/bin/bash
# Setup script for Label Studio annotation environment
#
# This script:
# 1. Installs Label Studio in the virtual environment
# 2. Creates a project configuration file
# 3. Provides instructions for starting the server
#
# Usage:
#   chmod +x setup_label_studio.sh
#   ./setup_label_studio.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/MLTableDetection/data"
IMAGES_DIR="$DATA_DIR/images_for_annotation"

echo "=========================================="
echo "Label Studio Setup for Table Detection"
echo "=========================================="
echo ""

# Activate virtual environment
echo "[1/4] Activating virtual environment..."
source /home/marco/venv/bin/activate

# Install Label Studio
echo "[2/4] Installing Label Studio..."
pip install --quiet label-studio

# Create data directories
echo "[3/4] Creating data directories..."
mkdir -p "$IMAGES_DIR"
mkdir -p "$DATA_DIR/annotations"
mkdir -p "$DATA_DIR/models"

# Create Label Studio project config
echo "[4/4] Creating project configuration..."
cat > "$DATA_DIR/label_config.xml" << 'EOF'
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="table" background="green"/>
    <Label value="panel_schedule" background="blue"/>
    <Label value="motor_schedule" background="orange"/>
    <Label value="riser_diagram" background="purple"/>
  </RectangleLabels>
</View>
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory structure created:"
echo "  $IMAGES_DIR"
echo "  $DATA_DIR/annotations"
echo "  $DATA_DIR/models"
echo ""
echo "Next steps:"
echo ""
echo "1. Render your PDFs to images:"
echo "   cd $PROJECT_ROOT"
echo "   python MLTableDetection/render_pdfs_for_annotation.py \\"
echo "       --input /path/to/your/pdfs \\"
echo "       --output $IMAGES_DIR"
echo ""
echo "2. Start Label Studio:"
echo "   source /home/marco/venv/bin/activate"
echo "   label-studio start --data-dir $DATA_DIR/label_studio_data"
echo ""
echo "3. In Label Studio (http://localhost:8080):"
echo "   a. Create a new project"
echo "   b. Go to Settings > Labeling Interface"
echo "   c. Paste the contents of: $DATA_DIR/label_config.xml"
echo "   d. Import your images from: $IMAGES_DIR"
echo "   e. Start annotating!"
echo ""
echo "4. After annotating, export as 'COCO' format to:"
echo "   $DATA_DIR/annotations/"
echo ""
echo "5. Fine-tune Table Transformer (only if zero-shot isn't good enough):"
echo "   python MLTableDetection/train_table_transformer.py \\"
echo "       --data $DATA_DIR/annotations \\"
echo "       --epochs 10"
echo ""
echo "NOTE: Try zero-shot detection first! Table Transformer is pretrained"
echo "      on 1M+ tables and often works without fine-tuning."
echo ""
