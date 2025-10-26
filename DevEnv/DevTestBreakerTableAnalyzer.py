import sys 
import os 
from pathlib import Path
# Path setup 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(script_dir) 
if project_root not in sys.path: sys.path.append(project_root) 

from OcrLibrary.BreakerTableAnalyzer3 import BreakerTableAnalyzer

a = BreakerTableAnalyzer(debug=True)
res = a.analyze('~/ElectricalDiagramAnalyzer/DevEnv/PanelSearchOuput/generic2_electrical_filtered_page001_table01_rect.png')