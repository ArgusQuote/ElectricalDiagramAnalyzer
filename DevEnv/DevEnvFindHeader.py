#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys
from glob import glob
import cv2, numpy as np

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from OcrLibrary.BreakerTableAnalyzer6 import BreakerTableAnalyzer

# ---------- USER SETTINGS ----------
INPUT_DIR  = os.path.expanduser("~/Documents/Diagrams/CaseStudy_VectorCrop_Run13")  # folder with PNGs
OUTPUT_DIR = os.path.expanduser("~/Documents/Diagrams/DebugOutput1")                 # where overlays go
FORCE_CPU_OCR = True                                                                # avoid CUDA OOM

# ---------- IO ----------
def gather_images(input_dir):
    d = os.path.expanduser(input_dir)
    if not os.path.isdir(d):
        print(f"[ERROR] INPUT_DIR not found or not a folder: {d}")
        return []
    files = sorted(glob(os.path.join(d, "*.png")))
    print(f"[INFO] Found {len(files)} PNG(s) in {d}")
    for f in files[:10]: print("       ", f)
    if len(files) > 10: print(f"       ... +{len(files)-10} more")
    return files

# ---------- ANALYZER HEADER (same as before) ----------
def find_header_like_analyzer(gray, analyzer: BreakerTableAnalyzer):
    H, W = gray.shape
    header_y = analyzer._find_header_by_tokens(gray)
    if header_y is None:
        header_y = analyzer._find_cct_header_y(gray)
    if header_y is not None and header_y > int(0.62 * H):
        header_y = None
    return header_y

# ---------- EASYOCR helpers ----------
def _rebuild_reader_cpu(analyzer):
    try:
        import easyocr
        analyzer.reader = easyocr.Reader(['en'], gpu=False)
        return True
    except Exception:
        analyzer.reader = None
        return False

def ocr_tokens_in_header_band(gray, analyzer: BreakerTableAnalyzer):
    if analyzer.reader is None:
        return []
    H, W = gray.shape
    y1, y2 = int(0.08*H), int(0.65*H)
    roi = gray[y1:y2, :]

    def _ocr_pass(mag):
        return analyzer.reader.readtext(
            roi, detail=1, paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/",
            mag_ratio=mag, contrast_ths=0.05, adjust_contrast=0.7,
            text_threshold=0.4, low_text=0.25
        )

    try:
        det = _ocr_pass(1.6) + _ocr_pass(2.0)
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            _rebuild_reader_cpu(analyzer)
            try:
                det = _ocr_pass(1.6) + _ocr_pass(2.0)
            except Exception:
                det = []
        else:
            det = []

    def N(s): return re.sub(r"[^A-Z0-9]", "", (s or "").upper().replace("1","I").replace("0","O"))
    CATEGORY = {
        "CKT","CCT","TRIP","POLES","AMP","AMPS","BREAKER","BKR","SIZE",
        "DESCRIPTION","DESIGNATION","NAME","LOADDESCRIPTION","CIRCUITDESCRIPTION"
    }

    toks=[]
    for box, txt, _ in det:
        if not txt:
            continue
        xs=[p[0] for p in box]; ys=[p[1] for p in box]
        toks.append({
            "text": txt,
            "is_header_word": any(k in N(txt) for k in CATEGORY),
            "x1": int(min(xs)), "y1": int(min(ys))+y1,
            "x2": int(max(xs)), "y2": int(max(ys))+y1
        })
    return toks

# ---------- BINARIZE + HORIZONTAL LINES (ported from your line checker) ----------
def binarize(gray):
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw_inv  # white=ink

def horiz_mask_from_bin(bw_inv, W, horiz_kernel_frac=0.035):
    fracs = [
        max(0.02, horiz_kernel_frac * 0.7),
        horiz_kernel_frac,
        min(0.08, horiz_kernel_frac * 1.8),
    ]
    masks = []
    for f in fracs:
        klen = max(5, int(round(W * f)))
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        m = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, kh, iterations=1)
        m = cv2.dilate(m, None, iterations=1)
        m = cv2.erode(m,  None, iterations=1)
        masks.append(m)
    out = masks[0]
    for m in masks[1:]:
        out = cv2.bitwise_or(out, m)
    return out

def extract_horiz_lines(gray, horiz_kernel_frac=0.035, min_width_frac=0.25,
                        max_thickness_px=6, y_merge_tol_px=4):
    H, W = gray.shape
    bw_inv = binarize(gray)                            # like your _binarize_ink
    horiz  = horiz_mask_from_bin(bw_inv, W, horiz_kernel_frac)
    cnts, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_w = int(round(W * min_width_frac))
    raws=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < min_w or h < 1 or h > max_thickness_px:
            continue
        y_mid = y + h//2
        row = horiz[y_mid, :]
        xs = np.where(row > 0)[0]
        if xs.size == 0: continue
        xL, xR = int(xs.min()), int(xs.max())
        if (xR - xL + 1) < min_w:
            continue
        raws.append((y, y+h-1, xL, xR))

    if not raws:
        return [], bw_inv, horiz

    # merge by Y
    raws.sort(key=lambda r: (r[0]+r[1]) / 2.0)
    merged, cur = [], [raws[0]]
    for r in raws[1:]:
        y0,y1,_,_=r
        py0,py1,_,_=cur[-1]
        if abs(((y0+y1)/2.0) - ((py0+py1)/2.0)) <= y_merge_tol_px:
            cur.append(r)
        else:
            merged.append(cur); cur=[r]
    merged.append(cur)

    lines=[]
    for grp in merged:
        y_t = min(g[0] for g in grp)
        y_b = max(g[1] for g in grp)
        x_l = min(g[2] for g in grp)
        x_r = max(g[3] for g in grp)
        y_c = (y_t + y_b)//2
        lines.append({
            "y_center": int(y_c),
            "y_top":    int(y_t),
            "y_bottom": int(y_b),
            "x_left":   int(x_l),
            "x_right":  int(x_r),
            "width":    int(x_r - x_l + 1),
            "height":   int(max(1, y_b - y_t + 1)),
        })
    lines.sort(key=lambda d: d["y_center"])
    return lines, bw_inv, horiz

def pick_line_above(horiz_lines, ref_y, gap_min=2):
    """Pick the nearest horizontal line strictly above ref_y."""
    if ref_y is None or not horiz_lines:
        return None
    below = [ln for ln in horiz_lines if ln["y_center"] <= int(ref_y) - gap_min]
    if not below:
        return None
    return below[-1]["y_center"]

# ---------- DRAWING ----------
def draw_tokens_overlay(gray, tokens, banner=None):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if banner:
        cv2.putText(vis, banner, (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
    for t in tokens:
        color = (255,0,255) if t["is_header_word"] else (0,255,255)
        cv2.rectangle(vis,(t["x1"],t["y1"]),(t["x2"],t["y2"]),color,1)
        cv2.putText(vis,t["text"],(t["x1"],max(10,t["y1"]-3)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,color,1,cv2.LINE_AA)
    return vis

def draw_full_overlay(gray, tokens, header_y, line_y):
    vis = draw_tokens_overlay(gray, tokens, banner=None)
    H, W = gray.shape
    if header_y is not None:
        y=int(header_y)
        cv2.line(vis,(0,y),(W-1,y),(0,255,0),2)
        cv2.putText(vis,"HEADER_TEXT",(10,max(20,y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
    if line_y is not None:
        y=int(line_y)
        cv2.line(vis,(0,y),(W-1,y),(255,0,0),2)
        cv2.putText(vis,"HEADER_LINE",(10,max(20,y-24)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
    return vis

# ---------- MAIN ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = gather_images(INPUT_DIR)
    if not files:
        print("[ERROR] No PNGs found to process.")
        return

    analyzer = BreakerTableAnalyzer(debug=False)
    if FORCE_CPU_OCR:
        _rebuild_reader_cpu(analyzer)

    for img_path in files:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {img_path}")
            continue
        gray = analyzer._prep(bgr)

        # 1) OCR tokens FIRST
        tokens = ocr_tokens_in_header_band(gray, analyzer)
        base = os.path.splitext(os.path.basename(img_path))[0]

        # If no tokens, write tokens-only + intermediates and SKIP header
        if not tokens:
            print(f"[ERROR] No tokens found -> skipping header detection: {img_path}")
            vis_tokens = draw_tokens_overlay(gray, tokens, banner="NO TOKENS FOUND (skipped header)")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_TOKENS_ONLY.png"), vis_tokens)

            # also save bin + horiz mask so you can confirm lines are visible
            lines, bw, hm = extract_horiz_lines(gray)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_BIN.png"), bw)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HORIZ_MASK.png"), hm)
            continue

        # 2) We have tokens: find header text row and binarized lines
        header_y = find_header_like_analyzer(gray, analyzer)

        # robust line extraction (whole image), then pick closest line ABOVE header (or token band)
        horiz_lines, bw, hm = extract_horiz_lines(gray,
                                horiz_kernel_frac=0.035,   # match your checker
                                min_width_frac=0.25,       # a bit less strict for crops
                                max_thickness_px=6,
                                y_merge_tol_px=4)
        # If header is None, fall back to top of token band
        ref_y = header_y
        if ref_y is None:
            ref_y = min(t["y1"] for t in tokens) if tokens else None
        line_y = pick_line_above(horiz_lines, ref_y, gap_min=2)

        # 3) Write main overlay + intermediates
        vis_full = draw_full_overlay(gray, tokens, header_y, line_y)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HEADER_OVERLAY.png"), vis_full)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_BIN.png"), bw)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HORIZ_MASK.png"), hm)

        # quick console summary
        print(f"[OK] {base}: tokens={len(tokens)} header={header_y} header_line={line_y}  "
              f"lines_total={len(horiz_lines)}")

if __name__ == "__main__":
    main()
