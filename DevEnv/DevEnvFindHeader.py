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
INPUT_DIR        = os.path.expanduser("~/Documents/Diagrams/CaseStudy_VectorCrop_Run11")
OUTPUT_DIR       = os.path.expanduser("~/Documents/Diagrams/DebugOutput2")
FORCE_CPU_OCR    = True
HORIZ_THICKEN_PX = 3

# Search strategy near header text line (we do NOT draw the header line)
SEARCH_UP_FIRST     = True   # try above the header first
MAX_DELTA_PX        = 80     # search radius in pixels (up and down)
MIN_COVERAGE_FRAC   = 0.25   # required white coverage across the row (fraction of width)
MERGE_Y_TOLERANCE   = 3      # merge contiguous rows into a single band if within this many px

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

# ---------- ANALYZER HEADER ----------
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

# ---------- BINARIZE + HORIZONTAL LINES ----------
def binarize(gray):
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw_inv  # white=ink

def horiz_mask_from_bin(bw_inv, W, horiz_kernel_frac=0.035, border_trim_px=2):
    m = bw_inv.copy()
    if border_trim_px > 0:
        m[:border_trim_px,:] = 0; m[-border_trim_px:,:] = 0
        m[:,:border_trim_px] = 0; m[:,-border_trim_px:] = 0

    fracs = [
        max(0.02, horiz_kernel_frac * 0.7),
        horiz_kernel_frac,
        min(0.10, horiz_kernel_frac * 1.8),
    ]
    out = None
    for f in fracs:
        klen = max(5, int(round(W * f)))
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        t = cv2.morphologyEx(m, cv2.MORPH_OPEN, kh, iterations=1)
        t = cv2.dilate(t, None, iterations=1)
        t = cv2.erode(t,  None, iterations=1)
        out = t if out is None else cv2.bitwise_or(out, t)
    return out

def thicken_horizontal_mask(hmask: np.ndarray, thicken_px: int) -> np.ndarray:
    if thicken_px <= 0:
        return hmask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, thicken_px)))
    return cv2.dilate(hmask, k, iterations=1)

def extract_horiz_lines(gray, horiz_kernel_frac=0.035, min_width_frac=0.25,
                        max_thickness_px=6, y_merge_tol_px=4):
    H, W = gray.shape
    bw_inv = binarize(gray)
    horiz  = horiz_mask_from_bin(bw_inv, W, horiz_kernel_frac)

    cnts, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w = int(round(W * min_width_frac))
    raws=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < min_w or h < 1 or h > max_thickness_px:
            continue
        y_mid = y + h//2
        xs = np.where(horiz[y_mid, :] > 0)[0]
        if xs.size == 0: continue
        xL, xR = int(xs.min()), int(xs.max())
        if (xR - xL + 1) < min_w: continue
        raws.append((y, y+h-1, xL, xR))

    if not raws:
        return [], horiz

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
    return lines, horiz

# ---------- FIND NEAR-HEADER WHITE BAND IN H-MASK ----------
def find_near_header_band(hmask: np.ndarray, header_y: int | None,
                          search_up_first=SEARCH_UP_FIRST,
                          max_delta=MAX_DELTA_PX,
                          min_cov_frac=MIN_COVERAGE_FRAC,
                          merge_tol=MERGE_Y_TOLERANCE):
    """
    hmask: horizontal-only mask (WHITE=horizontal lines)
    Return y (int) for the first strong white band near header_y,
    preferring 'up' (or down if configured).
    """
    if header_y is None:
        return None
    H, W = hmask.shape
    ref = int(header_y)
    ref = max(0, min(H-1, ref))

    row_cov = (hmask > 0).sum(axis=1)  # white pixels per row
    thr = int(round(min_cov_frac * W))
    # build a list of candidate y's where coverage >= thr
    ys = np.where(row_cov >= thr)[0].astype(int)
    if ys.size == 0:
        return None

    # Merge into bands (contiguous rows)
    bands = []
    s = ys[0]; p = ys[0]
    for y in ys[1:]:
        if y - p <= merge_tol:
            p = y
        else:
            bands.append((s, p))
            s = p = y
    bands.append((s, p))

    # For each band, define its representative as the mid-row
    reps = [int((a + b) // 2) for a, b in bands]

    # filter reps within window
    lo = max(0, ref - max_delta)
    hi = min(H - 1, ref + max_delta)
    near = [y for y in reps if lo <= y <= hi]
    if not near:
        return None

    # preference: up (<= ref), then down; else nearest abs
    above = [y for y in near if y <= ref]
    below = [y for y in near if y > ref]

    if search_up_first:
        if above:
            return max(above)
        return min(below, key=lambda y: abs(y - ref)) if below else None
    else:
        if below:
            return min(below)
        return max(above) if above else None

# ---------- DRAWING ----------
def draw_tokens_overlay(gray, tokens, banner=None):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if banner:
        cv2.putText(vis, banner, (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
    for t in tokens:
        color = (255,0,255) if t["is_header_word"] else (0,255,255)
        cv2.rectangle(vis,(t["x1"],t["y1"]),(t["x2"],t["y2"]),color,1)
    return vis

def draw_found_on_mask(hmask_thick: np.ndarray, found_y: int | None) -> np.ndarray:
    vis = cv2.cvtColor(hmask_thick, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]
    if found_y is not None:
        y = max(0, min(H-1, int(found_y)))
        cv2.line(vis, (0, y), (W-1, y), (0, 0, 255), 2)
        cv2.putText(vis, "FOUND_HEADER_RULE", (10, max(20, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return vis

def draw_full_overlay(gray, tokens, line_y):
    vis = draw_tokens_overlay(gray, tokens)
    if line_y is not None:
        y = int(line_y)
        cv2.line(vis, (0, y), (gray.shape[1]-1, y), (0,0,255), 2)
        cv2.putText(vis, "FOUND_HEADER_RULE", (10, max(20, y-24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
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
        base = os.path.splitext(os.path.basename(img_path))[0]

        # 1) OCR tokens FIRST
        tokens = ocr_tokens_in_header_band(gray, analyzer)
        if not tokens:
            print(f"[ERROR] No tokens found -> skipping header detection: {img_path}")
            # still produce horizontal-only mask for inspection
            _, horiz_only = extract_horiz_lines(gray)
            thick = thicken_horizontal_mask(horiz_only, HORIZ_THICKEN_PX)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HORIZ_ONLY.png"), thick)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_FOUND_ON_MASK.png"), draw_found_on_mask(thick, None))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_TOKENS_ONLY.png"), draw_tokens_overlay(gray, tokens, banner="NO TOKENS"))
            continue

        # 2) Header text row (remember it; don't draw it)
        header_y = find_header_like_analyzer(gray, analyzer)

        # 3) Horizontal-only mask (+thicken)
        horiz_lines, horiz_only = extract_horiz_lines(
            gray,
            horiz_kernel_frac=0.035,
            min_width_frac=0.25,
            max_thickness_px=6,
            y_merge_tol_px=4
        )
        thick_mask = thicken_horizontal_mask(horiz_only, HORIZ_THICKEN_PX)

        # 4) From header_y, search up/down for the nearest strong white band on the mask
        found_y = find_near_header_band(
            horiz_only, header_y,
            search_up_first=SEARCH_UP_FIRST,
            max_delta=MAX_DELTA_PX,
            min_cov_frac=MIN_COVERAGE_FRAC,
            merge_tol=MERGE_Y_TOLERANCE
        )

        # 5) Write outputs
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HORIZ_ONLY.png"), thick_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_FOUND_ON_MASK.png"), draw_found_on_mask(thick_mask, found_y))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HEADER_RULE_OVERLAY.png"), draw_full_overlay(gray, tokens, found_y))

        print(f"[OK] {base}: header_y={header_y}  found_rule_y={found_y}  horiz_lines={len(horiz_lines)}")

if __name__ == "__main__":
    main()
