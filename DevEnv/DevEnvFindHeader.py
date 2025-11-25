#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys
from glob import glob
import cv2, numpy as np

# ---------------- PATH SETUP (optional) ----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------------- USER SETTINGS ----------------
INPUT_DIR  = os.path.expanduser("~/Documents/Diagrams/CaseStudy_VectorCrop_Run13")  # folder with PNGs
OUTPUT_DIR = os.path.expanduser("~/Documents/Diagrams/DebugOutput_FH_CUDA")         # where overlays go

# CUDA prefer + safe fallback
FORCE_CPU_OCR = False       # <-- prefer CUDA; will fallback to CPU automatically if CUDA fails
PREFER_GPU    = True
MIN_FREE_MB   = 600         # minimum free VRAM to allow GPU

# Mask render knobs
HORIZ_KERNEL_FRAC = 0.035
MIN_WIDTH_FRAC    = 0.25
MAX_THICK_PX      = 6
Y_MERGE_TOL_PX    = 4
THICKEN_MASK_PX   = 3

# Footer line pick: prefer the line **just above** the footer text
FOOTER_PREFER_ABOVE = True
FOOTER_SEARCH_PX    = 200

# Header line pick: prefer line **just above** the header text
HEADER_PREFER_ABOVE = True
HEADER_SEARCH_PX    = 120

# If no header tokens at all, skip header-line search (still write tokens & masks)
SKIP_HEADER_IF_NO_TOKENS = True

# Env tweak to reduce CUDA fragmentation issues with EasyOCR (PyTorch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------- OCR: GPU-aware EasyOCR builder ----------------
def make_easyocr_reader(force_cpu: bool, prefer_gpu: bool, min_free_mb: int):
    try:
        import easyocr
        if force_cpu:
            r = easyocr.Reader(['en'], gpu=False)
            print("[OCR] Using CPU.")
            return r
        # else: prefer GPU if possible
        gpu_ok = False
        try:
            import torch
            if prefer_gpu and torch.cuda.is_available():
                try:
                    free, total = torch.cuda.mem_get_info()  # bytes
                    gpu_ok = (free // (1024 * 1024)) >= int(min_free_mb)
                except Exception:
                    # mem_get_info not available on older drivers – still try GPU
                    gpu_ok = True
        except Exception:
            gpu_ok = False
        try:
            r = easyocr.Reader(['en'], gpu=bool(gpu_ok))
            print(f"[OCR] Using {'GPU' if gpu_ok else 'CPU'} (initial).")
            return r
        except Exception as e:
            print(f"[OCR] GPU init failed ({e}); falling back to CPU.")
            r = easyocr.Reader(['en'], gpu=False)
            print("[OCR] Using CPU.")
            return r
    except Exception as e:
        print(f"[OCR] EasyOCR unavailable: {e}")
        return None


class OCRCtx:
    """Mutable holder so we can swap reader to CPU if CUDA errors occur mid-run."""
    def __init__(self, reader):
        self.reader = reader


def _safe_readtext(ctx: OCRCtx, roi, **kwargs):
    """
    Try OCR once (possibly on GPU). If it throws a CUDA/OOM-related error,
    rebuild the reader as CPU-only and retry once.
    """
    if ctx.reader is None:
        return []
    try:
        return ctx.reader.readtext(roi, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if ("cuda" in msg) or ("out of memory" in msg) or ("cublas" in msg) or ("cudnn" in msg):
            print("[OCR] CUDA failure during readtext → switching to CPU and retrying once...")
            # Rebuild CPU reader
            ctx.reader = make_easyocr_reader(force_cpu=True, prefer_gpu=False, min_free_mb=MIN_FREE_MB)
            if ctx.reader is None:
                return []
            try:
                return ctx.reader.readtext(roi, **kwargs)
            except Exception as e2:
                print(f"[OCR] CPU retry failed: {e2}")
                return []
        else:
            # Not a CUDA-related issue
            print(f"[OCR] readtext failed: {e}")
            return []


# ---------------- IO helpers ----------------
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


# ---------------- Preproc & masks ----------------
def prep_gray(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    H, W = g.shape
    if H < 1600:
        s = 1600.0 / H
        g = cv2.resize(g, (int(W*s), int(H*s)), interpolation=cv2.INTER_CUBIC)
    return g

def binarize_ink(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw_inv  # white=ink (lines/text)

def horiz_mask_from_bin(bw_inv: np.ndarray, W: int, horiz_kernel_frac=0.035) -> np.ndarray:
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

def extract_horiz_lines_from_mask(hmask: np.ndarray, W: int,
                                  min_width_frac=0.25, max_thickness_px=6,
                                  y_merge_tol_px=4):
    cnts, _ = cv2.findContours(hmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w = int(round(W * min_width_frac))
    raws=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < min_w or h < 1 or h > max_thickness_px:
            continue
        y_mid = y + h//2
        row = hmask[y_mid, :]
        xs = np.where(row > 0)[0]
        if xs.size == 0: continue
        xL, xR = int(xs.min()), int(xs.max())
        if (xR - xL + 1) < min_w:
            continue
        raws.append((y, y+h-1, xL, xR))

    if not raws:
        return []

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
    return lines

def thicken_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    if pixels <= 1: 
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (pixels, pixels))
    return cv2.dilate(mask, k, iterations=1)


# ---------------- OCR token passes ----------------
def NORM(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper().replace("1","I").replace("0","O"))

def ocr_tokens_in_band(gray: np.ndarray, ctx: OCRCtx, y1f: float, y2f: float,
                       allowlist: str, mags=(1.6, 2.0)):
    if ctx.reader is None:
        return []
    H, W = gray.shape
    y1, y2 = int(max(0, y1f * H)), int(min(H, y2f * H))
    roi = gray[y1:y2, :]
    if roi.size == 0:
        return []

    det = []
    for m in mags:
        part = _safe_readtext(
            ctx, roi,
            detail=1, paragraph=False,
            allowlist=allowlist,
            mag_ratio=m, contrast_ths=0.05, adjust_contrast=0.7,
            text_threshold=0.4, low_text=0.25
        )
        det += (part or [])

    out=[]
    for box, txt, conf in det:
        if not txt: 
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        out.append({
            "text": txt,
            "conf": float(conf or 0.0),
            "x1": int(min(xs)),
            "x2": int(max(xs)),
            "y1": int(min(ys)) + y1,
            "y2": int(max(ys)) + y1,
        })
    return out

def header_tokens(gray: np.ndarray, ctx: OCRCtx):
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/"
    det = ocr_tokens_in_band(gray, ctx, 0.08, 0.65, allow)
    CATEGORY = {
        "CKT","CCT","TRIP","POLES","AMP","AMPS","BREAKER","BKR","SIZE",
        "DESCRIPTION","DESIGNATION","NAME","LOADDESCRIPTION","CIRCUITDESCRIPTION"
    }
    hdr = []
    for t in det:
        n = NORM(t["text"])
        if n and any(k in n for k in CATEGORY):
            hdr.append(t)
    return det, hdr

def footer_tokens(gray: np.ndarray, ctx: OCRCtx):
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :/.-"
    det = ocr_tokens_in_band(gray, ctx, 0.45, 0.95, allow)
    total_hits, load_hits = [], []
    for t in det:
        n = NORM(t["text"])
        if not n: 
            continue
        sim_total = 0.0
        try:
            import difflib
            sim_total = difflib.SequenceMatcher(a=n, b="TOTALLOAD").ratio()
        except Exception:
            pass
        is_total = (("TOTAL" in n) and ("LOAD" in n)) or (sim_total >= 0.70)
        if is_total:
            total_hits.append(t)
        elif ("LOAD" in n) and ("CLASS" not in n):
            load_hits.append(t)
    return det, (total_hits if total_hits else load_hits)


# ---------------- picking helpers ----------------
def pick_nearest_line(lines, ref_y, prefer_above=None, gap_min=2, search_px=None):
    if ref_y is None or not lines:
        return None
    ref_y = int(ref_y)
    cands = lines
    if prefer_above is True:
        cands = [ln for ln in cands if ln["y_center"] <= ref_y - gap_min]
    elif prefer_above is False:
        cands = [ln for ln in cands if ln["y_center"] >= ref_y + gap_min]
    if search_px is not None:
        lo, hi = ref_y - search_px, ref_y + search_px
        cands = [ln for ln in cands if lo <= ln["y_center"] <= hi]
    if not cands:
        return None
    best = min(cands, key=lambda ln: abs(ln["y_center"] - ref_y))
    return int(best["y_center"])


# ---------------- drawing ----------------
def draw_tokens(gray, tokens, banner=None):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if banner:
        cv2.putText(vis, banner, (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
    for t in tokens:
        color = (255,0,255)  # magenta for all tokens here
        cv2.rectangle(vis,(t["x1"],t["y1"]),(t["x2"],t["y2"]),color,1)
        cv2.putText(vis,t["text"],(t["x1"],max(10,t["y1"]-3)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,color,1,cv2.LINE_AA)
    return vis

def draw_full_overlay(gray, hdr_tokens, ftr_tokens, hdr_text_y, hdr_line_y, ftr_text_y, ftr_line_y):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape

    for t in hdr_tokens:
        cv2.rectangle(vis,(t["x1"],t["y1"]),(t["x2"],t["y2"]),(0,255,255),1)  # header-ish tokens (yellow)
    for t in ftr_tokens:
        cv2.rectangle(vis,(t["x1"],t["y1"]),(t["x2"],t["y2"]),(255,0,255),1)  # footer-ish tokens (magenta)

    if hdr_text_y is not None:
        y=int(hdr_text_y)
        cv2.line(vis,(0,y),(W-1,y),(0,150,0),1)
        cv2.putText(vis,"HEADER_TEXT",(10,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,150,0),1,cv2.LINE_AA)
    if ftr_text_y is not None:
        y=int(ftr_text_y)
        cv2.line(vis,(0,y),(W-1,y),(150,0,150),1)
        cv2.putText(vis,"FOOTER_TEXT",(10,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,0,150),1,cv2.LINE_AA)

    if hdr_line_y is not None:
        y=int(hdr_line_y)
        cv2.line(vis,(0,y),(W-1,y),(0,255,0),2)
        cv2.putText(vis,"HEADER_LINE",(10,max(20,y-24)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)
    if ftr_line_y is not None:
        y=int(ftr_line_y)
        cv2.line(vis,(0,y),(W-1,y),(0,0,255),2)
        cv2.putText(vis,"FOOTER_LINE",(10,max(20,y+18)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)

    return vis

def draw_hmask_with_markers(gray, hmask_thick, hdr_text_y, hdr_line_y, ftr_text_y, ftr_line_y):
    H, W = gray.shape
    vis = cv2.cvtColor(hmask_thick, cv2.COLOR_GRAY2BGR)
    if hdr_text_y is not None:
        y=int(hdr_text_y); cv2.line(vis,(0,y),(W-1,y),(0,150,0),1)
        cv2.putText(vis,"HDR_TEXT",(10,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,150,0),1,cv2.LINE_AA)
    if hdr_line_y is not None:
        y=int(hdr_line_y); cv2.line(vis,(0,y),(W-1,y),(0,255,0),2)
        cv2.putText(vis,"HDR_LINE",(120,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1,cv2.LINE_AA)
    if ftr_text_y is not None:
        y=int(ftr_text_y); cv2.line(vis,(0,y),(W-1,y),(150,0,150),1)
        cv2.putText(vis,"FTR_TEXT",(10,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,0,150),1,cv2.LINE_AA)
    if ftr_line_y is not None:
        y=int(ftr_line_y); cv2.line(vis,(0,y),(W-1,y),(0,0,255),2)
        cv2.putText(vis,"FTR_LINE",(120,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),1,cv2.LINE_AA)
    return vis


# ---------------- main ----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = gather_images(INPUT_DIR)
    if not files:
        print("[ERROR] No PNGs found to process.")
        return

    # Build OCR (prefer GPU, auto-fallback)
    reader = make_easyocr_reader(FORCE_CPU_OCR, PREFER_GPU, MIN_FREE_MB)
    ctx = OCRCtx(reader)

    for img_path in files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        bgr  = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {img_path}")
            continue

        gray = prep_gray(bgr)
        H, W = gray.shape

        # 1) OCR tokens
        det_all_hdr, hdr_like = header_tokens(gray, ctx)
        det_all_ftr, ftr_hits = footer_tokens(gray, ctx)

        no_header_tokens = (len(hdr_like) == 0)

        hdr_text_y = int(min(t["y1"] for t in hdr_like)) if hdr_like else None
        ftr_text_y = int(min(t["y1"] for t in ftr_hits)) if ftr_hits else None

        # 2) Build horizontal mask once
        bw_inv = binarize_ink(gray)
        hmask  = horiz_mask_from_bin(bw_inv, W, HORIZ_KERNEL_FRAC)
        hmask_thick = thicken_mask(hmask, THICKEN_MASK_PX)

        # Extract lines
        lines = extract_horiz_lines_from_mask(hmask, W,
                                              min_width_frac=MIN_WIDTH_FRAC,
                                              max_thickness_px=MAX_THICK_PX,
                                              y_merge_tol_px=Y_MERGE_TOL_PX)

        # 3) Choose header/footer lines
        hdr_line_y = None
        ftr_line_y = None

        if hdr_text_y is not None:
            hdr_line_y = pick_nearest_line(
                lines, hdr_text_y,
                prefer_above=HEADER_PREFER_ABOVE,
                gap_min=2,
                search_px=HEADER_SEARCH_PX
            )
        elif not SKIP_HEADER_IF_NO_TOKENS:
            approx_ref = int(0.08 * H)
            hdr_line_y = pick_nearest_line(
                lines, approx_ref,
                prefer_above=None, gap_min=2, search_px=HEADER_SEARCH_PX
            )

        if ftr_text_y is not None:
            ftr_line_y = pick_nearest_line(
                lines, ftr_text_y,
                prefer_above=FOOTER_PREFER_ABOVE,
                gap_min=2,
                search_px=FOOTER_SEARCH_PX
            )

        # 4) Write outputs
        vis_tokens = draw_tokens(gray, det_all_hdr + det_all_ftr,
                                 banner=None if (hdr_like or ftr_hits) else "NO TOKENS FOUND")
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_TOKENS.png"), vis_tokens)

        vis_hmask = draw_hmask_with_markers(gray, hmask_thick, hdr_text_y, hdr_line_y, ftr_text_y, ftr_line_y)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HORIZ_MASK.png"), vis_hmask)

        vis_full = draw_full_overlay(gray, hdr_like, ftr_hits, hdr_text_y, hdr_line_y, ftr_text_y, ftr_line_y)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_OVERLAY.png"), vis_full)

        # 5) Console summary
        if no_header_tokens and SKIP_HEADER_IF_NO_TOKENS:
            print(f"[OK] {base}: NO HEADER TOKENS → header line skipped | "
                  f"footer_text_y={ftr_text_y} footer_line={ftr_line_y}  lines={len(lines)}")
        else:
            print(f"[OK] {base}: header_text_y={hdr_text_y} header_line={hdr_line_y} | "
                  f"footer_text_y={ftr_text_y} footer_line={ftr_line_y}  lines={len(lines)}")


if __name__ == "__main__":
    main()
