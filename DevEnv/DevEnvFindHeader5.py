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
INPUT_DIR  = os.path.expanduser("~/Documents/Diagrams/CaseStudy_VectorCrop_Run13")
OUTPUT_DIR = os.path.expanduser("~/Documents/Diagrams/DebugOutput_FH_CUDA")

# OCR / CUDA
FORCE_CPU_OCR = False
PREFER_GPU    = True
MIN_FREE_MB   = 600
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Horizontal-mask params (header work)
HORIZ_KERNEL_FRAC = 0.035
MIN_WIDTH_FRAC    = 0.20
MAX_THICK_PX      = 6
Y_MERGE_TOL_PX    = 4
THICKEN_MASK_PX   = 3

# Header picking
HEADER_PREFER_ABOVE = True
HEADER_SEARCH_PX    = 140
SKIP_HEADER_IF_NO_TOKENS = True

# “Start of breakers” picking (line after header)
SOB_PREFER_ABOVE = False
SOB_SEARCH_PX    = 200

# CKT/CCT column bracketing
CCT_COL_RIGHT_FRAC = 0.30   # search tokens only in left ~30% of image

# Footer picking (near the footer text baseline)
# >>> CHANGE #1: force footer line BELOW footer text + tighten window <<<
FOOTER_PREFER_ABOVE = False
FOOTER_SEARCH_PX    = 100

# ---------------- OCR utils ----------------
def make_easyocr_reader(force_cpu: bool, prefer_gpu: bool, min_free_mb: int):
    try:
        import easyocr
        if force_cpu:
            r = easyocr.Reader(['en'], gpu=False); print("[OCR] Using CPU."); return r
        gpu_ok = False
        try:
            import torch
            if prefer_gpu and torch.cuda.is_available():
                try:
                    free, _ = torch.cuda.mem_get_info()
                    gpu_ok = (free // (1024*1024)) >= int(min_free_mb)
                except Exception:
                    gpu_ok = True
        except Exception:
            gpu_ok = False
        try:
            r = easyocr.Reader(['en'], gpu=bool(gpu_ok))
            print(f"[OCR] Using {'GPU' if gpu_ok else 'CPU'} (initial).")
            return r
        except Exception as e:
            print(f"[OCR] GPU init failed ({e}); falling back to CPU.")
            r = easyocr.Reader(['en'], gpu=False); print("[OCR] Using CPU."); return r
    except Exception as e:
        print(f"[OCR] EasyOCR unavailable: {e}")
        return None

class OCRCtx:
    def __init__(self, reader): self.reader = reader

def _safe_readtext(ctx: OCRCtx, roi, **kwargs):
    if ctx.reader is None: return []
    try:
        return ctx.reader.readtext(roi, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("cuda","out of memory","cublas","cudnn")):
            print("[OCR] CUDA failure → switching to CPU and retrying once...")
            ctx.reader = make_easyocr_reader(force_cpu=True, prefer_gpu=False, min_free_mb=MIN_FREE_MB)
            if ctx.reader is None: return []
            try:   return ctx.reader.readtext(roi, **kwargs)
            except Exception as e2:
                print(f"[OCR] CPU retry failed: {e2}"); return []
        print(f"[OCR] readtext failed: {e}"); return []

# ---------------- IO ----------------
def gather_images(input_dir):
    d = os.path.expanduser(input_dir)
    if not os.path.isdir(d):
        print(f"[ERROR] INPUT_DIR not found or not a folder: {d}"); return []
    files = sorted(glob(os.path.join(d, "*.png")))
    print(f"[INFO] Found {len(files)} PNG(s) in {d}")
    for f in files[:10]: print("       ", f)
    if len(files)>10: print(f"       ... +{len(files)-10} more")
    return files

# ---------------- Preproc & masks ----------------
def prep_gray(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)
    H, W = g.shape
    if H < 1600:
        s = 1600.0 / H
        g = cv2.resize(g, (int(W*s), int(H*s)), interpolation=cv2.INTER_CUBIC)
    return g

def binarize_ink(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # NOTE: cv2.THRESH_OTSU (not cv2.OTSU)
    _, bw_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw_inv  # white=ink (lines/text)

def horiz_mask_from_bin(bw_inv: np.ndarray, W: int, horiz_kernel_frac=0.035) -> np.ndarray:
    fracs = [max(0.02, horiz_kernel_frac*0.7), horiz_kernel_frac, min(0.08, horiz_kernel_frac*1.8)]
    masks=[]
    for f in fracs:
        klen = max(5, int(round(W*f)))
        kh   = cv2.getStructuringElement(cv2.MORPH_RECT, (klen,1))
        m    = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, kh, iterations=1)
        m    = cv2.dilate(m, None, iterations=1)
        m    = cv2.erode(m,  None, iterations=1)
        masks.append(m)
    out = masks[0]
    for m in masks[1:]: out = cv2.bitwise_or(out, m)
    return out

def thicken_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    if pixels <= 1: return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (pixels, pixels))
    return cv2.dilate(mask, k, iterations=1)

# ---- horizontal lines from mask ----
def extract_horiz_lines_from_mask(hmask: np.ndarray, W: int,
                                  min_width_frac=0.20, max_thickness_px=8,
                                  y_merge_tol_px=4):
    cnts, _ = cv2.findContours(hmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w = int(round(W * min_width_frac))
    raws=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w < min_w or h < 1 or h > max_thickness_px: continue
        band = hmask[max(0,y-1):min(hmask.shape[0], y+h+2), x:x+w]
        colsum = band.sum(axis=1)
        if colsum.size==0 or colsum.max()==0: continue
        ym = int(np.argmax(colsum)) + max(0,y-1)
        xs = np.where(hmask[ym, :] > 0)[0]
        if xs.size == 0: continue
        xL, xR = int(xs.min()), int(xs.max())
        if (xR - xL + 1) < min_w: continue
        raws.append((y, y+h-1, xL, xR))

    if not raws: return []

    raws.sort(key=lambda r: (r[0]+r[1]) / 2.0)
    merged, cur = [], [raws[0]]
    for r in raws[1:]:
        y0,y1,_,_ = r; py0,py1,_,_ = cur[-1]
        if abs(((y0+y1)/2.0) - ((py0+py1)/2.0)) <= y_merge_tol_px: cur.append(r)
        else: merged.append(cur); cur=[r]
    merged.append(cur)

    lines=[]
    for grp in merged:
        y_t = min(g[0] for g in grp); y_b = max(g[1] for g in grp)
        x_l = min(g[2] for g in grp); x_r = max(g[3] for g in grp)
        y_c = (y_t + y_b)//2
        lines.append({"y_center":int(y_c),"y_top":int(y_t),"y_bottom":int(y_b),
                      "x_left":int(x_l),"x_right":int(x_r),
                      "width":int(x_r-x_l+1),"height":int(max(1,y_b-y_t+1))})
    lines.sort(key=lambda d: d["y_center"])
    return lines

# ---------------- picking helpers ----------------
def pick_nearest_line(lines, ref_y, prefer_above=None, gap_min=2, search_px=None):
    if ref_y is None or not lines: return None
    ref_y = int(ref_y)
    cands = lines
    if prefer_above is True:
        cands = [ln for ln in cands if ln["y_center"] <= ref_y - gap_min]
    elif prefer_above is False:
        cands = [ln for ln in cands if ln["y_center"] >= ref_y + gap_min]
    if search_px is not None:
        lo, hi = ref_y - search_px, ref_y + search_px
        cands = [ln for ln in cands if lo <= ln["y_center"] <= hi]
    if not cands: return None
    best = min(cands, key=lambda ln: abs(ln["y_center"] - ref_y))
    return int(best["y_center"])

def scan_white_band_near(ref_y:int, hmask_thick:np.ndarray, W:int,
                         prefer_above:bool|None, search_px:int, min_run_frac:float=0.15):
    if ref_y is None: return None
    ref = int(ref_y)
    lo, hi = max(0, ref-search_px), min(hmask_thick.shape[0]-1, ref+search_px)
    best_y, best_len = None, 0
    min_run = int(W * min_run_frac)
    rows = list(range(lo, hi+1))
    rows.sort(key=lambda y: (abs(y-ref), 0 if (prefer_above is None) else (0 if ((prefer_above and y<=ref) or ((prefer_above is False) and y>=ref)) else 1)))
    for y in rows:
        xs = np.where(hmask_thick[y,:] > 0)[0]
        if xs.size==0: continue
        run = xs.max()-xs.min()+1
        if run >= min_run and run > best_len:
            best_len, best_y = run, y
    return best_y

# ---------------- OCR token passes ----------------
def NORM(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper().replace("1","I").replace("0","O"))

def ocr_tokens_in_band(gray: np.ndarray, ctx: OCRCtx, y1f: float, y2f: float,
                       allowlist: str, mags=(1.6, 2.0)):
    if ctx.reader is None: return []
    H, W = gray.shape
    y1, y2 = int(max(0, y1f*H)), int(min(H, y2f*H))
    roi = gray[y1:y2, :]
    if roi.size == 0: return []

    det=[]
    for m in mags:
        part = _safe_readtext(ctx, roi, detail=1, paragraph=False,
                              allowlist=allowlist, mag_ratio=m,
                              contrast_ths=0.05, adjust_contrast=0.7,
                              text_threshold=0.4, low_text=0.25) or []
        det += part

    out=[]
    for box, txt, conf in det:
        if not txt: continue
        xs=[int(p[0]) for p in box]; ys=[int(p[1]) for p in box]
        out.append({"text":txt,"conf":float(conf or 0.0),
                    "x1":int(min(xs)),"x2":int(max(xs)),
                    "y1":int(min(ys))+y1,"y2":int(max(ys))+y1})
    return out

def header_tokens(gray: np.ndarray, ctx: OCRCtx):
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/"
    det = ocr_tokens_in_band(gray, ctx, 0.08, 0.65, allow)
    CATEGORY = {"CKT","CCT","TRIP","POLES","AMP","AMPS","BREAKER","BKR","SIZE",
                "DESCRIPTION","DESIGNATION","NAME","LOADDESCRIPTION","CIRCUITDESCRIPTION"}
    hdr=[]
    for t in det:
        n=NORM(t["text"])
        if n and any(k in n for k in CATEGORY): hdr.append(t)
    return det, hdr

def first_cct_token(gray: np.ndarray, ctx: OCRCtx, right_frac=0.30):
    """Return the **top-most** token whose text contains CKT/CCT, within the left region."""
    det = ocr_tokens_in_band(
        gray, ctx, 0.06, 0.70,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -/",
        mags=(1.4, 1.8, 2.2)
    )
    H, W = gray.shape
    xR = int(max(10, min(W-1, right_frac * W)))
    hits = []
    for t in det:
        n = NORM(t["text"])
        if "CCT" in n or "CKT" in n:
            if t["x2"] <= xR:
                hits.append(t)
    if not hits:
        return None
    hits.sort(key=lambda t: t["y1"])
    return hits[0]

# ---------------- vertical column bracketing (ROI below SoB) ----------------
def vert_mask_strong_roi(bw_inv: np.ndarray, x1:int, y1:int, x2:int, y2:int):
    roi = bw_inv[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros_like(bw_inv)
    H, W = roi.shape
    scales = [0.12, 0.20, 0.28]
    masks=[]
    for f in scales:
        klen = max(7, int(round(H*f)))
        kv   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
        m    = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kv, iterations=1)
        m    = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        masks.append(m)
    vraw = masks[0]
    for m in masks[1:]: vraw = cv2.bitwise_or(vraw, m)

    vmask = np.zeros_like(roi, dtype=np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(vraw, connectivity=8)
    min_h = int(0.30 * H)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if h >= min_h and w <= 8 and area > 0:
            vmask[labels == i] = 255

    out = np.zeros_like(bw_inv, dtype=np.uint8)
    out[y1:y2, x1:x2] = vmask
    return out

def find_column_bounds(vmask: np.ndarray, x_ref: int, y_top: int, H: int, search_margin_px: int = 400):
    x_lo = max(0, x_ref - search_margin_px)
    x_hi = min(vmask.shape[1] - 1, x_ref + search_margin_px)
    colsum = vmask[y_top:H, :].sum(axis=0)
    xs = np.where(colsum > 0)[0]
    xs = xs[(xs >= x_lo) & (xs <= x_hi)]
    if xs.size == 0:
        return None, None
    xs_sorted = np.sort(xs)
    cols = [[xs_sorted[0]]]
    for x in xs_sorted[1:]:
        if x - cols[-1][-1] <= 4:
            cols[-1].append(x)
        else:
            cols.append([x])
    centers = [int(round(np.mean(c))) for c in cols]
    lefts  = [x for x in centers if x <= x_ref]
    rights = [x for x in centers if x >= x_ref]
    if not lefts or not rights:
        return None, None
    xL = max(lefts); xR = min(rights)
    if xR - xL < 12:
        return None, None
    return xL, xR

# ---------------- grid removal on the crop ----------------
def degrid_crop(gray_crop: np.ndarray) -> np.ndarray:
    Hc, Wc = gray_crop.shape
    blur = cv2.GaussianBlur(gray_crop, (3,3), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    Kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, int(0.15*Hc))))
    Kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(0.18*Wc)), 1))
    v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)
    h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)
    mask = cv2.bitwise_or(v_candidates, h_candidates)
    if mask.sum() == 0:
        return gray_crop.copy()
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)
    crop_bgr = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    cleaned  = cv2.inpaint(crop_bgr, mask, 2, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

# ---------------- OCR numeric detection in the cropped CKT column ----------------

# Strictly numeric only (after cleaning common OCR confusions)
def _fix_ocr_digits(s: str) -> str:
    s = s.strip()
    s = s.replace("O", "0").replace("o", "0")
    s = s.replace("I", "1").replace("l", "1").replace("|", "1")
    s = s.replace("S", "5")
    s = s.replace("B", "8")
    return s

def is_pure_numeric(txt: str) -> bool:
    if not txt:
        return False
    t = _fix_ocr_digits(txt)
    t = re.sub(r"\s+", "", t)   # remove spaces
    return bool(re.fullmatch(r"\d+", t))

def ocr_numeric_tokens_in_crop(crop_gray: np.ndarray, ctx: OCRCtx):
    det = []
    for m in (1.4, 1.8, 2.2):
        part = _safe_readtext(
            ctx, crop_gray, detail=1, paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .-/",
            mag_ratio=m, contrast_ths=0.05, adjust_contrast=0.7,
            text_threshold=0.4, low_text=0.25
        ) or []
        det += part

    out=[]
    for box, txt, conf in det:
        if not txt or not box: continue
        if not is_pure_numeric(txt):  # STRICT: token must be digits only
            continue
        xs=[int(p[0]) for p in box]; ys=[int(p[1]) for p in box]
        out.append({
            "text": txt, "conf": float(conf or 0.0),
            "x1": int(min(xs)), "x2": int(max(xs)),
            "y1": int(min(ys)), "y2": int(max(ys)),
        })
    if not out:
        return None, []
    out.sort(key=lambda t: t["y2"])  # bottom-most = footer text
    return out[-1], out

# ---------------- drawing ----------------
def draw_header_overlay_full(gray, hdr_text_y, hdr_line_y, sob_line_y, ftr_text_y, ftr_line_y):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape
    if hdr_text_y is not None:
        y=int(hdr_text_y)
        cv2.line(vis,(0,y),(W-1,y),(0,150,0),1)
        cv2.putText(vis,"HEADER_TEXT",(10,max(20,y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,150,0),1,cv2.LINE_AA)
    if hdr_line_y is not None:
        y=int(hdr_line_y)
        cv2.line(vis,(0,y),(W-1,y),(0,255,0),2)
        cv2.putText(vis,"HEADER_LINE",(10,max(20,y-24)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)
    if sob_line_y is not None:
        y=int(sob_line_y)
        cv2.line(vis,(0,y),(W-1,y),(255,0,0),2)
        cv2.putText(vis,"START_OF_BREAKERS",(10,max(20,y+18)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2,cv2.LINE_AA)
    if ftr_text_y is not None:
        y=int(ftr_text_y)
        cv2.line(vis,(0,y),(W-1,y),(150,0,150),1)
        cv2.putText(vis,"FOOTER_TEXT",(10,max(20,y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,0,150),1,cv2.LINE_AA)
    if ftr_line_y is not None:
        y=int(ftr_line_y)
        cv2.line(vis,(0,y),(W-1,y),(0,0,255),2)
        cv2.putText(vis,"FOOTER_LINE",(10,max(20,y+18)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)
    return vis

def draw_hmask_with_markers(gray, hmask_thick, hdr_text_y, hdr_line_y, sob_line_y, ftr_text_y, ftr_line_y):
    H, W = gray.shape
    vis = cv2.cvtColor(hmask_thick, cv2.COLOR_GRAY2BGR)
    if hdr_text_y is not None:
        y=int(hdr_text_y); cv2.line(vis,(0,y),(W-1,y),(0,150,0),1)
        cv2.putText(vis,"HDR_TEXT",(10,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,150,0),1,cv2.LINE_AA)
    if hdr_line_y is not None:
        y=int(hdr_line_y); cv2.line(vis,(0,y),(W-1,y),(0,255,0),2)
        cv2.putText(vis,"HDR_LINE",(120,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1,cv2.LINE_AA)
    if sob_line_y is not None:
        y=int(sob_line_y); cv2.line(vis,(0,y),(W-1,y),(255,0,0),2)
        cv2.putText(vis,"SOB",(120,max(20,y+18)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1,cv2.LINE_AA)
    if ftr_text_y is not None:
        y=int(ftr_text_y); cv2.line(vis,(0,y),(W-1,y),(150,0,150),1)
        cv2.putText(vis,"FTR_TEXT",(10,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(150,0,150),1,cv2.LINE_AA)
    if ftr_line_y is not None:
        y=int(ftr_line_y); cv2.line(vis,(0,y),(W-1,y),(0,0,255),2)
        cv2.putText(vis,"FTR_LINE",(120,max(20,y+18)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),1,cv2.LINE_AA)
    return vis

# ---------------- main ----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = gather_images(INPUT_DIR)
    if not files:
        print("[ERROR] No PNGs found to process."); return

    reader = make_easyocr_reader(FORCE_CPU_OCR, PREFER_GPU, MIN_FREE_MB)
    ctx = OCRCtx(reader)

    for img_path in files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        bgr  = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] cannot read: {img_path}"); continue

        gray = prep_gray(bgr)
        H, W = gray.shape

        # ---------- OCR header tokens ----------
        det_all_hdr, hdr_like = header_tokens(gray, ctx)
        hdr_text_y = int(min(t["y1"] for t in hdr_like)) if hdr_like else None

        # ---------- Horizontal mask + lines ----------
        bw_inv = binarize_ink(gray)
        hmask  = horiz_mask_from_bin(bw_inv, W, HORIZ_KERNEL_FRAC)
        hmask_thick = thicken_mask(hmask, THICKEN_MASK_PX)

        lines = extract_horiz_lines_from_mask(
            hmask_thick, W,
            min_width_frac=MIN_WIDTH_FRAC,
            max_thickness_px=MAX_THICK_PX+2,
            y_merge_tol_px=Y_MERGE_TOL_PX
        )

        # ----- header line -----
        hdr_line_y = None
        if hdr_text_y is not None:
            hdr_line_y = pick_nearest_line(
                lines, hdr_text_y,
                prefer_above=HEADER_PREFER_ABOVE,
                gap_min=1,
                search_px=HEADER_SEARCH_PX
            )
            if hdr_line_y is None:
                hdr_line_y = scan_white_band_near(
                    hdr_text_y, hmask_thick, W,
                    prefer_above=HEADER_PREFER_ABOVE,
                    search_px=HEADER_SEARCH_PX,
                    min_run_frac=MIN_WIDTH_FRAC
                )
        elif not SKIP_HEADER_IF_NO_TOKENS:
            approx_ref = int(0.08 * H)
            hdr_line_y = pick_nearest_line(lines, approx_ref, prefer_above=None, gap_min=1, search_px=HEADER_SEARCH_PX)
            if hdr_line_y is None:
                hdr_line_y = scan_white_band_near(approx_ref, hmask_thick, W, prefer_above=None, search_px=HEADER_SEARCH_PX)

        # ----- start-of-breakers (next line after header) -----
        sob_line_y = None
        if hdr_line_y is not None:
            sob_line_y = pick_nearest_line(
                lines, hdr_line_y + 2,
                prefer_above=SOB_PREFER_ABOVE,  # below
                gap_min=1,
                search_px=SOB_SEARCH_PX
            )
            if sob_line_y is None:
                sob_line_y = scan_white_band_near(
                    hdr_line_y + 2, hmask_thick, W,
                    prefer_above=False,
                    search_px=SOB_SEARCH_PX,
                    min_run_frac=MIN_WIDTH_FRAC
                )

        # Write the horizontal mask NOW (header + SOB only)
        vis_hmask = draw_hmask_with_markers(gray, hmask_thick, hdr_text_y, hdr_line_y, sob_line_y, None, None)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HORIZ_MASK.png"), vis_hmask)

        # ---------- Column crop (from SOURCE image), below SoB ----------
        cct_tok = first_cct_token(gray, ctx, right_frac=CCT_COL_RIGHT_FRAC)
        ftr_text_y = None
        ftr_line_y = None

        if (hdr_line_y is None) or (sob_line_y is None) or (cct_tok is None):
            print(f"[SKIP-CROP] {base}: missing pieces (hdr_line={hdr_line_y} sob={sob_line_y} cct_tok={'yes' if cct_tok else 'no'})")
        else:
            y_top = int(sob_line_y + 1)
            vmask = vert_mask_strong_roi(bw_inv, 0, y_top, W, H)
            x_ref = int((cct_tok["x1"] + cct_tok["x2"]) // 2)
            xL, xR = find_column_bounds(vmask, x_ref, y_top, H, search_margin_px=400)
            if xL is None or xR is None:
                print(f"[SKIP-CROP] {base}: could not bracket CKT column (xL={xL}, xR={xR}).")
            else:
                # crop from SOURCE image
                crop_bgr = bgr[y_top:H, xL:xR].copy()
                crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                crop_clean = degrid_crop(crop_gray)
                out_col = os.path.join(OUTPUT_DIR, f"{base}_CCT_COLUMN_GRIDLESS.png")
                cv2.imwrite(out_col, crop_clean)

                # ---------- STRICT numeric OCR on the crop ----------
                last_num_token, _all_num = ocr_numeric_tokens_in_crop(crop_clean, ctx)
                if last_num_token is None:
                    print(f"[WARN] {base}: no PURE numeric tokens found in CKT column crop.")
                else:
                    # Map crop coordinates back to full image
                    ftr_text_y = int(y_top + last_num_token["y2"])

                    # ---------- Find footer line: nearest *below* footer text ----------
                    # >>> CHANGE #2: inline logic that forces "below" <<<
                    ftr_line_y = None
                    if ftr_text_y is not None:
                        ref = int(ftr_text_y)
                        lo, hi = ref - int(FOOTER_SEARCH_PX), ref + int(FOOTER_SEARCH_PX)

                        cands = [ln for ln in lines
                                 if (ln["y_center"] >= ref + 1) and (lo <= ln["y_center"] <= hi)]

                        if cands:
                            ftr_line_y = int(min(cands, key=lambda ln: abs(ln["y_center"] - ref))["y_center"])
                        else:
                            ftr_line_y = scan_white_band_near(
                                ref, hmask_thick, W,
                                prefer_above=False,  # strictly below
                                search_px=FOOTER_SEARCH_PX,
                                min_run_frac=MIN_WIDTH_FRAC
                            )

        # ---------- Update overlays with footer ----------
        vis_hmask = draw_hmask_with_markers(gray, hmask_thick, hdr_text_y, hdr_line_y, sob_line_y, ftr_text_y, ftr_line_y)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HORIZ_MASK.png"), vis_hmask)

        vis_header = draw_header_overlay_full(gray, hdr_text_y, hdr_line_y, sob_line_y, ftr_text_y, ftr_line_y)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HEADER_OVERLAY.png"), vis_header)

        # ---------- Console summary ----------
        print(f"[OK] {base}: "
              f"hdr_text_y={hdr_text_y} hdr_line={hdr_line_y} "
              f"SOB={sob_line_y} | "
              f"footer_text_y={ftr_text_y} footer_line={ftr_line_y}")

if __name__ == "__main__":
    main()
