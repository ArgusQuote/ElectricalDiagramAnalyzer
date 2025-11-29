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

# Horizontal mask params (header work)
HORIZ_KERNEL_FRAC = 0.035
MIN_WIDTH_FRAC    = 0.20
MAX_THICK_PX      = 6
Y_MERGE_TOL_PX    = 4
THICKEN_MASK_PX   = 3

# Header picking
HEADER_PREFER_ABOVE = True
HEADER_SEARCH_PX    = 140
SKIP_HEADER_IF_NO_TOKENS = True

# CKT/CCT column logic
CCT_COL_RIGHT_FRAC = 0.30          # only search tokens in left ~30% of the page
MIN_COL_WIDTH_PX   = 30            # fallback min width if we can't find both verticals
PAD_COL_PX         = 8             # pad left/right a bit around detected verticals

# Vertical detector params
V_MIN_HEIGHT_FRAC   = 0.30   # keep components at least 30% of image height
V_MAX_THICK_PX      = 8      # max thickness to still consider a “line”
V_HOUGH_ENABLE      = True   # fallback if morphology returns nothing
V_HOUGH_MINLEN_FRAC = 0.25   # minimum vertical length (fraction of H) for Hough
V_HOUGH_GAP_PX      = 6

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
    # IMPORTANT: it's THRESH_OTSU (not cv2.OTSU)
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

def vert_mask_strong(bw_inv: np.ndarray, H: int, W: int, y_min:int=0) -> np.ndarray:
    """
    Strong vertical detector below a given y_min:
      - multi-scale vertical kernels (OPEN → CLOSE)
      - component filter (tall & skinny)
      - Hough fallback (optional)
    """
    work = bw_inv.copy()
    if y_min > 0:
        work[:y_min,:] = 0

    # 1) Multi-scale morphology
    scales = [0.10, 0.16, 0.22, 0.30]
    masks=[]
    for f in scales:
        klen = max(7, int(round(H*f)))
        kv   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
        m    = cv2.morphologyEx(work, cv2.MORPH_OPEN, kv, iterations=1)
        m    = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        masks.append(m)
    vraw = masks[0]
    for m in masks[1:]:
        vraw = cv2.bitwise_or(vraw, m)

    # 2) Component filter (keep tall & skinny)
    vmask = np.zeros_like(vraw, dtype=np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(vraw, connectivity=8)
    min_h = int(max(10, V_MIN_HEIGHT_FRAC * (H - y_min)))
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if y < y_min:   # enforce "below header"
            continue
        if h >= min_h and w <= V_MAX_THICK_PX and area > 0:
            vmask[labels == i] = 255

    # 3) Hough fallback if empty
    if vmask.sum() == 0 and V_HOUGH_ENABLE:
        edges = cv2.Canny(work, 60, 160)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=140,
            minLineLength=int(V_HOUGH_MINLEN_FRAC*H),
            maxLineGap=V_HOUGH_GAP_PX
        )
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0]:
                if y1 < y_min and y2 < y_min:  # ignore above-header artifacts
                    continue
                if abs(x1-x2) <= 2 and abs(y2-y1) >= int(V_HOUGH_MINLEN_FRAC*H):
                    cv2.line(vmask, (x1, y1), (x2, y2), 255, 2)

    return vmask

def thicken_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    if pixels <= 1: return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (pixels, pixels))
    return cv2.dilate(mask, k, iterations=1)

# ---- horizontal lines from mask (for header work) ----
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

def first_cct_token_below_header(all_hdr_tokens, header_y, W, right_frac_limit=0.30):
    """Pick the first token that *looks* like CKT/CCT and is below header_y and in left block."""
    if header_y is None: return None
    x_max = int(W * right_frac_limit)
    hits = []
    for t in all_hdr_tokens:
        n = NORM(t["text"])
        if ("CKT" in n) or ("CCT" in n):
            if t["y1"] > header_y + 4 and t["x1"] < x_max:
                hits.append(t)
    if not hits:
        return None
    # choose the topmost below header (closest to header)
    return sorted(hits, key=lambda d: d["y1"])[0]

# ---------------- drawing ----------------
def draw_hmask_with_markers(gray, hmask_thick, hdr_text_y, hdr_line_y):
    H, W = gray.shape
    vis = cv2.cvtColor(hmask_thick, cv2.COLOR_GRAY2BGR)
    if hdr_text_y is not None:
        y=int(hdr_text_y); cv2.line(vis,(0,y),(W-1,y),(0,150,0),1)
        cv2.putText(vis,"HDR_TEXT",(10,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,150,0),1,cv2.LINE_AA)
    if hdr_line_y is not None:
        y=int(hdr_line_y); cv2.line(vis,(0,y),(W-1,y),(0,255,0),2)
        cv2.putText(vis,"HDR_LINE",(120,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1,cv2.LINE_AA)
    return vis

def draw_col_debug(vis_shape, token, xL, xR, header_y):
    H, W = vis_shape[:2]
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    if header_y is not None:
        cv2.line(vis, (0,int(header_y)), (W-1,int(header_y)), (0,150,0), 1)
        cv2.putText(vis,"HDR_LINE",(10,max(18,int(header_y)-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,150,0),1,cv2.LINE_AA)
    if token:
        cv2.rectangle(vis,(token["x1"],token["y1"]), (token["x2"],token["y2"]), (255,0,255), 1)
        cv2.putText(vis, "CKT_TOK", (token["x1"], max(10, token["y1"]-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,255), 1, cv2.LINE_AA)
    if xL is not None:
        cv2.line(vis,(int(xL),0),(int(xL),H-1),(255,255,0),1)
    if xR is not None:
        cv2.line(vis,(int(xR),0),(int(xR),H-1),(255,255,0),1)
    return vis

# ---------------- grid removal in crop ----------------
def remove_grids_in_crop(crop_gray: np.ndarray) -> np.ndarray:
    Hc, Wc = crop_gray.shape
    blur = cv2.GaussianBlur(crop_gray, (3,3), 0)
    bw   = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 10)

    # detect long vertical and horizontal lines
    Kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, int(0.20*Hc))))
    Kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(0.20*Wc)), 1))
    v_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kv, iterations=1)
    h_candidates = cv2.morphologyEx(bw, cv2.MORPH_OPEN, Kh, iterations=1)

    # filter long & skinny comp
    mask = np.zeros_like(bw, dtype=np.uint8)
    for cand in (v_candidates, h_candidates):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for i in range(1, num):
            x,y,w,h,area = stats[i]
            if (h >= int(0.35*Hc) and w <= 4) or (w >= int(0.35*Wc) and h <= 4):
                mask[labels == i] = 255

    if mask.sum() == 0:
        return crop_gray

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
    inpainted = cv2.inpaint(cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR), mask, 2, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)

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

        # ---------- OCR tokens (header band) ----------
        det_all_hdr, hdr_like = header_tokens(gray, ctx)
        hdr_text_y = int(min(t["y1"] for t in hdr_like)) if hdr_like else None

        # ---------- Horizontal lines + header line (ALWAYS save overlay) ----------
        bw_inv = binarize_ink(gray)
        hmask  = horiz_mask_from_bin(bw_inv, W, HORIZ_KERNEL_FRAC)
        hmask_thick = thicken_mask(hmask, THICKEN_MASK_PX)
        lines = extract_horiz_lines_from_mask(
            hmask_thick, W,
            min_width_frac=MIN_WIDTH_FRAC,
            max_thickness_px=MAX_THICK_PX+2,
            y_merge_tol_px=Y_MERGE_TOL_PX
        )

        hdr_line_y = None
        if hdr_text_y is not None:
            hdr_line_y = pick_nearest_line(
                lines, hdr_text_y,
                prefer_above=HEADER_PREFER_ABOVE,
                gap_min=1,
                search_px=HEADER_SEARCH_PX
            )
            if hdr_line_y is None:
                # brute band scan near header text
                hdr_line_y = int(np.clip(hdr_text_y+2, 0, H-1))
                # try to find the first strong white run near it
                row = hmask_thick[hdr_line_y, :]
                if row.sum()==0 and hdr_line_y+1 < H:
                    hdr_line_y += 1
        elif not SKIP_HEADER_IF_NO_TOKENS and lines:
            hdr_line_y = int(lines[0]["y_center"])

        # SAVE horizontal overlay no matter what
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_HORIZ_MASK.png"),
                    draw_hmask_with_markers(gray, hmask_thick, hdr_text_y, hdr_line_y))

        # ---------- CKT/CCT token below header ----------
        cct_tok = first_cct_token_below_header(det_all_hdr, hdr_line_y, W, right_frac_limit=CCT_COL_RIGHT_FRAC)

        # ---------- Vertical bracketing BELOW header ----------
        xL, xR = None, None
        vmask  = vert_mask_strong(bw_inv, H, W, y_min=(hdr_line_y+2 if hdr_line_y else 0))
        # estimate candidate vertical xs by column density
        col_density = vmask.sum(axis=0)
        thr = 0.25 * float(col_density.max() or 1)
        xs = np.where(col_density >= thr)[0]

        if cct_tok:
            x_center = int(0.5*(cct_tok["x1"] + cct_tok["x2"]))
            # nearest left and right vertical guides
            lefts  = [x for x in xs if x <  x_center]
            rights = [x for x in xs if x >  x_center]
            if lefts:  xL = int(max(lefts))
            if rights: xR = int(min(rights))

        # Fallbacks if one/both sides missing
        if cct_tok and (xL is None or xR is None):
            # try gentle morphology on bw_inv to find vertical runs near token
            x1t, x2t = cct_tok["x1"], cct_tok["x2"]
            xL = xL if xL is not None else max(0, x1t - PAD_COL_PX - MIN_COL_WIDTH_PX)
            xR = xR if xR is not None else min(W-1, x2t + PAD_COL_PX + MIN_COL_WIDTH_PX)

        # Final clamp + min width
        if cct_tok:
            if xL is None or xR is None or (xR - xL + 1) < MIN_COL_WIDTH_PX:
                # absolute worst-case: local token-based window
                xc = int(0.5*(cct_tok["x1"] + cct_tok["x2"]))
                half = max(MIN_COL_WIDTH_PX//2, (cct_tok["x2"] - cct_tok["x1"])//2 + 10)
                xL = int(np.clip(xc - half, 0, W-2))
                xR = int(np.clip(xc + half, xL + MIN_COL_WIDTH_PX, W-1))

        # ---------- Crop from SOURCE BGR + degrid ----------
        wrote_any = False
        if cct_tok and xL is not None and xR is not None and hdr_line_y is not None:
            y1 = int(min(H-1, max(0, hdr_line_y + 2)))
            y2 = H - 1
            x1 = int(np.clip(xL - PAD_COL_PX, 0, W-1))
            x2 = int(np.clip(xR + PAD_COL_PX, x1 + 1, W-1))
            col_bgr  = bgr[y1:y2, x1:x2]
            col_gray = gray[y1:y2, x1:x2]

            # save raw crop
            raw_path = os.path.join(OUTPUT_DIR, f"{base}_COL_CROP_raw.png")
            cv2.imwrite(raw_path, col_bgr)

            # remove grids
            gridless_gray = remove_grids_in_crop(col_gray)
            gridless_bgr  = cv2.cvtColor(gridless_gray, cv2.COLOR_GRAY2BGR)
            gl_path = os.path.join(OUTPUT_DIR, f"{base}_COL_CROP_gridless.png")
            cv2.imwrite(gl_path, gridless_bgr)
            wrote_any = True

            # save debug column bracket overlay
            dbg_path = os.path.join(OUTPUT_DIR, f"{base}_COL_DEBUG.png")
            cv2.imwrite(dbg_path, draw_col_debug((H,W,3), cct_tok, xL, xR, hdr_line_y))

            print(f"[OK] {base}: header_line={hdr_line_y} | CKT token @ x=({cct_tok['x1']},{cct_tok['x2']}) "
                  f"→ col [{xL},{xR}]  raw={raw_path} gridless={gl_path}")
        else:
            # even if we failed bracketing, write a debug image so you can see why
            dbg_path = os.path.join(OUTPUT_DIR, f"{base}_COL_DEBUG.png")
            cv2.imwrite(dbg_path, draw_col_debug((H,W,3), cct_tok, xL, xR, hdr_line_y))
            print(f"[WARN] {base}: could not confidently bracket CKT column "
                  f"(hdr_line={hdr_line_y}, token={'yes' if cct_tok else 'no'}) → wrote COL_DEBUG only.")

        # summary line (and we *already* wrote HORIZ_MASK.png earlier)
        print(f"       vmask={'yes' if vmask.sum()>0 else 'no'} | wrote_crops={'yes' if wrote_any else 'no'}")

if __name__ == "__main__":
    main()
