#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two PNGs and reveal subtle differences that can affect OCR/thresholding.

Dependencies: pip install pillow opencv-python numpy
"""

import os, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# ---------- EDIT THESE ----------
IMG_A   = os.path.expanduser("~/Documents/Diagrams/CaseStudy1/Panels_Example_page001_table01_rect.png")
IMG_B   = os.path.expanduser("~/Documents/Diagrams/CaseStudy5/Last4pgs_page002_table01_rect.png")
OUT_DIR = os.path.expanduser("~/Documents/Diagrams/CompareOut")
# Align mode: "resize_to_A" (default) or "error_if_mismatch"
ALIGN_MODE = "resize_to_A"
# -------------------------------

def _pil_meta(p: str):
    info = {}
    with Image.open(p) as im:
        info["format"]   = im.format
        info["mode"]     = im.mode
        info["size_px"]  = {"w": im.size[0], "h": im.size[1]}
        dpi = im.info.get("dpi")
        if isinstance(dpi, tuple) and len(dpi) >= 2:
            info["dpi"] = {"x": dpi[0], "y": dpi[1]}
        elif dpi is not None:
            info["dpi"] = {"x": dpi, "y": dpi}
        else:
            info["dpi"] = None
        # PNG-affecting chunks
        info["png_chunks"] = {}
        for k in ("icc_profile", "gamma", "gAMA", "sRGB", "transparency", "background"):
            if k in im.info:
                v = im.info[k]
                info["png_chunks"][k] = f"<{len(v)} bytes>" if isinstance(v, (bytes, bytearray)) else v
    return info

def _ensure_bgr(img):
    if img.ndim == 2:  # gray
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # BGRA -> BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def _to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def _ssim(img1, img2):
    img1 = img1.astype(np.float32); img2 = img2.astype(np.float32)
    C1 = (0.01 * 255)**2; C2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5); window = kernel @ kernel.T
    mu1 = cv2.filter2D(img1, -1, window); mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.filter2D(img1*img1, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2*img2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1*img2, -1, window) - mu1_mu2
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / denom
    return float(ssim_map.mean())

def _side_by_side(A, B, amp):
    h = max(A.shape[0], B.shape[0], amp.shape[0])
    def pad(img):
        if img.shape[0] == h: return img
        top = (h - img.shape[0]) // 2
        bottom = h - img.shape[0] - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    return np.hstack([pad(A), pad(B), pad(amp)])

def main():
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)

    # Metadata (via Pillow)
    metaA = _pil_meta(IMG_A)
    metaB = _pil_meta(IMG_B)

    # Read raw (preserve channels)
    A_raw = cv2.imread(IMG_A, cv2.IMREAD_UNCHANGED)
    B_raw = cv2.imread(IMG_B, cv2.IMREAD_UNCHANGED)
    if A_raw is None or B_raw is None:
        raise FileNotFoundError("Could not read one or both images. Check paths.")
    A = _ensure_bgr(A_raw); B = _ensure_bgr(B_raw)

    same_size = (A.shape[0] == B.shape[0]) and (A.shape[1] == B.shape[1])
    if not same_size and ALIGN_MODE == "error_if_mismatch":
        raise ValueError(f"Different sizes: A={A.shape[:2]} B={B.shape[:2]}")
    if not same_size:
        # Resize B to A
        B_aligned = cv2.resize(B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "aligned_A.png"), A)
        cv2.imwrite(str(out / "aligned_B.png"), B_aligned)
    else:
        B_aligned = B

    # Grayscale for metrics & structural ops
    Ag = _to_gray(A); Bg = _to_gray(B_aligned)

    # Metrics
    psnr_val = cv2.PSNR(Ag, Bg)
    ssim_val = _ssim(Ag, Bg)

    # Differences
    absdiff = cv2.absdiff(Ag, Bg)
    cv2.imwrite(str(out / "absdiff.png"), absdiff)

    amp = cv2.convertScaleAbs(absdiff, alpha=4.0, beta=0)  # amplify differences
    cv2.imwrite(str(out / "diff_amplified.png"), amp)

    heat = cv2.applyColorMap(amp, cv2.COLORMAP_JET)
    cv2.imwrite(str(out / "heatmap.png"), heat)

    _, A_bin = cv2.threshold(cv2.GaussianBlur(Ag, (3,3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, B_bin = cv2.threshold(cv2.GaussianBlur(Bg, (3,3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    xor = cv2.bitwise_xor(A_bin, B_bin)
    cv2.imwrite(str(out / "xor_binarized.png"), xor)

    # Side-by-side visualization (A | B | amplified diff)
    A_vis = A.copy()
    B_vis = B_aligned.copy()
    amp_vis = cv2.cvtColor(amp, cv2.COLOR_GRAY2BGR)
    sbs = _side_by_side(A_vis, B_vis, amp_vis)
    cv2.imwrite(str(out / "side_by_side.png"), sbs)

    # Report
    nz = int(np.count_nonzero(absdiff))
    report = {
        "image_A": {"path": IMG_A, **metaA},
        "image_B": {"path": IMG_B, **metaB},
        "comparison": {
            "same_dimensions": same_size,
            "align_mode": ALIGN_MODE,
            "metrics": {"psnr": psnr_val, "ssim": ssim_val},
            "nonzero_diff_pixels": nz,
            "diff_fraction": float(nz / absdiff.size)
        },
        "notes": [
            "High SSIM (~0.99+) & PSNR (>=30 dB) → visually near-identical.",
            "If xor_binarized.png lights up, tiny pixel/gamma/size changes are shifting thresholds.",
            "To stabilize your pipeline, normalize crops to fixed size and strip alpha/ICC before analysis."
        ]
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Wrote outputs to: {out}")

if __name__ == "__main__":
    main()
