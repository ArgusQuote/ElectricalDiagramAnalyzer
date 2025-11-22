#!/usr/bin/env python3
import os, json, glob
from pathlib import Path
import cv2
import numpy as np

# ===================== USER VARIABLES =====================
SOURCE_DIR  = "~/Documents/Diagrams/CaseStudy_VectorCrop_Run10"   # <- folder with PNGs
OUTPUT_DIR  = "~/Documents/Diagrams/LineCounting2"               # <- where results go
RUN_LEN     = 5                                    # consecutive next_gap count
TOLERANCE   = 8                                     # +/- tolerance in px
MIN_WIDTH_FRAC   = 0.30                             # min line width vs image width
HKERNEL_FRAC     = 0.035                            # horizontal kernel length vs width
MAX_THICKNESS_PX = 5                                # max line thickness to accept
Y_MERGE_TOL_PX   = 2                                # merge detections within this y-distance
# ==========================================================

def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def detect_horizontal_lines(
    img_bgr: np.ndarray,
    horiz_kernel_frac: float = 0.035,   # kept for API; we’ll still use it
    min_width_frac: float = 0.20,       # was 0.30 → more forgiving
    max_thickness_px: int = 6,          # was 5 → allow a hair more
    y_merge_tol_px: int = 4,            # was 2 → AA / DPI jitter
):
    """
    Adaptive + multi-scale horizontal line extraction.
    Returns [{y_center,y_top,y_bottom,x_left,x_right,width,height}, ...]
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold → robust to shaded rows
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 10
    )

    # Multi-scale kernels: short, medium, long → union
    fracs = [max(0.02, horiz_kernel_frac * 0.7), horiz_kernel_frac, min(0.08, horiz_kernel_frac * 1.8)]
    masks = []
    for f in fracs:
        klen = max(5, int(round(W * f)))
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        m = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kh, iterations=1)
        m = cv2.dilate(m, None, iterations=1)
        m = cv2.erode(m, None, iterations=1)
        masks.append(m)

    horiz = masks[0]
    for m in masks[1:]:
        horiz = cv2.bitwise_or(horiz, m)

    cnts, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_width_px = int(round(W * min_width_frac))
    raws = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_width_px or h < 1 or h > max_thickness_px:
            continue
        # Re-find true run across the row center to ignore tiny gaps
        y_mid = y + h // 2
        row = horiz[y_mid, :]
        xs = np.where(row > 0)[0]
        if xs.size == 0:
            continue
        x_left = int(xs.min())
        x_right = int(xs.max())
        if (x_right - x_left + 1) < min_width_px:
            continue
        raws.append((y, y + h - 1, x_left, x_right))

    if not raws:
        return []

    # Merge near-duplicate y’s
    raws.sort(key=lambda r: (r[0] + r[1]) / 2.0)
    merged, cur = [], [raws[0]]
    for r in raws[1:]:
        y0, y1, _, _ = r
        py0, py1, _, _ = cur[-1]
        if abs(((y0 + y1) / 2.0) - ((py0 + py1) / 2.0)) <= y_merge_tol_px:
            cur.append(r)
        else:
            merged.append(cur); cur = [r]
    merged.append(cur)

    lines = []
    for grp in merged:
        y_t = min(g[0] for g in grp)
        y_b = max(g[1] for g in grp)
        x_l = min(g[2] for g in grp)
        x_r = max(g[3] for g in grp)
        y_c = (y_t + y_b) // 2
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

def add_gaps(lines):
    n = len(lines)
    for i in range(n):
        prev_gap = None if i == 0 else int(lines[i]["y_center"] - lines[i-1]["y_center"])
        next_gap = None if i == n - 1 else int(lines[i+1]["y_center"] - lines[i]["y_center"])
        lines[i]["prev_gap"] = prev_gap
        lines[i]["next_gap"] = next_gap
    return lines

def overlay_lines(img_bgr: np.ndarray, lines, thickness: int = 2) -> np.ndarray:
    out = img_bgr.copy()
    for ln in lines:
        y = (ln["y_top"] + ln["y_bottom"]) // 2
        cv2.line(out, (ln["x_left"], y), (ln["x_right"], y), (0, 0, 255), thickness)
    return out

def uniform_run_found(next_gaps, run_len=20, tol=8):
    """
    True if there exists a consecutive run of length >= run_len
    (without hitting a None) where every element stays within +/- tol
    of the running mean band.
    """
    i = 0
    N = len(next_gaps)
    while i < N:
        while i < N and next_gaps[i] is None:
            i += 1
        if i >= N:
            break

        total = 0.0
        count = 0
        lo = hi = None
        length = 0
        j = i
        while j < N:
            g = next_gaps[j]
            if g is None:
                break
            if count == 0:
                total = float(g); count = 1; length = 1
                lo = g - tol; hi = g + tol
            else:
                mean = total / count
                if abs(g - mean) <= tol and lo <= g <= hi:
                    total += g; count += 1; length += 1
                    lo = max(lo, int(mean - tol))
                    hi = min(hi, int(mean + tol))
                else:
                    break
            if length >= run_len:
                return True
            j += 1
        i = j + 1
    return False

def process_one(png_path: Path, out_root: Path) -> dict:
    """Process a single PNG, write outputs under out_root/<stem>/, and report pass/fail."""
    img = load_bgr(str(png_path))
    lines = detect_horizontal_lines(
        img,
        horiz_kernel_frac=HKERNEL_FRAC,
        min_width_frac=MIN_WIDTH_FRAC,
        max_thickness_px=MAX_THICKNESS_PX,
        y_merge_tol_px=Y_MERGE_TOL_PX,
    )
    lines = add_gaps(lines)

    # per-image output folder
    subdir = out_root / png_path.stem
    subdir.mkdir(parents=True, exist_ok=True)

    # overlay
    overlay = overlay_lines(img, lines, thickness=2)
    overlay_path = subdir / "overlay_horizontal_lines.png"
    cv2.imwrite(str(overlay_path), overlay)

    # full geometry + gaps
    lines_json_path = subdir / "lines.json"
    with open(lines_json_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2)

    # measurements-only (ordered next_gap including nulls)
    next_gaps = [ln.get("next_gap", None) for ln in lines]
    meas_json_path = subdir / "line_measurements.json"
    with open(meas_json_path, "w", encoding="utf-8") as f:
        json.dump({"next_gaps": next_gaps}, f, indent=2)

    # uniform run check
    has_uniform = uniform_run_found(next_gaps, run_len=RUN_LEN, tol=TOLERANCE)

    # summary
    summary = {
        "input_image": str(png_path.resolve()),
        "overlay_image": str(overlay_path.resolve()),
        "lines_json": str(lines_json_path.resolve()),
        "measurements_json": str(meas_json_path.resolve()),
        "run_len": RUN_LEN,
        "tolerance": TOLERANCE,
        "uniform_run_found": bool(has_uniform),
        "num_lines": len(lines),
    }
    summary_json_path = subdir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    src_dir = Path(os.path.expanduser(SOURCE_DIR))
    out_dir = Path(os.path.expanduser(OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted([Path(p) for p in glob.glob(str(src_dir / "*.png"))])

    results = []
    for p in pngs:
        try:
            res = process_one(p, out_dir)
            status = "PASS" if res["uniform_run_found"] else "FAIL"
            print(f"{p.name}: {status} | lines={res['num_lines']}")
            results.append(res)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            results.append({
                "input_image": str(p.resolve()),
                "error": str(e),
            })

    # write an index of all runs
    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_dir": str(src_dir.resolve()),
            "output_dir": str(out_dir.resolve()),
            "run_len": RUN_LEN,
            "tolerance": TOLERANCE,
            "min_width_frac": MIN_WIDTH_FRAC,
            "hkernel_frac": HKERNEL_FRAC,
            "max_thickness_px": MAX_THICKNESS_PX,
            "y_merge_tol_px": Y_MERGE_TOL_PX,
            "items": results,
        }, f, indent=2)

    print(f"\nProcessed {len(pngs)} PNGs")
    print(f"Index: {index_path}")

if __name__ == "__main__":
    main()
