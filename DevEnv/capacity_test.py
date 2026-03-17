#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import traceback
from pathlib import Path
from multiprocessing import get_context

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- IMPORTS ----------
from PageFilter.PageFilterV3 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch
from OcrLibrary.BreakerTableParserAPIv9 import BreakerTablePipeline, API_VERSION

# ---------- INPUT ----------
INPUT_PDF = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/derekfirst.pdf").expanduser()
TEST_ROOT = Path("~/ElectricalDiagramAnalyzer/DevEnv/CapacityTestOutput").expanduser()
TEST_ROOT.mkdir(parents=True, exist_ok=True)

JOB_TIMEOUT_SEC = 600  # 10 minutes

def run_one_job(job_idx: int, input_pdf: str, result_path: str):
    start = time.time()
    result = {
        "job_idx": job_idx,
        "ok": False,
        "error": None,
        "crop_count": 0,
        "header_breakers": 0,
        "table_breakers": 0,
        "elapsed_sec": None,
    }

    try:
        job_root = Path(TEST_ROOT) / f"job_{job_idx}"
        filter_out_dir = job_root / "PdfOutput"
        finder_out_dir = job_root / "PanelSearchOutput"
        pipe_out_dir = job_root / "ParserOutput"

        for d in (filter_out_dir, finder_out_dir, pipe_out_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 1) PageFilter
        filt = PageFilter(
            output_dir=str(filter_out_dir),
            dpi=400,
            longest_cap_px=9000,
            proc_scale=0.5,
            use_ocr=True,
            ocr_gpu=False,
            verbose=True,
            debug=False,
            rect_w_fr_range=(0.10, 0.55),
            rect_h_fr_range=(0.10, 0.60),
            min_rectangularity=0.70,
            min_rect_count=2,
        )
        kept_pages, dropped_pages, filtered_pdf, log_json = filt.readPdf(str(input_pdf))
        pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept_pages) > 0) else Path(input_pdf)

        # 2) PanelBoardSearch
        finder = PanelBoardSearch(
            output_dir=str(finder_out_dir),
            dpi=400,
            render_dpi=1400,
            aa_level=8,
            render_colorspace="gray",
            min_void_area_fr=0.004,
            min_void_w_px=90,
            min_void_h_px=90,
            max_void_area_fr=0.30,
            void_w_fr_range=(0.20, 0.60),
            void_h_fr_range=(0.15, 0.55),
            min_whitespace_area_fr=0.01,
            margin_shave_px=6,
            pad=6,
            verbose=True,
        )
        crops = finder.readPdf(str(pdf_for_finder))
        result["crop_count"] = len(crops)

        if not crops:
            raise RuntimeError("No panel crops found")

        # 3) BreakerTablePipeline
        pipe = BreakerTablePipeline(debug=True)

        total_hdr_breakers = 0
        total_tbl_breakers = 0

        for img_path in crops:
            parsed = pipe.run(img_path)
            stages = parsed.get("results") or {}
            hdr_res = stages.get("header") or {}
            tbl_res = stages.get("parser") or {}

            hdr_breakers = ((hdr_res or {}).get("attrs") or {}).get("detected_breakers") or []
            tbl_breakers = (tbl_res or {}).get("detected_breakers") or []

            total_hdr_breakers += len(hdr_breakers)
            total_tbl_breakers += len(tbl_breakers)

        result["header_breakers"] = total_hdr_breakers
        result["table_breakers"] = total_tbl_breakers
        result["ok"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    finally:
        result["elapsed_sec"] = round(time.time() - start, 2)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

def run_concurrency_test(concurrency: int, baseline_single_job_time: float | None = None):
    print(f"\n{'='*70}")
    print(f"TESTING CONCURRENCY = {concurrency}")
    print(f"{'='*70}")

    batch_root = TEST_ROOT / f"concurrency_{concurrency}"
    batch_root.mkdir(parents=True, exist_ok=True)

    ctx = get_context("spawn")
    procs = []
    result_files = []

    batch_start = time.time()

    for i in range(concurrency):
        result_path = batch_root / f"result_{i}.json"
        p = ctx.Process(
            target=run_one_job,
            args=(i, str(INPUT_PDF), str(result_path))
        )
        p.start()
        procs.append(p)
        result_files.append(result_path)

    for p in procs:
        p.join(timeout=JOB_TIMEOUT_SEC)
        if p.is_alive():
            print(f"  WARNING: Process {p.pid} exceeded timeout ({JOB_TIMEOUT_SEC}s). Terminating.")
            p.terminate()
            p.join(5)

    batch_elapsed = round(time.time() - batch_start, 2)

    results = []
    for rp in result_files:
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            results.append({
                "ok": False,
                "error": "missing result file",
                "elapsed_sec": None
            })

    success_count = sum(1 for r in results if r.get("ok"))
    failure_count = len(results) - success_count

    elapsed_values = [
        r.get("elapsed_sec")
        for r in results
        if isinstance(r.get("elapsed_sec"), (int, float))
    ]

    avg_elapsed = round(sum(elapsed_values) / max(1, len(elapsed_values)), 2) if elapsed_values else None
    min_elapsed = round(min(elapsed_values), 2) if elapsed_values else None
    max_elapsed = round(max(elapsed_values), 2) if elapsed_values else None

    slowdown_vs_baseline = None
    if baseline_single_job_time and avg_elapsed:
        slowdown_vs_baseline = round(avg_elapsed / baseline_single_job_time, 2)

    summary = {
        "concurrency": concurrency,
        "batch_elapsed_sec": batch_elapsed,
        "success_count": success_count,
        "failure_count": failure_count,
        "avg_job_elapsed_sec": avg_elapsed,
        "min_job_elapsed_sec": min_elapsed,
        "max_job_elapsed_sec": max_elapsed,
        "slowdown_vs_1x": slowdown_vs_baseline,
        "results": results,
    }

    summary_path = batch_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -------- Clean terminal output --------
    print(f"\nRESULTS FOR CONCURRENCY {concurrency}")
    print(f"  Success:              {success_count}/{len(results)}")
    print(f"  Failures:             {failure_count}")
    print(f"  Total batch time:     {batch_elapsed} sec")
    print(f"  Avg job time:         {avg_elapsed} sec")
    print(f"  Fastest job:          {min_elapsed} sec")
    print(f"  Slowest job:          {max_elapsed} sec")

    if slowdown_vs_baseline is not None:
        print(f"  Slowdown vs 1 job:    {slowdown_vs_baseline}x")

    print("\n  Per-job status:")
    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        job_idx = r.get("job_idx", "?")
        elapsed = r.get("elapsed_sec")
        crops = r.get("crop_count")
        err = r.get("error")
        line = f"    Job {job_idx}: {status} | time={elapsed}s | crops={crops}"
        if err:
            line += f" | error={err}"
        print(line)

    # -------- Simple interpretation --------
    print("\n  Quick read:")
    if failure_count > 0:
        print("    -> Not stable at this concurrency. Treat this as too high or investigate errors.")
    elif slowdown_vs_baseline is not None and slowdown_vs_baseline >= 3.0:
        print("    -> Stable, but slowdown is heavy. Usable only if queue delay is acceptable.")
    elif slowdown_vs_baseline is not None and slowdown_vs_baseline >= 2.0:
        print("    -> Likely workable, but this is starting to add noticeable load.")
    else:
        print("    -> Looks healthy based on completion/time alone.")

    print(f"\n  Saved summary to: {summary_path}")
    return summary

if __name__ == "__main__":
    print("API_VERSION:", API_VERSION)

    all_summaries = []

    baseline_summary = run_concurrency_test(1)
    all_summaries.append(baseline_summary)

    baseline_time = baseline_summary.get("avg_job_elapsed_sec")

    for n in [2, 3, 4]:
        summary = run_concurrency_test(n, baseline_single_job_time=baseline_time)
        all_summaries.append(summary)

    print(f"\n{'='*70}")
    print("FINAL CAPACITY SUMMARY")
    print(f"{'='*70}")

    for s in all_summaries:
        print(
            f"Concurrency {s['concurrency']}: "
            f"success={s['success_count']}/{s['success_count'] + s['failure_count']} | "
            f"avg={s['avg_job_elapsed_sec']} sec | "
            f"batch={s['batch_elapsed_sec']} sec | "
            f"slowdown_vs_1x={s.get('slowdown_vs_1x')}"
        )