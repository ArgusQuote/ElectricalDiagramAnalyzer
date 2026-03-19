#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Server-mirroring simultaneous capacity test.

Goal:
- Simulate 1, 2, 3, 4 jobs being submitted at the same time.
- Mirror the real server architecture where it matters:
    * spawn multiprocessing
    * cap thread pools before heavy imports
    * persistent worker-style OCR/model loading
    * EasyOCR GPU reader injection
    * BreakerTablePipeline warmup via _ensure_analyzer/_ensure_header_parser
    * same PageFilter / PanelBoardSearch / BreakerTablePipeline / RulesEngine flow
- Exclude queueing behavior entirely.

What this measures:
- Whether N simultaneous jobs complete successfully
- Per-job elapsed processing time
- Total wall-clock time for all simultaneous jobs
- Slowdown vs 1-job baseline
- Whether 2/3/4 simultaneous jobs are production-viable
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from multiprocessing import get_context
from typing import Optional

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------- THREAD / DETERMINISM SETUP ----------
# Must happen before heavy libs initialize
from WorkerSetup import cap_thread_pools, set_runtime_determinism
cap_thread_pools()

# ---------- IMPORTS ----------
from PageFilter.PageFilterV3 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch
from OcrLibrary.BreakerTableParserAPIv9 import (
    BreakerTablePipeline,
    API_VERSION,
    reset_name_deduper,
)
import RulesEngine.RulesEngine4 as RE2

# ---------- CONFIG ----------
INPUT_PDF = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/chucklarge.pdf").expanduser()
TEST_ROOT = Path("~/ElectricalDiagramAnalyzer/DevEnv/CapacityTestOutput").expanduser()
TEST_ROOT.mkdir(parents=True, exist_ok=True)

WORKER_READY_TIMEOUT_SEC = 240
JOB_TIMEOUT_SEC = 600  # 10 min hard cap per simultaneous run

CONCURRENCY_LEVELS = [1, 2, 3, 4]

# Keep these aligned with your real server
SERVER_PANEL_FINDER_DEFAULTS = {
    "render_dpi": 1400,
    "aa_level": 8,
    "render_colorspace": "gray",
    "min_void_area_fr": 0.004,
    "min_void_w_px": 90,
    "min_void_h_px": 90,
    "max_void_area_fr": 0.30,
    "void_w_fr_range": (0.20, 0.60),
    "void_h_fr_range": (0.15, 0.55),
    "min_whitespace_area_fr": 0.01,
    "margin_shave_px": 6,
    "pad": 6,
    "verbose": True,
}

SERVER_UI_DEFAULTS = {
    "panelboards": {
        "bussing_material": "ALUMINUM",
        "allow_plug_on_breakers": True,
        "rating_type": "FULLY_RATED",
        "allow_feed_thru_lugs": True,
        "default_trim_style": "FLUSH",
        "enclosure": "NEMA1",
        "allow_square_d_spd": True,
    },
    "transformers": {
        "winding_material": "ALUMINUM",
        "temperature_rating": 150,
        "default_type": "3PHASESTANDARD",
        "weathershield": False,
        "mounting": "FLOOR",
        "resin_enclosure": "3R",
    },
    "disconnects": {
        "allow_littlefuse": True,
        "default_switch_type": "GENERAL_DUTY",
        "default_enclosure": "NEMA1",
        "default_fusible": True,
        "default_ground_required": True,
        "default_solid_neutral": True,
    },
}


# ---------- HELPERS ----------
def _to_int_or_none(x):
    try:
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None


def _count_would_skip_breakers(breakers: list[dict], panel_limit: Optional[int]) -> int:
    """
    Matches server behavior for 2-strikes suppression.
    """
    total_bad = 0

    def _amp_of(b):
        try:
            return int(str(b.get("amperage", "")).replace(",", "").strip())
        except Exception:
            return None

    for b in (breakers or []):
        amps = _amp_of(b)
        try:
            qty = int(b.get("count", 1))
        except Exception:
            qty = 1

        bad = False
        if amps is None:
            bad = True
        elif amps < 15 or amps > 1200:
            bad = True
        elif isinstance(panel_limit, int) and panel_limit > 0 and amps > panel_limit:
            bad = True

        if bad:
            total_bad += max(1, qty)

    return total_bad


def _normalize_component_for_none(obj):
    import re
    try:
        import numpy as np
        NP_INT = np.integer
        NP_FLT = np.floating
    except Exception:
        class _NP:
            integer = ()
            floating = ()
        NP_INT, NP_FLT = _NP().integer, _NP().floating

    num_pat = re.compile(r"^-?\d+(\.\d+)?$")

    def coerce(v):
        if v is None:
            return "NONE"
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, NP_INT):
            return int(v)
        if isinstance(v, NP_FLT):
            return int(v) if float(v).is_integer() else float(v)
        if isinstance(v, (int, float)):
            return int(v) if float(v).is_integer() else float(v)
        if isinstance(v, str):
            s = v.strip()
            if s.upper() == "NONE":
                return "NONE"
            if num_pat.match(s):
                try:
                    f = float(s)
                    return int(f) if f.is_integer() else f
                except Exception:
                    return s
            return s
        return v

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return coerce(x)

    return walk(obj)


def _build_rules_payload(defaults: dict, items: list[dict]) -> dict:
    return {"defaults": defaults or {}, "items": items or []}


def _merge_component_from_btp(result_dict: dict, src_img: str) -> dict:
    """
    Same mapping logic as your server.
    """
    stages = (result_dict or {}).get("results") or {}
    hdr = stages.get("header") or {}
    prs = stages.get("parser") or {}

    name = hdr.get("name") or ""
    h_attrs = hdr.get("attrs") or {}

    def _get_int_from_header(*keys):
        for k in keys:
            v = h_attrs.get(k)
            if v is not None:
                iv = _to_int_or_none(v)
                if iv is not None:
                    return iv
        return None

    amperage = _get_int_from_header("amperage", "main_amp", "mainAmperage")
    spaces_h = _get_int_from_header("spaces")
    voltage = _get_int_from_header("voltage")
    int_rating = _get_int_from_header("intRating", "interrupt_rating", "interruptRating", "kaic", "kaic_rating")
    main_amp = _get_int_from_header("mainBreakerAmperage", "main_breaker_amperage", "main_breaker", "mainBreaker")

    hdr_brkrs = list(h_attrs.get("detected_breakers") or [])
    spaces_t = _to_int_or_none((prs or {}).get("spaces"))
    tbl_brkrs = list((prs or {}).get("detected_breakers") or [])

    spaces = spaces_t if spaces_t is not None else spaces_h
    det_brkrs = hdr_brkrs + tbl_brkrs

    panel_limit = next((v for v in (main_amp, amperage) if isinstance(v, int) and v > 0), None)
    would_skip = _count_would_skip_breakers(det_brkrs, panel_limit)

    notes = []
    if would_skip >= 2:
        det_brkrs = []
        panel_name = name or Path(src_img).stem
        notes.append(
            f"No breakers supplied for '{panel_name}': {would_skip} breaker(s) failed validation "
            f"(2 or more breakers had detection errors - User review required)."
        )

    comp = {
        "type": "panelboard",
        "name": name,
        "source": src_img,
        "attrs": {
            "amperage": amperage,
            "spaces": spaces,
            "voltage": voltage,
            "intRating": int_rating,
            "mainBreakerAmperage": main_amp,
            "detected_breakers": det_brkrs,
        },
    }

    if notes:
        comp["notes"] = notes
        comp["attrs"]["breaker_data_suppressed"] = True

    return comp


def render_pdf_to_images(saved_pdf: Path, img_dir: Path, dpi: int = 400) -> list[str]:
    """
    Mirrors the server render path.
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> rendering PDF → images: {saved_pdf} -> {img_dir} (dpi={dpi})")

    try:
        pf = PageFilter(
            output_dir=str(img_dir.parent),
            dpi=400,
            longest_cap_px=9000,
            proc_scale=0.5,
            use_ocr=True,
            ocr_gpu=False,
            verbose=True,
            debug=False,
            rect_w_fr_range=(0.20, 0.60),
            rect_h_fr_range=(0.20, 0.60),
            min_rectangularity=0.70,
            min_rect_count=2,
            min_whitespace_area_fr=0.004,
            use_ghostscript_letter=True,
            letter_orientation="landscape",
            gs_use_cropbox=True,
            gs_compat="1.7",
        )
        kept_pages, dropped_pages, filtered_pdf, log_json = pf.readPdf(str(saved_pdf))
        print(f">>> PageFilter: kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")
    except Exception as e:
        print(f">>> PageFilter error: {e}")
        kept_pages, filtered_pdf = [], None

    pdf_for_finder = filtered_pdf if (filtered_pdf and len(kept_pages) > 0) else str(saved_pdf)
    if pdf_for_finder == str(saved_pdf) and (filtered_pdf is not None) and len(kept_pages) == 0:
        print(">>> PageFilter kept 0 pages — falling back to original PDF")

    finder = PanelBoardSearch(
        output_dir=str(img_dir),
        dpi=dpi,
        render_dpi=SERVER_PANEL_FINDER_DEFAULTS["render_dpi"],
        aa_level=SERVER_PANEL_FINDER_DEFAULTS["aa_level"],
        render_colorspace=SERVER_PANEL_FINDER_DEFAULTS["render_colorspace"],
        min_void_area_fr=SERVER_PANEL_FINDER_DEFAULTS["min_void_area_fr"],
        min_void_w_px=SERVER_PANEL_FINDER_DEFAULTS["min_void_w_px"],
        min_void_h_px=SERVER_PANEL_FINDER_DEFAULTS["min_void_h_px"],
        max_void_area_fr=SERVER_PANEL_FINDER_DEFAULTS["max_void_area_fr"],
        void_w_fr_range=SERVER_PANEL_FINDER_DEFAULTS["void_w_fr_range"],
        void_h_fr_range=SERVER_PANEL_FINDER_DEFAULTS["void_h_fr_range"],
        min_whitespace_area_fr=SERVER_PANEL_FINDER_DEFAULTS["min_whitespace_area_fr"],
        margin_shave_px=SERVER_PANEL_FINDER_DEFAULTS["margin_shave_px"],
        pad=SERVER_PANEL_FINDER_DEFAULTS["pad"],
        verbose=SERVER_PANEL_FINDER_DEFAULTS["verbose"],
    )

    crops = finder.readPdf(pdf_for_finder)
    print(f">>> rendered {len(crops)} image(s)")
    return crops


def run_server_like_job(
    job_idx: int,
    input_pdf: str,
    test_run_root: Path,
    pipeline,
) -> dict:
    """
    Mirrors your server job body but without queue/status/anvil plumbing.
    """
    start = time.time()

    job_root = test_run_root / f"job_{job_idx}"
    pdf_out_dir = job_root / "uploaded_pdfs"
    img_dir = job_root / "pdf_images"
    debug_dir = job_root / "debug"

    for d in (pdf_out_dir, img_dir, debug_dir):
        d.mkdir(parents=True, exist_ok=True)

    result = {
        "job_idx": job_idx,
        "ok": False,
        "error": None,
        "crop_count": 0,
        "component_count": 0,
        "header_breakers": 0,
        "table_breakers": 0,
        "rules_error": None,
        "elapsed_sec": None,
        "job_root": str(job_root),
    }

    try:
        reset_name_deduper()

        # Save/copy PDF into job folder so structure resembles server
        src_pdf = Path(input_pdf)
        saved_pdf = pdf_out_dir / src_pdf.name
        if not saved_pdf.exists():
            saved_pdf.write_bytes(src_pdf.read_bytes())

        imgs = render_pdf_to_images(saved_pdf, img_dir)
        result["crop_count"] = len(imgs)

        if not imgs:
            raise RuntimeError("PDF rendered but produced no crops/images.")

        components = []
        total_hdr_breakers = 0
        total_tbl_breakers = 0

        for idx, img_path in enumerate(imgs):
            try:
                raw = pipeline.run(
                    img_path,
                    run_analyzer=True,
                    run_parser=True,
                    run_header=True,
                )

                if isinstance(raw, dict) and "_error" in raw:
                    raise RuntimeError(raw["_error"])

                stages = raw.get("results") or {}
                hdr_res = stages.get("header") or {}
                tbl_res = stages.get("parser") or {}

                hdr_breakers = ((hdr_res or {}).get("attrs") or {}).get("detected_breakers") or []
                tbl_breakers = (tbl_res or {}).get("detected_breakers") or []

                total_hdr_breakers += len(hdr_breakers)
                total_tbl_breakers += len(tbl_breakers)

                comp = _merge_component_from_btp(raw or {}, img_path)
                comp = _normalize_component_for_none(comp or {})
                components.append(comp)

            except Exception as e:
                print(f">>> ERROR analyzing image {idx + 1}: {e}")
                print(traceback.format_exc())
                components.append({
                    "type": "panelboard",
                    "name": f"image_{idx + 1}",
                    "_skipped": True,
                    "reason": f"Parse error: {e}",
                    "attrs": {},
                    "source": img_path,
                })

        result["component_count"] = len(components)
        result["header_breakers"] = total_hdr_breakers
        result["table_breakers"] = total_tbl_breakers

        rules_payload = _build_rules_payload(SERVER_UI_DEFAULTS, components)
        try:
            rules_result = RE2.process_job(rules_payload) or {}
            if isinstance(rules_result, dict) and rules_result.get("error"):
                result["rules_error"] = rules_result.get("error")
        except Exception as re_err:
            rules_result = {"error": f"{type(re_err).__name__}: {re_err}"}
            result["rules_error"] = rules_result["error"]

        result_json_path = job_root / "result.json"
        result_dump = {
            "ok": True,
            "job_idx": job_idx,
            "saved_pdf": str(saved_pdf),
            "images": imgs,
            "image_count": len(imgs),
            "components": components,
            "rules_result": rules_result,
        }
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result_dump, f, indent=2, default=str)

        result["ok"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    finally:
        result["elapsed_sec"] = round(time.time() - start, 2)

    return result


# ---------- PERSISTENT WORKER ----------
def persistent_capacity_worker(
    worker_idx: int,
    input_pdf: str,
    test_run_root: str,
    ready_queue,
    done_queue,
    start_event,
):
    """
    Mirrors your persistent server worker:
    - set determinism
    - load easyocr gpu reader once
    - build BreakerTablePipeline once
    - warm analyzer/header once
    - wait for simultaneous release
    - run one job
    """
    set_runtime_determinism()

    init_start = time.time()

    try:
        import easyocr

        print(f">>> Worker {worker_idx}: loading EasyOCR GPU reader...")
        gpu_reader = easyocr.Reader(["en"], gpu=True)

        print(f">>> Worker {worker_idx}: building pipeline...")
        pipeline = BreakerTablePipeline(debug=True, reader=gpu_reader)

        print(f">>> Worker {worker_idx}: warming analyzer/header...")
        pipeline._ensure_analyzer()
        pipeline._ensure_header_parser()

        warmup_elapsed = round(time.time() - init_start, 2)

        ready_queue.put({
            "worker_idx": worker_idx,
            "ok": True,
            "warmup_sec": warmup_elapsed,
            "error": None,
        })

    except Exception as e:
        ready_queue.put({
            "worker_idx": worker_idx,
            "ok": False,
            "warmup_sec": round(time.time() - init_start, 2),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
        return

    # Wait until parent releases all jobs simultaneously
    start_event.wait()

    # Run one job
    try:
        result = run_server_like_job(
            job_idx=worker_idx,
            input_pdf=input_pdf,
            test_run_root=Path(test_run_root),
            pipeline=pipeline,
        )
        done_queue.put(result)

    except Exception as e:
        done_queue.put({
            "job_idx": worker_idx,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": None,
        })


# ---------- TEST RUNNER ----------
def run_simultaneous_capacity_test(
    simultaneous_jobs: int,
    baseline_single_job_time: Optional[float] = None,
) -> dict:
    """
    Run N jobs at the exact same time with N persistent workers.
    This is the closest non-queue approximation to:
    'How many jobs can production handle if they hit at once?'
    """
    print(f"\n{'=' * 80}")
    print(f"SIMULTANEOUS SERVER-LIKE JOB TEST = {simultaneous_jobs}")
    print(f"{'=' * 80}")

    test_run_root = TEST_ROOT / f"server_like_simul_{simultaneous_jobs}"
    test_run_root.mkdir(parents=True, exist_ok=True)

    ctx = get_context("spawn")
    ready_queue = ctx.Queue()
    done_queue = ctx.Queue()
    start_event = ctx.Event()

    procs = []
    ready_results = []
    job_results = []

    # Spawn workers
    spawn_start = time.time()
    for i in range(simultaneous_jobs):
        p = ctx.Process(
            target=persistent_capacity_worker,
            args=(
                i,
                str(INPUT_PDF),
                str(test_run_root),
                ready_queue,
                done_queue,
                start_event,
            ),
        )
        p.start()
        procs.append(p)

    # Wait for all workers to finish warmup
    all_ready_ok = True
    for _ in range(simultaneous_jobs):
        try:
            msg = ready_queue.get(timeout=WORKER_READY_TIMEOUT_SEC)
            ready_results.append(msg)
            if not msg.get("ok"):
                all_ready_ok = False
        except Exception:
            ready_results.append({
                "worker_idx": "?",
                "ok": False,
                "warmup_sec": None,
                "error": f"Worker did not report ready within {WORKER_READY_TIMEOUT_SEC}s",
            })
            all_ready_ok = False

    if not all_ready_ok:
        print("\nOne or more workers failed during warmup.")
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(5)

        summary = {
            "simultaneous_jobs": simultaneous_jobs,
            "ok": False,
            "phase": "warmup",
            "ready_results": ready_results,
            "results": [],
        }

        summary_path = test_run_root / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to: {summary_path}")
        return summary

    # Release all workers simultaneously
    batch_start = time.time()
    start_event.set()

    # Collect all results
    deadline = time.time() + JOB_TIMEOUT_SEC
    while len(job_results) < simultaneous_jobs and time.time() < deadline:
        try:
            timeout_remaining = max(0.1, deadline - time.time())
            msg = done_queue.get(timeout=timeout_remaining)
            job_results.append(msg)
        except Exception:
            break

    batch_elapsed = round(time.time() - batch_start, 2)

    # Kill any stragglers
    for p in procs:
        p.join(timeout=2)
        if p.is_alive():
            print(f"  WARNING: Worker pid={p.pid} exceeded timeout. Terminating.")
            p.terminate()
            p.join(5)

    # Fill in missing results if needed
    seen_job_idxs = {r.get("job_idx") for r in job_results}
    for i in range(simultaneous_jobs):
        if i not in seen_job_idxs:
            job_results.append({
                "job_idx": i,
                "ok": False,
                "error": "missing result (timeout/crash/no response)",
                "elapsed_sec": None,
            })

    job_results.sort(key=lambda r: r.get("job_idx", 999999))

    success_count = sum(1 for r in job_results if r.get("ok"))
    failure_count = len(job_results) - success_count

    elapsed_values = [
        r.get("elapsed_sec")
        for r in job_results
        if isinstance(r.get("elapsed_sec"), (int, float))
    ]

    avg_elapsed = round(sum(elapsed_values) / max(1, len(elapsed_values)), 2) if elapsed_values else None
    min_elapsed = round(min(elapsed_values), 2) if elapsed_values else None
    max_elapsed = round(max(elapsed_values), 2) if elapsed_values else None

    warmup_values = [
        r.get("warmup_sec")
        for r in ready_results
        if isinstance(r.get("warmup_sec"), (int, float))
    ]
    avg_warmup = round(sum(warmup_values) / max(1, len(warmup_values)), 2) if warmup_values else None
    max_warmup = round(max(warmup_values), 2) if warmup_values else None

    slowdown_vs_baseline = None
    if baseline_single_job_time and avg_elapsed:
        slowdown_vs_baseline = round(avg_elapsed / baseline_single_job_time, 2)

    summary = {
        "simultaneous_jobs": simultaneous_jobs,
        "api_version": API_VERSION,
        "input_pdf": str(INPUT_PDF),
        "batch_elapsed_sec": batch_elapsed,
        "success_count": success_count,
        "failure_count": failure_count,
        "avg_job_elapsed_sec": avg_elapsed,
        "min_job_elapsed_sec": min_elapsed,
        "max_job_elapsed_sec": max_elapsed,
        "avg_worker_warmup_sec": avg_warmup,
        "max_worker_warmup_sec": max_warmup,
        "slowdown_vs_1x": slowdown_vs_baseline,
        "ready_results": ready_results,
        "results": job_results,
    }

    summary_path = test_run_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---------- Clean terminal output ----------
    print(f"\nRESULTS FOR {simultaneous_jobs} SIMULTANEOUS JOB(S)")
    print(f"  Success:              {success_count}/{len(job_results)}")
    print(f"  Failures:             {failure_count}")
    print(f"  Total wall time:      {batch_elapsed} sec")
    print(f"  Avg job time:         {avg_elapsed} sec")
    print(f"  Fastest job:          {min_elapsed} sec")
    print(f"  Slowest job:          {max_elapsed} sec")
    print(f"  Avg worker warmup:    {avg_warmup} sec")
    print(f"  Max worker warmup:    {max_warmup} sec")

    if slowdown_vs_baseline is not None:
        print(f"  Slowdown vs 1 job:    {slowdown_vs_baseline}x")

    print("\n  Per-worker warmup:")
    for r in sorted(ready_results, key=lambda x: x.get("worker_idx", 999999)):
        status = "READY" if r.get("ok") else "FAIL"
        print(
            f"    Worker {r.get('worker_idx')}: {status} | "
            f"warmup={r.get('warmup_sec')}s"
            + (f" | error={r.get('error')}" if r.get("error") else "")
        )

    print("\n  Per-job status:")
    for r in job_results:
        status = "OK" if r.get("ok") else "FAIL"
        line = (
            f"    Job {r.get('job_idx')}: {status} | "
            f"time={r.get('elapsed_sec')}s | "
            f"crops={r.get('crop_count')} | "
            f"components={r.get('component_count')} | "
            f"hdr_breakers={r.get('header_breakers')} | "
            f"tbl_breakers={r.get('table_breakers')}"
        )
        if r.get("rules_error"):
            line += f" | rules_error={r.get('rules_error')}"
        if r.get("error"):
            line += f" | error={r.get('error')}"
        print(line)

    print("\n  Quick read:")
    if failure_count > 0:
        print("    -> Not stable at this simultaneous load.")
    elif slowdown_vs_baseline is not None and slowdown_vs_baseline >= 3.0:
        print("    -> Stable, but slowdown is heavy. Production use would likely feel rough.")
    elif slowdown_vs_baseline is not None and slowdown_vs_baseline >= 2.0:
        print("    -> Probably workable, but load is becoming noticeable.")
    else:
        print("    -> Looks healthy based on simultaneous completion and timing.")

    print(f"\n  Saved summary to: {summary_path}")
    return summary


# ---------- MAIN ----------
if __name__ == "__main__":
    print("API_VERSION:", API_VERSION)
    print("INPUT_PDF:", INPUT_PDF)

    if not INPUT_PDF.exists():
        raise FileNotFoundError(f"Input PDF not found: {INPUT_PDF}")

    all_summaries = []

    # Baseline
    baseline_summary = run_simultaneous_capacity_test(1)
    all_summaries.append(baseline_summary)
    baseline_time = baseline_summary.get("avg_job_elapsed_sec")

    # Simultaneous loads
    for n in [2, 3, 4]:
        summary = run_simultaneous_capacity_test(n, baseline_single_job_time=baseline_time)
        all_summaries.append(summary)

    print(f"\n{'=' * 80}")
    print("FINAL SIMULTANEOUS CAPACITY SUMMARY")
    print(f"{'=' * 80}")

    for s in all_summaries:
        print(
            f"Simultaneous {s['simultaneous_jobs']}: "
            f"success={s.get('success_count', 0)}/{s.get('success_count', 0) + s.get('failure_count', 0)} | "
            f"avg={s.get('avg_job_elapsed_sec')} sec | "
            f"wall={s.get('batch_elapsed_sec')} sec | "
            f"slowdown_vs_1x={s.get('slowdown_vs_1x')}"
        )