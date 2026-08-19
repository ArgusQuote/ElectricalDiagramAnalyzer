#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the DevTestEnv full-pipeline stack (PageFilterV5 + PanelSearchToolV26 +
BreakerTableParserAPIv14) over every PDF in a directory.

Each PDF gets its own intermediate artifact subdir under BATCH_ARTIFACT_ROOT and
a JobSummaries entry matching DevTestEnv.py output shape.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- IMPORTS ----------
from PageFilter.PageFilterV5 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV26 import PanelBoardSearch

# Parser API is selectable so a batch can be re-run against an older generation
# for a like-for-like comparison without editing this file.
_PARSER_APIS = {
    "13": "OcrLibrary.BreakerTableParserAPIv13",
    "14": "OcrLibrary.BreakerTableParserAPIv14",
    "15": "OcrLibrary.BreakerTableParserAPIv15",
}
DEFAULT_PARSER_API = "15"


def load_parser_api(version: str):
    """Import the requested BreakerTableParser API and return (pipeline_cls, version_str)."""
    import importlib

    try:
        module_name = _PARSER_APIS[version]
    except KeyError:
        raise ValueError(
            f"Unknown parser API {version!r}; choose one of {sorted(_PARSER_APIS)}"
        ) from None

    module = importlib.import_module(module_name)
    return module.BreakerTablePipeline, module.API_VERSION

DEFAULT_INPUT_DIR = Path("~/Documents/NewTest").expanduser()
BATCH_ARTIFACT_ROOT = Path("~/ElectricalDiagramAnalyzer/DevEnv/BatchRuns").expanduser()
SUMMARY_ROOT_DIR = Path("~/ElectricalDiagramAnalyzer/DevEnv/JobSummaries").expanduser()


def ms_to_readable(ms: int | float | None) -> str:
    if ms is None:
        return "N/A"

    total_ms = int(ms)
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    seconds = remainder // 1_000
    milliseconds = remainder % 1_000
    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"


def now_ts_ms() -> int:
    return int(time.time() * 1000)


def build_job_id(pdf_path: Path) -> str:
    base = pdf_path.stem.upper()
    safe = re.sub(r"[^A-Z0-9]+", "_", base).strip("_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe}__{stamp}"


def ensure_json_safe(value):
    if isinstance(value, dict):
        return {str(k): ensure_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [ensure_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [ensure_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_header_attrs(attrs: dict | None) -> dict:
    attrs = attrs or {}
    return {
        "amperage": attrs.get("amperage", "NONE"),
        "spaces": attrs.get("spaces", "NONE"),
        "voltage": attrs.get("voltage", "NONE"),
        "intRating": attrs.get("intRating", "NONE"),
        "mainBreakerAmperage": attrs.get("mainBreakerAmperage", "NONE"),
        "detected_breakers": ensure_json_safe(attrs.get("detected_breakers", [])),
    }


def build_component_summary(img_path: str, result: dict, unnamed_counts: dict[str, int]) -> dict:
    stages = result.get("results") or {}
    hdr_res = stages.get("header") or {}

    raw_name = str((hdr_res.get("name") or "")).strip()
    base_name = raw_name if raw_name else "(unnamed)"

    unnamed_counts[base_name] = unnamed_counts.get(base_name, 0) + 1
    occurrence = unnamed_counts[base_name]

    if occurrence == 1:
        final_name = base_name
    else:
        final_name = f"{base_name} ({occurrence})"

    attrs = normalize_header_attrs((hdr_res.get("attrs") or {}))

    panel_note = str(hdr_res.get("panelNote") or "").strip()
    special_header_type = hdr_res.get("specialHeaderType")
    panel_status = str((result.get("panelStatus") or "")).strip()

    return {
        "type": "panelboard",
        "name": final_name,
        "source": str(img_path),
        "panelStatus": panel_status,
        "panelNote": panel_note,
        "specialHeaderType": ensure_json_safe(special_header_type) if isinstance(special_header_type, dict) else None,
        "attrs": attrs,
    }


def build_default_ui_overrides() -> dict:
    return {
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


def build_rules_result_from_components(components: list[dict]) -> dict:
    rules_result = {}

    for component in components:
        name = component["name"]
        attrs = component["attrs"]

        missing = []
        if attrs.get("amperage") in (None, "", "NONE"):
            missing.append("amperage")
        if attrs.get("voltage") in (None, "", "NONE"):
            missing.append("voltage")

        spaces = attrs.get("spaces")
        if spaces in (None, "", "NONE"):
            missing.append("spaces")

        if missing:
            panel_note = str(component.get("panelNote") or "").strip()
            if panel_note:
                skipped_msg = panel_note
            else:
                skipped_msg = f"Missing required attributes: {', '.join(missing)}"

            rules_result[name] = {"Skipped": skipped_msg}
        else:
            rules_result[name] = {"ReadyForRules": True}

    return rules_result


def process_one_pdf(
    input_pdf: Path,
    filter_out_dir: Path,
    finder_out_dir: Path,
    pipe,
    *,
    debug: bool,
    api_version: str,
) -> dict:
    run_start_ts_ms = now_ts_ms()
    run_start_perf = time.perf_counter()

    job_id = build_job_id(input_pdf)
    job_dir = SUMMARY_ROOT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = job_dir / "summary.json"
    raw_dump_path = job_dir / "full_pipeline_breaker_dump.json"

    for d in (filter_out_dir, finder_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[JOB] job_id={job_id}")
    print(f"[JOB] job_dir={job_dir}")
    print(f"[JOB] pdf={input_pdf}")

    # ---- 1) PageFilter ----
    print("\n[PageFilter] starting…")
    page_filter = PageFilter(
        output_dir=str(filter_out_dir),
        dpi=400,
        longest_cap_px=9000,
        proc_scale=0.5,
        use_ocr=True,
        ocr_gpu=True,
        verbose=True,
        debug=False,
        rect_w_fr_range=(0.10, 0.55),
        rect_h_fr_range=(0.10, 0.60),
        min_rectangularity=0.70,
        min_rect_count=2,
    )
    kept_pages, dropped_pages, filtered_pdf, log_json = page_filter.readPdf(str(input_pdf))
    print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")

    pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept_pages) > 0) else input_pdf
    print(f"[PanelFinder] using PDF: {pdf_for_finder}")

    # ---- 2) PanelBoardSearch ----
    print("\n[PanelFinder] starting…")
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
    print(f"[PanelFinder] wrote {len(crops)} crop(s) to {finder_out_dir}")

    all_results = []
    components = []
    unnamed_counts: dict[str, int] = {}
    total_hdr_breakers = 0
    total_tbl_breakers = 0

    for img_path in crops:
        print(f"\n\n=========================")
        print(f"Analyzing crop: {img_path}")
        print(f"=========================")

        result = pipe.run(img_path)
        stages = result.get("results") or {}
        hdr_res = stages.get("header") or {}
        tbl_res = stages.get("parser") or {}

        hdr_breakers = ((hdr_res.get("attrs") or {}).get("detected_breakers") or [])
        detected_breakers = (tbl_res.get("detected_breakers") or []) if tbl_res else []

        total_hdr_breakers += len(hdr_breakers)
        total_tbl_breakers += len(detected_breakers)

        component = build_component_summary(img_path, result, unnamed_counts)
        components.append(component)

        all_results.append({
            "image": str(img_path),
            "results": ensure_json_safe(result),
            "header_breakers": ensure_json_safe(hdr_breakers),
            "table_breakers": ensure_json_safe(detected_breakers),
        })

    parse_done_ts_ms = now_ts_ms()
    cycle_time_ms = int((time.perf_counter() - run_start_perf) * 1000)
    rules_result = build_rules_result_from_components(components)

    panels_with_status = sum(1 for c in components if c.get("panelStatus"))
    panels_ready = sum(1 for v in rules_result.values() if v.get("ReadyForRules"))

    summary = {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "saved_pdf": str(input_pdf),
        "output_dir": str(job_dir),
        "images": [str(c) for c in crops],
        "image_count": len(crops),
        "components": components,
        "rules_result": rules_result,
        "ui_overrides": build_default_ui_overrides(),
        "cycle_time_ms": cycle_time_ms,
        "cycle_time_str": ms_to_readable(cycle_time_ms),
        "noticed_ts_ms": run_start_ts_ms,
        "parse_done_ts_ms": parse_done_ts_ms,
        "debug": {
            "kept_pages": ensure_json_safe(kept_pages),
            "dropped_pages": ensure_json_safe(dropped_pages),
            "filtered_pdf": str(filtered_pdf) if filtered_pdf else "",
            "pagefilter_log_json": str(log_json) if log_json else "",
            "api_version": api_version,
            "header_breakers_total": total_hdr_breakers,
            "table_breakers_total": total_tbl_breakers,
            "parser_debug": debug,
        },
    }

    with open(raw_dump_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[WROTE RAW DUMP] {raw_dump_path}")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[WROTE SUMMARY] {summary_json_path}")

    print(
        f"\n[Pipeline] Done. crops={len(crops)} | "
        f"header_breakers={total_hdr_breakers} | table_breakers={total_tbl_breakers}"
    )
    print(f"[Pipeline] cycle_time={summary['cycle_time_str']}")

    return {
        "pdf": str(input_pdf),
        "pdf_name": input_pdf.name,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "summary_path": str(summary_json_path),
        "panel_count": len(crops),
        "table_breakers_total": total_tbl_breakers,
        "panels_with_status": panels_with_status,
        "panels_ready_for_rules": panels_ready,
        "cycle_time_ms": cycle_time_ms,
        "cycle_time_str": summary["cycle_time_str"],
        "ok": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch full-pipeline dev harness (current production stack).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing PDFs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=BATCH_ARTIFACT_ROOT,
        help=f"Per-PDF intermediate outputs root (default: {BATCH_ARTIFACT_ROOT})",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable parser debug overlays (faster, less disk use)",
    )
    parser.add_argument(
        "--api",
        choices=sorted(_PARSER_APIS),
        default=DEFAULT_PARSER_API,
        help=f"BreakerTableParser API generation to run (default: {DEFAULT_PARSER_API})",
    )
    parser.add_argument(
        "--batch-label",
        default="NewTestBatch",
        help="Prefix for the batch job id (default: NewTestBatch)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser()
    artifact_root = args.artifact_root.expanduser()

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return 1

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDFs found in {input_dir}")
        return 1

    SUMMARY_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    pipeline_cls, api_version = load_parser_api(args.api)

    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"{args.batch_label}__{api_version}__{batch_stamp}"
    batch_dir = SUMMARY_ROOT_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    debug = not args.no_debug
    pipe = pipeline_cls(debug=debug)

    print(f"[Batch] id={batch_id}")
    print(f"[Batch] input_dir={input_dir}")
    print(f"[Batch] pdfs={len(pdfs)}")
    print("[ParserAPI] API_VERSION:", api_version)
    print("[ParserAPI] debug:", debug)

    batch_start_perf = time.perf_counter()
    results = []
    failures = []

    for i, input_pdf in enumerate(pdfs, 1):
        stem = input_pdf.stem
        print(f"\n{'=' * 72}")
        print(f"PDF {i}/{len(pdfs)}: {input_pdf.name}")
        print(f"{'=' * 72}")

        filter_out_dir = artifact_root / stem / "PdfOutput"
        finder_out_dir = artifact_root / stem / "PanelSearchOutput"

        try:
            result = process_one_pdf(
                input_pdf,
                filter_out_dir,
                finder_out_dir,
                pipe,
                debug=debug,
                api_version=api_version,
            )
            results.append(result)
        except Exception as exc:
            print(f"[ERROR] Failed on {input_pdf.name}: {exc}")
            failures.append({
                "pdf": str(input_pdf),
                "pdf_name": input_pdf.name,
                "error": str(exc),
            })

    batch_cycle_ms = int((time.perf_counter() - batch_start_perf) * 1000)

    batch_summary = {
        "batch_id": batch_id,
        "batch_dir": str(batch_dir),
        "input_dir": str(input_dir),
        "artifact_root": str(artifact_root),
        "api_version": api_version,
        "parser_debug": debug,
        "pdf_count": len(pdfs),
        "success_count": len(results),
        "failure_count": len(failures),
        "cycle_time_ms": batch_cycle_ms,
        "cycle_time_str": ms_to_readable(batch_cycle_ms),
        "results": results,
        "failures": failures,
        "totals": {
            "panels": sum(r.get("panel_count", 0) for r in results),
            "table_breakers": sum(r.get("table_breakers_total", 0) for r in results),
            "panels_with_status": sum(r.get("panels_with_status", 0) for r in results),
            "panels_ready_for_rules": sum(r.get("panels_ready_for_rules", 0) for r in results),
        },
    }

    batch_summary_path = batch_dir / "batch_summary.json"
    with open(batch_summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False, default=str)

    overlay_dir = batch_dir / "review_overlays"
    try:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from CollectReviewOverlays import collect_from_batch_summary

        overlay_manifest = collect_from_batch_summary(batch_summary_path, overlay_dir)
        batch_summary["review_overlays_dir"] = str(overlay_dir)
        batch_summary["review_overlay_count"] = len(overlay_manifest)
        with open(batch_summary_path, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False, default=str)
    except Exception as exc:
        print(f"[WARN] Could not collect review overlays: {exc}")

    print(f"\n{'=' * 72}")
    print(f"[BATCH DONE] {len(results)}/{len(pdfs)} succeeded")
    print(f"[BATCH DONE] wall_time={batch_summary['cycle_time_str']}")
    print(f"[BATCH DONE] panels={batch_summary['totals']['panels']} breakers={batch_summary['totals']['table_breakers']}")
    print(f"[WROTE] {batch_summary_path}")

    for row in results:
        print(
            f"  {row['pdf_name']:24} panels={row['panel_count']:3} "
            f"breakers={row['table_breakers_total']:4} "
            f"status={row['panels_with_status']:2} time={row['cycle_time_str']} "
            f"-> {row['job_id']}"
        )

    if failures:
        print("\nFailures:")
        for row in failures:
            print(f"  {row['pdf_name']}: {row['error']}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
