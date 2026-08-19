#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B the breaker-table body OCR before and after body-cell grid-line removal.

APIv13 (BreakerTableParser11) is the baseline and APIv14 (BreakerTableParser12,
which de-grids each body cell via _prep_cell_for_ocr) is the candidate. Page
filtering and panel finding run ONCE and both parsers then read the exact same
crops, so any difference in the report is attributable to the parser change.

Each parser gets its own copy of the crops so their debug/ folders do not
overwrite each other:

    <out>/<pdf_stem>/baseline_v13/{crop.png,debug/...}
    <out>/<pdf_stem>/candidate_v14/{crop.png,debug/...}

Usage:
    python DevEnv/CompareCellPrep_APIv13_vs_APIv14.py PDF [PDF ...] [--out DIR]
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from PageFilter.PageFilterV5 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV26 import PanelBoardSearch

from OcrLibrary.BreakerTableParserAPIv13 import (
    BreakerTablePipeline as PipelineV13,
    API_VERSION as API_VERSION_V13,
    reset_name_deduper as reset_names_v13,
)
from OcrLibrary.BreakerTableParserAPIv14 import (
    BreakerTablePipeline as PipelineV14,
    API_VERSION as API_VERSION_V14,
    reset_name_deduper as reset_names_v14,
)

DEFAULT_OUT_DIR = Path("~/ElectricalDiagramAnalyzer/DevEnv/CellPrepCompare").expanduser()


def summarize_panel(crop_path, result):
    """Reduce a pipeline result to the fields a body-OCR change can move."""
    stages = result.get("results") or {}
    header = stages.get("header") or {}
    parser = stages.get("parser") or {}
    attrs = header.get("attrs") or {}

    # Keep the raw OCR text alongside the parsed value: a body-OCR change shows
    # up there first, even when the parsed amps/poles happen to land the same.
    breakers = []
    for breaker in parser.get("detected_breakers") or []:
        breakers.append(
            {
                "side": breaker.get("side"),
                "rowIndex": breaker.get("rowIndex"),
                "amperage": breaker.get("amperage"),
                "poles": breaker.get("poles"),
                "specialFeatures": breaker.get("specialFeatures"),
                "tripText": breaker.get("tripText"),
                "polesText": breaker.get("polesText"),
                "comboText": breaker.get("comboText"),
            }
        )

    return {
        "crop": os.path.basename(str(crop_path)),
        "name": header.get("name"),
        "panelStatus": result.get("panelStatus"),
        "layout": parser.get("layout"),
        "spaces": parser.get("spaces"),
        "amperage": attrs.get("amperage"),
        "voltage": attrs.get("voltage"),
        "breakerCounts": parser.get("breakerCounts") or {},
        "gfiBreakerCounts": parser.get("gfiBreakerCounts") or {},
        "reviewCellCount": len(parser.get("reviewCells") or []),
        "detectedBreakerCount": len(parser.get("detected_breakers") or []),
        "breakers": breakers,
    }


def run_pipeline(pipeline_cls, reset_names, crop_paths, label):
    reset_names()
    pipe = pipeline_cls(debug=True)

    panels = []
    started = time.perf_counter()
    for crop_path in crop_paths:
        print(f"  [{label}] {os.path.basename(str(crop_path))}")
        result = pipe.run(str(crop_path))
        panels.append(summarize_panel(crop_path, result))

    elapsed = time.perf_counter() - started
    print(f"  [{label}] {len(panels)} panel(s) in {elapsed:.1f}s")
    return panels


def stage_crops(crop_paths, dest_dir):
    """Give one parser its own copy of the crops so debug output stays separate."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for crop_path in crop_paths:
        target = dest_dir / os.path.basename(str(crop_path))
        shutil.copy2(str(crop_path), str(target))
        staged.append(target)
    return staged


def diff_panels(baseline, candidate):
    """Compare the two parsers panel by panel; returns (rows, totals)."""
    by_crop = {panel["crop"]: panel for panel in candidate}
    rows = []

    totals = {
        "panels": 0,
        "panels_changed": 0,
        "baseline_breakers": 0,
        "candidate_breakers": 0,
        "baseline_review_cells": 0,
        "candidate_review_cells": 0,
    }

    for base_panel in baseline:
        cand_panel = by_crop.get(base_panel["crop"])
        totals["panels"] += 1
        totals["baseline_breakers"] += base_panel["detectedBreakerCount"]
        totals["baseline_review_cells"] += base_panel["reviewCellCount"]

        if cand_panel is None:
            rows.append({"crop": base_panel["crop"], "status": "MISSING_IN_CANDIDATE"})
            totals["panels_changed"] += 1
            continue

        totals["candidate_breakers"] += cand_panel["detectedBreakerCount"]
        totals["candidate_review_cells"] += cand_panel["reviewCellCount"]

        changed_fields = {}
        for field in (
            "name",
            "panelStatus",
            "layout",
            "spaces",
            "amperage",
            "voltage",
            "breakerCounts",
            "gfiBreakerCounts",
            "detectedBreakerCount",
            "reviewCellCount",
        ):
            if base_panel.get(field) != cand_panel.get(field):
                changed_fields[field] = {
                    "v13": base_panel.get(field),
                    "v14": cand_panel.get(field),
                }

        row_diffs = []
        for base_row, cand_row in zip(base_panel["breakers"], cand_panel["breakers"]):
            if base_row != cand_row:
                row_diffs.append({"v13": base_row, "v14": cand_row})

        if changed_fields or row_diffs:
            totals["panels_changed"] += 1

        rows.append(
            {
                "crop": base_panel["crop"],
                "name": base_panel.get("name"),
                "status": "CHANGED" if (changed_fields or row_diffs) else "IDENTICAL",
                "changedFields": changed_fields,
                "rowDiffs": row_diffs,
            }
        )

    return rows, totals


def process_pdf(pdf_path, out_root):
    pdf_path = Path(pdf_path).expanduser()
    stem = pdf_path.stem
    work_dir = out_root / stem
    filter_dir = work_dir / "pagefilter"
    finder_dir = work_dir / "panelfinder"

    for directory in (filter_dir, finder_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\n[PDF] {pdf_path}\n{'=' * 70}")

    print("[PageFilter] running…")
    page_filter = PageFilter(
        output_dir=str(filter_dir),
        dpi=400,
        longest_cap_px=9000,
        proc_scale=0.5,
        use_ocr=True,
        ocr_gpu=True,
        verbose=False,
        debug=False,
        rect_w_fr_range=(0.10, 0.55),
        rect_h_fr_range=(0.10, 0.60),
        min_rectangularity=0.70,
        min_rect_count=2,
    )
    kept_pages, dropped_pages, filtered_pdf, _log_json = page_filter.readPdf(str(pdf_path))
    print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)}")

    pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and kept_pages) else pdf_path

    print("[PanelFinder] running…")
    finder = PanelBoardSearch(
        output_dir=str(finder_dir),
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
        verbose=False,
    )
    crops = finder.readPdf(str(pdf_for_finder))
    print(f"[PanelFinder] {len(crops)} crop(s)")

    if not crops:
        print("[SKIP] no panel crops found")
        return None

    baseline_crops = stage_crops(crops, work_dir / "baseline_v13")
    candidate_crops = stage_crops(crops, work_dir / "candidate_v14")

    print(f"\n[Parse] baseline {API_VERSION_V13}")
    baseline = run_pipeline(PipelineV13, reset_names_v13, baseline_crops, API_VERSION_V13)

    print(f"\n[Parse] candidate {API_VERSION_V14}")
    candidate = run_pipeline(PipelineV14, reset_names_v14, candidate_crops, API_VERSION_V14)

    rows, totals = diff_panels(baseline, candidate)

    report = {
        "pdf": str(pdf_path),
        "baselineApi": API_VERSION_V13,
        "candidateApi": API_VERSION_V14,
        "totals": totals,
        "panels": rows,
        "baseline": baseline,
        "candidate": candidate,
        "cellPrepImages": sorted(
            str(p) for p in (work_dir / "candidate_v14" / "debug").glob("*_cellprep.png")
        ),
    }

    report_path = work_dir / "comparison.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)

    print(f"\n--- {stem}: {totals['panels_changed']}/{totals['panels']} panel(s) changed ---")
    print(
        f"    breakers  v13={totals['baseline_breakers']}  v14={totals['candidate_breakers']}\n"
        f"    review    v13={totals['baseline_review_cells']}  v14={totals['candidate_review_cells']}"
    )

    for row in rows:
        if row["status"] == "IDENTICAL":
            continue
        print(f"\n  [{row['status']}] {row['crop']}  name={row.get('name')!r}")
        for field, values in (row.get("changedFields") or {}).items():
            print(f"      {field}: {values['v13']}  ->  {values['v14']}")
        for row_diff in (row.get("rowDiffs") or [])[:20]:
            print(f"      row: {row_diff['v13']}  ->  {row_diff['v14']}")

    print(f"\n[WROTE] {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", help="PDF file(s) to compare")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output root directory")
    args = parser.parse_args()

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    reports = [report for report in (process_pdf(pdf, out_root) for pdf in args.pdfs) if report]

    print(f"\n\n{'=' * 70}\nOVERALL\n{'=' * 70}")
    for report in reports:
        totals = report["totals"]
        print(
            f"{Path(report['pdf']).stem:<28} "
            f"panels={totals['panels']:<3} changed={totals['panels_changed']:<3} "
            f"breakers {totals['baseline_breakers']}->{totals['candidate_breakers']}  "
            f"review {totals['baseline_review_cells']}->{totals['candidate_review_cells']}"
        )


if __name__ == "__main__":
    main()
