#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ---------- IMPORTS ----------
from PageFilter.PageFilterV3 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch
from OcrLibrary.BreakerTableParserAPIv12 import BreakerTablePipeline, API_VERSION

# ---------- IO PATHS (fixed typos: PdfOutput / PanelSearchOutput) ----------
INPUT_PDF       = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf/chucksmall.pdf").expanduser()
FILTER_OUT_DIR  = Path("~/ElectricalDiagramAnalyzer/DevEnv/PdfOutput").expanduser()
FINDER_OUT_DIR  = Path("~/ElectricalDiagramAnalyzer/DevEnv/PanelSearchOutput").expanduser()
PIPE_OUT_DIR    = Path("~/ElectricalDiagramAnalyzer/DevEnv/ParserOutput").expanduser()
SUMMARY_ROOT_DIR   = Path("~/ElectricalDiagramAnalyzer/DevEnv/JobSummaries").expanduser()

for d in (FILTER_OUT_DIR, FINDER_OUT_DIR, PIPE_OUT_DIR, SUMMARY_ROOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

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

            rules_result[name] = {
                "Skipped": skipped_msg
            }
        else:
            rules_result[name] = {
                "ReadyForRules": True
            }

    return rules_result

def main():
    run_start_ts_ms = now_ts_ms()
    run_start_perf = time.perf_counter()

    job_id = build_job_id(INPUT_PDF)
    job_dir = SUMMARY_ROOT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = job_dir / "summary.json"
    raw_dump_path = job_dir / "full_pipeline_breaker_dump.json"

    # Treat source PDF as the saved PDF in dev env
    saved_pdf = str(INPUT_PDF)

    print(f"\n[JOB] job_id={job_id}")
    print(f"[JOB] job_dir={job_dir}")

    # ---- 1) PageFilter ----
    print("\n[PageFilter] starting…")
    FILTER = PageFilter(
        output_dir=str(FILTER_OUT_DIR),
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
    kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(INPUT_PDF))
    print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")

    pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept_pages) > 0) else INPUT_PDF
    print(f"[PanelFinder] using PDF: {pdf_for_finder}")

    # ---- 2) PanelBoardSearch ----
    print("\n[PanelFinder] starting…")
    FINDER = PanelBoardSearch(
        output_dir=str(FINDER_OUT_DIR),
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
    crops = FINDER.readPdf(str(pdf_for_finder))
    print(f"[PanelFinder] wrote {len(crops)} crop(s) to {FINDER_OUT_DIR}")

    # ---- 3) BreakerTableParser API ----
    print("\n[ParserAPI] starting…")
    pipe = BreakerTablePipeline(debug=True)
    print("[ParserAPI] API_VERSION:", API_VERSION)

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
        ana_res = stages.get("analyzer") or {}
        hdr_res = stages.get("header") or {}
        tbl_res = stages.get("parser") or {}

        print("\n=== ANALYZER ===")
        print("header_y        :", ana_res.get("header_y"))
        print("footer_y        :", ana_res.get("footer_y"))

        print("\n=== HEADER PARSER ===")
        print("name    :", hdr_res.get("name"))
        print("attrs   :", hdr_res.get("attrs"))
        print("panelNote :", hdr_res.get("panelNote"))
        print("specialHeaderType :", hdr_res.get("specialHeaderType"))
        print("panelStatus :", result.get("panelStatus"))
        hdr_breakers = ((hdr_res.get("attrs") or {}).get("detected_breakers") or [])

        print("\n=== TABLE PARSER (summary) ===")
        if tbl_res:
            spaces = tbl_res.get("spaces")
            detected_breakers = tbl_res.get("detected_breakers") or []
            breaker_counts = tbl_res.get("breakerCounts") or {}
            gfi_counts = tbl_res.get("gfiBreakerCounts") or {}

            print("spaces               :", spaces)
            print("detected breakers    :", len(detected_breakers))
            print("Breakers (tally):")

            if not breaker_counts:
                print("  (none detected)")
            else:
                def _sort_key(item):
                    key, _count = item
                    m = re.match(r"(\d+)P_(\d+)A", key)
                    if not m:
                        return (9999, 9999, key)
                    p = int(m.group(1))
                    a = int(m.group(2))
                    return (p, a, key)

                for key, count in sorted(breaker_counts.items(), key=_sort_key):
                    m = re.match(r"(\d+)P_(\d+)A", key)
                    if m:
                        poles = int(m.group(1))
                        amps = int(m.group(2))
                        print(f"  {poles} P, {amps} A, count - {count}")
                    else:
                        print(f"  {key}, count - {count}")

            if gfi_counts:
                print("GFI Breakers (tally):")
                for key, count in sorted(gfi_counts.items(), key=_sort_key):
                    m = re.match(r"(\d+)P_(\d+)A", key)
                    if m:
                        poles = int(m.group(1))
                        amps = int(m.group(2))
                        print(f"  {poles} P, {amps} A, GFI count - {count}")
                    else:
                        print(f"  {key}, GFI count - {count}")
        else:
            print("parser  : None")
            detected_breakers = []

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

    summary = {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "saved_pdf": saved_pdf,
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
            "api_version": API_VERSION,
            "header_breakers_total": total_hdr_breakers,
            "table_breakers_total": total_tbl_breakers,
        },
    }

    try:
        with open(raw_dump_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[WROTE RAW DUMP] {raw_dump_path}")
    except Exception as e:
        print(f"[WARN] Could not write raw dump: {e}")

    try:
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"[WROTE SUMMARY] {summary_json_path}")
    except Exception as e:
        print(f"[ERROR] Could not write summary: {e}")
        raise

    print(f"\n[Pipeline] Done. crops={len(crops)} | header_breakers={total_hdr_breakers} | table_breakers={total_tbl_breakers}")
    print(f"[Pipeline] cycle_time={summary['cycle_time_str']}")

if __name__ == "__main__":
    main()
