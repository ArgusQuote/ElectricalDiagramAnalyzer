#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import sys
import json
import csv
import traceback
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
from OcrLibrary.BreakerTableParserAPIv18 import BreakerTablePipeline, API_VERSION
from OcrLibrary.BreakerTableParserAPIv17 import BreakerTablePipeline as BreakerTablePipeline17

# ---------- IO PATHS (fixed typos: PdfOutput / PanelSearchOutput) ----------

SOURCE_PDF_DIR = Path("~/ElectricalDiagramAnalyzer/DevEnv/SourcePdf").expanduser()
INPUT_PDF = SOURCE_PDF_DIR / "E.pdf"

# ==================== TEST CONTROLS ====================

# "ONE", "SELECTED", "FIRST_N", or "ALL"
RUN_MODE = "ALL"

ONE_JOB = "chucksmall.pdf"

SELECTED_JOBS = [
    "A.pdf",
    "B.pdf",
    "E.pdf",
]

FIRST_N_JOBS = 5

# S1.pdf through S23.pdf are always excluded.
# =======================================================

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


def build_component_summary(
    img_path: str,
    result: dict,
    unnamed_counts: dict[str, int]
) -> dict:
    stages = result.get("results") or {}
    hdr_res = stages.get("header") or {}
    tbl_res = stages.get("parser") or {}

    raw_name = str(
        hdr_res.get("name") or ""
    ).strip()

    base_name = (
        raw_name
        if raw_name
        else "(unnamed)"
    )

    unnamed_counts[base_name] = (
        unnamed_counts.get(base_name, 0) + 1
    )

    occurrence = unnamed_counts[base_name]

    if occurrence == 1:
        final_name = base_name
    else:
        final_name = (
            f"{base_name} ({occurrence})"
        )

    attrs = normalize_header_attrs(
        hdr_res.get("attrs") or {}
    )

    panel_note = str(
        hdr_res.get("panelNote") or ""
    ).strip()

    # --- Existing breakers from description column ---
    existing_breaker_count = int(
        tbl_res.get("existingBreakerCount") or 0
    )

    existing_breaker_note = str(
        tbl_res.get("existingBreakerNote") or ""
    ).strip()

    if existing_breaker_note and existing_breaker_note not in panel_note:
        panel_note = (
            f"{panel_note} | {existing_breaker_note}"
            if panel_note
            else existing_breaker_note
        )

    special_header_type = hdr_res.get(
        "specialHeaderType"
    )

    panel_status = str(
        result.get("panelStatus") or ""
    ).strip()

    # -----------------------------------------
    # Existing-panel metadata
    # Mirror the production server behavior.
    # This does NOT skip or alter the panel.
    # -----------------------------------------
    suspected_existing = bool(
        hdr_res.get(
            "suspectedExisting",
            False,
        )
    )

    existing_reason = str(
        hdr_res.get(
            "existingReason"
        )
        or ""
    ).strip()

    existing_detection = str(
        hdr_res.get(
            "existingDetection"
        )
        or ""
    ).strip()

    return {
        "type": "panelboard",
        "name": final_name,
        "source": str(img_path),

        "panelStatus": panel_status,
        "panelNote": panel_note,

        "hasExistingBreakers": bool(existing_breaker_count),
        "existingBreakerCount": existing_breaker_count,
        "existingBreakerNote": existing_breaker_note,

        "specialHeaderType": (
            ensure_json_safe(
                special_header_type
            )
            if isinstance(
                special_header_type,
                dict,
            )
            else None
        ),

        # Existing-equipment metadata
        "suspectedExisting": (
            suspected_existing
        ),
        "existingReason": (
            existing_reason
        ),
        "existingDetection": (
            existing_detection
        ),

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


def choose_test_pdfs():
    """Select jobs, always excluding S1 through S23."""

    all_pdfs = sorted(
        (
            p for p in SOURCE_PDF_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf"
        ),
        key=lambda p: p.as_posix().lower(),
    )

    eligible = [
        p for p in all_pdfs
        if not re.fullmatch(
            r"S0*(?:[1-9]|1[0-9]|2[0-3])",
            p.stem,
            re.I,
        )
    ]

    def resolve(name):
        target = str(name).strip().replace("\\", "/").lower()

        matches = [
            p for p in eligible
            if p.name.lower() == target
            or p.relative_to(
                SOURCE_PDF_DIR
            ).as_posix().lower() == target
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one eligible PDF for {name!r}; "
                f"found {len(matches)}"
            )

        return matches[0]

    mode = RUN_MODE.upper().strip()

    if mode == "ALL":
        selected = eligible

    elif mode == "ONE":
        selected = [resolve(ONE_JOB)]

    elif mode == "SELECTED":
        selected = list(
            dict.fromkeys(
                resolve(name) for name in SELECTED_JOBS
            )
        )

    elif mode == "FIRST_N":
        if FIRST_N_JOBS < 1:
            raise ValueError(
                "FIRST_N_JOBS must be positive"
            )
        selected = eligible[:FIRST_N_JOBS]

    else:
        raise ValueError(
            f"Invalid RUN_MODE: {RUN_MODE}"
        )

    if not selected:
        raise ValueError("No eligible PDFs selected")

    print(
        f"[SELECTION] {len(selected)} jobs | "
        "excluded S1-S23"
    )

    return selected



# Keys that only name the code version that produced a result. They always
# differ between API 17 and API 18, so comparing them would flag every panel.
VERSION_TAG_KEYS = frozenset({"parserVersion", "apiVersion", "analyzerVersion"})


def without_version_tags(value):
    """Return a copy of value with VERSION_TAG_KEYS removed at every depth."""
    if isinstance(value, dict):
        return {
            k: without_version_tags(v)
            for k, v in value.items()
            if k not in VERSION_TAG_KEYS
        }

    if isinstance(value, (list, tuple)):
        return [without_version_tags(v) for v in value]

    return value


def find_raw_differences(old, new, prefix, normalize, limit=2000):
    """Find exact changed paths within parser output."""
    differences = []
    missing = object()

    def display(value):
        if value is missing:
            return "<MISSING>"

        rendered = json.dumps(
            value, ensure_ascii=False, default=str
        )

        return (
            rendered if len(rendered) <= 400
            else rendered[:397] + "..."
        )

    def walk(left, right, path):
        if len(differences) >= limit:
            return

        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right), key=str):
                walk(
                    left.get(key, missing),
                    right.get(key, missing),
                    f"{path}.{key}",
                )
            return

        if (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
        ):
            for index in range(max(len(left), len(right))):
                walk(
                    left[index] if index < len(left) else missing,
                    right[index] if index < len(right) else missing,
                    f"{path}[{index}]",
                )
            return

        if (
            left is missing
            or right is missing
            or normalize(left) != normalize(right)
        ):
            differences.append({
                "path": path,
                "v17": display(left),
                "v18": display(right),
            })

    walk(old, new, prefix)
    return differences


def compare_parser_outputs(
    pdf, crop, old, new, old_error=""
):
    """Compare API 17 and API 18 panel results."""

    def norm(value):
        if isinstance(value, dict):
            return {
                str(k): norm(v)
                for k, v in sorted(
                    value.items(),
                    key=lambda x: str(x[0]),
                )
            }

        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]

        if value is None or (
            isinstance(value, str)
            and value.strip().upper() in ("", "NONE")
        ):
            return "NONE"

        if isinstance(value, bool):
            return value

        return str(value).strip().upper()

    def extract(result):
        stages = result.get("results") or {}
        header = stages.get("header") or {}
        table = stages.get("parser") or {}
        attrs = header.get("attrs") or {}

        fields = {
            "name": header.get("name"),
            "panelStatus": result.get("panelStatus"),
            "panelNote": header.get("panelNote"),
            "specialHeaderType": header.get("specialHeaderType"),
            "suspectedExisting": header.get("suspectedExisting"),
            "existingReason": header.get("existingReason"),
            "existingDetection": header.get("existingDetection"),
            "existingBreakerCount": table.get("existingBreakerCount"),
            "existingBreakerNote": table.get("existingBreakerNote"),
            "table.spaces": table.get("spaces"),
            "header.detected_breakers_count": len(
                attrs.get("detected_breakers") or []
            ),

            "table.detected_breakers_count": len(
                table.get("detected_breakers") or []
            ),
            "header.detected_breakers": (
                attrs.get("detected_breakers") or []
            ),
            "table.detected_breakers": (
                table.get("detected_breakers") or []
            ),

        }

        for key in (
            "amperage",
            "spaces",
            "voltage",
            "intRating",
            "mainBreakerAmperage",
        ):
            fields[f"attrs.{key}"] = attrs.get(key)

        def tally(key):
            values = table.get(key) or {}

            if not isinstance(values, dict):
                raise TypeError(
                    f"{key} is not a dictionary"
                )

            return {
                str(k).strip().upper(): int(v)
                for k, v in values.items()
            }

        return (
            fields,
            tally("breakerCounts"),
            tally("gfiBreakerCounts"),
            header,
            table,
        )

    record = {
        "job": pdf.name,
        "crop": Path(crop).name,
        "panel_v17": "",
        "panel_v18": "",
        "status": "ERROR",
        "field_differences": [],
        "breaker_differences": [],
        "v17_breakers": 0,
        "v18_breakers": 0,
        "raw_changed": False,
        "raw_differences": [],
        "error": old_error,
    }

    try:
        if old is None or new is None:
            raise ValueError(
                "One parser returned no result"
            )

        f17, b17, g17, h17, t17 = extract(old)
        f18, b18, g18, h18, t18 = extract(new)
        h17, t17, h18, t18 = (
            without_version_tags(x) for x in (h17, t17, h18, t18)
        )

        record["panel_v17"] = str(f17["name"] or "")
        record["panel_v18"] = str(f18["name"] or "")

        record["v17_breakers"] = sum(b17.values())
        record["v18_breakers"] = sum(b18.values())

        record["field_differences"] = [
            {
                "field": key,
                "v17": f17[key],
                "v18": f18[key],
            }
            for key in f17
            if norm(f17[key]) != norm(f18[key])
        ]

        for category, left, right in (
            ("breaker", b17, b18),
            ("gfi", g17, g18),
        ):
            for breaker_type in sorted(
                set(left) | set(right)
            ):
                v17 = left.get(breaker_type, 0)
                v18 = right.get(breaker_type, 0)

                if v17 != v18:
                    record["breaker_differences"].append({
                        "category": category,
                        "breaker_type": breaker_type,
                        "v17_qty": v17,
                        "v18_qty": v18,
                        "difference": v18 - v17,
                    })


        record["raw_changed"] = (
            norm(h17) != norm(h18)
            or norm(t17) != norm(t18)
        )

        # Only investigate raw data when a difference exists.
        if record["raw_changed"]:
            record["raw_differences"] = (
                find_raw_differences(
                    h17, h18, "header", norm
                )
                + find_raw_differences(
                    t17, t18, "parser", norm
                )
            )



        breaker_record_changed = any(
            ".detected_breakers[" in d["path"]
            for d in record["raw_differences"]
        )

        if (
            record["field_differences"]
            or record["breaker_differences"]
            or breaker_record_changed
        ):
            record["status"] = "CHANGED"


        elif not h17 or not t17 or not h18 or not t18:
            record["status"] = "INCOMPLETE"

        elif record["raw_changed"]:
            record["status"] = "RAW_ONLY"

        else:
            record["status"] = "MATCH"

    except Exception as exc:
        record["error"] = (
            record["error"] + " | " + str(exc)
        ).strip(" |")

    return record


def write_comparison_reports(jobs, report_dir):
    """Save cumulative reports after each completed job."""

    report_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        p
        for job in jobs
        for p in job.get("panels", [])
    ]

    panel_rows = []
    breaker_rows = []
    detail_rows = []

    for p in panels:
        panel_rows.append({
            "job": p["job"],
            "crop": p["crop"],
            "panel_v17": p["panel_v17"],
            "panel_v18": p["panel_v18"],
            "status": p["status"],
            "changed_fields": "; ".join(
                x["field"]
                for x in p["field_differences"]
            ),
            "breaker_differences": len(
                p["breaker_differences"]
            ),
            "v17_breakers": p["v17_breakers"],
            "v18_breakers": p["v18_breakers"],
            "raw_changed": p["raw_changed"],
            "error": p["error"],
        })

        for difference in p["breaker_differences"]:
            breaker_rows.append({
                "job": p["job"],
                "crop": p["crop"],
                "panel_v17": p["panel_v17"],
                "panel_v18": p["panel_v18"],
                **difference,
            })


    # Build detailed records ONLY for panels with differences.
    for p in panels:
        if p["status"] == "MATCH":
            continue

        def add_detail(kind, path, old_value, new_value):
            detail_rows.append({
                "job": p["job"],
                "panel": (
                    p["panel_v18"] or p["panel_v17"]
                ),
                "crop": p["crop"],
                "status": p["status"],
                "kind": kind,
                "field_or_breaker": path,
                "api17": str(old_value),
                "api18": str(new_value),
            })

        # Panel attributes: voltage, amperage, spaces, etc.
        for d in p["field_differences"]:
            add_detail(
                "PANEL_FIELD",
                d["field"],
                d["v17"],
                d["v18"],
            )

        # Breaker and GFI quantity changes.
        for d in p["breaker_differences"]:
            kind = (
                "BREAKER_QTY"
                if d["category"] == "breaker"
                else "GFI_QTY"
            )

            add_detail(
                kind,
                d["breaker_type"],
                d["v17_qty"],
                d["v18_qty"],
            )

        # Exact differences in raw parser outputs.
        for d in p.get("raw_differences", []):
            path = d["path"]

            kind = (
                "BREAKER_RECORD"
                if ".detected_breakers[" in path
                else "RAW_DATA"
            )

            add_detail(
                kind,
                path,
                d["v17"],
                d["v18"],
            )

        if p["status"] == "ERROR":
            add_detail(
                "ERROR",
                "parser_error",
                p["error"],
                "",
            )


    def save_csv(filename, rows, columns):
        with (
            report_dir / filename
        ).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=columns
            )
            writer.writeheader()
            writer.writerows(rows)

    save_csv(
        "panel_comparison.csv",
        panel_rows,
        [
            "job", "crop", "panel_v17", "panel_v18",
            "status", "changed_fields",
            "breaker_differences",
            "v17_breakers", "v18_breakers",
            "raw_changed", "error",
        ],
    )

    save_csv(
        "breaker_differences.csv",
        breaker_rows,
        [
            "job", "crop", "panel_v17", "panel_v18",
            "category", "breaker_type",
            "v17_qty", "v18_qty", "difference",
        ],
    )


    # Write detailed report only when differences exist.
    if detail_rows:
        save_csv(
            "difference_details.csv",
            detail_rows,
            [
                "job",
                "panel",
                "crop",
                "status",
                "kind",
                "field_or_breaker",
                "api17",
                "api18",
            ],
        )


    job_rows = []

    for j in jobs:
        jp = j.get("panels", [])

        job_rows.append({
            "job": j["job"],
            "status": j["status"],
            "panels": len(jp),
            "matches": sum(
                p["status"] == "MATCH" for p in jp
            ),
            "changed": sum(
                p["status"] == "CHANGED" for p in jp
            ),
            "errors": sum(
                p["status"] == "ERROR" for p in jp
            ),
            "time_seconds": j.get("time_seconds", 0),
            "error": j.get("error", ""),
        })

    save_csv(
        "job_summary.csv",
        job_rows,
        [
            "job", "status", "panels", "matches",
            "changed", "errors", "time_seconds",
            "error",
        ],
    )

    summary = {
        "jobs_processed": len(jobs),
        "job_failures": sum(
            j["status"] == "ERROR" for j in jobs
        ),
        "panels_compared": len(panels),
        "matches": sum(
            p["status"] == "MATCH" for p in panels
        ),
        "changed": sum(
            p["status"] == "CHANGED" for p in panels
        ),
        "incomplete": sum(
            p["status"] == "INCOMPLETE" for p in panels
        ),
        "raw_only": sum(
            p["status"] == "RAW_ONLY" for p in panels
        ),
        "errors": sum(
            p["status"] == "ERROR" for p in panels
        ),
        "breaker_type_differences": sum(
            d["category"] == "breaker"
            for p in panels
            for d in p["breaker_differences"]
        ),
        "gfi_type_differences": sum(
            d["category"] == "gfi"
            for p in panels
            for d in p["breaker_differences"]
        ),
        "absolute_breaker_quantity_differences": sum(
            abs(d["difference"])
            for p in panels
            for d in p["breaker_differences"]
            if d["category"] == "breaker"
        ),
    }

    with (
        report_dir / "comparison.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "jobs": jobs,
            },
            f,
            indent=2,
            default=str,
        )

    return summary


def run_all_jobs():
    """Run API 17, API 18, and comparison one job at a time."""

    global INPUT_PDF

    pdfs = choose_test_pdfs()

    comparison_dir = (
        SUMMARY_ROOT_DIR
        / "APIComparison"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    jobs = []

    for index, pdf in enumerate(pdfs, 1):
        INPUT_PDF = pdf

        print(
            f"\n========== JOB {index}/{len(pdfs)}: "
            f"{pdf.name} ==========",
            flush=True,
        )

        started = time.perf_counter()

        try:
            report = main()

            if not report["panels"]:
                report["status"] = "NO_PANELS"

            elif any(
                p["status"] == "ERROR"
                for p in report["panels"]
            ):
                report["status"] = "PANEL_ERRORS"

            else:
                report["status"] = "DONE"

        except Exception:
            report = {
                "job": pdf.name,
                "status": "ERROR",
                "panels": [],
                "error": traceback.format_exc(),
            }

            print(
                f"[JOB ERROR] "
                f"{report['error'].splitlines()[-1]}",
                flush=True,
            )

        report["time_seconds"] = round(
            time.perf_counter() - started, 2
        )

        jobs.append(report)

        summary = write_comparison_reports(
            jobs, comparison_dir
        )

        print(
            f"[JOB DONE] {pdf.name} | "
            f"panels={len(report['panels'])} | "
            f"changed={sum(p['status'] == 'CHANGED' for p in report['panels'])} | "
            f"time={report['time_seconds']}s",
            flush=True,
        )

    print(
        "\n========== FINAL API 17 VS 18 COMPARISON ==========",
        flush=True,
    )

    for key, value in summary.items():
        print(
            f"{key}: {value}",
            flush=True,
        )

    print(
        f"Reports: {comparison_dir}",
        flush=True,
    )

    print(
        "Note: discrepancies indicate differences, "
        "not verified errors.",
        flush=True,
    )



def main():
    run_start_ts_ms = now_ts_ms()
    run_start_perf = time.perf_counter()

    job_id = build_job_id(INPUT_PDF)
    job_dir = SUMMARY_ROOT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    filter_dir = job_dir / "PdfOutput"
    finder_dir = job_dir / "PanelSearchOutput"

    filter_dir.mkdir(exist_ok=True)
    finder_dir.mkdir(exist_ok=True)

    summary_json_path = job_dir / "summary.json"
    raw_dump_path = job_dir / "full_pipeline_breaker_dump.json"

    # Treat source PDF as the saved PDF in dev env
    saved_pdf = str(INPUT_PDF)

    print(f"\n[JOB] job_id={job_id}")
    print(f"[JOB] job_dir={job_dir}")

    # ---- 1) PageFilter ----
    print("\n[PageFilter] starting…")
    FILTER = PageFilter(
        output_dir=str(filter_dir),
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
    filter_start = time.perf_counter()
    kept_pages, dropped_pages, filtered_pdf, log_json = FILTER.readPdf(str(INPUT_PDF))
    filter_elapsed = time.perf_counter() - filter_start
    print(f"[PageFilter] kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")

    pdf_for_finder = Path(filtered_pdf) if (filtered_pdf and len(kept_pages) > 0) else INPUT_PDF
    print(f"[PanelFinder] using PDF: {pdf_for_finder}")

    # ---- 2) PanelBoardSearch ----
    print("\n[PanelFinder] starting…")
    FINDER = PanelBoardSearch(
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
        verbose=True,
    )
    finder_start = time.perf_counter()
    crops = FINDER.readPdf(str(pdf_for_finder))
    finder_elapsed = time.perf_counter() - finder_start
    print(f"[PanelFinder] wrote {len(crops)} crop(s) to {finder_dir}")


    # ---- 3) API 17 ----
    print("\n[API17] starting…", flush=True)

    api17_start = time.perf_counter()

    pipe17 = BreakerTablePipeline17(debug=True)

    api17_results = []
    api17_errors = []

    for img_path in crops:
        try:
            api17_results.append(
                pipe17.run(img_path)
            )
            api17_errors.append("")

        except Exception as exc:
            api17_results.append(None)
            api17_errors.append(
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"[API17 ERROR] {img_path}: {exc}",
                flush=True,
            )

    del pipe17

    api17_elapsed = (
        time.perf_counter() - api17_start
    )

    print(
        f"[API17] done in {api17_elapsed:.1f}s",
        flush=True,
    )

    # ---- 4) API 18 ----
    print("\n[ParserAPI] starting…")

    pipe = BreakerTablePipeline(debug=True)

    print("[ParserAPI] API_VERSION:", API_VERSION)

    api18_start = time.perf_counter()

    comparisons = []


    all_results = []
    components = []
    unnamed_counts: dict[str, int] = {}

    total_hdr_breakers = 0
    total_tbl_breakers = 0

    for crop_index, img_path in enumerate(crops):
        print(f"\n\n=========================")
        print(f"Analyzing crop: {img_path}")
        print(f"=========================")



        try:
            result = pipe.run(img_path)

        except Exception as exc:
            error_text = f"API18 {type(exc).__name__}: {exc}"

            print(
                f"[API18 ERROR] {img_path}: {error_text}",
                flush=True,
            )

            comparison = compare_parser_outputs(
                INPUT_PDF,
                img_path,
                api17_results[crop_index],
                None,
                error_text,
            )

            comparisons.append(comparison)
            continue

        comparison = compare_parser_outputs(

            INPUT_PDF,
            img_path,
            api17_results[crop_index],
            result,
            api17_errors[crop_index],
        )

        comparisons.append(comparison)


        if comparison["status"] != "MATCH":
            panel_name = (
                comparison["panel_v18"]
                or comparison["panel_v17"]
                or "(unnamed)"
            )

            print(
                f"\n[DIFFERENCE] Job={INPUT_PDF.name} | "
                f"Panel={panel_name} | "
                f"Crop={Path(img_path).name}",
                flush=True,
            )

            for d in comparison["field_differences"]:
                print(
                    f"  FIELD {d['field']}: "
                    f"v17={d['v17']!r} | "
                    f"v18={d['v18']!r}",
                    flush=True,
                )

            for d in comparison["breaker_differences"]:
                print(
                    f"  {d['category'].upper()} "
                    f"{d['breaker_type']}: "
                    f"v17={d['v17_qty']} | "
                    f"v18={d['v18_qty']}",
                    flush=True,
                )

            raw_changes = comparison.get(
                "raw_differences", []
            )

            for d in raw_changes[:8]:
                print(
                    f"  RAW {d['path']}: "
                    f"v17={d['v17']} | "
                    f"v18={d['v18']}",
                    flush=True,
                )

            if len(raw_changes) > 8:
                print(
                    f"  ... {len(raw_changes) - 8} "
                    "additional raw differences. "
                    "See difference_details.csv",
                    flush=True,
                )

            if comparison["error"]:
                print(
                    f"  ERROR: {comparison['error']}",
                    flush=True,
                )


        print(
            f"[COMPARE] {Path(img_path).name}: "
            f"{comparison['status']} | "
            f"fields={len(comparison['field_differences'])} | "
            f"breaker types={len(comparison['breaker_differences'])}",
            flush=True,
        )

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
        print(
            "suspectedExisting :",
            hdr_res.get(
                "suspectedExisting",
                False,
            ),
        )

        print(
            "existingDetection :",
            hdr_res.get(
                "existingDetection",
            ),
        )

        print(
            "existingReason :",
            hdr_res.get(
                "existingReason",
            ),
        )
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

    api18_elapsed = time.perf_counter() - api18_start
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


    # Save complete API 17 raw outputs
    with (
        job_dir / "api17_raw_results.json"
    ).open("w", encoding="utf-8") as handle:

        json.dump(
            [
                {
                    "crop": str(c),
                    "result": r,
                    "error": e,
                }
                for c, r, e in zip(
                    crops,
                    api17_results,
                    api17_errors,
                )
            ],
            handle,
            indent=2,
            default=str,
        )

    # Save this job's comparison
    with (
        job_dir / "api17_vs_18_comparison.json"
    ).open("w", encoding="utf-8") as handle:

        json.dump(
            comparisons,
            handle,
            indent=2,
            default=str,
        )

    print(
        f"[TIMING] filter={filter_elapsed:.1f}s | "
        f"finder={finder_elapsed:.1f}s | "
        f"API17={api17_elapsed:.1f}s | "
        f"API18={api18_elapsed:.1f}s",
        flush=True,
    )

    return {
        "job": INPUT_PDF.name,
        "pdf": str(INPUT_PDF),
        "job_dir": str(job_dir),
        "panels": comparisons,
    }


if __name__ == "__main__":
    run_all_jobs()