#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-way breaker comparison across parser generations on the same crops.

    v13 = BreakerTableParserAPIv13 / BreakerTableParser11   (pre cell-prep)
    v14 = BreakerTableParserAPIv14 / BreakerTableParser12   (cell-prep)
    v15 = BreakerTableParserAPIv15 / BreakerTableParser13   (cell-prep + raw fallback)

The acceptance bar for v15 is "never below v13 on breaker count, and at least
v14 everywhere", so the report prints both deltas per panel.

Usage:
    python DevEnv/Compare3Way_v13_v14_v15.py CROP [CROP ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from OcrLibrary.BreakerTableParserAPIv13 import (
    BreakerTablePipeline as PipelineV13,
    reset_name_deduper as reset_v13,
)
from OcrLibrary.BreakerTableParserAPIv14 import (
    BreakerTablePipeline as PipelineV14,
    reset_name_deduper as reset_v14,
)
from OcrLibrary.BreakerTableParserAPIv15 import (
    BreakerTablePipeline as PipelineV15,
    reset_name_deduper as reset_v15,
)

VERSIONS = (
    ("v13", PipelineV13, reset_v13),
    ("v14", PipelineV14, reset_v14),
    ("v15", PipelineV15, reset_v15),
)


def summarize(result: dict) -> dict:
    stages = result.get("results") or {}
    header = stages.get("header") or {}
    parser = stages.get("parser") or {}
    rows = {}
    for breaker in parser.get("detected_breakers") or []:
        rows[(breaker.get("side"), breaker.get("rowIndex"))] = (
            breaker.get("amperage"),
            breaker.get("poles"),
        )
    return {
        "name": header.get("name"),
        "panelStatus": result.get("panelStatus"),
        "spaces": parser.get("spaces"),
        "breakerCounts": parser.get("breakerCounts") or {},
        "reviewCells": len(parser.get("reviewCells") or []),
        "count": len(parser.get("detected_breakers") or []),
        "rows": rows,
    }


def run_all(crop: Path) -> dict:
    out = {}
    for label, pipeline_cls, reset in VERSIONS:
        reset()
        started = time.perf_counter()
        result = pipeline_cls(debug=False).run(str(crop))
        summary = summarize(result)
        summary["seconds"] = round(time.perf_counter() - started, 1)
        out[label] = summary
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crops", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="write full report here")
    args = parser.parse_args()

    report = []
    for crop in args.crops:
        crop = Path(crop).expanduser()
        res = run_all(crop)
        v13, v14, v15 = res["v13"], res["v14"], res["v15"]

        recovered = sorted(set(v13["rows"]) - set(v14["rows"]) & set(v15["rows"]))
        still_lost = sorted(set(v13["rows"]) - set(v15["rows"]))

        print(f"\n{'=' * 76}")
        print(f"{crop.name}")
        print(
            f"  breakers   v13={v13['count']:>4}  v14={v14['count']:>4}  v15={v15['count']:>4}"
            f"   (v15-v13={v15['count'] - v13['count']:+d}, v15-v14={v15['count'] - v14['count']:+d})"
        )
        print(
            f"  review     v13={v13['reviewCells']:>4}  v14={v14['reviewCells']:>4}  v15={v15['reviewCells']:>4}"
        )
        print(f"  seconds    v13={v13['seconds']}  v14={v14['seconds']}  v15={v15['seconds']}")
        if recovered:
            print(f"  RECOVERED vs v14 ({len(recovered)}): {recovered}")
        if still_lost:
            print(f"  STILL LOST vs v13 ({len(still_lost)}): {still_lost}")
            for key in still_lost:
                print(f"      {key}: v13 had {v13['rows'][key]}")

        report.append(
            {
                "crop": crop.name,
                "path": str(crop),
                "v13": {k: v for k, v in v13.items() if k != "rows"},
                "v14": {k: v for k, v in v14.items() if k != "rows"},
                "v15": {k: v for k, v in v15.items() if k != "rows"},
                "recovered_vs_v14": [list(k) for k in recovered],
                "still_lost_vs_v13": [list(k) for k in still_lost],
            }
        )

    t13 = sum(r["v13"]["count"] for r in report)
    t14 = sum(r["v14"]["count"] for r in report)
    t15 = sum(r["v15"]["count"] for r in report)
    regressions = [r["crop"] for r in report if r["v15"]["count"] < r["v13"]["count"]]

    print(f"\n\n{'=' * 76}\nTOTALS\n{'=' * 76}")
    print(f"  breakers   v13={t13}   v14={t14}   v15={t15}")
    print(f"  v15 vs v13 {t15 - t13:+d}      v15 vs v14 {t15 - t14:+d}")
    print(f"  panels below v13: {len(regressions)}  {regressions}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {"totals": {"v13": t13, "v14": t14, "v15": t15}, "panels": report},
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        print(f"\n[WROTE] {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
