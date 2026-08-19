#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Row-level diff of APIv13 (Parser11) vs APIv14 (Parser12) on specific crops.

Prints every (side, rowIndex) slot where the two parsers disagree, including the
raw OCR text, so a lost breaker can be traced to either an empty/failed OCR read
or a rejected parse.

Usage:
    python DevEnv/DiagnoseV14Regressions.py CROP [CROP ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from OcrLibrary.BreakerTableParserAPIv13 import (
    BreakerTablePipeline as PipelineV13,
    reset_name_deduper as reset_names_v13,
)
from OcrLibrary.BreakerTableParserAPIv14 import (
    BreakerTablePipeline as PipelineV14,
    reset_name_deduper as reset_names_v14,
)


def row_map(result: dict) -> dict:
    """Key detected breakers by (side, rowIndex) for slot-wise comparison."""
    stages = result.get("results") or {}
    parser = stages.get("parser") or {}
    rows = {}
    for breaker in parser.get("detected_breakers") or []:
        key = (breaker.get("side"), breaker.get("rowIndex"))
        rows[key] = {
            "amperage": breaker.get("amperage"),
            "poles": breaker.get("poles"),
            "tripText": breaker.get("tripText"),
            "polesText": breaker.get("polesText"),
            "comboText": breaker.get("comboText"),
        }
    return rows


def review_map(result: dict) -> dict:
    """Key review cells by (side, rowIndex, role) so we can see why a row failed."""
    stages = result.get("results") or {}
    parser = stages.get("parser") or {}
    cells = {}
    for cell in parser.get("reviewCells") or []:
        key = (cell.get("side"), cell.get("rowIndex"), cell.get("role") or cell.get("column"))
        cells[key] = {
            "status": cell.get("status"),
            "text": cell.get("text") or cell.get("ocrText"),
            "conf": cell.get("conf") or cell.get("confidence"),
            "reason": cell.get("reason") or cell.get("message"),
        }
    return cells


def fmt(row: dict | None) -> str:
    if row is None:
        return "-- none --"
    amps = row.get("amperage")
    poles = row.get("poles")
    texts = [t for t in (row.get("tripText"), row.get("polesText"), row.get("comboText")) if t]
    return f"{poles}P {amps}A   raw={texts}"


def diagnose(crop: Path) -> dict:
    reset_names_v13()
    res13 = PipelineV13(debug=False).run(str(crop))
    reset_names_v14()
    res14 = PipelineV14(debug=False).run(str(crop))

    rows13 = row_map(res13)
    rows14 = row_map(res14)
    rev14 = review_map(res14)
    rev13 = review_map(res13)

    only13 = sorted(set(rows13) - set(rows14), key=lambda k: (str(k[0]), k[1] or 0))
    only14 = sorted(set(rows14) - set(rows13), key=lambda k: (str(k[0]), k[1] or 0))
    shared_changed = sorted(
        (k for k in set(rows13) & set(rows14) if rows13[k] != rows14[k]),
        key=lambda k: (str(k[0]), k[1] or 0),
    )

    print(f"\n{'=' * 78}")
    print(f"{crop.name}")
    print(f"  v13 breakers={len(rows13)}   v14 breakers={len(rows14)}   delta={len(rows14) - len(rows13):+d}")
    print(f"{'=' * 78}")

    if only13:
        print(f"\n  LOST by v14 ({len(only13)} slot(s)) -- present in v13, absent in v14:")
        for key in only13:
            print(f"    {key[0]:>5} row {key[1]}: {fmt(rows13[key])}")
            for rk, rv in rev14.items():
                if rk[0] == key[0] and rk[1] == key[1]:
                    print(f"          v14 review[{rk[2]}]: status={rv['status']} text={rv['text']!r} conf={rv['conf']}")

    if only14:
        print(f"\n  GAINED by v14 ({len(only14)} slot(s)):")
        for key in only14:
            print(f"    {key[0]:>5} row {key[1]}: {fmt(rows14[key])}")

    if shared_changed:
        print(f"\n  VALUE CHANGED in shared slots ({len(shared_changed)}):")
        for key in shared_changed:
            print(f"    {key[0]:>5} row {key[1]}:")
            print(f"          v13: {fmt(rows13[key])}")
            print(f"          v14: {fmt(rows14[key])}")

    if not (only13 or only14 or shared_changed):
        print("\n  identical")

    return {
        "crop": crop.name,
        "v13": len(rows13),
        "v14": len(rows14),
        "lost": len(only13),
        "gained": len(only14),
        "changed": len(shared_changed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crops", nargs="+", type=Path)
    args = parser.parse_args()

    summaries = [diagnose(Path(c).expanduser()) for c in args.crops]

    print(f"\n\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for s in summaries:
        print(
            f"{s['crop']:<52} v13={s['v13']:>4} v14={s['v14']:>4} "
            f"lost={s['lost']:>3} gained={s['gained']:>3} changed={s['changed']:>3}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
