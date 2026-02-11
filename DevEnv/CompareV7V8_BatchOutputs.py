#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare header parser outputs from DevTestEnv (V7) vs DevTestEnvV2 (V8) batch runs.
Reads pipeline_dump.json from each output base and prints per-PDF, per-crop differences.
"""
import json
from pathlib import Path

OUT_V7 = Path("~/Documents/DevEnv_FullAnalysis").expanduser()
OUT_V8 = Path("~/Documents/DevEnv2_FullAnalysis").expanduser()


def header_attrs(entry: dict) -> dict:
    """Extract header name + attrs from a pipeline dump entry."""
    results = entry.get("results") or {}
    hdr = (results.get("results") or {}).get("header") or {}
    attrs = dict(hdr.get("attrs") or {})
    return {"name": hdr.get("name"), **attrs}


def main():
    if not OUT_V7.is_dir():
        print(f"V7 output dir not found: {OUT_V7}")
        return
    if not OUT_V8.is_dir():
        print(f"V8 output dir not found: {OUT_V8}")
        return

    stems_v7 = {d.name for d in OUT_V7.iterdir() if d.is_dir() and (d / "ParserOutput" / "pipeline_dump.json").exists()}
    stems_v8 = {d.name for d in OUT_V8.iterdir() if d.is_dir() and (d / "ParserOutput" / "pipeline_dump.json").exists()}
    common = sorted(stems_v7 & stems_v8)
    only_v7 = sorted(stems_v7 - stems_v8)
    only_v8 = sorted(stems_v8 - stems_v7)

    print("PanelHeaderParser V7 vs V8 — batch output comparison")
    print("=" * 70)
    print(f"V7 base: {OUT_V7}")
    print(f"V8 base: {OUT_V8}")
    print(f"PDFs in both: {len(common)}  |  Only in V7: {len(only_v7)}  |  Only in V8: {len(only_v8)}")
    if only_v7:
        print(f"  Only V7: {only_v7[:10]}{'...' if len(only_v7) > 10 else ''}")
    if only_v8:
        print(f"  Only V8: {only_v8[:10]}{'...' if len(only_v8) > 10 else ''}")
    print()

    fields = ["name", "voltage", "amperage", "mainBreakerAmperage", "intRating"]
    diffs = []

    for stem in common:
        with open(OUT_V7 / stem / "ParserOutput" / "pipeline_dump.json", encoding="utf-8") as f:
            list_v7 = json.load(f)
        with open(OUT_V8 / stem / "ParserOutput" / "pipeline_dump.json", encoding="utf-8") as f:
            list_v8 = json.load(f)

        n = max(len(list_v7), len(list_v8))
        for i in range(n):
            e7 = list_v7[i] if i < len(list_v7) else {}
            e8 = list_v8[i] if i < len(list_v8) else {}
            a7 = header_attrs(e7)
            a8 = header_attrs(e8)
            for k in fields:
                v7 = a7.get(k)
                v8 = a8.get(k)
                if v7 != v8:
                    diffs.append((stem, i + 1, k, v7, v8))

    if not diffs:
        print("No header differences between V7 and V8 for compared PDFs.")
        return

    print("Differences (PDF, crop, field, V7 value, V8 value):")
    print("-" * 70)
    for stem, crop, field, v7, v8 in diffs:
        print(f"  {stem}  crop{crop}  {field}:  {v7!r}  ->  {v8!r}")
    print(f"\nTotal: {len(diffs)} difference(s)")


if __name__ == "__main__":
    main()
