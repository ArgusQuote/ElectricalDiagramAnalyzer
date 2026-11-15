#!/usr/bin/env python3
# MLCellReader/harvest_cell_crops.py
"""
Harvest every trip / poles / combo body-cell crop the production parser reads,
together with what EasyOCR + the parser made of it, as raw material for
labeling and training the cell classifier.

It runs the normal dev batch stack (PageFilterV5 + PanelSearchToolV26 +
BreakerTableParserAPIv15) and intercepts BreakerTableParser13 *in this process
only*, so the crops are pixel-identical to what the parser OCRs. Production
code is not modified.

Usage (from the repo root, venv active):

    python MLCellReader/harvest_cell_crops.py                  # default corpus
    python MLCellReader/harvest_cell_crops.py --input ~/Documents/NewTest/H.pdf
    python MLCellReader/harvest_cell_crops.py --limit-pdfs 1   # quick smoke run

PDFs already in the manifest (matched by md5) are skipped, so it is safe to
re-run after adding new documents.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "DevEnv")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import OcrLibrary.BreakerTableParser13 as parser13  # noqa: E402
from MLCellReader.cell_data import (  # noqa: E402
    CELL_ROLES,
    ROLE_COMBO,
    ROLE_POLES,
    ROLE_TRIP,
    append_jsonl,
    crops_dir,
    data_root,
    load_manifest,
    manifest_path,
)

DEFAULT_INPUTS = (
    Path("~/Documents/pdfToScan").expanduser(),
    Path("~/Documents/Diagrams").expanduser(),
)
# PageFilter writes "<stem>_electrical_filtered.pdf" copies next to some sources;
# harvesting those would duplicate pages that are already in the corpus.
SKIP_NAME_FRAGMENTS = ("_electrical_filtered",)
DEFAULT_ARTIFACT_ROOT = Path(_REPO_ROOT) / "DevEnv" / "BatchRuns_cellharvest"
PDF_ID_PREFIX_LEN = 10
# Row bands this short are slivers of grid line between real rows, not cells.
MIN_CROP_HEIGHT_PX = 12


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_pdfs(inputs: Iterable[Path]) -> List[Path]:
    """Expand files/dirs into a list of unique PDFs (deduplicated by content)."""
    candidates: List[Path] = []
    for item in inputs:
        item = item.expanduser()
        if item.is_dir():
            candidates.extend(sorted(item.glob("*.pdf")))
        elif item.suffix.lower() == ".pdf" and item.exists():
            candidates.append(item)
        else:
            print(f"[harvest] Ignoring missing/non-PDF input: {item}")

    unique: List[Path] = []
    seen_md5 = set()
    for pdf in candidates:
        if any(fragment in pdf.stem for fragment in SKIP_NAME_FRAGMENTS):
            continue
        md5 = file_md5(pdf)
        if md5 in seen_md5:
            print(f"[harvest] Skipping duplicate content: {pdf}")
            continue
        seen_md5.add(md5)
        unique.append(pdf)
    return unique


class CellHarvester:
    """
    Intercepts BreakerTableParser13 body OCR and writes one crop + manifest
    record per row band of every trip / poles / combo column.
    """

    def __init__(self, root: Path):
        self.root = root
        self.known_ids = set(load_manifest(root))
        self.counts: Counter = Counter()
        self._pdf_name: Optional[str] = None
        self._pdf_md5: Optional[str] = None
        self._panel: Optional[dict] = None
        self._pending: List[dict] = []
        for role in CELL_ROLES:
            crops_dir(root, role).mkdir(parents=True, exist_ok=True)

    def set_pdf(self, pdf: Path, md5: str) -> None:
        self._pdf_name = pdf.name
        self._pdf_md5 = md5

    def install(self) -> None:
        for cls in (parser13.SeparatedLayoutParser, parser13.CombinedLayoutParser):
            self._wrap_parse(cls)
            self._wrap_column_ocr(cls)

    def _wrap_parse(self, cls) -> None:
        original = cls.parse
        harvester = self

        def parse_with_column_roles(parser_self, analyzer_result, header_scan):
            columns = ((header_scan or {}).get("normalizedColumns") or {}).get("columns") or []
            # The parser crops each column at max(0, int(x_left)), which is the
            # only key the per-column OCR call carries back to us.
            role_by_x = {}
            for col in columns:
                try:
                    role_by_x[max(0, int(col.get("x_left", 0)))] = col.get("role")
                except (TypeError, ValueError):
                    continue
            harvester._panel = {
                "src_path": (analyzer_result or {}).get("src_path") or "",
                "role_by_x": role_by_x,
            }
            try:
                return original(parser_self, analyzer_result, header_scan)
            finally:
                harvester._panel = None
                harvester._flush()

        cls.parse = parse_with_column_roles

    def _wrap_column_ocr(self, cls) -> None:
        original = cls._ocr_body_column_by_rows
        signature = inspect.signature(original)
        harvester = self

        def ocr_and_harvest(parser_self, *args, **kwargs):
            tokens = original(parser_self, *args, **kwargs)
            try:
                bound = signature.bind(parser_self, *args, **kwargs)
                bound.apply_defaults()
                harvester._record_column(parser_self, bound.arguments, tokens)
            except Exception as exc:  # harvesting must never break the parse itself
                print(f"[harvest] WARNING: could not record column: {type(exc).__name__}: {exc}")
            return tokens

        cls._ocr_body_column_by_rows = ocr_and_harvest

    def _record_column(self, parser_self, arguments: dict, tokens: list) -> None:
        if self._panel is None:
            return

        x_left = int(arguments["x_left"])
        role = self._panel["role_by_x"].get(x_left)
        if role not in CELL_ROLES:
            return

        strip_gray = arguments["strip_gray"]
        if strip_gray is None or strip_gray.size == 0:
            return

        body_y_top = int(arguments["body_y_top"])
        panel_path = self._panel["src_path"]
        panel_stem = Path(panel_path).stem or "panel"
        strip_h, strip_w = strip_gray.shape[:2]

        for row_index, (row_top, row_bottom) in enumerate(arguments["row_bands_strip"]):
            # Same clamping as BreakerTableParser13._ocr_body_column_by_rows.
            row_top = max(0, min(strip_h - 1, int(row_top)))
            row_bottom = max(row_top + 1, min(strip_h, int(row_bottom)))
            crop = strip_gray[row_top:row_bottom, :]
            if crop.size == 0 or crop.shape[0] < MIN_CROP_HEIGHT_PX:
                continue

            page_top = body_y_top + row_top
            page_bottom = body_y_top + row_bottom
            cell_id = (
                f"{self._pdf_md5[:PDF_ID_PREFIX_LEN]}__{panel_stem}__{role}"
                f"__x{x_left}_y{page_top}"
            )
            if cell_id in self.known_ids:
                continue

            row_tokens = [
                t for t in tokens
                if page_top <= (t["box_page"][1] + t["box_page"][3]) / 2.0 < page_bottom
            ]
            row_tokens.sort(key=lambda t: t["box_page"][0])
            ocr_text = " ".join(t["text"] for t in row_tokens).strip()
            ocr_conf = min((float(t["conf"]) for t in row_tokens), default=None)
            weak_amps, weak_poles = self._weak_values(parser_self, role, ocr_text)

            rel_image = Path("crops") / role / f"{cell_id}.png"
            if not cv2.imwrite(str(self.root / rel_image), crop):
                print(f"[harvest] WARNING: failed to write {rel_image}")
                continue

            self.known_ids.add(cell_id)
            self.counts[role] += 1
            self._pending.append(
                {
                    "id": cell_id,
                    "role": role,
                    "image": str(rel_image),
                    "source_pdf": self._pdf_name,
                    "source_pdf_md5": self._pdf_md5,
                    "panel_image": panel_path,
                    "parser_version": parser13.PARSER_VERSION,
                    "row_index": row_index,
                    "box_page": [x_left, page_top, x_left + strip_w, page_bottom],
                    "ocr_text": ocr_text,
                    "ocr_conf": ocr_conf,
                    "weak_amps": weak_amps,
                    "weak_poles": weak_poles,
                }
            )

    @staticmethod
    def _weak_values(parser_self, role: str, text: str):
        """What the current parser would conclude from this cell's OCR text."""
        if role == ROLE_TRIP:
            return parser_self._parse_trip_value(text), None
        if role == ROLE_POLES:
            return None, parser_self._parse_poles_value(text)
        if role == ROLE_COMBO:
            pairs = parser_self._parse_combo_cell(text)
            if pairs:
                return int(pairs[0][0]), int(pairs[0][1])
        return None, None

    def _flush(self) -> None:
        if self._pending:
            append_jsonl(manifest_path(self.root), self._pending)
            self._pending = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest breaker body-cell crops for labeling.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="PDF files and/or directories of PDFs (default: pdfToScan + Diagrams)",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Annotation data root")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Where page-filter / panel-finder intermediates go",
    )
    parser.add_argument("--limit-pdfs", type=int, default=None, help="Only process the first N PDFs")
    parser.add_argument(
        "--reharvest",
        action="store_true",
        help="Process PDFs even if their md5 is already in the manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = data_root(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(args.input)
    harvested_md5 = {r.get("source_pdf_md5") for r in load_manifest(root).values()}

    # Imported late: these pull in EasyOCR / torch and print a lot on import.
    from DevTestEnvBatch import load_parser_api, process_one_pdf

    pipeline_cls, api_version = load_parser_api("15")
    harvester = CellHarvester(root)
    harvester.install()
    pipe = pipeline_cls(debug=False)

    todo = []
    for pdf in pdfs:
        md5 = file_md5(pdf)
        if md5 in harvested_md5 and not args.reharvest:
            print(f"[harvest] Already harvested, skipping: {pdf.name}")
            continue
        todo.append((pdf, md5))
    if args.limit_pdfs is not None:
        todo = todo[: args.limit_pdfs]

    print(f"[harvest] data_root={root}")
    print(f"[harvest] {len(todo)} PDF(s) to process")

    failures = []
    start = time.perf_counter()
    for index, (pdf, md5) in enumerate(todo, 1):
        print(f"\n[harvest] ===== PDF {index}/{len(todo)}: {pdf.name} =====")
        before = sum(harvester.counts.values())
        harvester.set_pdf(pdf, md5)
        artifact_dir = args.artifact_root / pdf.stem
        try:
            process_one_pdf(
                pdf,
                artifact_dir / "PdfOutput",
                artifact_dir / "PanelSearchOutput",
                pipe,
                debug=False,
                api_version=api_version,
            )
        except Exception as exc:
            print(f"[harvest] ERROR on {pdf.name}: {type(exc).__name__}: {exc}")
            failures.append(pdf.name)
        print(f"[harvest] {pdf.name}: +{sum(harvester.counts.values()) - before} crops")

    elapsed_min = (time.perf_counter() - start) / 60.0
    print("\n[harvest] ===== DONE =====")
    print(f"[harvest] new crops by role: {dict(harvester.counts)}")
    print(f"[harvest] total crops in manifest: {len(harvester.known_ids)}")
    print(f"[harvest] wall time: {elapsed_min:.1f} min")
    if failures:
        print(f"[harvest] failed PDFs: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
