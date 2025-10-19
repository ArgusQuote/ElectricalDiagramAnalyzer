# MS: Changed dpi to 400.
# -------------------------------
# Anvil Uplink VM (disk only) + Rules Engine defaults + cycle-time
# -------------------------------
import os, re, json, sys, threading, traceback
from pathlib import Path
from datetime import datetime, timezone
import anvil.server
# from anvil.tables import app_tables  # (disabled in this version)

# ---------- CONFIG ----------
# Ensure your repo is on sys.path (for imports below)
REPO_ROOT = Path("/home/paperspace/ElectricalDiagramAnalyzer").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Put jobs directly under the home directory
BASE_JOBS_DIR = Path.home() / "jobs"
BASE_JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Keep legacy dir around (not used directly)
(Path.home() / "uploaded_pdfs").mkdir(parents=True, exist_ok=True)

# ---------- IMPORTS FROM REPO ----------
from VisualDetectionToolLibrary.PanelSearchToolV6 import PanelBoardSearch
from OcrLibrary.PanelHeaderParserV4 import PanelParser
from OcrLibrary.PanelAnalysisAPI import PanelAnalysisAPI, analyze_many as analyze_many_panels
import RulesEngine.RulesEngine2 as RE2  # must expose process_job(payload)

# ---------- CONNECT UPLINK ----------
ANVIL_UPLINK_KEY = os.environ.get("ANVIL_UPLINK_KEY", "")
if not ANVIL_UPLINK_KEY:
    raise RuntimeError("Set ANVIL_UPLINK_KEY in environment (ANVIL_UPLINK_KEY).")
anvil.server.connect(ANVIL_UPLINK_KEY)
print(">>> ENTRY OK")

# ---------- SINGLETONS ----------
PARSER = PanelParser(debug=False)
ANALYZER = PanelAnalysisAPI(debug=False)
FINDER = PanelBoardSearch(
    output_dir=str(BASE_JOBS_DIR / "pdf_images"),
    dpi=300,
    min_rel_area=0.06,
    max_rel_area=0.40,
    aspect_ratio_range=(0.6, 2.2),
    pad=4,
    use_ocr=True,  # Enable OCR (default: True)
    panel_keywords=['panel', 'panelboard'],
    verbose=True,   # See OCR detection results   
    max_tables=10
)

_FINDER_LOCK = threading.Lock()

# ---------- OCR warmup (best-effort) ----------
def _warmup_ocr_once():
    try:
        import numpy as np
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)

        # Warm header OCR (EasyOCR) if present
        hdr = getattr(PARSER, "reader", None) or getattr(ANALYZER, "header_reader", None)
        if hdr:
            try:
                hdr.readtext(dummy, detail=0)
                print(">>> Header OCR warmup complete")
            except Exception as e:
                print(f">>> Header OCR warmup skipped: {e}")

        # Warm table OCR (if BreakerTableExtractor exposes a reader)
        tbl = getattr(ANALYZER, "table_reader", None)
        if tbl:
            try:
                tbl.readtext(dummy, detail=0)
                print(">>> Table OCR warmup complete")
            except Exception as e:
                print(f">>> Table OCR warmup skipped: {e}")

    except Exception as e:
        print(f">>> OCR warmup skipped: {e}")

_warmup_ocr_once()

# ---------- UTILITIES ----------
def _now_utc():
    return datetime.now(timezone.utc)

def _epoch_ms(dt=None) -> int:
    """UTC epoch milliseconds."""
    dt = dt or datetime.now(timezone.utc)
    return int(dt.timestamp() * 1000)

def _fmt_cycle_time(ms: int) -> str:
    """Format milliseconds as HH:MM:SS:MMM."""
    if ms is None or ms < 0:
        return "00:00:00:000"
    hours = ms // 3_600_000
    rem = ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    millis  = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{millis:03d}"

def _slugify(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s or "untitled"

def _json_read_or_none(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _parse_job_note(job_note: str) -> dict:
    """
    Expected client format (free-form tolerated):
      'job_name=Panel A | submitted_at_utc=2025-09-08T15:12:34.567Z | submitted_local=... | tz_offset_min=... | user=email'
    """
    out = {"job_name": None, "submitted_at_utc": None, "submitted_local": None, "tz_offset_min": None, "user": None}
    if not job_note:
        return out
    try:
        parts = [p.strip() for p in job_note.split("|")]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip()] = v.strip()
    except Exception:
        pass
    return out

def _iso_to_stamp(s: str) -> str:
    # '2025-09-08T15:12:34.567Z' -> '20250908_151234'
    try:
        s2 = s.rstrip("Z")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _make_job_dir(job_note: str, fallback_filename: str) -> Path:
    meta = _parse_job_note(job_note)
    job_name = _slugify(meta.get("job_name") or Path(fallback_filename).stem)
    stamp = _iso_to_stamp(meta.get("submitted_at_utc") or "")
    job_dir = BASE_JOBS_DIR / f"{job_name}__{stamp}"
    (job_dir / "uploaded_pdfs").mkdir(parents=True, exist_ok=True)
    (job_dir / "pdf_images").mkdir(parents=True, exist_ok=True)
    return job_dir

def _save_media_to_disk(media, dest_dir: Path) -> Path:
    fname = _slugify(getattr(media, "name", None) or "uploaded.pdf")
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    dst = dest_dir / fname
    with open(dst, "wb") as f:
        f.write(media.get_bytes())
    return dst

def _normalize_component_for_none(obj):
    """
    Normalize for JSON:
      - None -> "NONE"
      - numeric-like strings -> ints/floats
      - numpy scalars -> ints/floats
      - Path -> str
      - recurse lists/dicts
    """
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

# ----- status.json / result.json on disk -----
def _status_paths(dir_path: Path):
    dir_path = Path(dir_path)
    return {"status": dir_path / "status.json", "result": dir_path / "result.json"}

def _status_write(dir_path: Path, state: str, **extras):
    paths = _status_paths(dir_path)
    payload = {"state": state, **extras, "ts": datetime.now(timezone.utc).isoformat()}
    with open(paths["status"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str, indent=2)

def _result_write(dir_path: Path, result: dict):
    paths = _status_paths(dir_path)
    with open(paths["result"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, default=str, indent=2)

# ----- Data Tables helpers (disabled here; leave no-ops) -----
def _jobs_upsert(job_id: str, **fields):
    return

# ---------- UI OVERRIDES ----------
_DEFAULT_OVERRIDES = {
    "panelboards": {
        "bussing_material":        "ALUMINUM",
        "allow_plug_on_breakers":  True,
        "rating_type":             "FULLY_RATED",
        "allow_feed_thru_lugs":    True,
        "default_trim_style":      "FLUSH",
        "enclosure":               "NEMA1",
        "allow_square_d_spd":      True,
    },
    "transformers": {
        "winding_material":   "ALUMINUM",
        "temperature_rating": 150,
        "default_type":       "3PHASESTANDARD",
        "weathershield":      False,
        "mounting":           "FLOOR",
        "resin_enclosure":    "3R",
    },
    "disconnects": {
        "allow_littlefuse":         True,
        "default_switch_type":      "GENERAL_DUTY",
        "default_enclosure":        "NEMA1",
        "default_fusible":          True,
        "default_ground_required":  True,
        "default_solid_neutral":    True,
    },
}

def _deep_merge(dst: dict, src: dict) -> dict:
    out = dict(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _coerce_types(overrides: dict) -> dict:
    def coerce(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "false"):
                return s == "true"
            if s.isdigit():
                try:
                    return int(s)
                except Exception:
                    return v
        return v
    def walk(d):
        if isinstance(d, dict):
            return {k: walk(coerce(v)) for k, v in d.items()}
        if isinstance(d, list):
            return [walk(coerce(x)) for x in d]
        return coerce(d)
    return walk(overrides or {})

def _normalize_ui_overrides(overrides: dict | None) -> dict:
    return _deep_merge(_DEFAULT_OVERRIDES, _coerce_types(overrides or {}))

@anvil.server.callable
def vm_get_default_overrides() -> dict:
    # return a deep copy that the client can tweak safely
    return json.loads(json.dumps(_DEFAULT_OVERRIDES))

# ---------- PDF → images ----------
def render_pdf_to_images(saved_pdf: Path, img_dir: Path, dpi: int = 250) -> list[str]:
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> rendering PDF → images: {saved_pdf} -> {img_dir} (dpi={dpi})")
    with _FINDER_LOCK:
        FINDER.output_dir = str(img_dir)  # per-job output folder
        FINDER.dpi = dpi                  # per-job DPI
        try:
            crops = FINDER.readPdf(str(saved_pdf))
        except Exception as e:
            print(f">>> render error: {e}")
            raise
    print(f">>> rendered {len(crops)} image(s)")
    return crops

# ---------- Rules payload helper ----------
def _build_rules_payload(defaults: dict, items: list[dict]) -> dict:
    """Rules engine expects: {'defaults': <dict>, 'items': [<component dicts>]}"""
    return {"defaults": defaults or {}, "items": items or []}

# ---------- Core job processing ----------
def _process_job(job_id: str):
    """
    Worker thread: read images → parse panel → run rules (with defaults) → write result.
    """
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)

    try:
        print(f">>> worker start: {job_id}")
        prev = _json_read_or_none(sp["status"]) or {}

        # carry forward the "noticed" timestamp we wrote at submit time
        noticed_ts_ms = prev.get("noticed_ts_ms")

        # IMPORTANT: don't pass noticed_ts_ms twice
        _prev_carry = {k: v for k, v in prev.items() if k not in ("state", "ts", "noticed_ts_ms")}
        _status_write(
            job_dir,
            "running",
            **_prev_carry,
            noticed_ts_ms=noticed_ts_ms,
        )
        _jobs_upsert(job_id, state="running", updated_at=_now_utc())

        # Pull normalized overrides from status (saved during submit)
        ui_overrides = prev.get("ui_overrides") or _DEFAULT_OVERRIDES

        # Ensure images are present
        pdf_dir = job_dir / "uploaded_pdfs"
        img_dir = job_dir / "pdf_images"
        imgs = sorted(str(p) for p in img_dir.glob("*.png"))  # <-- ensure deterministic order
        if not imgs:
            pdfs = sorted(pdf_dir.glob("*.pdf"))              # <-- also deterministic for PDFs
            if not pdfs:
                raise RuntimeError("No PDF found to render.")
            imgs = render_pdf_to_images(pdfs[0], img_dir)

        if not imgs:                                          # <-- keep your guard
            raise RuntimeError("PDF rendered but produced no crops/images.")

        print(f">>> parsing {len(imgs)} images")
        _status_write(
            job_dir,
            "running",
            step="parsing",
            image_count=len(imgs),
            noticed_ts_ms=noticed_ts_ms,
        )

        # Parse all images → components
        components = []
        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # ---- BATCHED PARSING (max 8 at a time) ----
        BATCH_SIZE = 8
        components = []

        total = len(imgs)
        workers = max(1, min(4, os.cpu_count() or 1))  # safe default; tune if desired

        for start in range(0, total, BATCH_SIZE):
            batch_imgs = imgs[start:start + BATCH_SIZE]
            # Precompute debug overlay paths for this batch (order-aligned)
            batch_overlays = [
                str(debug_dir / f"breaker_debug_{start + j + 1:03d}.png")
                for j in range(len(batch_imgs))
            ]

            print(f">>> parsing batch {start + 1}-{start + len(batch_imgs)} / {total}")
            try:
                batch_results = analyze_many_panels(
                    batch_imgs,
                    workers=workers,
                    debug=False,
                    overlays=batch_overlays
                )
            except Exception as e:
                # Catastrophic batch failure (rare). Fallback: mark all in this batch as errors.
                print(f">>> ERROR batch analyzing images {start + 1}-{start + len(batch_imgs)}: {e}")
                print(f">>> Traceback: {traceback.format_exc()}")
                batch_results = [
                    {"type": "panelboard", "name": f"image_{start + j + 1}", "_skipped": True,
                    "reason": f"Batch error: {e}", "attrs": {}, "source": batch_imgs[j]}
                    for j in range(len(batch_imgs))
                ]

            # Normalize + log each result in batch order (deterministic)
            for j, comp in enumerate(batch_results):
                idx = start + j
                img_path = batch_imgs[j]
                try:
                    comp = comp or {}
                    comp = _normalize_component_for_none(comp)

                    attrs = comp.get("attrs") or {}
                    print(f">>> PANEL RESULTS for {img_path}:")
                    print(f"    Name: {comp.get('name')}")
                    print(f"    Amperage: {attrs.get('amperage')}")
                    print(f"    Voltage: {attrs.get('voltage')}")
                    print(f"    IntRating: {attrs.get('intRating')}")
                    print(f"    MainBreakerAmperage: {attrs.get('mainBreakerAmperage')}")
                    print(f"    Spaces (merged): {attrs.get('spaces')}")
                    print(f"    Detected breakers: {len(attrs.get('detected_breakers') or [])}")

                    components.append(comp)
                    print(f">>> parsed component: {comp.get('name', 'unnamed')} (skipped: {comp.get('_skipped', False)})")

                except Exception as e:
                    print(f">>> ERROR normalizing/logging image {idx + 1} ({img_path}): {e}")
                    print(f">>> Traceback: {traceback.format_exc()}")
                    components.append({
                        "type": "panelboard",
                        "name": f"image_{idx + 1}",
                        "_skipped": True,
                        "reason": f"Post-parse error: {e}",
                        "attrs": {},
                        "source": img_path
                    })

        print(f">>> parse done - {len(components)} components processed")

        # --- cycle time: from when we noticed the doc to parse completion ---
        parse_done_ts_ms = _epoch_ms()
        cycle_time_ms = (parse_done_ts_ms - noticed_ts_ms) if isinstance(noticed_ts_ms, int) else None
        cycle_time_str = _fmt_cycle_time(cycle_time_ms if cycle_time_ms is not None else 0)

        _status_write(
            job_dir,
            "running",
            step="parsed",
            component_count=len(components),
            image_count=len(imgs),
            noticed_ts_ms=noticed_ts_ms,
            parse_done_ts_ms=parse_done_ts_ms,
            cycle_time_ms=cycle_time_ms,
            cycle_time_str=cycle_time_str,
        )

        # ---- RUN RULES with defaults ----
        rules_payload = _build_rules_payload(ui_overrides, components)
        try:
            rules_result = RE2.process_job(rules_payload) or {}
        except Exception as re_err:
            rules_result = {"error": f"{type(re_err).__name__}: {re_err}"}
            print(f">>> rules engine error: {rules_result['error']}")

        # ---- RESULT JSON ----
        try:
            first_pdf = str(next(pdf_dir.glob("*.pdf")))
        except StopIteration:
            first_pdf = ""

        result = {
            "ok": True,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "saved_pdf": first_pdf,
            "output_dir": str(img_dir),
            "images": imgs,
            "image_count": len(imgs),
            "components": components,
            "rules_result": rules_result,
            "ui_overrides": ui_overrides,
            # cycle time
            "cycle_time_ms": cycle_time_ms,
            "cycle_time_str": cycle_time_str,
            "noticed_ts_ms": noticed_ts_ms,
            "parse_done_ts_ms": parse_done_ts_ms,
        }

        _result_write(job_dir, result)
        _status_write(job_dir, "done", result_path=str(sp["result"]))
        _jobs_upsert(job_id, state="done", updated_at=_now_utc(), result_json=result)

        print(f">>> worker done: {job_id}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f">>> worker error [{job_id}]: {e}\n{tb}")
        _status_write(job_dir, "error", error=f"{type(e).__name__}: {e}")
        _jobs_upsert(job_id, state="error", updated_at=_now_utc(), error=f"{type(e).__name__}: {e}")

def _start_worker(job_id: str):
    t = threading.Thread(target=_process_job, args=(job_id,), daemon=True)
    t.start()

@anvil.server.callable
def vm_submit_for_detection(media, ui_overrides=None, job_note=None):
    """
    Create job folder, save PDF, record 'noticed' time, render images, persist normalized overrides, spawn worker.
    Returns job meta immediately (with overrides echoed).
    """
    original_name = getattr(media, "name", "uploaded.pdf")
    job_dir = _make_job_dir(job_note, original_name)
    job_id = job_dir.name
    print(f">>> vm_submit_for_detection: job_dir={job_dir}")

    # 1) Save the uploaded PDF
    pdf_dir = job_dir / "uploaded_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf = _save_media_to_disk(media, pdf_dir)     # <-- brings back saved_pdf
    print(f">>> saved_pdf={saved_pdf}")

    # 2) Normalize overrides
    normalized_overrides = _normalize_ui_overrides(
        ui_overrides if isinstance(ui_overrides, dict) else {}
    )

    # 3) Record "noticed" time BEFORE rendering so cycle-time includes render
    noticed_ts_ms = _epoch_ms()

    # 4) Render images now so the client can show a count
    img_dir = job_dir / "pdf_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    images = render_pdf_to_images(saved_pdf, img_dir, dpi=400)  # or 250; your choice

    # 5) Persist status (disk)
    _status_write(
        job_dir,
        "queued",
        created_at=_now_utc().isoformat(),
        file_path=str(saved_pdf),
        job_dir_path=str(job_dir),
        ui_overrides=normalized_overrides,
        job_note=(job_note or ""),
        image_count=len(images),
        step="queued",
        noticed_ts_ms=noticed_ts_ms,
    )

    # 6) (Optional) Data Table upsert (no-op here)
    meta = _parse_job_note(job_note or "")
    _jobs_upsert(
        job_id,
        state="queued",
        created_at=_now_utc(),
        updated_at=_now_utc(),
        job_dir=str(job_dir),
        file_path=str(saved_pdf),
        job_note=(job_note or ""),
        user_email=meta.get("user") or "",
        image_count=len(images),
        ui_overrides=normalized_overrides,
    )

    # 7) Kick worker
    _start_worker(job_id)

    print(f">>> Job queued: {job_id} (images={len(images)})")
    return {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "saved_pdf": str(saved_pdf),
        "output_dir": str(img_dir),
        "images": images,
        "image_count": len(images),
        "ui_overrides": normalized_overrides,
        "noticed_ts_ms": noticed_ts_ms,
    }

@anvil.server.callable
def vm_get_job_status(job_id: str) -> dict:
    """Status primarily from disk; returns result when done."""
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)

    st = _json_read_or_none(sp["status"])
    if not st:
        return {"state": "error", "error": f"Unknown job_id {job_id}"}

    state = (st.get("state") or "unknown").lower()
    if state == "done":
        res = _json_read_or_none(sp["result"]) or {}
        return {"state": "done", "result": res}
    if state == "error":
        return {"state": "error", "error": st.get("error") or "Unknown error"}

    # running/queued + progress hints
    out = {"state": state}
    for k in ("step", "image_count", "chosen", "ui_overrides", "noticed_ts_ms", "cycle_time_str", "cycle_time_ms"):
        if k in st:
            out[k] = st[k]

    # Optional: live elapsed display while queued/running
    if "noticed_ts_ms" in st and isinstance(st["noticed_ts_ms"], int):
        elapsed_ms = max(0, _epoch_ms() - int(st["noticed_ts_ms"]))
        out["elapsed_ms"]  = elapsed_ms
        out["elapsed_str"] = _fmt_cycle_time(elapsed_ms)

    return out

# ---------- MAIN ----------
if __name__ == "__main__":
    print(">>> Uplink ready; waiting for calls")
    anvil.server.wait_forever()
