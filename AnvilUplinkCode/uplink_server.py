# Anvil Uplink VM (disk only) + Rules Engine defaults + cycle-time
# -------------------------------
import os, re, json, sys, threading, traceback
import contextlib
from multiprocessing import get_context
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime, timezone
import anvil.server
import platform
import os as _os
from anvil import BlobMedia

# ---------- CONFIG ----------
REPO_ROOT = Path("/home/paperspace/ElectricalDiagramAnalyzer").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Put jobs directly under the home directory
BASE_JOBS_DIR = Path.home() / "jobs"
BASE_JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- PANEL FINDER CONFIG (PanelSearchToolV18) ----------
PANEL_FINDER_DEFAULTS = {
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

# Keep legacy dir around (not used directly)
(Path.home() / "uploaded_pdfs").mkdir(parents=True, exist_ok=True)

# Worker pool limits
MAX_WORKERS = 4
MAX_INFLIGHT_PER_USER = 1

# Recycle the persistent worker after this many jobs to prevent resource
# degradation (RSS creep, GPU memory fragmentation, leaked OCR threads).
WORKER_RECYCLE_AFTER_JOBS = int(os.environ.get("WORKER_RECYCLE_JOBS", "25"))

# ===== Determinism & Thread Caps (must run before heavy libs init) =====
from WorkerSetup import cap_thread_pools, set_runtime_determinism
cap_thread_pools()

# ---------- WATCHDOG CONFIG ----------
WATCHDOG_TIMEOUT_MIN = int(os.environ.get("WATCHDOG_TIMEOUT_MIN", "10"))  # dial in prod
WATCHDOG_KILL_GRACE_SEC = int(os.environ.get("WATCHDOG_KILL_GRACE_SEC", "3"))
WATCHDOG_ERROR_MSG = (
  "This job took over {mins} minutes to process. "
  "Please trim the PDF to only relevant pages and try again."
)

# ---------- QUEUE TIMEOUT CONFIG ----------
QUEUE_TIMEOUT_MIN = int(os.environ.get("QUEUE_TIMEOUT_MIN", "25"))
QUEUE_TIMEOUT_ERROR_MSG = (
  "This job waited in the processing queue too long due to current demand. "
  "Please try again in a few minutes."
)

def _set_runtime_determinism():
    """Delegate to shared WorkerSetup module."""
    set_runtime_determinism()

def _log_run_fingerprint(tag: str = ""):
    try:
        import torch
        devs = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devs.append(torch.cuda.get_device_name(i))
        cudnn_det = getattr(getattr(torch.backends, "cudnn", None), "deterministic", None)
        cudnn_bmk = getattr(getattr(torch.backends, "cudnn", None), "benchmark", None)
        print(f">>> FPRINT {tag} | torch_cuda={torch.cuda.is_available()} devices={devs} cudnn.det={cudnn_det} cudnn.bmk={cudnn_bmk}")
    except Exception:
        print(f">>> FPRINT {tag} | torch not present")

_set_runtime_determinism()
_log_run_fingerprint("init")

# ---------- IMPORTS FROM REPO ----------
from PageFilter.PageFilterV3 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch
from OcrLibrary.BreakerTableParserAPIv10 import BreakerTablePipeline, API_VERSION, reset_name_deduper
import RulesEngine.RulesEngine4 as RE2  # must expose process_job(payload)

# Persistent worker subprocesses set this env var so module-level
# initialization (Anvil connection, warmup, worker threads) is skipped.
_IS_WORKER_SUBPROCESS = os.environ.get("_EDA_WORKER_SUBPROCESS") == "1"

# ---------- CONNECT UPLINK ----------
NODE_ID = ""
if not _IS_WORKER_SUBPROCESS:
    ANVIL_UPLINK_KEY = os.environ.get("ANVIL_UPLINK_KEY", "")
    if not ANVIL_UPLINK_KEY:
        raise RuntimeError("Set ANVIL_UPLINK_KEY in environment (ANVIL_UPLINK_KEY).")
    anvil.server.connect(ANVIL_UPLINK_KEY)
    print(">>> ENTRY OK")
    NODE_ID = f"{platform.node()}:{_os.getpid()}"
    print(f">>> NODE_ID={NODE_ID}")

# ---------- OCR warmup (via BreakerTablePipeline) ----------
def _warmup_ocr_once():
    try:
        _log_run_fingerprint("warmup")
        import numpy as np, cv2, tempfile
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, img)
            # Warm analyzer + header (table parser not needed for warmup)
            pipe = BreakerTablePipeline(debug=False)
            _ = pipe.run(
                tmp_path,
                run_analyzer=True,
                run_parser=False,
                run_header=True,
            )
            print(f">>> OCR warmup complete (analyzer + header) | API_VERSION={API_VERSION}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as e:
        print(f">>> OCR warmup skipped: {e}")

if not _IS_WORKER_SUBPROCESS:
    # Warmup skipped: main process does not do OCR.  Each worker subprocess
    # loads its own GPU EasyOCR reader, so warming up here just wastes ~10 GiB
    # VRAM that the worker pool needs.
    print(">>> Main process: skipping OCR warmup (workers load their own models)")

# ---------- UTILITIES ----------
def _now_utc():
    return datetime.now(timezone.utc)

def _epoch_ms(dt=None) -> int:
    dt = dt or datetime.now(timezone.utc)
    return int(dt.timestamp() * 1000)

def _fmt_cycle_time(ms: int) -> str:
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
    Parse a job_note string like:
      "job_name=My Project | submitted_at_utc=2025-03-01T12:34:56Z | user=jane@example.com"
    into a dict:
      {"job_name": "...", "submitted_at_utc": "...", "submitted_local": None, "tz_offset_min": None, "user": "..."}
    Unknown/missing fields are left as None.
    """
    out = {
        "job_name": None,
        "submitted_at_utc": None,
        "submitted_local": None,
        "tz_offset_min": None,
        "user": None,
    }
    if not job_note:
        return out

    try:
        parts = [p.strip() for p in job_note.split("|")]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)   # split on first '='
                out[k.strip()] = v.strip()
    except Exception:
        pass

    return out

def _iso_to_stamp(s: str) -> str:
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
    """
    Write status.json while preserving existing fields unless explicitly overwritten.
    This prevents losing owner_id/owner_email on error/timeouts.
    """
    paths = _status_paths(dir_path)

    prev = {}
    try:
        prev = _json_read_or_none(paths["status"]) or {}
    except Exception:
        prev = {}

    # Merge: previous -> extras -> required fields
    payload = dict(prev)
    payload.update(extras or {})
    payload["state"] = state
    payload["ts"] = datetime.now(timezone.utc).isoformat()

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

# Delete unused images and folders for storage
def _rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root)).replace("\\", "/")

def _collect_keep_relpaths(job_dir: Path, keep_pdf: bool = True) -> set[str]:
    """
    Return a set of job-relative file paths to keep.
    Everything else in the job folder will be deleted.
    """
    keep: set[str] = set()

    # Always keep status/result
    keep.add("status.json")
    keep.add("result.json")

    pdf_images = job_dir / "pdf_images"
    if not pdf_images.is_dir():
        return keep

    # 1) Keep REVIEW overlays (per-panel overlays)
    review_dir = pdf_images / "review_overlays"
    if review_dir.is_dir():
        for p in review_dir.rglob("*"):
            if p.is_file():
                keep.add(_rel(p, job_dir))

    # 2) Keep MAGENTA overlays exactly like vm_list_magenta_overlay_images()
    candidate_dirs = [
        pdf_images / "magenta_overlays",
        pdf_images / "magenta_overlay",
        pdf_images / "page_overlays",
        pdf_images / "full_overlays",
        pdf_images / "overlays",
    ]

    found: list[Path] = []
    for d in candidate_dirs:
        if d.is_dir():
            found.extend(list(d.glob("*.png")))

    # Fallback: any png containing "magenta" (excluding review_overlays)
    if not found:
        for p in pdf_images.rglob("*.png"):
            if "review_overlays" in p.parts:
                continue
            if "magenta" in p.name.lower():
                found.append(p)

    for p in found:
        if p.is_file():
            keep.add(_rel(p, job_dir))

    # Optional: keep original uploaded pdf(s)
    if keep_pdf:
        up = job_dir / "uploaded_pdfs"
        if up.is_dir():
            for p in up.rglob("*.pdf"):
                keep.add(_rel(p, job_dir))

    return keep

def _cleanup_job_dir(job_dir: Path, keep_relpaths: set[str]):
    """
    Delete everything in job_dir except keep_relpaths.
    Then remove empty directories.
    """
    job_dir = Path(job_dir).resolve()

    # 1) Delete files not in keep list
    for p in job_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = _rel(p, job_dir)
        if rel not in keep_relpaths:
            try:
                p.unlink()
            except Exception:
                pass

    # 2) Remove empty directories (bottom-up)
    for d in sorted([x for x in job_dir.rglob("*") if x.is_dir()], reverse=True):
        try:
            next(d.iterdir())
        except StopIteration:
            try:
                d.rmdir()
            except Exception:
                pass

@anvil.server.callable
def vm_get_default_overrides() -> dict:
    return json.loads(json.dumps(_DEFAULT_OVERRIDES))

# ---------- PDF → images ----------
def render_pdf_to_images(saved_pdf: Path, img_dir: Path, dpi: int = 400, status_cb=None) -> list[str]:
    """
    Run PageFilter first to keep only probable electrical/panel pages,
    then pass the (possibly filtered) PDF to PanelBoardSearch to produce crops.
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> rendering PDF → images: {saved_pdf} -> {img_dir} (dpi={dpi})")
    def _emit(step: str, progress: float | None = None, **extra):
        if callable(status_cb):
            try:
                payload = {}
                if step is not None:
                    payload["step"] = step
                if progress is not None:
                    payload["progress"] = progress
                payload.update(extra or {})
                status_cb(**payload)
            except Exception:
                pass

    # --- 1) Filter pages (OCR first, footprints only if undecided) ---
    _emit("finding_relevant_pages", 2.0)    
    try:
        pf = PageFilter(
            output_dir=str(img_dir.parent),   # keep filtered PDF alongside job folders
            dpi=400,                          # raster DPI used only for undecided pages
            longest_cap_px=9000,
            proc_scale=0.5,
            use_ocr=True,
            ocr_gpu=False,
            verbose=True,
            debug=False,                      # set True to write JSON log at output_dir/filter_debug/
            rect_w_fr_range=(0.20, 0.60),
            rect_h_fr_range=(0.20, 0.60),
            min_rectangularity=0.70,
            min_rect_count=2,
            # A bit more permissive area cut
            min_whitespace_area_fr=0.004,
            use_ghostscript_letter=True,       # turn GS letter step on/off
            letter_orientation="landscape",    # "portrait" or "landscape"
            gs_use_cropbox=True,               # True: fit what's inside CropBox; False: use MediaBox
            gs_compat="1.7"                    # PDF compatibility level            
        )
        kept_pages, dropped_pages, filtered_pdf, log_json = pf.readPdf(str(saved_pdf))
        print(f">>> PageFilter: kept={len(kept_pages)} dropped={len(dropped_pages)} filtered_pdf={filtered_pdf}")
        _emit(
            "finding_components",
            8.0,
            kept_pages=len(kept_pages),
            dropped_pages=len(dropped_pages),
        )
    except Exception as e:
        print(f">>> PageFilter error: {e}")
        kept_pages, filtered_pdf = [], None

    # Choose which PDF to feed into the finder:
    # - if filter kept at least one page, use filtered_pdf
    # - else fall back to the original PDF
    pdf_for_finder = filtered_pdf if (filtered_pdf and len(kept_pages) > 0) else str(saved_pdf)
    if pdf_for_finder == str(saved_pdf) and (filtered_pdf is not None) and len(kept_pages) == 0:
        print(">>> PageFilter kept 0 pages — falling back to original PDF")

    # --- 2) Run the panel finder (PanelSearchToolV18) on the chosen PDF ---
    local_finder = PanelBoardSearch(
        output_dir=str(img_dir),
        dpi=dpi,
        # All other knobs pulled from PANEL_FINDER_DEFAULTS so they match dev env
        render_dpi=PANEL_FINDER_DEFAULTS["render_dpi"],
        aa_level=PANEL_FINDER_DEFAULTS["aa_level"],
        render_colorspace=PANEL_FINDER_DEFAULTS["render_colorspace"],
        min_void_area_fr=PANEL_FINDER_DEFAULTS["min_void_area_fr"],
        min_void_w_px=PANEL_FINDER_DEFAULTS["min_void_w_px"],
        min_void_h_px=PANEL_FINDER_DEFAULTS["min_void_h_px"],
        max_void_area_fr=PANEL_FINDER_DEFAULTS["max_void_area_fr"],
        void_w_fr_range=PANEL_FINDER_DEFAULTS["void_w_fr_range"],
        void_h_fr_range=PANEL_FINDER_DEFAULTS["void_h_fr_range"],
        min_whitespace_area_fr=PANEL_FINDER_DEFAULTS["min_whitespace_area_fr"],
        margin_shave_px=PANEL_FINDER_DEFAULTS["margin_shave_px"],
        pad=PANEL_FINDER_DEFAULTS["pad"],
        verbose=PANEL_FINDER_DEFAULTS["verbose"],
    )

    try:
        crops = local_finder.readPdf(pdf_for_finder)
        _emit("removing_false_positives", 18.0, image_count=len(crops))
    except Exception as e:
        print(f">>> render error: {e}")
        raise

    print(f">>> rendered {len(crops)} image(s)")
    return crops

# ---------- Rules payload helper ----------
def _build_rules_payload(defaults: dict, items: list[dict]) -> dict:
    return {"defaults": defaults or {}, "items": items or []}

# ---------- Queue / Pool state ----------
_JOB_Q: "Queue[tuple[str,str]]" = Queue()
_INFLIGHT_BY_USER: dict[str, int] = {}
_Q_LOCK = threading.RLock()
_WORKERS: list[threading.Thread] = []
_STOP = threading.Event()
_SPECS_RUNNING: dict[str, bool] = {}
_SPECS_LOCK = threading.RLock()

# Per-slot worker pool state: each dequeue thread gets its own worker subprocess
_WORKER_READY_TIMEOUT = 120  # seconds to wait for worker model loading

class _WorkerSlot:
    """State for one worker subprocess slot."""
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None       # multiprocessing.Process
        self.job_q = None      # mp Queue: main → worker (job_id str or None for shutdown)
        self.done_q = None     # mp Queue: worker → main (status, job_id, error_msg)

_WORKER_SLOTS: list[_WorkerSlot] = [_WorkerSlot() for _ in range(MAX_WORKERS)]
_SPAWN_LOCK = threading.Lock()  # serializes env-var set/start/unset across slots

def _enqueue_job(job_id: str, owner_id: str):
    _JOB_Q.put((job_id, owner_id))

def _enter_inflight(owner_id: str) -> bool:
    with _Q_LOCK:
        c = _INFLIGHT_BY_USER.get(owner_id, 0)
        if c >= MAX_INFLIGHT_PER_USER:
            return False
        _INFLIGHT_BY_USER[owner_id] = c + 1
        return True

def _leave_inflight(owner_id: str):
    with _Q_LOCK:
        c = _INFLIGHT_BY_USER.get(owner_id, 0)
        _INFLIGHT_BY_USER[owner_id] = max(0, c - 1)

# ---------- Cancel helpers ----------
def _cancel_path(job_dir: Path) -> Path:
    return job_dir / ".cancel"

def _is_canceled(job_dir: Path) -> bool:
    return _cancel_path(job_dir).exists()

def _peek_owner_id(job_dir: Path) -> str:
    """Return the owner_id currently on disk for this job (empty if missing)."""
    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}
    return str(st.get("owner_id") or "").strip().lower()

def _is_queue_timed_out(job_dir: Path) -> tuple[bool, int | None]:
    """
    Returns (timed_out, age_ms).
    Only checks total queued age from noticed_ts_ms.
    """
    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}

    noticed_ts_ms = st.get("noticed_ts_ms")
    if not isinstance(noticed_ts_ms, int):
        return (False, None)

    age_ms = max(0, _epoch_ms() - noticed_ts_ms)
    queue_limit_ms = max(1, int(QUEUE_TIMEOUT_MIN)) * 60 * 1000
    return (age_ms > queue_limit_ms, age_ms)

# ---------- Shared helpers for component mapping ----------
def _to_int_or_none(x):
    try:
        return int(str(x).replace(",", "").strip())
    except Exception:
        return None

def _count_would_skip_breakers(breakers: list[dict], panel_limit: int | None) -> int:
    """
    Count how many individual breakers would be 'skipped/rejected':
      - amperage not numeric
      - amperage outside supported range [15..1200]
      - amperage exceeds panel/main limit (if provided)
    Uses the 'count' field to account for aggregated entries.
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

def _merge_component_from_btp(result_dict: dict, src_img: str) -> dict:
    """
    Map BreakerTablePipeline result → component schema expected by RulesEngine.
    Applies 4-strikes suppression rule.
    """
    stages = (result_dict or {}).get("results") or {}
    hdr    = stages.get("header")  or {}
    prs    = stages.get("parser")  or {}

    # Header fields
    name   = hdr.get("name") or ""
    h_attrs = hdr.get("attrs") or {}

    def _get_int_from_header(*keys):
        for k in keys:
            v = h_attrs.get(k)
            if v is not None:
                iv = _to_int_or_none(v)
                if iv is not None:
                    return iv
        return None

    def _get_text_from_header(*keys):
        for k in keys:
            v = h_attrs.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return None

    amperage   = _get_int_from_header("amperage", "main_amp", "mainAmperage")
    spaces_h   = _get_int_from_header("spaces")
    voltage    = _get_int_from_header("voltage")
    intRating  = _get_int_from_header("intRating", "interrupt_rating", "interruptRating", "kaic", "kaic_rating")
    main_amp   = _get_int_from_header("mainBreakerAmperage", "main_breaker_amperage", "main_breaker", "mainBreaker")
    trim_style = _get_text_from_header("trimStyle", "trim_style")
    enclosure  = _get_text_from_header("enclosure")
    if trim_style and not enclosure:
        enclosure = "Nema1"
    if str(enclosure or "").strip().upper() == "NEMA3R":
        trim_style = None
    hdr_brkrs  = list(h_attrs.get("detected_breakers") or [])

    # Table fields
    spaces_t   = _to_int_or_none((prs or {}).get("spaces"))
    tbl_brkrs  = list((prs or {}).get("detected_breakers") or [])

    spaces     = spaces_t if spaces_t is not None else spaces_h
    det_brkrs  = hdr_brkrs + tbl_brkrs

    # ===== 2-strikes pre-check =====
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
            "intRating": intRating,
            "mainBreakerAmperage": main_amp,
            "trimStyle": trim_style,
            "enclosure": enclosure,
            "detected_breakers": det_brkrs,
        },
    }

    if notes:
        comp["notes"] = notes
        comp["attrs"]["breaker_data_suppressed"] = True

    return comp

# ---------- Core job processing ----------
def _process_job(job_id: str, pipeline: "BreakerTablePipeline | None" = None):
    """
    Process a single job: render PDF, parse panels via BTP, run rules, write result.
    When ``pipeline`` is provided (persistent worker), it is reused across all
    images, avoiding per-image model reload.  When None, a fresh pipeline is
    created once for this job.
    """
    if pipeline is None:
        pipeline = BreakerTablePipeline(debug=True)
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)

    try:
        print(f">>> worker start: {job_id}")
        _log_run_fingerprint(f"job_start:{job_id}")
        prev = _json_read_or_none(sp["status"]) or {}
        print(f">>> DIAG worker prev keys: {sorted(list(prev.keys()))}")
        print(f">>> DIAG worker prev.owner_id: {str(prev.get('owner_id') or '').strip().lower()!r}")

        noticed_ts_ms = prev.get("noticed_ts_ms")
        _prev_carry = {
            k: v for k, v in prev.items()
            if k not in ("state", "ts", "noticed_ts_ms", "progress")
        }

        # Respect early cancel
        if _is_canceled(job_dir):
            _status_write(job_dir, "canceled", **_prev_carry, noticed_ts_ms=noticed_ts_ms, progress=0.0)
            _jobs_upsert(job_id, state="canceled", updated_at=_now_utc())
            print(f">>> worker canceled before start: {job_id}")
            return

        # First running write — owner_id should still be present in _prev_carry
        _status_write(job_dir, "running", **_prev_carry, noticed_ts_ms=noticed_ts_ms, progress=0.0)

        # DIAG: check again after writing
        peek = _peek_owner_id(job_dir)
        print(f">>> DIAG after running _status_write: job_id={job_id} owner_id_written={peek!r}")

        _jobs_upsert(job_id, state="running", updated_at=_now_utc())

        ui_overrides = prev.get("ui_overrides") or _DEFAULT_OVERRIDES
        def _render_status_cb(**kwargs):
            payload = {"noticed_ts_ms": noticed_ts_ms}
            payload.update(kwargs or {})
            _status_write(job_dir, "running", **payload)

        # Ensure images are present
        pdf_dir = job_dir / "uploaded_pdfs"
        img_dir = job_dir / "pdf_images"
        imgs = sorted(str(p) for p in img_dir.glob("*.png"))

        if not imgs:
            pdfs = sorted(pdf_dir.glob("*.pdf"))
            if not pdfs:
                raise RuntimeError("No PDF found to render.")

            # Early heartbeat: start relevant-page search
            _status_write(
                job_dir,
                "running",
                step="finding_relevant_pages",
                noticed_ts_ms=noticed_ts_ms,
                progress=2.0
            )

            # Render / page-filter / component-find with live phase updates
            imgs = []

            try:
                imgs = render_pdf_to_images(
                    pdfs[0],
                    img_dir,
                    status_cb=_render_status_cb
                )
            finally:
                # Final heartbeat after image generation finishes
                _status_write(
                    job_dir,
                    "running",
                    step="rendered",
                    image_count=len(imgs),
                    noticed_ts_ms=noticed_ts_ms,
                    progress=9.0
                )

        if not imgs:
            raise RuntimeError("PDF rendered but produced no crops/images.")

        if _is_canceled(job_dir):
            _status_write(job_dir, "canceled", step="rendered", noticed_ts_ms=noticed_ts_ms, progress=5.0)
            _jobs_upsert(job_id, state="canceled", updated_at=_now_utc())
            print(f">>> worker canceled after render: {job_id}")
            return

        print(f">>> parsing {len(imgs)} images")
        try:
            print(f">>> BreakerTable API Version: {API_VERSION}")
        except Exception:
            pass
        _status_write(job_dir, "running", step="parsing", image_count=len(imgs), noticed_ts_ms=noticed_ts_ms, progress=20.0)

        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        total = len(imgs)
        done = 0

        # ---------- SEQUENTIAL IMAGE PARSING (deterministic per job) ----------
        print(f">>> sequential parsing for {total} images (per-job determinism)")
        components = [None] * total  # preserve order
        for idx, img_path in enumerate(imgs):
            if _is_canceled(job_dir):
                pct = 20.0 + (done / max(1, total)) * 80.0
                _status_write(job_dir, "canceled", step="parsing", image_count=total, noticed_ts_ms=noticed_ts_ms, progress=pct)
                _jobs_upsert(job_id, state="canceled", updated_at=_now_utc())
                print(f">>> worker canceled mid-parse: {job_id}")
                return

            try:
                raw = pipeline.run(
                    img_path,
                    run_analyzer=True,
                    run_parser=True,
                    run_header=True,
                )
                if isinstance(raw, dict) and "_error" in raw:
                    raise RuntimeError(raw["_error"])

                comp = _merge_component_from_btp(raw or {}, img_path)
                comp = _normalize_component_for_none(comp or {})
                attrs = comp.get("attrs") or {}
                print(f">>> PANEL RESULTS for {img_path}:")
                print(f"    Name: {comp.get('name')}")
                print(f"    Amperage: {attrs.get('amperage')}")
                print(f"    Voltage: {attrs.get('voltage')}")
                print(f"    IntRating: {attrs.get('intRating')}")
                print(f"    MainBreakerAmperage: {attrs.get('mainBreakerAmperage')}")
                print(f"    TrimStyle: {attrs.get('trimStyle')}")
                print(f"    Enclosure: {attrs.get('enclosure')}")
                print(f"    Spaces (merged): {attrs.get('spaces')}")
                print(f"    Detected breakers: {len(attrs.get('detected_breakers') or [])}")
                components[idx] = comp
            except Exception as e:
                print(f">>> ERROR analyzing image {idx + 1}: {e}")
                print(f">>> Traceback: {traceback.format_exc()}")
                components[idx] = {
                    "type": "panelboard",
                    "name": f"image_{idx + 1}",
                    "_skipped": True,
                    "reason": f"Parse error: {e}",
                    "attrs": {},
                    "source": img_path
                }

            # progress after each image completes fully (A→P→H)
            done += 1
            pct = 20.0 + (done / max(1, total)) * 80.0
            _status_write(
                job_dir,
                "running",
                step="parsing",
                image_count=total,
                noticed_ts_ms=noticed_ts_ms,
                progress=pct
            )

        print(f">>> parse done - {len(components)} components processed")

        parse_done_ts_ms = _epoch_ms()
        cycle_time_ms = (parse_done_ts_ms - noticed_ts_ms) if isinstance(noticed_ts_ms, int) else None
        cycle_time_str = _fmt_cycle_time(cycle_time_ms if cycle_time_ms is not None else 0)

        _status_write(job_dir, "running", step="parsed", component_count=len(components), image_count=len(imgs),
                      noticed_ts_ms=noticed_ts_ms, parse_done_ts_ms=parse_done_ts_ms,
                      cycle_time_ms=cycle_time_ms, cycle_time_str=cycle_time_str, progress=92.0)

        # ---- RUN RULES with defaults ----
        _status_write(job_dir, "running", step="rules", image_count=len(imgs), noticed_ts_ms=noticed_ts_ms, progress=95.0)
        if _is_canceled(job_dir):
            _status_write(job_dir, "canceled", step="rules", noticed_ts_ms=noticed_ts_ms, progress=95.0)
            _jobs_upsert(job_id, state="canceled", updated_at=_now_utc())
            print(f">>> worker canceled before rules: {job_id}")
            return

        ui_defaults = prev.get("ui_overrides") or _DEFAULT_OVERRIDES

        panel_defaults = (ui_defaults.get("panelboards") or {})
        default_trim = str(panel_defaults.get("default_trim_style") or "").strip().upper()
        default_enclosure = str(panel_defaults.get("enclosure") or "").strip()

        for comp in components:
            if not isinstance(comp, dict):
                continue
            if str(comp.get("type") or "").strip().lower() != "panelboard":
                continue

            attrs = comp.get("attrs") or {}

            trim_style = str(attrs.get("trimStyle") or "").strip().upper()
            enclosure = str(attrs.get("enclosure") or "").strip().upper()

            if trim_style in ("", "NONE", "X", "-"):
                attrs["trimStyle"] = default_trim

            if enclosure in ("", "NONE", "X", "-"):
                attrs["enclosure"] = default_enclosure

            comp["attrs"] = attrs

        rules_payload = _build_rules_payload(ui_defaults, components)
        try:
            rules_result = RE2.process_job(rules_payload) or {}

        except Exception as re_err:
            rules_result = {"error": f"{type(re_err).__name__}: {re_err}"}
            print(f">>> rules engine error: {rules_result['error']}")

        # ---- RESULT JSON ----
        try:
            first_pdf = str(next((job_dir / "uploaded_pdfs").glob("*.pdf")))
        except StopIteration:
            first_pdf = ""

        result = {
            "ok": True,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "saved_pdf": first_pdf,
            "output_dir": "",
            "images": [],
            "image_count": len(imgs),
            "components": components,
            "rules_result": rules_result,
            "ui_overrides": prev.get("ui_overrides") or _DEFAULT_OVERRIDES,
            "cycle_time_ms": cycle_time_ms,
            "cycle_time_str": cycle_time_str,
            "noticed_ts_ms": noticed_ts_ms,
            "parse_done_ts_ms": parse_done_ts_ms,
        }

        _result_write(job_dir, result)
        _status_write(job_dir, "done", result_path=str(_status_paths(job_dir)["result"]), progress=100.0)
        _jobs_upsert(job_id, state="done", updated_at=_now_utc(), result_json=result)
        print(f">>> worker done: {job_id}")

        # ---- AUTO CLEANUP (keep only what UI uses) ----
        try:
            keep = _collect_keep_relpaths(job_dir, keep_pdf=True)
            _cleanup_job_dir(job_dir, keep)
            print(f">>> cleanup complete: kept {len(keep)} files")
        except Exception as ce:
            print(f">>> cleanup failed: {ce}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f">>> worker error [{job_id}]: {e}\n{tb}")
        _status_write(job_dir, "error", error=f"{type(e).__name__}: {e}")
        _jobs_upsert(job_id, state="error", updated_at=_now_utc(), error=f"{type(e).__name__}: {e}")

# ---------- Persistent worker subprocess ----------

def _persistent_worker_main(slot_idx, job_q, done_q):
    """Long-lived subprocess that loads GPU models once and processes jobs.

    Runs between-job hygiene (gc.collect + torch.cuda.empty_cache) after
    every job. Recycles after WORKER_RECYCLE_AFTER_JOBS jobs to reclaim
    RSS and eliminate any leaked daemon threads from OCR timeouts.

    NOTE: _set_runtime_determinism() is NOT called here because the
    ``spawn`` context re-imports the module from scratch, so the
    module-level call already runs in this subprocess. Calling it a second
    time would raise ``RuntimeError: cannot set number of interop threads``
    from PyTorch.
    """
    tag = f"slot-{slot_idx}"
    import gc
    import torch
    _log_run_fingerprint(f"persistent_worker_init:{tag}")

    import easyocr
    print(f">>> Worker [{tag}]: loading GPU EasyOCR reader ...")
    gpu_reader = easyocr.Reader(["en"], gpu=True)
    pipeline = BreakerTablePipeline(debug=True, reader=gpu_reader)
    pipeline._ensure_analyzer()
    pipeline._ensure_header_parser()
    print(f">>> Worker [{tag}]: models loaded | API_VERSION={API_VERSION}")

    try:
        free_vram, total_vram = torch.cuda.mem_get_info()
        print(
            f">>> Worker [{tag}]: VRAM after model load — "
            f"free={free_vram / 1024**3:.1f} GiB, "
            f"total={total_vram / 1024**3:.1f} GiB"
        )
    except Exception:
        pass

    done_q.put(("ready", None, None))

    jobs_processed = 0
    while True:
        try:
            msg = job_q.get()
        except Exception:
            continue
        if msg is None:
            break
        job_id = msg
        try:
            reset_name_deduper()
            _process_job(job_id, pipeline)
            done_q.put(("done", job_id, None))
        except Exception as e:
            tb = traceback.format_exc()
            print(f">>> Worker [{tag}] job error [{job_id}]: {e}\n{tb}")
            try:
                job_dir = BASE_JOBS_DIR / job_id
                _status_write(job_dir, "error", error=f"{type(e).__name__}: {e}")
            except Exception:
                pass
            done_q.put(("error", job_id, f"{type(e).__name__}: {e}"))
        finally:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        jobs_processed += 1
        if jobs_processed >= WORKER_RECYCLE_AFTER_JOBS:
            print(f">>> Worker [{tag}]: recycling after {jobs_processed} jobs")
            break

    print(f">>> Worker [{tag}]: shutdown")


# ---------- Worker lifecycle ----------

def _spawn_persistent_worker(idx: int):
    """Spawn (or respawn) worker subprocess for slot ``idx``. Blocks until ready."""
    slot = _WORKER_SLOTS[idx]
    tag = f"slot-{idx}"

    mp_ctx = get_context("spawn")
    slot.job_q = mp_ctx.Queue()
    slot.done_q = mp_ctx.Queue()

    with _SPAWN_LOCK:
        os.environ["_EDA_WORKER_SUBPROCESS"] = "1"
        slot.proc = mp_ctx.Process(
            target=_persistent_worker_main,
            args=(idx, slot.job_q, slot.done_q),
            daemon=True,
        )
        slot.proc.start()
        os.environ.pop("_EDA_WORKER_SUBPROCESS", None)

    print(f">>> Worker [{tag}] spawned (pid={slot.proc.pid}), waiting for ready ...")
    try:
        status, _, _ = slot.done_q.get(timeout=_WORKER_READY_TIMEOUT)
        if status != "ready":
            raise RuntimeError(f"unexpected worker init status: {status}")
    except Empty:
        raise RuntimeError(
            f"Worker [{tag}] did not become ready within {_WORKER_READY_TIMEOUT}s"
        )
    print(f">>> Worker [{tag}] ready")


def _kill_persistent_worker(idx: int):
    """Terminate and join the worker subprocess in slot ``idx``."""
    slot = _WORKER_SLOTS[idx]
    if slot.proc is None:
        return
    try:
        slot.proc.terminate()
    except Exception:
        pass
    slot.proc.join(timeout=WATCHDOG_KILL_GRACE_SEC)
    if slot.proc.is_alive():
        try:
            slot.proc.kill()
        except Exception:
            pass
        slot.proc.join(timeout=1)
    slot.proc = None


def _ensure_worker_alive(idx: int):
    """If the worker in slot ``idx`` exited (recycle or crash), respawn it."""
    slot = _WORKER_SLOTS[idx]
    tag = f"slot-{idx}"
    if slot.proc is not None and slot.proc.is_alive():
        return
    if slot.proc is not None:
        exitcode = getattr(slot.proc, "exitcode", None)
        if exitcode is not None and exitcode == 0:
            print(f">>> Worker [{tag}] exited (planned recycle) — respawning")
        else:
            print(f">>> Worker [{tag}] died (exitcode={exitcode}) — respawning")
    _kill_persistent_worker(idx)
    _spawn_persistent_worker(idx)


# ---------- Dequeue loop (per-slot worker pool) ----------

def _dequeue_loop(idx: int):
    """Each dequeue thread ``idx`` owns worker slot ``idx``.

    Pulls jobs from the shared ``_JOB_Q``, dispatches to its own slot's
    ``job_q``, and waits on its own slot's ``done_q``.  No cross-slot
    queue access, no race conditions.
    """
    threading.current_thread().name = f"pool-worker-{idx}"
    slot = _WORKER_SLOTS[idx]
    tag = f"slot-{idx}"

    while not _STOP.is_set():
        try:
            job_id, owner_id = _JOB_Q.get(timeout=0.5)
        except Empty:
            continue

        job_dir = BASE_JOBS_DIR / job_id

        if not job_dir.exists():
            _JOB_Q.task_done()
            continue

        sp = _status_paths(job_dir)
        st = _json_read_or_none(sp["status"]) or {}
        current_state = str(st.get("state") or "").lower()

        if current_state in ("done", "error", "canceled"):
            _JOB_Q.task_done()
            continue

        # Queue timeout check
        timed_out, age_ms = _is_queue_timed_out(job_dir)
        if current_state == "queued" and timed_out:
            msg = QUEUE_TIMEOUT_ERROR_MSG
            age_str = _fmt_cycle_time(age_ms if age_ms is not None else 0)
            _status_write(
                job_dir, "error", error=msg,
                queue_timeout=True, queue_timeout_min=QUEUE_TIMEOUT_MIN,
                queue_age_ms=age_ms, queue_age_str=age_str, progress=0.0,
            )
            _jobs_upsert(job_id, state="error", updated_at=_now_utc(), error=msg)
            print(f">>> queue timeout: {job_id} | age={age_str} | limit={QUEUE_TIMEOUT_MIN} min")
            _JOB_Q.task_done()
            continue

        if not _enter_inflight(owner_id):
            threading.Timer(0.05, lambda j=job_id, o=owner_id: _enqueue_job(j, o)).start()
            _JOB_Q.task_done()
            continue

        try:
            # Re-check queue timeout right before dispatching
            st = _json_read_or_none(sp["status"]) or {}
            current_state = str(st.get("state") or "").lower()
            timed_out, age_ms = _is_queue_timed_out(job_dir)
            if current_state == "queued" and timed_out:
                msg = QUEUE_TIMEOUT_ERROR_MSG
                age_str = _fmt_cycle_time(age_ms if age_ms is not None else 0)
                _status_write(
                    job_dir, "error", error=msg, step="queue_timeout",
                    queue_timeout=True, queue_timeout_min=QUEUE_TIMEOUT_MIN,
                    queue_age_ms=age_ms, queue_age_str=age_str, progress=0.0,
                )
                _jobs_upsert(job_id, state="error", updated_at=_now_utc(), error=msg)
                print(f">>> queue timeout (pre-start): {job_id} | age={age_str}")
                continue

            with slot.lock:
                _ensure_worker_alive(idx)

            slot.job_q.put(job_id)

            # Wait for completion with watchdog timeout
            timeout_sec = max(1, int(WATCHDOG_TIMEOUT_MIN) * 60)
            worker_ok = True
            try:
                status, done_job_id, err_msg = slot.done_q.get(timeout=timeout_sec)
            except Empty:
                worker_ok = False

            if not worker_ok or (slot.proc is not None and not slot.proc.is_alive()):
                try:
                    with open(_cancel_path(job_dir), "w") as f:
                        f.write("1")
                except Exception:
                    pass

                if not worker_ok:
                    msg = WATCHDOG_ERROR_MSG.format(mins=WATCHDOG_TIMEOUT_MIN)
                    print(f">>> watchdog timeout [{tag}]: {job_id}")
                else:
                    msg = "Worker process crashed during this job. Please try again."
                    print(f">>> worker crash [{tag}] during: {job_id}")

                _status_write(job_dir, "error", error=msg)
                try:
                    _cleanup_job_dir(job_dir, {"status.json"})
                except Exception:
                    pass
                _jobs_upsert(job_id, state="error", updated_at=_now_utc(), error=msg)

                with slot.lock:
                    _kill_persistent_worker(idx)
                    _spawn_persistent_worker(idx)

        finally:
            _leave_inflight(owner_id)
            _JOB_Q.task_done()

def _run_specs_analysis_job(job_id: str):
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}

    owner_email = str(st.get("owner_email") or "").strip().lower()
    saved_pdf = Path(st.get("file_path") or "").resolve()

    try:
        _status_write(
            job_dir,
            "running",
            created_at=st.get("created_at"),
            file_path=str(saved_pdf),
            job_dir_path=str(job_dir),
            owner_email=owner_email,
            owner_id=owner_email,
            node_id=NODE_ID,
            step="specs_analyzing",
            progress=15.0
        )

        module_path = REPO_ROOT / "Spec_Sheet_Analysis" / "Specs_AnalyzerV5.py"
        print(f">>> SPECS DEBUG selected module_path={module_path}")

        if not module_path.is_file():
            raise FileNotFoundError(f"Specs analyzer file not found: {module_path}")

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "argus_specs_analyzer_v5",
            str(module_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {module_path}")

        spec_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(spec_module)

        analyze_specs_pdf_for_ui = getattr(spec_module, "analyze_specs_pdf_for_ui", None)
        if analyze_specs_pdf_for_ui is None:
            raise AttributeError("Specs_AnalyzerV5.py does not define analyze_specs_pdf_for_ui")

        result = analyze_specs_pdf_for_ui(
            pdf_path=str(saved_pdf),
            job_dir=str(job_dir)
        ) or {}

        result = dict(result)
        result["job_id"] = job_id
        result["job_dir"] = str(job_dir)
        result["saved_pdf"] = str(saved_pdf)
        result["owner_email"] = owner_email
        result["owner_id"] = owner_email

        _result_write(job_dir, result)

        _status_write(
            job_dir,
            "done",
            created_at=st.get("created_at"),
            file_path=str(saved_pdf),
            job_dir_path=str(job_dir),
            owner_email=owner_email,
            owner_id=owner_email,
            node_id=NODE_ID,
            step="specs_complete",
            progress=100.0
        )

        print(f">>> specs analysis done: {job_id}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f">>> specs analysis error [{job_id}]: {e}\n{tb}")

        _status_write(
            job_dir,
            "error",
            created_at=st.get("created_at"),
            file_path=str(saved_pdf),
            job_dir_path=str(job_dir),
            owner_email=owner_email,
            owner_id=owner_email,
            node_id=NODE_ID,
            step="specs_error",
            error=f"{type(e).__name__}: {e}",
            traceback=tb,
            progress=100.0
        )

    finally:
        with _SPECS_LOCK:
            _SPECS_RUNNING.pop(job_id, None)

# ---------- Start worker pool (per-slot) ----------
if not _IS_WORKER_SUBPROCESS:
    try:
        for i in range(MAX_WORKERS):
            _spawn_persistent_worker(i)
        for i in range(MAX_WORKERS):
            t = threading.Thread(target=_dequeue_loop, args=(i,), daemon=True)
            t.start()
            _WORKERS.append(t)
        print(
            f">>> Worker pool started: {MAX_WORKERS} slots, "
            f"{MAX_WORKERS} dequeue threads, per-user cap={MAX_INFLIGHT_PER_USER}"
        )
    except Exception as e:
        print(f">>> Worker pool startup failed: {e}")
        print(traceback.format_exc())

# ---------- API: submit / status / list / cancel ----------
@anvil.server.callable
def vm_ping():
    print(f">>> vm_ping called | NODE_ID={NODE_ID}")
    return {"ok": True, "node_id": NODE_ID}

@anvil.server.callable
def vm_submit_for_detection(media, ui_overrides=None, job_note=None, owner_email=None):
    """
    Create job folder, save PDF, record 'noticed' time, render images, persist normalized overrides, enqueue worker.
    Ownership is always the lowercased email address.
    """
    if not owner_email or not str(owner_email).strip():
        raise RuntimeError("owner_email required")

    owner_email = str(owner_email).strip().lower()

    original_name = getattr(media, "name", "uploaded.pdf")
    job_dir = _make_job_dir(job_note, original_name)
    job_id = job_dir.name
    print(f">>> vm_submit_for_detection: job_dir={job_dir}, owner_email={owner_email!r}")

    # 1) Save the uploaded PDF
    pdf_dir = job_dir / "uploaded_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf = _save_media_to_disk(media, pdf_dir)
    print(f">>> saved_pdf={saved_pdf}")

    # 2) Normalize overrides
    normalized_overrides = _normalize_ui_overrides(ui_overrides if isinstance(ui_overrides, dict) else {})

    # 3) Record "noticed" time immediately
    noticed_ts_ms = _epoch_ms()

    # 4) Persist initial status BEFORE any heavy work
    _status_write(
        job_dir,
        "queued",
        created_at=_now_utc().isoformat(),
        file_path=str(saved_pdf),
        job_dir_path=str(job_dir),
        ui_overrides=normalized_overrides,
        job_note=(job_note or ""),
        image_count=0,                 # unknown yet
        step="received",               # explicit early step
        noticed_ts_ms=noticed_ts_ms,
        owner_email=owner_email,
        owner_id=owner_email,
        node_id=NODE_ID,
        canceled=False,
        progress=0.0
    )

    # 5) (Optional) upsert – store meta with zero images for now
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
        image_count=0,                 # unknown yet
        ui_overrides=normalized_overrides,
        owner_email=owner_email,
        owner_id=owner_email,
        node_id=NODE_ID,
    )

    # 6) Enqueue (worker will render/parse/rule)
    _enqueue_job(job_id, owner_email)

    print(f">>> Job queued: {job_id} (deferred render)")
    return {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "saved_pdf": str(saved_pdf),
        "output_dir": str(job_dir / "pdf_images"),
        "images": [],                  # not rendered yet
        "image_count": 0,              # not rendered yet
        "ui_overrides": normalized_overrides,
        "noticed_ts_ms": noticed_ts_ms,
        "owner_email": owner_email,
        "owner_id": owner_email,
        "node_id": NODE_ID,
        "state": "queued",
        "deferred_render": True
    }

@anvil.server.callable
def vm_upload_specs_pdf(media, owner_email=None, job_name=""):
    """
    Upload/save only. Do NOT analyze yet.
    """
    if not owner_email or not str(owner_email).strip():
        raise RuntimeError("owner_email required")

    owner_email = str(owner_email).strip().lower()

    safe_job_name = _slugify(job_name or Path(getattr(media, "name", "specs.pdf")).stem)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_id = f"specs_{safe_job_name}__{stamp}"

    job_dir = BASE_JOBS_DIR / job_id
    pdf_dir = job_dir / "uploaded_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    saved_pdf = _save_media_to_disk(media, pdf_dir)

    _status_write(
        job_dir,
        "uploaded",
        created_at=_now_utc().isoformat(),
        file_path=str(saved_pdf),
        job_dir_path=str(job_dir),
        owner_email=owner_email,
        owner_id=owner_email,
        node_id=NODE_ID,
        step="specs_uploaded",
        progress=5.0
    )

    return {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "saved_pdf": str(saved_pdf),
        "owner_email": owner_email,
        "owner_id": owner_email,
        "node_id": NODE_ID,
        "state": "uploaded"
    }


@anvil.server.callable
def vm_start_specs_analysis(job_id: str, owner_email: str):
    """
    Start analysis only after the PDF has already been uploaded/saved.
    """
    if not job_id or not owner_email:
        raise RuntimeError("job_id and owner_email required")

    owner_email = str(owner_email).strip().lower()
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}

    if not st:
        raise RuntimeError(f"Unknown specs job_id: {job_id}")

    job_owner = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
    if job_owner != owner_email:
        raise RuntimeError("Owner mismatch")

    with _SPECS_LOCK:
        if _SPECS_RUNNING.get(job_id):
            return {"ok": True, "job_id": job_id, "state": "running"}

        _SPECS_RUNNING[job_id] = True

    t = threading.Thread(target=_run_specs_analysis_job, args=(job_id,), daemon=True)
    t.start()

    return {"ok": True, "job_id": job_id, "state": "running"}


@anvil.server.callable
def vm_get_specs_status(job_id: str, owner_email: str) -> dict:
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)

    st = _json_read_or_none(sp["status"])
    if not st:
        return {
            "state": "error",
            "error": f"Unknown job_id {job_id}",
            "debug_job_dir": str(job_dir),
            "debug_status_path": str(sp["status"]),
            "debug_result_path": str(sp["result"]),
        }

    req_email = str(owner_email or "").strip().lower()
    job_email = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()

    if not req_email or not job_email or req_email != job_email:
        return {
            "state": "not_found",
            "debug_req_email": req_email,
            "debug_job_email": job_email,
            "debug_raw_state": st.get("state"),
            "debug_job_dir": str(job_dir),
        }

    state = (st.get("state") or "unknown").lower()

    if state == "done":
        res = _json_read_or_none(sp["result"]) or {}
        return {
            "state": "done",
            "result": res,
            "debug_raw_state": st.get("state"),
            "debug_job_dir": str(job_dir),
        }

    if state == "error":
        return {
            "state": "error",
            "error": st.get("error") or "Unknown error",
            "debug_raw_state": st.get("state"),
            "debug_job_dir": str(job_dir),
        }

    out = {
        "state": state,
        "debug_raw_state": st.get("state"),
        "debug_job_dir": str(job_dir),
    }
    for k in ("step", "progress"):
        if k in st:
            out[k] = st[k]
    return out

@anvil.server.callable
def vm_delete_specs_job(job_id: str, owner_email: str) -> bool:
    """
    Delete a specs-only temp job folder after the results modal is finished.
    Only allows deletion of specs_* jobs owned by the requesting user.
    """
    if not job_id or not owner_email:
        return False

    owner_email = str(owner_email).strip().lower()
    if not job_id.startswith("specs_"):
        return False

    job_dir = BASE_JOBS_DIR / job_id
    if not job_dir.exists() or not job_dir.is_dir():
        return False

    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}
    job_owner = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()

    if not job_owner or job_owner != owner_email:
        return False

    import shutil
    try:
        shutil.rmtree(job_dir, ignore_errors=False)
        print(f">>> deleted specs temp job folder: {job_dir}")
        return True
    except Exception as e:
        print(f">>> failed deleting specs temp job folder {job_dir}: {e}")
        return False

def _natural_key(p: Path):
    # Sort like page2 before page10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

@anvil.server.callable
def vm_list_magenta_overlay_images(job_id: str) -> list[str]:
    """
    Returns job-relative PNG paths for FULL-PAGE magenta overlays.
    Looks in common directories first, then falls back to any PNG containing 'magenta'
    (excluding pdf_images/review_overlays).
    """
    if not job_id:
        return []

    job_root = (BASE_JOBS_DIR / job_id).resolve()
    pdf_images = (job_root / "pdf_images")
    if not pdf_images.is_dir():
        return []

    # 1) Preferred / common directories
    candidate_dirs = [
        pdf_images / "magenta_overlays",
        pdf_images / "magenta_overlay",
        pdf_images / "page_overlays",
        pdf_images / "full_overlays",
        pdf_images / "overlays",
    ]

    found: list[Path] = []
    for d in candidate_dirs:
        if d.is_dir():
            found.extend(list(d.glob("*.png")))

    # 2) Fallback: search for filenames containing "magenta" anywhere under pdf_images,
    # but EXCLUDE review_overlays 
    if not found:
        for p in pdf_images.rglob("*.png"):
            if "review_overlays" in p.parts:
                continue
            if "magenta" in p.name.lower():
                found.append(p)

    # Deduplicate + sort
    uniq = {}
    for p in found:
        try:
            rp = p.relative_to(job_root)
        except Exception:
            continue
        uniq[str(rp).replace("\\", "/")] = p

    rel_paths = list(uniq.keys())
    rel_paths.sort(key=lambda s: _natural_key(Path(s)))
    return rel_paths

@anvil.server.callable
def vm_list_overlay_images(job_id: str) -> list[str]:
    if not job_id:
        return []
    job_root = (BASE_JOBS_DIR / job_id).resolve()
    overlay_dir = (job_root / "pdf_images" / "review_overlays")
    if not overlay_dir.is_dir():
        return []
    return [str(p.relative_to(job_root)) for p in sorted(overlay_dir.glob("*.png"))]

@anvil.server.callable
def vm_fetch_image(job_id: str, source_path: str):
    """
    Return an image as BlobMedia.
    Accepts either:
      - absolute paths inside this job folder, OR
      - job-relative paths like: 'pdf_images/review_overlays/foo.png'
    """
    if not job_id or not source_path:
        raise RuntimeError("job_id and source_path are required")

    job_root = (BASE_JOBS_DIR / job_id).resolve()

    raw = str(source_path).strip().replace("\\", "/")
    p_in = Path(raw)

    # If client sent a relative path, interpret it under the job folder.
    if not p_in.is_absolute():
        p = (job_root / p_in).resolve()
    else:
        p = p_in.resolve()

    # Security: ensure requested file is inside this job folder
    if not str(p).startswith(str(job_root)):
        raise RuntimeError(f"Invalid image path for this job: {raw}")

    if not p.is_file():
        raise RuntimeError(f"Image not found: {p}")

    ctype = "image/png" if p.suffix.lower() == ".png" else "application/octet-stream"
    return BlobMedia(ctype, p.read_bytes(), name=p.name)

@anvil.server.callable
def vm_set_queue_timeout(minutes: int) -> dict:
  """
  Set the queue timeout (minutes) at runtime.
  Persists only for this process lifetime.
  """
  global QUEUE_TIMEOUT_MIN
  try:
    m = int(minutes)
    if m < 1 or m > 120:
      raise ValueError("minutes must be between 1 and 120")
    QUEUE_TIMEOUT_MIN = m
    return {"ok": True, "queue_timeout_min": QUEUE_TIMEOUT_MIN}
  except Exception as e:
    return {"ok": False, "error": str(e), "queue_timeout_min": QUEUE_TIMEOUT_MIN}

@anvil.server.callable
def vm_get_queue_timeout() -> int:
  return int(QUEUE_TIMEOUT_MIN)

@anvil.server.callable
def vm_set_watchdog_timeout(minutes: int) -> dict:
  """
  Set the watchdog timeout (minutes) at runtime.
  Persists only for this process lifetime.
  """
  global WATCHDOG_TIMEOUT_MIN
  try:
    m = int(minutes)
    if m < 1 or m > 60:
      raise ValueError("minutes must be between 1 and 60")
    WATCHDOG_TIMEOUT_MIN = m
    return {"ok": True, "watchdog_min": WATCHDOG_TIMEOUT_MIN}
  except Exception as e:
    return {"ok": False, "error": str(e), "watchdog_min": WATCHDOG_TIMEOUT_MIN}

@anvil.server.callable
def vm_get_watchdog_timeout() -> int:
  return int(WATCHDOG_TIMEOUT_MIN)

def _get_queue_position(job_id: str, owner_email: str | None = None) -> tuple[int | None, int]:
    """
    Returns (queue_position, active_count)

    queue_position:
      0 -> currently running
      1 -> next in line
      2 -> one queued ahead, etc.
      None -> job missing / not active anymore

    active_count:
      count of jobs on this node that are still queued/running
    """
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)
    target = _json_read_or_none(sp["status"]) or {}
    if not target:
        return (None, 0)

    target_state = str(target.get("state") or "").lower()
    if target_state in ("done", "error", "canceled"):
        return (None, 0)

    target_noticed = target.get("noticed_ts_ms")
    if not isinstance(target_noticed, int):
        return (None, 0)

    target_node = str(target.get("node_id") or "").strip()
    if not target_node:
        target_node = NODE_ID

    rows = []
    for d in BASE_JOBS_DIR.iterdir():
        if not d.is_dir():
            continue

        st = _json_read_or_none(_status_paths(d)["status"]) or {}
        state = str(st.get("state") or "").lower()
        if state not in ("queued", "running"):
            continue

        node_id = str(st.get("node_id") or "").strip()
        if node_id and node_id != target_node:
            continue

        noticed = st.get("noticed_ts_ms")
        if not isinstance(noticed, int):
            continue

        rows.append({
            "job_id": d.name,
            "state": state,
            "noticed_ts_ms": noticed,
        })

    # Oldest first, then job_id for stable ordering
    rows.sort(key=lambda r: (r["noticed_ts_ms"], r["job_id"]))

    active_count = len(rows)

    for idx, row in enumerate(rows):
        if row["job_id"] == job_id:
            # idx is zero-based among active queued/running jobs
            return (idx, active_count)

    return (None, active_count)

@anvil.server.callable
def vm_get_job_status(job_id: str, owner_email: str) -> dict:
    """Status primarily from disk; returns result when done. Enforces ownership by email."""
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)

    st = _json_read_or_none(sp["status"])
    if not st:
        return {"state": "error", "error": f"Unknown job_id {job_id}"}

    req_email = str(owner_email or "").strip().lower()
    job_email = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()

    # Optional node hint (non-fatal)
    job_node = (st.get("node_id") or "").strip()
    node_hint = {"node_id": NODE_ID}
    if job_node and job_node != NODE_ID:
        node_hint.update({"job_node_id": job_node})

    if not req_email or not job_email:
        return {"state": "not_found", **node_hint}

    # ENFORCE OWNERSHIP
    if req_email != job_email:
        return {"state": "not_found", **node_hint}

    state = (st.get("state") or "unknown").lower()
    if st.get("canceled") is True and state != "done":
        return {"state": "canceled", **node_hint}

    if state == "done":
        res = _json_read_or_none(sp["result"]) or {}
        return {"state": "done", "result": res, **node_hint}
    if state == "error":
        out = {
            "state": "error",
            "error": st.get("error") or "Unknown error",
            **node_hint
        }
        for k in ("queue_timeout", "queue_timeout_min", "queue_age_ms", "queue_age_str"):
            if k in st:
                out[k] = st[k]
        return out

    out = {"state": state, **node_hint}
    for k in ("step", "image_count", "ui_overrides", "noticed_ts_ms",
            "cycle_time_str", "cycle_time_ms", "progress"):
        if k in st:
            out[k] = st[k]

    if state in ("queued", "running"):
        queue_position, active_count = _get_queue_position(job_id, req_email)
        if queue_position is not None:
            out["queue_position"] = int(queue_position)
        out["active_count"] = int(active_count)

    if "noticed_ts_ms" in st and isinstance(st["noticed_ts_ms"], int):
        elapsed_ms = max(0, _epoch_ms() - int(st["noticed_ts_ms"]))
        out["elapsed_ms"]  = elapsed_ms
        out["elapsed_str"] = _fmt_cycle_time(elapsed_ms)

    return out

@anvil.server.callable
def vm_list_jobs(owner_id: str, limit: int = 50) -> list[dict]:
    print(f">>> vm_list_jobs called | owner_id={owner_id!r} | NODE_ID={NODE_ID}")

    owner_id = str(owner_id or "").strip().lower()
    if not owner_id:
        print(">>> vm_list_jobs: empty owner_id")
        return []

    def _safe_iso(dt_s):
        if not isinstance(dt_s, str) or not dt_s.strip():
            return ""
        return dt_s.strip()

    rows = []
    try:
        for d in sorted(BASE_JOBS_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue

            st = _json_read_or_none(_status_paths(d)["status"]) or {}
            st_owner = str(st.get("owner_id") or st.get("owner_email") or "").strip().lower()
            if st_owner != owner_id:
                continue

            state = (st.get("state") or "unknown").lower()

            # TEMP: include error jobs while debugging
            if state == "error":
                continue

            meta = _parse_job_note(st.get("job_note") or "")
            job_name = (meta.get("job_name") or "").strip() or d.name
            submitted_at_utc = (meta.get("submitted_at_utc") or "").strip()

            row = {
                "job_id": d.name,
                "job_name": job_name,
                "submitted_at_utc": submitted_at_utc,
                "created_at": _safe_iso(st.get("created_at")),
                "state": (st.get("state") or "unknown"),
                "step": (st.get("step") or ""),
                "progress": float(st.get("progress", 0.0) or 0.0),
                "image_count": int(st.get("image_count", 0) or 0),
                "cycle_time_str": (st.get("cycle_time_str") or ""),
            }
            rows.append(row)

            if len(rows) >= int(limit):
                break

        print(f">>> vm_list_jobs returning {len(rows)} rows for {owner_id!r}")
        return rows

    except Exception as e:
        print(f">>> vm_list_jobs ERROR: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise

@anvil.server.callable
def vm_cancel_job(job_id: str, owner_id: str) -> bool:
    """Mark a job as canceled (queued or running), enforcing ownership."""
    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"])
    if not st or str(st.get("owner_id") or "").strip().lower() != str(owner_id or "").strip().lower():
        return False

    if st.get("state") in ("done", "error"):
        return False

    # mark cancellation file
    with open(_cancel_path(job_dir), "w") as f:
        f.write("1")

    # update status snapshot
    st["canceled"] = True
    st["state"] = "canceled" if st.get("state") == "queued" else st.get("state")
    _status_write(job_dir, st["state"], **{k: v for k, v in st.items() if k not in ("state", "ts")})
    return True

# ---------- MAIN ----------
if not _IS_WORKER_SUBPROCESS:
    print(">>> Uplink ready; waiting for calls")
    anvil.server.wait_forever()