# Anvil Uplink VM (disk only) + Rules Engine defaults + cycle-time
# -------------------------------
import os, re, json, sys, threading, traceback, uuid, time
import contextlib
import shutil
from multiprocessing import get_context
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime, timezone, timedelta
import anvil.server
import platform
import os as _os
from anvil import BlobMedia
import hashlib

# ---------- CONFIG ----------
# Resolve the repo root from this file's location so the server runs
# unchanged on both production hosts -- Paperspace
# (/home/paperspace/ElectricalDiagramAnalyzer) and the AWS dev box
# (/home/ubuntu/ElectricalDiagramAnalyzer) -- as well as on any future
# host with a different checkout path. uplink_server.py is one level
# below the repo root at AnvilUplinkCode/uplink_server.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Put jobs directly under the home directory
BASE_JOBS_DIR = Path.home() / "jobs"
BASE_JOBS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_JOB_RETENTION_HOURS = 24
EXCLUDED_SPECS_ARTIFACT_RETENTION_MINUTES = int(os.environ.get("EXCLUDED_SPECS_ARTIFACT_RETENTION_MINUTES", "10"))
EXCLUDED_JOB_MARKER_FILENAME = "ARGUS_EXCLUDED_FROM_IMPROVEMENT_REVIEW_README.txt"

STANDARD_JOB_RETENTION_DAYS = int(os.environ.get("STANDARD_JOB_RETENTION_DAYS", "90"))
STANDARD_JOB_HISTORY_LIMIT = int(os.environ.get("STANDARD_JOB_HISTORY_LIMIT", "30"))

EXCLUDED_JOB_MARKER_TEXT = """This job was excluded from Argus improvement review.

Raw PDF is deleted after processing.
Processed job artifacts are retained for up to 24 hours so the user can view, edit, regenerate, and download results.
After 24 hours, the full job folder should be automatically deleted.
Do not use this job for troubleshooting, QA review, style analysis, training, or product improvement unless the customer explicitly authorizes it.
"""

CLEANUP_SWEEP_INTERVAL_SEC = int(os.environ.get("CLEANUP_SWEEP_INTERVAL_SEC", "600"))  # 10 minutes

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

# Compatibility checks analyze candidates one at a time.
# Finder-level rejects do not count toward this limit.
COMPATIBILITY_MAX_PANEL_ATTEMPTS = int(
    os.environ.get("COMPATIBILITY_MAX_PANEL_ATTEMPTS", "3")
)

# Quick compatibility tests should never occupy a worker for
# as long as a full detection job.
COMPATIBILITY_TIMEOUT_SEC = int(
    os.environ.get(
        "COMPATIBILITY_TIMEOUT_SEC",
        "60",
    )
)

COMPATIBILITY_TIMEOUT_ERROR_MSG = (
    "The quick compatibility test took too long to process. "
    "Run the drawings as a normal job instead. The full run "
    "can handle larger and more complex or unusual drawing sets."
)

# Fallback cleanup for abandoned compatibility tests.
COMPATIBILITY_TEMP_RETENTION_MINUTES = int(
    os.environ.get(
        "COMPATIBILITY_TEMP_RETENTION_MINUTES",
        "20",
    )
)

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

# ---------- WATCHDOG / PROGRESS CONFIG ----------
# Hard cap: protects the worker pool from one monster job holding a worker forever.
WATCHDOG_TIMEOUT_MIN = int(os.environ.get("WATCHDOG_TIMEOUT_MIN", "20"))

# Soft cap: if there is no status movement AND no new image files for this long,
# assume the worker is wedged and requeue the job.
NO_SIGNAL_TIMEOUT_SEC = int(os.environ.get("NO_SIGNAL_TIMEOUT_SEC", "180"))

WATCHDOG_KILL_GRACE_SEC = int(os.environ.get("WATCHDOG_KILL_GRACE_SEC", "3"))

WATCHDOG_ERROR_MSG = (
  "This job took over {mins} minutes to process. "
  "Try trimming the document or splitting it into 2 documents and running them separately."
)

# ---------- QUEUE TIMEOUT CONFIG ----------
QUEUE_TIMEOUT_MIN = int(os.environ.get("QUEUE_TIMEOUT_MIN", "25"))
QUEUE_TIMEOUT_ERROR_MSG = (
  "This job waited in the processing queue too long due to current demand. "
  "Please try again in a few minutes."
)

# ---------- STALE / ORPHANED JOB RECOVERY CONFIG ----------
# Protects against the exact bad case:
# status.json says queued/running, but the in-memory queue/worker was lost
# because the uplink/VM process restarted or disconnected.

STALE_QUEUED_REQUEUE_SEC = int(os.environ.get("STALE_QUEUED_REQUEUE_SEC", "120"))

# On startup, a previous "running" job cannot still belong to this new uplink process.
# Give it a small grace window, then requeue it if the uploaded PDF still exists.
STARTUP_RUNNING_REQUEUE_SEC = int(os.environ.get("STARTUP_RUNNING_REQUEUE_SEC", "0"))

# Status polling fallback. Keep this longer than NO_SIGNAL_TIMEOUT_SEC because
# the dequeue loop should handle current-process no-signal recovery first.
STATUS_RUNNING_STALE_SEC = int(os.environ.get("STATUS_RUNNING_STALE_SEC", "300"))

RECOVERY_MAX_REQUEUE_ATTEMPTS = int(os.environ.get("RECOVERY_MAX_REQUEUE_ATTEMPTS", "3"))

INTERRUPTED_JOB_MSG = (
    "Processing was interrupted before completion. "
    "Please retry the job."
)

def _set_runtime_determinism():
    """Delegate to shared WorkerSetup module."""
    set_runtime_determinism()

def _log_run_fingerprint(tag: str = ""):
    """Log CUDA device names and cuDNN determinism settings for diagnostics."""
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
from VisualDetectionToolLibrary.PanelSearchToolV26 import PanelBoardSearch
from OcrLibrary.BreakerTableParserAPIv16 import BreakerTablePipeline, API_VERSION, reset_name_deduper
import RulesEngine.RulesEngine7 as RE2  # must expose process_job(payload)

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
    """Pre-load EasyOCR models by running BreakerTablePipeline on a dummy 32x32 image."""
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
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)

def _epoch_ms(dt=None) -> int:
    """Convert a datetime (default: now UTC) to epoch milliseconds."""
    dt = dt or datetime.now(timezone.utc)
    return int(dt.timestamp() * 1000)

def _fmt_cycle_time(ms: int) -> str:
    """Format milliseconds as HH:MM:SS:mmm for display in status payloads."""
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
    """Normalize a string to a filesystem-safe slug (alphanumeric, dots, hyphens, underscores)."""
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s or "untitled"

def _json_read_or_none(path: Path):
    """Load JSON from *path*; return None on any read/parse error."""
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
    """Parse an ISO-8601 datetime string into a YYYYMMDD_HHMMSS stamp for job directory names."""
    try:
        s2 = s.rstrip("Z")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _safe_folder_key(value: str, fallback: str = "ungrouped") -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or fallback


def _owner_folder_key(owner_email: str) -> str:
    # readable but safe
    s = str(owner_email or "").strip().lower().replace("@", "_at_")
    return _safe_folder_key(s, "user")


def _group_users_root(group_folder: str) -> Path:
    group_key = _safe_folder_key(group_folder, "ungrouped")
    return (BASE_JOBS_DIR / "groups" / group_key / "users").resolve()


def _user_jobs_root(owner_email: str, group_folder: str) -> Path:
    return (_group_users_root(group_folder) / _owner_folder_key(owner_email)).resolve()


def _assert_under_base(path: Path) -> Path:
    path = Path(path).resolve()
    base = BASE_JOBS_DIR.resolve()
    if base not in path.parents and path != base:
        raise RuntimeError("Path escaped jobs directory.")
    return path


def _resolve_job_dir_for_owner(job_id: str, owner_email: str, group_folder: str = None) -> Path | None:
    """
    Resolve a user-owned job path without trusting the client.
    Checks new grouped storage first, then legacy flat storage for old jobs.
    """
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")

    if not job_id or not owner_email:
        return None

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return None

    candidates = []

    # New grouped path
    candidates.append(_user_jobs_root(owner_email, group_folder) / job_id)

    # Legacy flat path fallback so current jobs do not disappear immediately.
    candidates.append(BASE_JOBS_DIR / job_id)

    for cand in candidates:
        try:
            cand = _assert_under_base(cand)
        except Exception:
            continue

        sp = _status_paths(cand)
        st = _json_read_or_none(sp["status"]) or {}
        if not isinstance(st, dict) or not st:
            continue

        job_owner = str(
            st.get("owner_email")
            or st.get("owner_id")
            or ""
        ).strip().lower()

        if job_owner != owner_email:
            continue

        # If status has group_folder, require it to match for grouped jobs.
        st_group = _safe_folder_key(st.get("group_folder") or group_folder, group_folder)
        if st_group and group_folder and st_group != group_folder:
            continue

        return cand

    return None


def _resolve_job_dir_any(job_id: str) -> Path | None:
    """
    Worker-only resolver.
    Finds a job by id under grouped storage or legacy flat storage.
    Job ids should be globally unique due to suffix in _make_job_dir().
    """
    job_id = str(job_id or "").strip()

    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        return None

    legacy = BASE_JOBS_DIR / job_id
    try:
        legacy = _assert_under_base(legacy)
        if (_status_paths(legacy)["status"]).exists():
            return legacy
    except Exception:
        pass

    groups = BASE_JOBS_DIR / "groups"
    try:
        if groups.is_dir():
            for p in groups.rglob(job_id):
                if not p.is_dir():
                    continue
                try:
                    p = _assert_under_base(p)
                except Exception:
                    continue
                if (_status_paths(p)["status"]).exists():
                    return p
    except Exception:
        pass

    return None


def _iter_job_dirs_for_owner(owner_email: str, group_folder: str = None):
    """
    Yield grouped jobs for this owner, plus legacy flat jobs for backwards compatibility.
    """
    owner_email = str(owner_email or "").strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")

    if not owner_email:
        return

    # New grouped jobs
    root = _user_jobs_root(owner_email, group_folder)
    try:
        root = _assert_under_base(root)
        if root.is_dir():
            for d in root.iterdir():
                if d.is_dir():
                    yield d
    except Exception:
        pass

    # Legacy flat jobs
    try:
        for d in BASE_JOBS_DIR.iterdir():
            if not d.is_dir():
                continue
            if d.name == "groups":
                continue

            st = _json_read_or_none(_status_paths(d)["status"]) or {}
            st_owner = str(st.get("owner_id") or st.get("owner_email") or "").strip().lower()
            if st_owner == owner_email:
                yield d
    except Exception:
        pass

def _make_job_dir(job_note: str, fallback_filename: str, owner_email: str = "", group_folder: str = "personal") -> Path:
    meta = _parse_job_note(job_note)
    job_name = _slugify(meta.get("job_name") or Path(fallback_filename).stem)
    stamp = _iso_to_stamp(meta.get("submitted_at_utc") or "")

    # Add a short suffix so job_id remains globally unique even inside grouped folders.
    suffix = uuid.uuid4().hex[:8]
    job_id = f"{job_name}__{stamp}__{suffix}"

    root = _user_jobs_root(owner_email, group_folder)
    root.mkdir(parents=True, exist_ok=True)

    job_dir = root / job_id
    job_dir = _assert_under_base(job_dir)

    (job_dir / "uploaded_pdfs").mkdir(parents=True, exist_ok=True)
    (job_dir / "pdf_images").mkdir(parents=True, exist_ok=True)
    return job_dir

def _make_compatibility_job_dir(
    owner_email: str,
    group_folder: str = "personal",
) -> Path:
    """
    Create a temporary compatibility-test job folder.
    """
    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    suffix = uuid.uuid4().hex[:8]

    job_id = (
        f"compatibility__{stamp}__{suffix}"
    )

    root = _user_jobs_root(
        owner_email,
        group_folder,
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    job_dir = _assert_under_base(
        root / job_id
    )

    (job_dir / "uploaded_pdfs").mkdir(
        parents=True,
        exist_ok=True,
    )

    (job_dir / "pdf_images").mkdir(
        parents=True,
        exist_ok=True,
    )

    return job_dir

def _specs_group_users_root(group_folder: str) -> Path:
    group_key = _safe_folder_key(group_folder, "ungrouped")
    return (BASE_JOBS_DIR / "specs" / "groups" / group_key / "users").resolve()


def _user_specs_root(owner_email: str, group_folder: str) -> Path:
    return (_specs_group_users_root(group_folder) / _owner_folder_key(owner_email)).resolve()


def _make_specs_job_dir(media, owner_email: str, group_folder: str = "personal", job_name: str = "") -> Path:
    safe_job_name = _slugify(job_name or Path(getattr(media, "name", "specs.pdf")).stem)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]

    job_id = f"specs_{safe_job_name}__{stamp}__{suffix}"

    root = _user_specs_root(owner_email, group_folder)
    root.mkdir(parents=True, exist_ok=True)

    job_dir = _assert_under_base(root / job_id)
    (job_dir / "uploaded_pdfs").mkdir(parents=True, exist_ok=True)

    return job_dir


def _resolve_specs_job_dir_for_owner(job_id: str, owner_email: str, group_folder: str = "personal") -> Path | None:
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")

    if not job_id or not owner_email:
        return None

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return None

    candidates = [
        _user_specs_root(owner_email, group_folder) / job_id,
        BASE_JOBS_DIR / job_id,  # legacy fallback
    ]

    for cand in candidates:
        try:
            cand = _assert_under_base(cand)
        except Exception:
            continue

        st = _json_read_or_none(_status_paths(cand)["status"]) or {}
        if not isinstance(st, dict) or not st:
            continue

        job_owner = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
        if job_owner != owner_email:
            continue

        return cand

    return None


def _resolve_specs_job_dir_any(job_id: str) -> Path | None:
    job_id = str(job_id or "").strip()

    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        return None

    legacy = BASE_JOBS_DIR / job_id
    try:
        legacy = _assert_under_base(legacy)
        if (_status_paths(legacy)["status"]).exists():
            return legacy
    except Exception:
        pass

    specs_root = BASE_JOBS_DIR / "specs"
    try:
        if specs_root.is_dir():
            for status_path in specs_root.rglob("status.json"):
                d = status_path.parent
                if d.name == job_id:
                    return _assert_under_base(d)
    except Exception:
        pass

    return None

def _save_media_to_disk(media, dest_dir: Path) -> Path:
    """Write an Anvil BlobMedia's bytes to *dest_dir* as a PDF file; return the saved path."""
    fname = _slugify(getattr(media, "name", None) or "uploaded.pdf")
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    dst = dest_dir / fname
    with open(dst, "wb") as f:
        f.write(media.get_bytes())
    return dst

def _normalize_component_for_none(obj):
    """Recursively normalize a component dict: convert None to 'NONE', numpy types to Python ints/floats, and numeric strings to numbers."""
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
    """Return a dict with 'status' and 'result' keys pointing to the respective JSON files in *dir_path*."""
    dir_path = Path(dir_path)
    return {"status": dir_path / "status.json", "result": dir_path / "result.json"}

def _status_write(dir_path: Path, state: str, **extras):
    """
    Write status.json while preserving existing fields unless explicitly overwritten.
    This prevents losing owner_id/owner_email on error/timeouts.

    Also writes heartbeat_ts so we can detect stale queued/running jobs and avoid
    the customer-facing "stuck forever" failure mode.
    """
    paths = _status_paths(dir_path)

    prev = {}
    try:
        prev = _json_read_or_none(paths["status"]) or {}
    except Exception:
        prev = {}

    now = datetime.now(timezone.utc).isoformat()

    # Merge: previous -> extras -> required fields
    payload = dict(prev)
    payload.update(extras or {})
    payload["state"] = state
    payload["ts"] = now
    payload["heartbeat_ts"] = now

    # Keep node_id refreshed on every write from the active uplink process.
    if NODE_ID:
        payload["node_id"] = NODE_ID

    with open(paths["status"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str, indent=2)

def _result_write(dir_path: Path, result: dict):
    """Write *result* dict to result.json in the job directory."""
    paths = _status_paths(dir_path)
    with open(paths["result"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, default=str, indent=2)

# ---------- Stale / orphaned job recovery helpers ----------

def _status_dt(value):
    """
    Best-effort parser for status timestamps.
    Returns timezone-aware UTC datetime or None.
    """
    s = str(value or "").strip()
    if not s:
        return None

    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _status_age_sec(st: dict) -> int:
    """
    Age in seconds since the last status heartbeat/write.
    If missing/bad, return a huge value so it is treated as stale.
    """
    if not isinstance(st, dict):
        return 999999

    dt = _status_dt(st.get("heartbeat_ts") or st.get("ts") or st.get("created_at"))
    if not dt:
        return 999999

    return max(0, int((_now_utc() - dt).total_seconds()))

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _job_png_activity(job_dir: Path) -> tuple[int, int]:
    """
    Returns:
      (png_count, latest_png_mtime_ns)

    Any new/modified PNG under pdf_images counts as real activity.
    This catches long 'finding components' stages where PanelBoardSearch is
    slowly writing crops but has not returned yet.
    """
    count = 0
    latest = 0

    try:
        img_dir = Path(job_dir) / "pdf_images"
        if not img_dir.is_dir():
            return 0, 0

        for p in img_dir.rglob("*.png"):
            try:
                if not p.is_file():
                    continue
                count += 1
                latest = max(latest, int(p.stat().st_mtime_ns))
            except Exception:
                pass

    except Exception:
        pass

    return count, latest


def _job_activity_signature(job_dir: Path) -> tuple:
    """
    Meaningful activity signature.

    Deliberately excludes heartbeat_ts/ts so our own heartbeat writes do not
    fake progress. This changes only when real status fields or artifacts change.
    """
    st = _json_read_or_none(_status_paths(job_dir)["status"]) or {}
    png_count, latest_png_mtime_ns = _job_png_activity(job_dir)

    return (
        str(st.get("state") or ""),
        str(st.get("step") or ""),
        str(st.get("progress") or ""),
        str(st.get("image_count") or ""),
        str(st.get("component_count") or ""),
        str(st.get("kept_pages") or ""),
        str(st.get("dropped_pages") or ""),
        str(st.get("parse_done_ts_ms") or ""),
        png_count,
        latest_png_mtime_ns,
    )


def _write_live_finding_components_heartbeat(job_dir: Path, owner_id: str, png_count: int):
    """
    While PanelBoardSearch is finding components, it may write image files before
    returning to Python. This turns those files into visible status movement.
    """
    if png_count <= 0:
        return

    st = _json_read_or_none(_status_paths(job_dir)["status"]) or {}
    state = str(st.get("state") or "").strip().lower()

    if state != "running":
        return

    step = str(st.get("step") or "").strip().lower()

    # Only interfere during the image/component-finding stage.
    if step not in (
        "finding_components",
        "finding_panels",
        "detecting_panels",
        "worker_dispatch",
        "finding_relevant_pages",
    ):
        return

    current_progress = _safe_float(st.get("progress"), 8.0)

    # Keep this phase capped before removing_false_positives/rendered/parsing.
    live_progress = max(current_progress, min(18.0, 8.0 + (png_count * 0.20)))

    _status_write(
        job_dir,
        "running",
        step="finding_components",
        progress=live_progress,
        image_count=png_count,
        live_image_count=png_count,
        owner_email=owner_id,
        owner_id=owner_id,
    )

def _job_has_uploaded_pdf(job_dir: Path, st: dict | None = None) -> bool:
    """
    True if the original PDF is still available, so the job can be safely re-run.
    """
    st = st or {}

    # Prefer status.file_path if present.
    try:
        fp = str(st.get("file_path") or "").strip()
        if fp and Path(fp).exists() and Path(fp).is_file():
            return True
    except Exception:
        pass

    # Fallback to uploaded_pdfs folder.
    try:
        up = Path(job_dir) / "uploaded_pdfs"
        if up.is_dir():
            for p in up.glob("*.pdf"):
                if p.is_file():
                    return True
    except Exception:
        pass

    return False

def _iter_detection_job_dirs_all():
    """
    Yield drawing/detection job folders only.
    Handles:
      - legacy flat jobs: ~/jobs/<job_id>
      - grouped jobs: ~/jobs/groups/<group>/users/<owner>/<job_id>

    Skips specs jobs.
    """
    seen = set()

    # Legacy flat jobs
    try:
        for d in BASE_JOBS_DIR.iterdir():
            if not d.is_dir():
                continue
            if d.name in ("groups", "specs"):
                continue
            if d.name.startswith("specs_"):
                continue

            sp = _status_paths(d)
            if sp["status"].exists():
                key = str(d.resolve())
                if key not in seen:
                    seen.add(key)
                    yield d
    except Exception:
        pass

    # Grouped detection jobs
    try:
        groups_root = BASE_JOBS_DIR / "groups"
        if groups_root.is_dir():
            for status_path in groups_root.rglob("status.json"):
                job_dir = status_path.parent

                # Defensive skip if anything specs-like ever lands under groups.
                parts_lower = {str(x).lower() for x in job_dir.parts}
                if "specs" in parts_lower or job_dir.name.startswith("specs_"):
                    continue

                key = str(job_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    yield job_dir
    except Exception:
        pass

def _mark_job_interrupted(job_dir: Path, st: dict, reason: str = ""):
    """
    Mark an unrecoverable stale job as error so the UI/customer is not stuck forever.
    """
    owner_email = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
    group_folder = _safe_folder_key(st.get("group_folder") or "personal", "personal")

    _status_write(
        job_dir,
        "error",
        step="interrupted",
        progress=float(st.get("progress") or 0.0),
        error=INTERRUPTED_JOB_MSG,
        interrupted=True,
        interrupted_reason=reason or "stale_job",
        owner_email=owner_email,
        owner_id=owner_email,
        group_folder=group_folder,
        previous_state=st.get("state"),
    )

def _requeue_existing_job(job_dir: Path, st: dict, reason: str) -> bool:
    """
    Requeue an existing disk job into the in-memory queue.

    Returns True if requeued.
    Returns False if marked as error or skipped.
    """
    try:
        job_id = Path(job_dir).name
        owner_email = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
        group_folder = _safe_folder_key(st.get("group_folder") or "personal", "personal")

        if not job_id or not owner_email:
            _mark_job_interrupted(job_dir, st, reason=f"{reason}: missing_owner_or_job_id")
            return False

        if _is_canceled(job_dir) or bool(st.get("canceled")):
            return False

        if not _job_has_uploaded_pdf(job_dir, st):
            _mark_job_interrupted(job_dir, st, reason=f"{reason}: missing_uploaded_pdf")
            return False

        attempts = int(st.get("recovery_requeue_attempts") or 0)
        if attempts >= RECOVERY_MAX_REQUEUE_ATTEMPTS:
            _mark_job_interrupted(job_dir, st, reason=f"{reason}: too_many_requeue_attempts")
            return False

        _status_write(
            job_dir,
            "queued",
            step="requeued_after_interruption",
            progress=0.0,
            owner_email=owner_email,
            owner_id=owner_email,
            group_folder=group_folder,
            recovered_after_interruption=True,
            recovery_reason=reason,
            recovery_requeue_attempts=attempts + 1,
            previous_state=st.get("state"),
            queue_reentered_at_utc=_now_utc().isoformat(),
        )

        _enqueue_job(job_id, owner_email)

        print(
            f">>> requeued stale/orphaned job: {job_id} | "
            f"owner={owner_email} | reason={reason} | attempt={attempts + 1}"
        )

        return True

    except Exception as e:
        print(f">>> failed to requeue stale job {job_dir}: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return False

def _recover_orphaned_detection_jobs_on_startup():
    """
    Called once when the uplink process starts.

    Fixes the dangerous case:
      - status.json survived
      - _JOB_Q did not survive
      - customer would otherwise see queued/running forever
    """
    recovered = 0
    marked_error = 0
    skipped = 0

    try:
        for job_dir in list(_iter_detection_job_dirs_all()):
            sp = _status_paths(job_dir)
            st = _json_read_or_none(sp["status"]) or {}

            if not isinstance(st, dict) or not st:
                skipped += 1
                continue

            job_type = str(
                st.get("job_type") or ""
            ).strip().lower()

            if (
                job_type == "compatibility"
                or job_dir.name.startswith(
                    "compatibility__"
                )
            ):
                shutil.rmtree(
                    job_dir,
                    ignore_errors=True,
                )

                print(
                    ">>> deleted abandoned compatibility "
                    f"job during startup: {job_dir}"
                )

                skipped += 1
                continue

            state = str(st.get("state") or "").strip().lower()

            if state in ("done", "error", "canceled", "cancelled"):
                skipped += 1
                continue

            if bool(st.get("canceled")) or _is_canceled(job_dir):
                skipped += 1
                continue

            if state not in ("queued", "running", "unknown"):
                skipped += 1
                continue

            age = _status_age_sec(st)

            # Queued jobs are always safe to requeue on startup because the
            # in-memory queue was lost during restart.
            if state == "queued":
                if _requeue_existing_job(job_dir, st, "startup_recovery_queued"):
                    recovered += 1
                else:
                    marked_error += 1
                continue

            # Running/unknown jobs from a previous uplink process are orphaned after restart.
            # The old worker/queue died, even if the heartbeat is only a few seconds old.
            if state in ("running", "unknown"):
                old_node = str(st.get("node_id") or "").strip()
                new_node = str(NODE_ID or "").strip()

                old_process = bool(old_node and new_node and old_node != new_node)

                if old_process or age >= STARTUP_RUNNING_REQUEUE_SEC:
                    if _requeue_existing_job(job_dir, st, f"startup_recovery_{state}_old_process"):
                        recovered += 1
                    else:
                        marked_error += 1
                    continue

            skipped += 1

        print(
            f">>> startup orphan recovery complete | "
            f"recovered={recovered} marked_error={marked_error} skipped={skipped}"
        )

    except Exception as e:
        print(f">>> startup orphan recovery failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())

def _repair_stale_active_job_from_status_poll(job_dir: Path, st: dict) -> dict:
    """
    Called from vm_get_job_status and _active_job_for_owner.

    For queued jobs:
      - if stale, requeue into _JOB_Q

    For running/unknown jobs:
      - do NOT quickly requeue while the process is alive, because some legit
        jobs can run for minutes.
      - if stale beyond watchdog+grace, mark error so the customer is not stuck forever.
    """
    if not isinstance(st, dict) or not st:
        return st or {}

    state = str(st.get("state") or "").strip().lower()

    if state not in ("queued", "running", "unknown"):
        return st

    if bool(st.get("canceled")) or _is_canceled(job_dir):
        return st

    age = _status_age_sec(st)

    if state == "queued" and age >= STALE_QUEUED_REQUEUE_SEC:
        _requeue_existing_job(job_dir, st, "status_poll_stale_queued")
        return _json_read_or_none(_status_paths(job_dir)["status"]) or st

    if state in ("running", "unknown"):
        old_node = str(st.get("node_id") or "").strip()
        new_node = str(NODE_ID or "").strip()
        old_process = bool(old_node and new_node and old_node != new_node)

        # If this status belongs to a previous uplink process, requeue it now.
        # This catches cases where startup recovery missed it.
        if old_process:
            _requeue_existing_job(job_dir, st, f"status_poll_{state}_old_process")
            return _json_read_or_none(_status_paths(job_dir)["status"]) or st

        # If it belongs to the current process but has gone stale, mark interrupted.
        # Don't requeue current-process running jobs automatically or you can duplicate work.
        if age >= STATUS_RUNNING_STALE_SEC:
            _mark_job_interrupted(job_dir, st, reason=f"status_poll_stale_{state}")
            return _json_read_or_none(_status_paths(job_dir)["status"]) or st

    return st

def _utc_iso_z(dt=None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _excluded_delete_after_utc() -> str:
    return _utc_iso_z(datetime.now(timezone.utc) + timedelta(hours=EXCLUDED_JOB_RETENTION_HOURS))

def _excluded_specs_delete_after_utc() -> str:
    return _utc_iso_z(
        datetime.now(timezone.utc) + timedelta(minutes=EXCLUDED_SPECS_ARTIFACT_RETENTION_MINUTES)
    )

def _write_excluded_job_marker(job_dir: Path):
    try:
        marker = Path(job_dir) / EXCLUDED_JOB_MARKER_FILENAME
        with open(marker, "w", encoding="utf-8") as f:
            f.write(EXCLUDED_JOB_MARKER_TEXT)
    except Exception as e:
        print(f">>> excluded marker write failed: {e}")


def _is_excluded_job_expired(status: dict) -> bool:
    if not isinstance(status, dict):
        return False

    if not bool(status.get("exclude_from_improvement")):
        return False

    # Never delete a job while it may still be processing.
    state = str(
        status.get("state") or ""
    ).strip().lower()

    if state in {
        "uploaded",
        "queued",
        "running",
        "unknown",
    }:
        return False

    delete_after = str(
        status.get("delete_after_utc") or ""
    ).strip()

    if not delete_after:
        return False

    try:
        dt = datetime.fromisoformat(delete_after.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= dt.astimezone(timezone.utc)
    except Exception:
        return False

def _parse_utc_dt(value):
    """
    Best-effort UTC datetime parser.
    Returns timezone-aware UTC datetime or None.
    """
    s = str(value or "").strip()
    if not s:
        return None

    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _job_created_dt(job_dir: Path, status: dict):
    """
    Determine a job's created/submitted timestamp.
    Prefer status.created_at, then job_note submitted_at_utc, then folder mtime.
    """
    status = status or {}

    dt = _parse_utc_dt(status.get("created_at"))
    if dt:
        return dt

    try:
        meta = _parse_job_note(status.get("job_note") or "")
        dt = _parse_utc_dt(meta.get("submitted_at_utc"))
        if dt:
            return dt
    except Exception:
        pass

    try:
        return datetime.fromtimestamp(Path(job_dir).stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _is_active_job_state(status: dict) -> bool:
    state = str((status or {}).get("state") or "").strip().lower()
    return state in ("queued", "running", "unknown")


def _is_standard_retention_job(status: dict) -> bool:
    """
    Standard retention applies only to non-excluded jobs.
    Excluded jobs follow the shorter 24-hour policy.
    """
    if not isinstance(status, dict):
        return False

    if bool(status.get("exclude_from_improvement")):
        return False

    retention_mode = str(status.get("retention_mode") or "standard").strip().lower()
    if retention_mode == "excluded_24h":
        return False

    return True

def _cleanup_expired_excluded_jobs():
    """
    Deletes full job folders for excluded jobs once their 24-hour retention expires.
    Handles:
      - legacy flat jobs
      - grouped drawing jobs
      - grouped specs jobs
    """
    try:
        if not BASE_JOBS_DIR.is_dir():
            return

        candidates = []

        try:
            for job_dir in BASE_JOBS_DIR.iterdir():
                if not job_dir.is_dir():
                    continue
                if job_dir.name in ("groups", "specs"):
                    continue
                candidates.append(job_dir)
        except Exception:
            pass

        for root_name in ("groups", "specs"):
            root = BASE_JOBS_DIR / root_name
            try:
                if root.is_dir():
                    for status_path in root.rglob("status.json"):
                        candidates.append(status_path.parent)
            except Exception:
                pass

        seen = set()
        for job_dir in candidates:
            try:
                job_dir = _assert_under_base(job_dir)
            except Exception:
                continue

            key = str(job_dir)
            if key in seen:
                continue
            seen.add(key)

            st = _json_read_or_none(_status_paths(job_dir)["status"]) or {}
            if not _is_excluded_job_expired(st):
                continue

            try:
                shutil.rmtree(job_dir, ignore_errors=True)
                print(f">>> deleted expired excluded job: {job_dir}")
            except Exception as e:
                print(f">>> failed deleting expired excluded job {job_dir}: {e}")

    except Exception as e:
        print(f">>> excluded cleanup sweep failed: {e}")

def _cleanup_standard_retention_for_owner(owner_email: str, group_folder: str = "personal", reserve_slots: int = 0):
    """
    Enforce standard/non-excluded retention for one owner:
      1. Delete standard jobs older than STANDARD_JOB_RETENTION_DAYS.
      2. Keep at most STANDARD_JOB_HISTORY_LIMIT standard jobs per user.
         If reserve_slots=1 before a new upload, make room for the new job.

    Does not delete queued/running/unknown jobs.
    Does not touch excluded jobs.
    Does not touch specs jobs.
    """
    owner_email = str(owner_email or "").strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")

    if not owner_email:
        return

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=STANDARD_JOB_RETENTION_DAYS)
        max_existing = max(0, int(STANDARD_JOB_HISTORY_LIMIT) - int(reserve_slots or 0))

        standard_jobs = []

        for job_dir in list(_iter_job_dirs_for_owner(owner_email, group_folder)):
            try:
                job_dir = _assert_under_base(job_dir)
            except Exception:
                continue

            if not job_dir.is_dir():
                continue

            if job_dir.name.startswith("specs_"):
                continue

            st = _json_read_or_none(_status_paths(job_dir)["status"]) or {}
            if not isinstance(st, dict) or not st:
                continue

            st_owner = str(st.get("owner_id") or st.get("owner_email") or "").strip().lower()
            if st_owner != owner_email:
                continue

            if _is_active_job_state(st):
                continue

            if not _is_standard_retention_job(st):
                continue

            created_dt = _job_created_dt(job_dir, st)
            if created_dt is None:
                try:
                    created_dt = datetime.fromtimestamp(job_dir.stat().st_mtime, tz=timezone.utc)
                except Exception:
                    continue

            # 1) Age-based deletion
            if created_dt < cutoff:
                try:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    print(
                        f">>> deleted standard job older than {STANDARD_JOB_RETENTION_DAYS} days: "
                        f"{job_dir}"
                    )
                except Exception as e:
                    print(f">>> failed deleting old standard job {job_dir}: {e}")
                continue

            standard_jobs.append({
                "job_dir": job_dir,
                "created_dt": created_dt,
            })

        # 2) Count-based deletion: newest kept, oldest deleted
        standard_jobs.sort(key=lambda x: x["created_dt"], reverse=True)

        if len(standard_jobs) > max_existing:
            to_delete = standard_jobs[max_existing:]

            for item in to_delete:
                job_dir = item["job_dir"]
                try:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    print(
                        f">>> deleted standard job over {STANDARD_JOB_HISTORY_LIMIT}-job limit: "
                        f"{job_dir}"
                    )
                except Exception as e:
                    print(f">>> failed deleting over-limit standard job {job_dir}: {e}")

    except Exception as e:
        print(f">>> standard retention cleanup failed for {owner_email!r}: {e}")


def _cleanup_standard_retention_all_users():
    """
    Background sweep for standard/non-excluded retention.
    Groups jobs by owner + group_folder, then applies the 90-day / 30-job policy.
    """
    try:
        if not BASE_JOBS_DIR.is_dir():
            return

        pairs = set()

        # Legacy flat jobs
        try:
            for job_dir in BASE_JOBS_DIR.iterdir():
                if not job_dir.is_dir():
                    continue
                if job_dir.name == "groups":
                    continue
                if job_dir.name.startswith("specs_"):
                    continue

                st = _json_read_or_none(_status_paths(job_dir)["status"]) or {}
                owner = str(st.get("owner_id") or st.get("owner_email") or "").strip().lower()
                group = _safe_folder_key(st.get("group_folder") or "personal", "personal")

                if owner:
                    pairs.add((owner, group))
        except Exception:
            pass

        # Grouped jobs
        groups_root = BASE_JOBS_DIR / "groups"
        try:
            if groups_root.is_dir():
                for status_path in groups_root.rglob("status.json"):
                    try:
                        job_dir = status_path.parent
                        if job_dir.name.startswith("specs_"):
                            continue

                        st = _json_read_or_none(status_path) or {}
                        owner = str(st.get("owner_id") or st.get("owner_email") or "").strip().lower()
                        group = _safe_folder_key(st.get("group_folder") or "personal", "personal")

                        if owner:
                            pairs.add((owner, group))
                    except Exception:
                        pass
        except Exception:
            pass

        for owner, group in sorted(pairs):
            _cleanup_standard_retention_for_owner(owner, group, reserve_slots=0)

    except Exception as e:
        print(f">>> standard retention global sweep failed: {e}")


def _cleanup_all_retention_policies():
    """
    One retention entry point:
      - excluded jobs: 24-hour full-folder cleanup
      - standard jobs: 90-day / 30-job-per-user cleanup
    """
    _cleanup_expired_excluded_jobs()
    _cleanup_standard_retention_all_users()

def _excluded_cleanup_loop():
    """
    Lightweight background maintenance loop.
    Runs only in the main uplink process, not worker subprocesses.

    Enforces:
      - excluded jobs: 24-hour cleanup
      - standard jobs: 90-day retention / 30-job per-user history limit
    """
    threading.current_thread().name = "job-retention-cleanup"

    while not _STOP.is_set():
        try:
            _cleanup_all_retention_policies()
        except Exception as e:
            print(f">>> retention cleanup loop error: {e}")

        # Wait is better than time.sleep because it can exit cleanly if _STOP is set.
        _STOP.wait(CLEANUP_SWEEP_INTERVAL_SEC)

# ----- Data Tables helpers (disabled here; leave no-ops) -----
def _jobs_upsert(job_id: str, **fields):
    """No-op placeholder for a Data Tables upsert (disabled in disk-only mode)."""
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
    """Recursively merge *src* into *dst*, returning a new dict (nested dicts are merged, scalars overwritten)."""
    out = dict(dst)
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _coerce_types(overrides: dict) -> dict:
    """Recursively coerce string values in UI overrides: 'true'/'false' to bool, digit strings to int."""
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
    """Merge coerced user overrides on top of _DEFAULT_OVERRIDES, returning the combined config."""
    return _deep_merge(_DEFAULT_OVERRIDES, _coerce_types(overrides or {}))

# Delete unused images and folders for storage
def _rel(p: Path, root: Path) -> str:
    """Return *p* relative to *root* with forward slashes (for portable JSON paths)."""
    return str(p.relative_to(root)).replace("\\", "/")

def _collect_keep_relpaths(job_dir: Path, keep_pdf: bool = True) -> set[str]:
    """
    Return a set of job-relative file paths to keep.
    Everything else in the job folder will be deleted.
    """
    keep: set[str] = set()

    # Always keep status/result/edit log/excluded marker
    keep.add("status.json")
    keep.add("result.json")
    keep.add("edits.json")
    keep.add(EXCLUDED_JOB_MARKER_FILENAME)

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

def _cleanup_excluded_specs_job_dir(job_dir: Path):
    """
    For excluded specs jobs:
      - delete raw uploaded PDFs
      - keep result.json, status.json, marker, and generated images/artifacts
        so the user can view results for the 24-hour window
    """
    job_dir = Path(job_dir).resolve()

    uploaded = job_dir / "uploaded_pdfs"
    if uploaded.is_dir():
        for p in uploaded.rglob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

        for d in sorted([x for x in uploaded.rglob("*") if x.is_dir()], reverse=True):
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                except Exception:
                    pass

        try:
            uploaded.rmdir()
        except Exception:
            pass

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
    """RPC callable: return a deep copy of the default UI overrides for panelboards/transformers/disconnects."""
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
    """Assemble the payload dict expected by RulesEngine4.process_job() from UI defaults and component items."""
    return {"defaults": defaults or {}, "items": items or []}

def _edit_log_path(job_dir: Path) -> Path:
    return Path(job_dir) / "edits.json"


def _deep_copy_jsonable(obj):
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return obj

def _is_problem_component_for_rules(comp: dict) -> bool:
    if not isinstance(comp, dict):
        return False

    if bool(comp.get("_skipped")):
        return True

    if str(comp.get("panelStatus") or "").strip():
        return True

    return False


def _rules_safe_components(components: list[dict]) -> list[dict]:
    """
    Build a RulesEngine-safe copy of components.

    Saved result components still keep raw/editable data.
    RulesEngine input gets sanitized so invalid/junk OCR cannot
    manufacture quoteable BOMs.
    """

    VALID_VOLTAGES = {120, 208, 240, 480, 600}
    VALID_INT_RATINGS = {10, 14, 18, 22, 25, 30, 35, 42, 50, 65, 100, 125, 150, 200}
    VALID_TRIMS = {"SURFACE", "FLUSH"}
    VALID_ENCLOSURES = {"NEMA1", "NEMA3R"}

    SNAP_SPACES = {
        16: 18, 20: 18,
        28: 30, 32: 30,
        40: 42, 44: 42,
        52: 54, 56: 54,
        64: 66, 68: 66,
        70: 72, 74: 72,
        82: 84, 86: 84,
    }

    def _int_or_none(v):
        try:
            if v in (None, "", "NONE", "-", "X"):
                return None
            return int(float(str(v).replace(",", "").strip()))
        except Exception:
            return None

    def _panel_amp_or_none(v):
        n = _int_or_none(v)
        if n is None:
            return None
        if n < 100 or n > 1200:
            return None
        if n % 10 not in (0, 5):
            return None
        return n

    def _main_amp_or_none(v):
        # Main breaker is optional, but if present it must be sane.
        return _panel_amp_or_none(v)

    def _voltage_or_none(v):
        n = _int_or_none(v)
        if n in VALID_VOLTAGES:
            return n
        return None

    def _int_rating_or_none(v):
        n = _int_or_none(v)
        if n in VALID_INT_RATINGS:
            return n
        return None

    def _spaces_or_none(v):
        n = _int_or_none(v)
        if n is None:
            return None

        n = SNAP_SPACES.get(n, n)

        # Hard sanity range. Prevents things like 123 spaces from becoming
        # a valid-looking panel selection input.
        if n <= 0 or n > 84:
            return None

        # Most panel spaces are even. If this ever blocks a real case,
        # relax this, but it is a good junk filter.
        if n % 2 != 0:
            return None

        return n

    def _norm_choice(v, allowed: set[str]):
        s = str(v or "").strip().upper().replace(" ", "")
        if s in ("", "NONE", "-", "X"):
            return None
        if s in allowed:
            return s
        return None

    def _clean_breakers(breakers, panel_limit: int | None):
        clean = []

        if not isinstance(breakers, list):
            return clean

        for b in breakers:
            if not isinstance(b, dict):
                continue

            amps = _int_or_none(b.get("amperage"))
            poles = _int_or_none(b.get("poles"))

            if amps is None or poles is None:
                continue

            if amps < 15 or amps > 1200:
                continue

            if poles not in (1, 2, 3):
                continue

            if isinstance(panel_limit, int) and panel_limit > 0 and amps > panel_limit:
                continue

            try:
                count = int(b.get("count", 1) or 1)
            except Exception:
                count = 1

            if count <= 0:
                continue

            nb = dict(b)
            nb["amperage"] = amps
            nb["poles"] = poles

            if count != 1:
                nb["count"] = count

            clean.append(nb)

        return clean

    safe = []

    for comp in components or []:
        if not isinstance(comp, dict):
            safe.append(comp)
            continue

        c = _deep_copy_jsonable(comp)

        if str(c.get("type") or "").strip().lower() != "panelboard":
            safe.append(c)
            continue

        attrs = c.get("attrs") or {}
        if not isinstance(attrs, dict):
            attrs = {}

        attrs = dict(attrs)

        # Problem panels: suppress completely for RulesEngine.
        # UI/result.json still keeps the original raw evidence.
        if _is_problem_component_for_rules(c):
            reason = (
                str(c.get("reason") or "").strip()
                or str(c.get("panelNote") or "").strip()
                or str(c.get("panelStatus") or "").strip()
                or "User review required."
            )

            attrs["amperage"] = None
            attrs["voltage"] = None
            attrs["spaces"] = None
            attrs["mainBreakerAmperage"] = None
            attrs["detected_breakers"] = []
            attrs["breaker_data_suppressed"] = True
            attrs["breaker_suppression_reason"] = reason

            c["attrs"] = attrs
            c["_skipped"] = True
            c["reason"] = reason

            safe.append(c)
            continue

        # Clean/usable panels: sanitize quote-driving values before rules.
        bus_amp = _panel_amp_or_none(attrs.get("amperage"))
        main_amp = _main_amp_or_none(attrs.get("mainBreakerAmperage"))
        voltage = _voltage_or_none(attrs.get("voltage"))
        spaces = _spaces_or_none(attrs.get("spaces"))
        int_rating = _int_rating_or_none(attrs.get("intRating"))

        panel_limit = next(
            (v for v in (main_amp, bus_amp) if isinstance(v, int) and v > 0),
            None
        )

        attrs["amperage"] = bus_amp
        attrs["mainBreakerAmperage"] = main_amp
        attrs["voltage"] = voltage
        attrs["spaces"] = spaces
        attrs["intRating"] = int_rating
        attrs["detected_breakers"] = _clean_breakers(
            attrs.get("detected_breakers") or [],
            panel_limit
        )

        trim = _norm_choice(
            attrs.get("trimStyle") or attrs.get("trim_style"),
            VALID_TRIMS
        )

        enclosure = _norm_choice(
            attrs.get("enclosure"),
            VALID_ENCLOSURES
        )

        if enclosure == "NEMA3R":
            trim = None

        attrs["trimStyle"] = trim
        attrs["enclosure"] = enclosure

        c["attrs"] = attrs
        safe.append(c)

    return safe

def _fmt_edit_value(value, suffix=""):
    if value in (None, "", "NONE", "-", "X"):
        return "-"
    return f"{value}{suffix}"


def _safe_int_for_edit(value):
    try:
        if value in (None, "", "NONE", "-", "X"):
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def _normalize_breaker_option(value):
    s = str(value or "").strip().upper()
    if s in ("", "NONE", "-", "X", "STANDARD"):
        return ""
    return s


def _breaker_group_map_for_edit(breakers) -> dict:
    """
    Convert breaker rows into grouped counts:
      {(amps, poles, option): count}
    """
    grouped = {}

    if not isinstance(breakers, list):
        return grouped

    for b in breakers:
        if not isinstance(b, dict):
            continue

        amps = _safe_int_for_edit(b.get("amperage"))
        poles = _safe_int_for_edit(b.get("poles"))

        if amps is None or poles is None:
            continue

        try:
            count = int(b.get("count", 1) or 1)
        except Exception:
            count = 1

        if count <= 0:
            continue

        option = _normalize_breaker_option(
            b.get("specialFeatures")
            or b.get("special_features")
            or b.get("breakerOption")
            or b.get("breaker_option")
            or ""
        )

        key = (amps, poles, option)
        grouped[key] = grouped.get(key, 0) + count

    return grouped


def _format_breaker_key_for_edit(key) -> str:
    amps, poles, option = key
    base = f"{poles}P {amps}A"
    if option:
        return f"{base} {option}"
    return base


def _build_panel_edit_changes(old_component: dict, new_component: dict) -> list[str]:
    """
    Build human-readable edit lines like:
      int rating: 30K -> 22K
      bus amps: 1200A -> 200A
      removed 1 - 1P 20A
      added 1 - 2P 20A
    """

    old_component = old_component or {}
    new_component = new_component or {}

    old_attrs = old_component.get("attrs") or {}
    new_attrs = new_component.get("attrs") or {}

    changes = []

    field_specs = [
        ("bus amps", old_attrs.get("amperage"), new_attrs.get("amperage"), "A", True),
        ("volts", old_attrs.get("voltage"), new_attrs.get("voltage"), "V", True),
        ("main amps", old_attrs.get("mainBreakerAmperage"), new_attrs.get("mainBreakerAmperage"), "A", True),
        ("int rating", old_attrs.get("intRating"), new_attrs.get("intRating"), "K", True),
        ("spaces", old_attrs.get("spaces"), new_attrs.get("spaces"), "", True),

        # These can be injected by modal/defaults, so only log if old value existed.
        ("material", old_attrs.get("material") or old_attrs.get("bussingMaterial") or old_attrs.get("bussing_material"), new_attrs.get("material") or new_attrs.get("bussingMaterial") or new_attrs.get("bussing_material"), "", False),
        ("rating", old_attrs.get("ratingType") or old_attrs.get("rating_type") or old_attrs.get("panelRatingType"), new_attrs.get("ratingType") or new_attrs.get("rating_type") or new_attrs.get("panelRatingType"), "", False),
        ("trim", old_attrs.get("trimStyle") or old_attrs.get("trim_style"), new_attrs.get("trimStyle") or new_attrs.get("trim_style"), "", False),
        ("enclosure", old_attrs.get("enclosure"), new_attrs.get("enclosure"), "", False),
    ]

    # Panel name is top-level, not attrs.
    old_name = old_component.get("name")
    new_name = new_component.get("name")
    if str(old_name or "").strip() != str(new_name or "").strip():
        changes.append(f"name: {_fmt_edit_value(old_name)} -> {_fmt_edit_value(new_name)}")

    for label, old_val, new_val, suffix, always_log_if_changed in field_specs:
        if not always_log_if_changed and old_val in (None, "", "NONE", "-", "X"):
            continue

        old_display = _fmt_edit_value(old_val, suffix)
        new_display = _fmt_edit_value(new_val, suffix)

        if str(old_display).strip().upper() != str(new_display).strip().upper():
            changes.append(f"{label}: {old_display} -> {new_display}")

    old_breakers = _breaker_group_map_for_edit(old_attrs.get("detected_breakers") or [])
    new_breakers = _breaker_group_map_for_edit(new_attrs.get("detected_breakers") or [])

    all_keys = sorted(
        set(old_breakers.keys()) | set(new_breakers.keys()),
        key=lambda k: (int(k[1]), int(k[0]), str(k[2] or ""))
    )

    for key in all_keys:
        old_count = int(old_breakers.get(key, 0) or 0)
        new_count = int(new_breakers.get(key, 0) or 0)

        if old_count == new_count:
            continue

        label = _format_breaker_key_for_edit(key)
        delta = new_count - old_count

        if old_count == 0 and new_count > 0:
            changes.append(f"added {new_count} - {label}")
        elif new_count == 0 and old_count > 0:
            changes.append(f"removed {old_count} - {label}")
        elif delta > 0:
            changes.append(f"added {delta} - {label}")
        else:
            changes.append(f"removed {abs(delta)} - {label}")

    return changes


def _append_panel_edit_log(job_dir: Path, job_id: str, old_component: dict, new_component: dict) -> dict:
    """
    Append one edit event to edits.json.

    File shape:
    {
      "job_id": "...",
      "updated_at_utc": "...",
      "edits": [
        {
          "edited_at_utc": "...",
          "panel_original": "HC",
          "panel_current": "HC",
          "changes": [...]
        }
      ]
    }
    """

    path = _edit_log_path(job_dir)

    existing = _json_read_or_none(path) or {}
    if not isinstance(existing, dict):
        existing = {}

    edits = existing.get("edits") or []
    if not isinstance(edits, list):
        edits = []

    changes = _build_panel_edit_changes(old_component, new_component)

    if not changes:
        changes = ["No meaningful field changes detected."]

    now = _now_utc().isoformat()

    event = {
        "edited_at_utc": now,
        "panel_original": str((old_component or {}).get("name") or "").strip(),
        "panel_current": str((new_component or {}).get("name") or "").strip(),
        "changes": changes,
    }

    edits.append(event)

    payload = {
        "job_id": job_id,
        "updated_at_utc": now,
        "edits": edits,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str, indent=2)

    return payload

@anvil.server.callable
def vm_rerun_rules_with_panel_edit(job_id: str, owner_email: str, group_folder: str, original_panel_name: str, edited_component: dict, original_source_path: str = None) -> dict:
    """
    Fast rules-only rerun after the user edits one panel.

    IMPORTANT:
    - Does NOT enter the normal job queue.
    - Does NOT rerun OCR.
    - Does NOT rerender PDF/images.
    - Does NOT touch worker subprocesses.
    - Only reads result.json, swaps one component, reruns RulesEngine, writes result.json.
    """

    if not job_id or not str(job_id).strip():
        return {"ok": False, "error": "Missing job_id."}

    if not owner_email or not str(owner_email).strip():
        return {"ok": False, "error": "Missing owner_email."}

    if (not original_panel_name or not str(original_panel_name).strip()) and (not original_source_path or not str(original_source_path).strip()):
        return {"ok": False, "error": "Missing original_panel_name or original_source_path."}

    if not isinstance(edited_component, dict):
        return {"ok": False, "error": "edited_component must be a dict."}

    job_id = str(job_id).strip()
    owner_email = str(owner_email).strip().lower()
    original_panel_name = str(original_panel_name or "").strip()
    original_source_path = str(original_source_path or "").strip().replace("\\", "/")

    job_dir = _resolve_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_dir is None:
        return {"ok": False, "error": "Job not found. Please resubmit your PDF."}
    sp = _status_paths(job_dir)

    status = _json_read_or_none(sp["status"]) or {}
    if not status:
        return {"ok": False, "error": f"Unknown job_id: {job_id}"}

    if _is_excluded_job_expired(status):
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f">>> deleted expired excluded job on edit rerun: {job_dir}")
        except Exception:
            pass
        return {"ok": False, "error": "Job not found. Please resubmit your PDF."}

    job_owner = str(
        status.get("owner_email")
        or status.get("owner_id")
        or ""
    ).strip().lower()

    if not job_owner or job_owner != owner_email:
        return {"ok": False, "error": "Owner mismatch."}

    state = str(status.get("state") or "").strip().lower()
    if state != "done":
        return {
            "ok": False,
            "error": f"Cannot edit this job yet. Current state: {state or 'unknown'}."
        }

    result = _json_read_or_none(sp["result"]) or {}
    if not isinstance(result, dict) or not result:
        return {"ok": False, "error": "Could not load saved result.json."}

    components = result.get("components") or []
    if not isinstance(components, list):
        return {"ok": False, "error": "Saved result does not contain a valid components list."}

    def _norm_name(value):
        return str(value or "").strip().upper()

    def _norm_path_for_edit(value):
        s = str(value or "").strip().replace("\\", "/")
        while "//" in s:
            s = s.replace("//", "/")
        return s

    target_norm = _norm_name(original_panel_name)
    target_source = _norm_path_for_edit(original_source_path)
    target_source_l = target_source.lower()
    target_source_base_l = target_source_l.split("/")[-1] if target_source_l else ""

    cleaned = dict(edited_component)
    cleaned["type"] = "panelboard"

    # Keep the stable/original panel name as the real rules/lookup key.
    # User-facing renames belong in display_name only.
    stable_name = str(
        cleaned.get("original_name")
        or original_panel_name
        or cleaned.get("name")
        or ""
    ).strip()

    display_name = str(cleaned.get("display_name") or "").strip()

    cleaned["name"] = stable_name
    cleaned["original_name"] = stable_name

    if display_name and _norm_name(display_name) != _norm_name(stable_name):
        cleaned["display_name"] = display_name
        cleaned["name_was_edited"] = True
    else:
        cleaned.pop("display_name", None)
        cleaned.pop("name_was_edited", None)

    attrs = cleaned.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    # Manual edit means stale parser/problem status should not survive.
    # The regenerated result should be judged from the edited attrs + rerun rules.
    for key in (
        "panelStatus",
        "panelNote",
        "specialHeaderType",
        "_skipped",
        "reason",
    ):
        cleaned.pop(key, None)

    for key in (
        "breaker_data_suppressed",
        "breaker_suppression_reason",
        "headerValidationStatus",
        "headerValidationMissing",
    ):
        attrs.pop(key, None)

    cleaned["attrs"] = attrs

    replaced = False
    old_component_for_edit_log = None

    def _component_source_matches(comp: dict) -> bool:
        if not target_source_l:
            return False

        candidates = (
            comp.get("source"),
            comp.get("overlay_source"),
            comp.get("overlaySource"),
            comp.get("preview_source"),
            comp.get("previewSource"),
            comp.get("reviewOverlayPath"),
            comp.get("review_overlay_path"),
        )

        for cand in candidates:
            cand_n = _norm_path_for_edit(cand).lower()
            if not cand_n:
                continue

            cand_base = cand_n.split("/")[-1]

            if cand_n == target_source_l or (target_source_base_l and cand_base == target_source_base_l):
                return True

        return False

    def _replace_component_at(idx: int, comp: dict):
        nonlocal replaced, old_component_for_edit_log

        # Preserve visual/source metadata only.
        # Do NOT preserve old parser/problem status after manual edit.
        for key in (
            "source",
            "overlay_source",
            "overlaySource",
            "preview_source",
            "previewSource",
            "reviewOverlayPath",
            "review_overlay_path",
        ):
            if key not in cleaned and key in comp:
                cleaned[key] = comp.get(key)

        old_component_for_edit_log = _deep_copy_jsonable(comp)
        components[idx] = cleaned
        replaced = True

    # 1) Source match first. This lets bad/problem edits target the exact crop.
    if target_source_l:
        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                continue

            if str(comp.get("type") or "").strip().lower() != "panelboard":
                continue

            if not _component_source_matches(comp):
                continue

            _replace_component_at(idx, comp)
            break

    # 2) Backward-compatible name match for normal BOM-card edits.
    if not replaced and target_norm:
        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                continue

            if str(comp.get("type") or "").strip().lower() != "panelboard":
                continue

            if _norm_name(comp.get("name")) != target_norm:
                continue

            _replace_component_at(idx, comp)
            break

    if not replaced:
        return {"ok": False, "error": f"Panel not found: {original_panel_name or original_source_path}"}

    edit_log_payload = _append_panel_edit_log(
        job_dir=job_dir,
        job_id=job_id,
        old_component=old_component_for_edit_log or {},
        new_component=cleaned,
    )

    ui_overrides = (
        result.get("ui_overrides")
        or status.get("ui_overrides")
        or _DEFAULT_OVERRIDES
    )

    # Mirror the normal final processing step, but only for rules.
    rules_payload = _build_rules_payload(ui_overrides, _rules_safe_components(components))

    try:
        new_rules_result = RE2.process_job(rules_payload) or {}
    except Exception as e:
        new_rules_result = {
            "error": f"{type(e).__name__}: {e}"
        }

    result["components"] = components
    result["rules_result"] = new_rules_result
    result["ui_overrides"] = ui_overrides
    result["manually_edited"] = True
    result["last_edited_panel"] = cleaned.get("name") or original_panel_name
    result["last_edited_at_utc"] = _now_utc().isoformat()
    result["edit_log_path"] = str(_edit_log_path(job_dir))
    result["edit_log"] = edit_log_payload

    _result_write(job_dir, result)

    _status_write(
        job_dir,
        "done",
        result_path=str(sp["result"]),
        progress=100.0,
        manually_edited=True,
        last_edited_panel=cleaned.get("name") or original_panel_name,
        last_edited_at_utc=result["last_edited_at_utc"],
        edit_log_path=str(_edit_log_path(job_dir)),
        edit_count=len((edit_log_payload or {}).get("edits") or []),
    )

    return {
        "ok": True,
        "job_id": job_id,
        "result": result
    }

@anvil.server.callable
def vm_rerun_rules_with_global_defaults(job_id: str, owner_email: str, group_folder: str, clean_defaults: dict) -> dict:
    """
    Fast rules-only rerun after editing whole-job defaults.

    Does not rerun OCR.
    Does not rerender images.
    Updates ui_overrides.panelboards, applies override-level fields where needed,
    reruns RulesEngine, writes result.json/status.json, and returns updated result.
    """

    if not job_id or not str(job_id).strip():
        return {"ok": False, "error": "Missing job_id."}

    if not owner_email or not str(owner_email).strip():
        return {"ok": False, "error": "Missing owner_email."}

    if not isinstance(clean_defaults, dict):
        return {"ok": False, "error": "clean_defaults must be a dict."}

    job_id = str(job_id).strip()
    owner_email = str(owner_email).strip().lower()

    job_dir = _resolve_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_dir is None:
        return {"ok": False, "error": "Job not found. Please resubmit your PDF."}

    sp = _status_paths(job_dir)

    status = _json_read_or_none(sp["status"]) or {}
    if not status:
        return {"ok": False, "error": f"Unknown job_id: {job_id}"}

    if _is_excluded_job_expired(status):
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f">>> deleted expired excluded job on global defaults rerun: {job_dir}")
        except Exception:
            pass
        return {"ok": False, "error": "Job not found. Please resubmit your PDF."}

    job_owner = str(
        status.get("owner_email")
        or status.get("owner_id")
        or ""
    ).strip().lower()

    if not job_owner or job_owner != owner_email:
        return {"ok": False, "error": "Owner mismatch."}

    state = str(status.get("state") or "").strip().lower()
    if state != "done":
        return {
            "ok": False,
            "error": f"Cannot edit this job yet. Current state: {state or 'unknown'}."
        }

    result = _json_read_or_none(sp["result"]) or {}
    if not isinstance(result, dict) or not result:
        return {"ok": False, "error": "Could not load saved result.json."}

    components = result.get("components") or []
    if not isinstance(components, list):
        return {"ok": False, "error": "Saved result does not contain a valid components list."}

    ui_overrides = (
        result.get("ui_overrides")
        or status.get("ui_overrides")
        or _DEFAULT_OVERRIDES
    )

    ui_overrides = _normalize_ui_overrides(ui_overrides)

    pb = dict((ui_overrides.get("panelboards") or {}))
    pb.update(clean_defaults or {})

    # Important compatibility key:
    # Your default schema already uses allow_plug_on_breakers.
    breaker_type = str(
        clean_defaults.get("breaker_type")
        or clean_defaults.get("branch_breaker_type")
        or clean_defaults.get("breaker_mounting")
        or ""
    ).strip().upper()

    if breaker_type == "PLUG_ON":
        pb["allow_plug_on_breakers"] = True
    elif breaker_type == "BOLT_ON":
        pb["allow_plug_on_breakers"] = False

    ui_overrides["panelboards"] = pb

    # Enclosure override means override detected enclosure/trim for the regenerated BOM.
    override_enclosure = str(pb.get("enclosure") or "").strip().upper()
    override_trim = str(
        pb.get("default_trim_style")
        or pb.get("trim_style")
        or ""
    ).strip().upper()

    override_material = str(
        pb.get("bussing_material")
        or pb.get("material")
        or ""
    ).strip().upper()

    override_rating = str(
        pb.get("rating_type")
        or pb.get("panel_rating_type")
        or ""
    ).strip().upper()

    for comp in components:
        if not isinstance(comp, dict):
            continue

        if str(comp.get("type") or "").strip().lower() != "panelboard":
            continue

        attrs = comp.get("attrs") or {}
        if not isinstance(attrs, dict):
            attrs = {}

        if override_material:
            attrs["material"] = override_material
            attrs["bussingMaterial"] = override_material
            attrs["bussing_material"] = override_material

        if override_rating:
            attrs["panelRatingType"] = override_rating
            attrs["ratingType"] = override_rating
            attrs["rating_type"] = override_rating

        if override_enclosure:
            attrs["enclosure"] = override_enclosure

        if override_trim:
            attrs["trimStyle"] = override_trim
            attrs["trim_style"] = override_trim

        comp["attrs"] = attrs

    rules_payload = _build_rules_payload(ui_overrides, _rules_safe_components(components))

    try:
        new_rules_result = RE2.process_job(rules_payload) or {}
    except Exception as e:
        new_rules_result = {
            "error": f"{type(e).__name__}: {e}"
        }

    now = _now_utc().isoformat()

    result["components"] = components
    result["rules_result"] = new_rules_result
    result["ui_overrides"] = ui_overrides
    result["manually_edited"] = True
    result["global_defaults_edited"] = True
    result["last_edited_panel"] = "GLOBAL_DEFAULTS"
    result["last_edited_at_utc"] = now

    _result_write(job_dir, result)

    _status_write(
        job_dir,
        "done",
        result_path=str(sp["result"]),
        progress=100.0,
        manually_edited=True,
        global_defaults_edited=True,
        last_edited_panel="GLOBAL_DEFAULTS",
        last_edited_at_utc=now,
    )

    return {
        "ok": True,
        "job_id": job_id,
        "result": result
    }

# ---------- Queue / Pool state ----------
_JOB_Q: "Queue[tuple[str,str]]" = Queue()
_INFLIGHT_BY_USER: dict[str, int] = {}
_Q_LOCK = threading.RLock()
_SUBMIT_LOCK = threading.RLock()
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
    """Put a (job_id, owner_id) tuple onto the shared job queue for worker threads to dequeue."""
    _JOB_Q.put((job_id, owner_id))

def _enter_inflight(owner_id: str) -> bool:
    """Try to increment the per-user inflight count; return False if at MAX_INFLIGHT_PER_USER."""
    with _Q_LOCK:
        c = _INFLIGHT_BY_USER.get(owner_id, 0)
        if c >= MAX_INFLIGHT_PER_USER:
            return False
        _INFLIGHT_BY_USER[owner_id] = c + 1
        return True

def _leave_inflight(owner_id: str):
    """Decrement the per-user inflight count (floor at 0)."""
    with _Q_LOCK:
        c = _INFLIGHT_BY_USER.get(owner_id, 0)
        _INFLIGHT_BY_USER[owner_id] = max(0, c - 1)

# ---------- Cancel helpers ----------
def _cancel_path(job_dir: Path) -> Path:
    """Return the path to the .cancel marker file used to signal job cancellation."""
    return job_dir / ".cancel"

def _is_canceled(job_dir: Path) -> bool:
    """Check whether a job has been marked as canceled by the presence of its .cancel file."""
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
    """Parse *x* as an integer (stripping commas); return None on failure."""
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
    panel_status = str((result_dict or {}).get("panelStatus") or "").strip()
    panel_note = str(hdr.get("panelNote") or "").strip()
    special_header_type = hdr.get("specialHeaderType") if isinstance(hdr.get("specialHeaderType"), dict) else None

    # Existing-panel metadata normally comes from the header parser.
    # Fall back to parser/root payload defensively in case the pipeline
    # nests the header-parser result differently.
    suspected_existing = bool(
        hdr.get("suspectedExisting", False)
        or prs.get("suspectedExisting", False)
        or (result_dict or {}).get("suspectedExisting", False)
    )

    existing_reason = str(
        hdr.get("existingReason")
        or prs.get("existingReason")
        or (result_dict or {}).get("existingReason")
        or ""
    ).strip()

    existing_detection = str(
        hdr.get("existingDetection")
        or prs.get("existingDetection")
        or (result_dict or {}).get("existingDetection")
        or ""
    ).strip()

    print(
        f"[EXISTING DEBUG] "
        f"name={hdr.get('name')!r} "
        f"flag={suspected_existing!r} "
        f"detection={existing_detection!r} "
        f"reason={existing_reason!r}"
    )

    # Header fields
    name   = hdr.get("name") or ""
    h_attrs = hdr.get("attrs") or {}

    header_missing = set()

    if isinstance(hdr.get("headerValidationMissing"), list):
        header_missing.update(str(x).strip().lower() for x in hdr.get("headerValidationMissing") or [])

    if isinstance(h_attrs.get("headerValidationMissing"), list):
        header_missing.update(str(x).strip().lower() for x in h_attrs.get("headerValidationMissing") or [])

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

    def _valid_panel_amp_for_rules(v):
        return isinstance(v, int) and 100 <= v <= 1200 and (v % 10 in (0, 5))

    # Do not let invalid OCR values like "Location: ELECTRICAL 123"
    # become real rule-engine amperage inputs.
    if "bus amps" in header_missing or not _valid_panel_amp_for_rules(amperage):
        amperage = None

    if "voltage" in header_missing or voltage not in (120, 208, 240, 480, 600):
        voltage = None

    if "main amps" in header_missing:
        main_amp = None
    elif main_amp is not None and not _valid_panel_amp_for_rules(main_amp):
        main_amp = None
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

        "panelStatus": panel_status,
        "panelNote": panel_note,
        "specialHeaderType": special_header_type,

        # Existing-equipment review metadata.
        # Does NOT affect Rules Engine processing.
        "suspectedExisting": suspected_existing,
        "existingReason": existing_reason,
        "existingDetection": existing_detection,

        "attrs": {
            "amperage": amperage,
            "spaces": spaces,
            "voltage": voltage,
            "intRating": intRating,
            "mainBreakerAmperage": main_amp,
            "trimStyle": trim_style,
            "enclosure": enclosure,
            "detected_breakers": det_brkrs,
            "breaker_position_issues": list(
                (prs or {}).get(
                    "breakerPositionIssues"
                )
                or []
            ),
        },
    }

    # If the parser flagged this panel, keep it available for review/edit,
    # but do not treat it as clean input for generated BOM logic.
    if panel_status:
        comp["_skipped"] = True
        comp["reason"] = panel_note or panel_status

    if notes:
        comp["notes"] = notes
        comp["attrs"]["breaker_data_suppressed"] = True

    return comp

# ---------- Compatibility panel selection ----------

_COMPATIBILITY_UNKNOWN_VALUES = {
    "",
    "none",
    "unknown",
    "null",
    "n/a",
    "na",
    "not found",
    "not available",
    "unavailable",
    "tbd",
    "x",
    "-",
    "0",
}


def _compatibility_value_present(value) -> bool:
    """
    Return True when a compatibility-selection value contains
    meaningful detected information.

    This is intentionally lenient. Its purpose is only to decide
    whether a candidate is probably a real panel schedule.

    The stricter compatibility grading happens later.
    """
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value > 0

    if isinstance(value, dict):
        return bool(value)

    if isinstance(value, (list, tuple, set)):
        return len(value) > 0

    normalized = str(value).strip().lower()

    if normalized.startswith("(unnamed)"):
        return False

    if normalized.startswith("image_"):
        return False

    return normalized not in _COMPATIBILITY_UNKNOWN_VALUES

def _compatibility_candidate_check(
    component: dict,
) -> tuple[bool, list[str], str]:
    """
    Decide whether a completed BreakerTablePipeline result is credible
    enough to use as the representative compatibility-test panel.

    Selection requirements:
      1. Component is a panelboard.
      2. A usable panel name was detected.
      3. At least one meaningful secondary attribute was detected.

    Important:
    panelStatus and _skipped do NOT automatically reject the candidate.

    A problem panel can still be a real panel schedule. We want to select
    it, then use its missing/problem fields when grading compatibility.
    """
    if not isinstance(component, dict):
        return False, [], "invalid_component"

    component_type = str(
        component.get("type") or ""
    ).strip().lower()

    if component_type != "panelboard":
        return False, [], "not_panelboard"

    special_header_type = component.get(
        "specialHeaderType"
    )

    if isinstance(special_header_type, dict):
        special_kind = str(
            special_header_type.get("kind") or ""
        ).strip().lower()

        # Every specialHeaderType produced by the
        # header parser represents something other
        # than a usable panel schedule.
        if special_kind:
            return (
                False,
                [],
                f"special_header_{special_kind}",
            )

    panel_name = component.get("name")

    if not _compatibility_value_present(panel_name):
        return False, [], "missing_valid_panel_name"

    attrs = component.get("attrs") or {}

    if not isinstance(attrs, dict):
        attrs = {}

    meaningful_fields = (
        "amperage",
        "voltage",
        "spaces",
        "intRating",
        "mainBreakerAmperage",
        "trimStyle",
        "enclosure",
    )

    detected_attributes = []

    for field_name in meaningful_fields:
        if _compatibility_value_present(attrs.get(field_name)):
            detected_attributes.append(field_name)

    detected_breakers = attrs.get("detected_breakers")

    if (
        isinstance(detected_breakers, list)
        and len(detected_breakers) > 0
    ):
        detected_attributes.append("detected_breakers")

    if not detected_attributes:
        return False, [], "missing_secondary_attributes"

    return True, detected_attributes, ""


def _find_compatibility_panel(
    saved_pdf: Path,
    img_dir: Path,
    pipeline: "BreakerTablePipeline",
    status_cb=None,
    max_attempts: int = COMPATIBILITY_MAX_PANEL_ATTEMPTS,
) -> dict:
    """
    Find and analyze one representative panel for a compatibility test.

    Flow:
      1. Run the normal PageFilter.
      2. Run PanelBoardSearch incrementally.
      3. Analyze one finder-valid candidate at a time.
      4. Accept the first candidate with:
           - a valid panel name
           - at least one meaningful secondary attribute
      5. Stop after max_attempts BreakerTablePipeline attempts.

    This function does not:
      - run the Rules Engine
      - create a BOM
      - persist a result
      - perform retention cleanup

    The compatibility-job wrapper added later will create the temporary
    directory and delete the entire directory in a finally block.
    """
    saved_pdf = Path(saved_pdf).resolve()
    img_dir = Path(img_dir).resolve()

    if not saved_pdf.is_file():
        raise FileNotFoundError(
            f"Compatibility PDF not found: {saved_pdf}"
        )

    if pipeline is None:
        raise ValueError(
            "A loaded BreakerTablePipeline instance is required."
        )

    try:
        max_attempts = int(max_attempts)
    except Exception:
        raise ValueError(
            "max_attempts must be a positive integer."
        )

    if max_attempts <= 0:
        raise ValueError(
            "max_attempts must be a positive integer."
        )

    img_dir.mkdir(parents=True, exist_ok=True)

    def _emit(step: str, progress: float | None = None, **extra):
        if not callable(status_cb):
            return

        try:
            payload = {
                "step": step,
            }

            if progress is not None:
                payload["progress"] = progress

            payload.update(extra or {})
            status_cb(**payload)

        except Exception:
            pass

    print(
        f">>> compatibility search started: "
        f"pdf={saved_pdf} max_attempts={max_attempts}"
    )

    # ---------------------------------------------------------
    # 1. Run the same PageFilter configuration as production
    # ---------------------------------------------------------
    _emit(
        "compatibility_filtering_pages",
        5.0,
        compatibility_attempt=0,
        compatibility_max_attempts=max_attempts,
    )

    kept_pages = []
    dropped_pages = []
    filtered_pdf = None
    filter_log_json = None
    page_filter_error = ""

    try:
        pf = PageFilter(
            output_dir=str(img_dir.parent),
            dpi=400,
            longest_cap_px=9000,
            proc_scale=0.5,
            use_ocr=True,
            ocr_gpu=False,
            verbose=True,
            debug=False,
            rect_w_fr_range=(0.20, 0.60),
            rect_h_fr_range=(0.20, 0.60),
            min_rectangularity=0.70,
            min_rect_count=2,
            min_whitespace_area_fr=0.004,
            use_ghostscript_letter=True,
            letter_orientation="landscape",
            gs_use_cropbox=True,
            gs_compat="1.7",
        )

        (
            kept_pages,
            dropped_pages,
            filtered_pdf,
            filter_log_json,
        ) = pf.readPdf(str(saved_pdf))

        print(
            f">>> compatibility PageFilter: "
            f"kept={len(kept_pages)} "
            f"dropped={len(dropped_pages)} "
            f"filtered_pdf={filtered_pdf}"
        )

    except Exception as e:
        page_filter_error = f"{type(e).__name__}: {e}"

        print(
            f">>> compatibility PageFilter error; "
            f"falling back to original PDF: {page_filter_error}"
        )

        kept_pages = []
        dropped_pages = []
        filtered_pdf = None
        filter_log_json = None

    if filtered_pdf and len(kept_pages) > 0:
        pdf_for_finder = str(filtered_pdf)
        used_filtered_pdf = True
    else:
        pdf_for_finder = str(saved_pdf)
        used_filtered_pdf = False

        if filtered_pdf is not None and len(kept_pages) == 0:
            print(
                ">>> compatibility PageFilter kept 0 pages — "
                "falling back to original PDF"
            )

    _emit(
        "compatibility_finding_panel",
        15.0,
        kept_pages=len(kept_pages),
        dropped_pages=len(dropped_pages),
        compatibility_attempt=0,
        compatibility_max_attempts=max_attempts,
    )

    # ---------------------------------------------------------
    # 2. Build the compatibility finder
    # ---------------------------------------------------------
    local_finder = PanelBoardSearch(
        output_dir=str(img_dir),
        dpi=400,
        render_dpi=PANEL_FINDER_DEFAULTS["render_dpi"],
        aa_level=PANEL_FINDER_DEFAULTS["aa_level"],
        render_colorspace=PANEL_FINDER_DEFAULTS[
            "render_colorspace"
        ],
        min_void_area_fr=PANEL_FINDER_DEFAULTS[
            "min_void_area_fr"
        ],
        min_void_w_px=PANEL_FINDER_DEFAULTS[
            "min_void_w_px"
        ],
        min_void_h_px=PANEL_FINDER_DEFAULTS[
            "min_void_h_px"
        ],
        max_void_area_fr=PANEL_FINDER_DEFAULTS[
            "max_void_area_fr"
        ],
        void_w_fr_range=PANEL_FINDER_DEFAULTS[
            "void_w_fr_range"
        ],
        void_h_fr_range=PANEL_FINDER_DEFAULTS[
            "void_h_fr_range"
        ],
        min_whitespace_area_fr=PANEL_FINDER_DEFAULTS[
            "min_whitespace_area_fr"
        ],
        margin_shave_px=PANEL_FINDER_DEFAULTS[
            "margin_shave_px"
        ],
        pad=PANEL_FINDER_DEFAULTS["pad"],
        verbose=PANEL_FINDER_DEFAULTS["verbose"],
    )

    state = {
        "attempts_used": 0,
        "limit_reached": False,
        "stop_reason": "",
        "selected_raw": None,
        "selected_component": None,
        "selected_candidate_path": "",
        "selected_attributes": [],
    }

    attempt_log = []

    # ---------------------------------------------------------
    # 3. Analyze candidates one at a time
    # ---------------------------------------------------------
    def _check_candidate(candidate_path: str) -> bool:
        """
        Return False:
            Reject this candidate and request the next finder candidate.

        Return True:
            Stop PanelBoardSearch.

        A True return can mean either:
            - a credible panel was selected
            - the six-attempt cap was reached
        """
        # Defensive guard. Normally the callback stops on the exact
        # max_attempts-th candidate below.
        if state["attempts_used"] >= max_attempts:
            state["limit_reached"] = True
            state["stop_reason"] = "attempt_limit"
            return True

        state["attempts_used"] += 1
        attempt_number = state["attempts_used"]

        progress = 20.0 + (
            attempt_number / max_attempts
        ) * 60.0

        _emit(
            "compatibility_testing_candidate",
            progress,
            compatibility_attempt=attempt_number,
            compatibility_max_attempts=max_attempts,
            candidate_name=Path(candidate_path).name,
        )

        print(
            f">>> compatibility candidate "
            f"{attempt_number}/{max_attempts}: "
            f"{candidate_path}"
        )

        attempt_record = {
            "attempt": attempt_number,
            "candidate_path": str(candidate_path),
            "candidate_name": Path(candidate_path).name,
            "credible_panel": False,
            "detected_name": "",
            "detected_attributes": [],
            "rejection_reason": "",
            "error": "",
        }

        try:
            # Rejected candidates should not affect name handling for
            # the next candidate. Each compatibility candidate is tested
            # independently.
            reset_name_deduper()

            raw = pipeline.run(
                candidate_path,
                run_analyzer=True,
                run_parser=True,
                run_header=True,
            )

            if isinstance(raw, dict) and "_error" in raw:
                raise RuntimeError(
                    str(raw.get("_error") or "Pipeline error")
                )

            component = _merge_component_from_btp(
                raw or {},
                candidate_path,
            )

            component = _normalize_component_for_none(
                component or {}
            )

            # Special schedules/equipment are not panel
            # attempts. Skip them and keep searching.
            special_header_type = component.get(
                "specialHeaderType"
            )

            if isinstance(
                special_header_type,
                dict
            ):
                special_kind = str(
                    special_header_type.get(
                        "kind"
                    ) or ""
                ).strip().lower()

                if special_kind:
                    # This candidate was counted before
                    # the pipeline identified its type.
                    # Remove it from the panel-attempt count.
                    state["attempts_used"] = max(
                        0,
                        state["attempts_used"] - 1,
                    )

                    attempt_record["attempt"] = None
                    attempt_record[
                        "skipped_special_header"
                    ] = True
                    attempt_record[
                        "special_header_kind"
                    ] = special_kind
                    attempt_record[
                        "rejection_reason"
                    ] = (
                        f"special_header_"
                        f"{special_kind}"
                    )

                    attempt_log.append(
                        attempt_record
                    )

                    print(
                        ">>> compatibility candidate "
                        "skipped without using a panel "
                        f"attempt: {special_kind} | "
                        f"{candidate_path}"
                    )

                    return False

            (
                credible,
                detected_attributes,
                rejection_reason,
            ) = _compatibility_candidate_check(component)

            attempt_record["credible_panel"] = bool(credible)
            attempt_record["detected_name"] = str(
                component.get("name") or ""
            ).strip()
            attempt_record["detected_attributes"] = list(
                detected_attributes
            )
            attempt_record["rejection_reason"] = (
                rejection_reason or ""
            )

            attempt_log.append(attempt_record)

            if credible:
                state["selected_raw"] = raw
                state["selected_component"] = component
                state["selected_candidate_path"] = str(
                    candidate_path
                )
                state["selected_attributes"] = list(
                    detected_attributes
                )
                state["stop_reason"] = "credible_panel_found"

                print(
                    f">>> compatibility panel selected: "
                    f"name={component.get('name')!r} "
                    f"attributes={detected_attributes}"
                )

                return True

            print(
                f">>> compatibility candidate rejected: "
                f"reason={rejection_reason} "
                f"name={component.get('name')!r}"
            )

        except Exception as e:
            attempt_record["error"] = (
                f"{type(e).__name__}: {e}"
            )
            attempt_record["rejection_reason"] = (
                "pipeline_error"
            )

            attempt_log.append(attempt_record)

            print(
                f">>> compatibility candidate error "
                f"{attempt_number}/{max_attempts}: "
                f"{type(e).__name__}: {e}"
            )
            print(traceback.format_exc())

        # The current candidate was not accepted.
        # Stop after the sixth actual pipeline attempt.
        if attempt_number >= max_attempts:
            state["limit_reached"] = True
            state["stop_reason"] = "attempt_limit"

            print(
                f">>> compatibility attempt limit reached: "
                f"{max_attempts}"
            )

            # PanelBoardSearch currently uses True to mean "stop."
            # selected_component remains None, so the caller knows
            # this was a limit stop rather than a selected panel.
            return True

        return False

    # The finder-returned list is intentionally not used as the source
    # of truth. The callback state determines whether a panel was selected.
    local_finder.readPdf(
        pdf_for_finder,
        candidate_callback=_check_candidate,
    )

    selected_component = state["selected_component"]
    found_panel = isinstance(selected_component, dict)

    if not found_panel:
        if state["limit_reached"]:
            stop_reason = "attempt_limit"
        elif state["attempts_used"] == 0:
            stop_reason = "no_finder_valid_candidates"
        else:
            stop_reason = "finder_candidates_exhausted"

        state["stop_reason"] = stop_reason

        print(
            f">>> compatibility search completed without "
            f"a credible panel: reason={stop_reason} "
            f"attempts={state['attempts_used']}"
        )

        _emit(
            "compatibility_no_panel_found",
            90.0,
            compatibility_attempt=state["attempts_used"],
            compatibility_max_attempts=max_attempts,
        )

    else:
        _emit(
            "compatibility_panel_selected",
            90.0,
            compatibility_attempt=state["attempts_used"],
            compatibility_max_attempts=max_attempts,
            panel_name=selected_component.get("name"),
        )

    return {
        "ok": True,
        "found_panel": found_panel,
        "attempts_used": state["attempts_used"],
        "max_attempts": max_attempts,
        "limit_reached": state["limit_reached"],
        "stop_reason": state["stop_reason"],

        "kept_pages": len(kept_pages),
        "dropped_pages": len(dropped_pages),
        "used_filtered_pdf": used_filtered_pdf,
        "page_filter_error": page_filter_error,

        "candidate_path": state[
            "selected_candidate_path"
        ],
        "detected_attributes": state[
            "selected_attributes"
        ],

        # Internal compatibility data.
        # The later grading function will use these directly,
        # then the entire temporary test directory will be deleted.
        "component": state["selected_component"],
        "raw_pipeline_result": state["selected_raw"],

        "attempts": attempt_log,
    }

def _build_compatibility_report(selection: dict) -> dict:
    """
    Build the three user-facing compatibility sections:
      1. Panel Identification / Structure
      2. Header Section
      3. Breaker Section
    """
    selection = selection or {}

    attempts_used = int(selection.get("attempts_used") or 0)
    max_attempts = int(
        selection.get("max_attempts")
        or COMPATIBILITY_MAX_PANEL_ATTEMPTS
    )

    if not selection.get("found_panel"):
        return {
            "grade": "unconfirmed",
            "headline": "Unable to Confirm Compatibility",
            "summary": (
                "Argus could not identify a usable representative "
                "panel schedule."
            ),
            "attempts_used": attempts_used,
            "max_attempts": max_attempts,
            "sections": [
                {
                    "title": "Panel Identification / Structure",
                    "status": "problem",
                    "message": (
                        "Argus could not confidently identify a panel schedule. "
                        "Possible causes include touching panel schedules, schedules "
                        "touching drawing boundaries, the header being below the "
                        "breaker section, a non-standard layout, unreadable text, "
                        "or poor drawing quality."
                    ),
                },
                {
                    "title": "Header Section",
                    "status": "not_tested",
                    "message": (
                        "Header attributes could not be tested because a usable "
                        "panel schedule was not identified."
                    ),
                },
                {
                    "title": "Breaker Section",
                    "status": "not_tested",
                    "message": (
                        "Breaker columns and rows could not be tested because a "
                        "usable panel schedule was not identified."
                    ),
                },
            ],
            "notes": [
                (
                    "Try using a PDF with at least one clear, fully visible, "
                    "separated panel schedule."
                )
            ],
        }

    component = selection.get("component") or {}
    raw = selection.get("raw_pipeline_result") or {}

    attrs = component.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    stages = raw.get("results") or {}
    if not isinstance(stages, dict):
        stages = {}

    header = stages.get("header") or {}
    parser = stages.get("parser") or {}

    if not isinstance(header, dict):
        header = {}

    if not isinstance(parser, dict):
        parser = {}

    header_attrs = header.get("attrs") or {}
    if not isinstance(header_attrs, dict):
        header_attrs = {}

    header_missing = set()

    for source in (header, header_attrs):
        missing = source.get("headerValidationMissing")

        if isinstance(missing, list):
            for item in missing:
                text = str(item or "").strip().lower()
                if text:
                    header_missing.add(text)

    panel_name = component.get("name")
    panel_status = str(
        component.get("panelStatus") or ""
    ).strip()

    panel_note = str(
        component.get("panelNote") or ""
    ).strip()

    sections = []
    notes = []
    problems = 0

    # ---------------------------------------------------------
    # 1. Panel Identification / Structure
    # ---------------------------------------------------------
    if panel_status:
        problems += 1

        sections.append({
            "title": "Panel Identification / Structure",
            "status": "questionable",
            "message": (
                panel_note
                or (
                    "Argus identified a possible panel schedule, but its "
                    "structure was flagged for review. This may be caused by "
                    "touching schedules, the header being below the breaker "
                    "section, a non-standard layout, revision markings, or "
                    "poor drawing quality."
                )
            ),
        })

    else:
        sections.append({
            "title": "Panel Identification / Structure",
            "status": "good",
            "message": (
                f"Argus identified panel “{panel_name}” as a usable "
                "panel schedule."
            ),
        })

    # ---------------------------------------------------------
    # 2. Header Section
    # ---------------------------------------------------------
    questionable_fields = []

    if not _compatibility_value_present(
        attrs.get("amperage")
    ) or "bus amps" in header_missing:
        questionable_fields.append("Bus Amps")

    if not _compatibility_value_present(
        attrs.get("voltage")
    ) or "voltage" in header_missing:
        questionable_fields.append("Voltage")

    if "main amps" in header_missing:
        questionable_fields.append("Main Breaker Amps")

    if questionable_fields:
        problems += 1

        sections.append({
            "title": "Header Section",
            "status": "questionable",
            "message": (
                "Argus could not confidently confirm: "
                + ", ".join(questionable_fields)
                + ". The information may be missing, may not have a clear "
                  "label-to-value relationship, may use unfamiliar wording, "
                  "or may not be readable enough to detect reliably."
            ),
        })

    else:
        sections.append({
            "title": "Header Section",
            "status": "good",
            "message": (
                "Argus was able to confidently analyze the header section."
            ),
        })

    if not _compatibility_value_present(
        attrs.get("intRating")
    ):
        notes.append(
            "kAIC was not confidently detected. Argus may use the "
            "configured default when this value is unavailable."
        )

    if not _compatibility_value_present(
        attrs.get("mainBreakerAmperage")
    ):
        notes.append(
            "Main Breaker Amps was not detected. This may be acceptable "
            "when the panel is main-lug-only or the value does not apply."
        )

    # ---------------------------------------------------------
    # 3. Breaker Section
    # ---------------------------------------------------------
    breakers = parser.get(
        "detected_breakers"
    )

    if not isinstance(breakers, list) or not breakers:
        breakers = attrs.get(
            "detected_breakers"
        )

    if not isinstance(breakers, list):
        breakers = []

    breaker_information_complete = bool(
        breakers
    )

    if breaker_information_complete:
        for breaker in breakers:
            if not isinstance(breaker, dict):
                breaker_information_complete = False
                break

            amperage_found = (
                _compatibility_value_present(
                    breaker.get("amperage")
                )
            )

            poles_found = (
                _compatibility_value_present(
                    breaker.get("poles")
                )
            )

            if not amperage_found or not poles_found:
                breaker_information_complete = False
                break

    if not breaker_information_complete:
        problems += 1

        sections.append({
            "title": "Breaker Section",
            "status": "questionable",
            "message": (
                "Some breaker information was difficult to detect. "
                "Breakers will need careful review."
            ),
        })

    else:
        sections.append({
            "title": "Breaker Section",
            "status": "good",
            "message": (
                "Argus located the breaker section and confidently "
                "detected information."
            ),
        })

    if problems:
        grade = "questionable"
        headline = "Compatibility Questionable"
        summary = (
            "Argus identified a representative panel, but parts of the "
            "schedule may require review or manual correction."
        )
    else:
        grade = "likely"
        headline = "Likely Compatible"
        summary = (
            "Argus successfully identified and analyzed a representative "
            "panel schedule."
        )

    return {
        "grade": grade,
        "headline": headline,
        "summary": summary,
        "panel_name": str(panel_name or "").strip(),
        "attempts_used": attempts_used,
        "max_attempts": max_attempts,
        "sections": sections,
        "notes": notes,
    }


def _process_compatibility_job(
    job_id: str,
    pipeline: "BreakerTablePipeline",
):
    """
    Run a compatibility test without running RulesEngine or creating a BOM.
    """
    job_dir = _resolve_job_dir_any(job_id)

    if job_dir is None:
        raise RuntimeError(
            f"Compatibility job not found: {job_id}"
        )

    sp = _status_paths(job_dir)
    previous = _json_read_or_none(sp["status"]) or {}
    noticed_ts_ms = previous.get("noticed_ts_ms")

    try:
        _status_write(
            job_dir,
            "running",
            job_type="compatibility",
            step="compatibility_starting",
            progress=1.0,
        )

        pdfs = sorted(
            (job_dir / "uploaded_pdfs").glob("*.pdf")
        )

        if not pdfs:
            raise RuntimeError(
                "No compatibility PDF was found."
            )

        def _status_cb(**kwargs):
            payload = {
                "job_type": "compatibility",
            }
            payload.update(kwargs or {})

            _status_write(
                job_dir,
                "running",
                **payload,
            )

        selection = _find_compatibility_panel(
            saved_pdf=pdfs[0],
            img_dir=job_dir / "pdf_images",
            pipeline=pipeline,
            status_cb=_status_cb,
            max_attempts=COMPATIBILITY_MAX_PANEL_ATTEMPTS,
        )

        report = _build_compatibility_report(
            selection
        )

        # Preserve the exact panel image/overlay used by
        # the quick compatibility analysis.
        selected_component = (
            selection.get("component")
            or selection.get("selected_component")
            or {}
        )

        if not isinstance(
            selected_component,
            dict
        ):
            selected_component = {}

        compatibility_image_path = str(
            selected_component.get("overlay_source")
            or selected_component.get("overlaySource")
            or selected_component.get("reviewOverlayPath")
            or selected_component.get("review_overlay_path")
            or selected_component.get("preview_source")
            or selected_component.get("previewSource")
            or selected_component.get("source")
            or selection.get("overlay_source")
            or selection.get("overlay_path")
            or selection.get("candidate_path")
            or selection.get("source")
            or ""
        ).strip()

        finished_ts_ms = _epoch_ms()

        cycle_time_ms = (
            finished_ts_ms - noticed_ts_ms
            if isinstance(noticed_ts_ms, int)
            else None
        )

        result = {
            "ok": True,
            "job_id": job_id,
            "job_type": "compatibility",
            "report": report,

            # Stored as a path in result.json.
            # The status callable converts it to BlobMedia
            # before sending it to the Anvil client.
            "compatibility_image_path": (
                compatibility_image_path
            ),

            "cycle_time_ms": cycle_time_ms,
            "cycle_time_str": _fmt_cycle_time(
                cycle_time_ms or 0
            ),
        }

        _result_write(job_dir, result)

        _status_write(
            job_dir,
            "done",
            job_type="compatibility",
            step="compatibility_complete",
            progress=100.0,
            result_path=str(sp["result"]),
            compatibility_attempt=selection.get(
                "attempts_used", 0
            ),
            compatibility_max_attempts=selection.get(
                "max_attempts",
                COMPATIBILITY_MAX_PANEL_ATTEMPTS,
            ),
            cycle_time_ms=cycle_time_ms,
            cycle_time_str=result["cycle_time_str"],
        )

        print(
            f">>> compatibility job complete: {job_id}"
        )

    except Exception as e:
        print(
            f">>> compatibility job error [{job_id}]: "
            f"{type(e).__name__}: {e}"
        )
        print(traceback.format_exc())

        _status_write(
            job_dir,
            "error",
            job_type="compatibility",
            step="compatibility_error",
            progress=100.0,
            error=f"{type(e).__name__}: {e}",
        )

def _finalize_canceled_job(job_dir: Path, job_id: str, noticed_ts_ms=None, step="canceled"):
    """
    Final cancel cleanup called by the worker once it notices .cancel.
    Removes the job folder so it disappears from My Jobs.
    """
    try:
        _status_write(
            job_dir,
            "canceled",
            canceled=True,
            step=step,
            noticed_ts_ms=noticed_ts_ms,
            progress=0.0,
        )
    except Exception:
        pass

    try:
        shutil.rmtree(job_dir, ignore_errors=True)
        print(f">>> canceled job folder deleted: {job_id}")
    except Exception as e:
        print(f">>> canceled job folder delete failed [{job_id}]: {e}")

    try:
        _jobs_upsert(job_id, state="canceled", updated_at=_now_utc())
    except Exception:
        pass

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
    job_dir = _resolve_job_dir_any(job_id)
    if job_dir is None:
        print(f">>> worker could not resolve job_id: {job_id}")
        return
    sp = _status_paths(job_dir)

    try:
        print(f">>> worker start: {job_id}")
        _log_run_fingerprint(f"job_start:{job_id}")
        prev = _json_read_or_none(sp["status"]) or {}
        exclude_from_improvement = bool(prev.get("exclude_from_improvement"))
        print(f">>> DIAG worker prev keys: {sorted(list(prev.keys()))}")
        print(f">>> DIAG worker prev.owner_id: {str(prev.get('owner_id') or '').strip().lower()!r}")

        noticed_ts_ms = prev.get("noticed_ts_ms")
        _prev_carry = {
            k: v for k, v in prev.items()
            if k not in ("state", "ts", "noticed_ts_ms", "progress")
        }

        # Respect early cancel
        if _is_canceled(job_dir):
            print(f">>> worker canceled before start: {job_id}")
            _finalize_canceled_job(job_dir, job_id, noticed_ts_ms=noticed_ts_ms, step="canceled_before_start")
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
            print(f">>> worker canceled after render: {job_id}")
            _finalize_canceled_job(job_dir, job_id, noticed_ts_ms=noticed_ts_ms, step="canceled_after_render")
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
                print(f">>> worker canceled mid-parse: {job_id}")
                _finalize_canceled_job(job_dir, job_id, noticed_ts_ms=noticed_ts_ms, step="canceled_mid_parse")
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
            print(f">>> worker canceled before rules: {job_id}")
            _finalize_canceled_job(job_dir, job_id, noticed_ts_ms=noticed_ts_ms, step="canceled_before_rules")
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

        rules_payload = _build_rules_payload(ui_defaults, _rules_safe_components(components))
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
            "saved_pdf": "" if exclude_from_improvement else first_pdf,
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
            "exclude_from_improvement": exclude_from_improvement,
            "retention_mode": "excluded_24h" if exclude_from_improvement else "standard",
            "delete_after_utc": prev.get("delete_after_utc"),
        }

        _result_write(job_dir, result)
        _status_write(
            job_dir,
            "done",
            result_path=str(_status_paths(job_dir)["result"]),
            image_count=len(imgs),
            component_count=len(components),
            progress=100.0,
            step="done",
            cycle_time_ms=cycle_time_ms,
            cycle_time_str=cycle_time_str,
            exclude_from_improvement=exclude_from_improvement,
            retention_mode="excluded_24h" if exclude_from_improvement else "standard",
            delete_after_utc=prev.get("delete_after_utc"),
            raw_pdf_deleted=exclude_from_improvement,
            file_path="" if exclude_from_improvement else prev.get("file_path", ""),
        )
        _jobs_upsert(job_id, state="done", updated_at=_now_utc(), result_json=result)
        print(f">>> worker done: {job_id}")

        # ---- AUTO CLEANUP (keep only what UI uses) ----
        try:
            if exclude_from_improvement:
                _write_excluded_job_marker(job_dir)

            keep = _collect_keep_relpaths(job_dir, keep_pdf=(not exclude_from_improvement))
            _cleanup_job_dir(job_dir, keep)

            print(
                f">>> cleanup complete: kept {len(keep)} files | "
                f"exclude_from_improvement={exclude_from_improvement}"
            )
        except Exception as ce:
            print(f">>> cleanup failed: {ce}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f">>> worker error [{job_id}]: {e}\n{tb}")

        try:
            prev = _json_read_or_none(sp["status"]) or {}
            exclude_from_improvement = bool(prev.get("exclude_from_improvement"))
        except Exception:
            exclude_from_improvement = False

        _status_write(
            job_dir,
            "error",
            error=f"{type(e).__name__}: {e}",
            raw_pdf_deleted=exclude_from_improvement,
            file_path="" if exclude_from_improvement else None,
        )

        if exclude_from_improvement:
            try:
                _write_excluded_job_marker(job_dir)
                _cleanup_job_dir(job_dir, {"status.json", EXCLUDED_JOB_MARKER_FILENAME})
            except Exception as ce:
                print(f">>> excluded error cleanup failed: {ce}")

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

            job_dir = _resolve_job_dir_any(job_id)
            status = {}

            if job_dir is not None:
                status = (
                    _json_read_or_none(
                        _status_paths(job_dir)["status"]
                    )
                    or {}
                )

            job_type = str(
                status.get("job_type") or "detection"
            ).strip().lower()

            if job_type == "compatibility":
                _process_compatibility_job(
                    job_id,
                    pipeline,
                )
            else:
                _process_job(
                    job_id,
                    pipeline,
                )

            done_q.put(("done", job_id, None))
        except Exception as e:
            tb = traceback.format_exc()
            print(f">>> Worker [{tag}] job error [{job_id}]: {e}\n{tb}")
            try:
                job_dir = _resolve_job_dir_any(job_id)
                if job_dir is not None:
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
    ``job_q``, and waits on its own slot's ``done_q``. No cross-slot
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

        job_dir = _resolve_job_dir_any(job_id)

        if job_dir is None or not job_dir.exists():
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
                job_dir,
                "error",
                error=msg,
                queue_timeout=True,
                queue_timeout_min=QUEUE_TIMEOUT_MIN,
                queue_age_ms=age_ms,
                queue_age_str=age_str,
                progress=0.0,
            )

            if bool(st.get("exclude_from_improvement")):
                try:
                    _write_excluded_job_marker(job_dir)
                    _cleanup_job_dir(job_dir, {"status.json", EXCLUDED_JOB_MARKER_FILENAME})
                except Exception as ce:
                    print(f">>> excluded queue-timeout cleanup failed: {ce}")

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
                    job_dir,
                    "error",
                    error=msg,
                    step="queue_timeout",
                    queue_timeout=True,
                    queue_timeout_min=QUEUE_TIMEOUT_MIN,
                    queue_age_ms=age_ms,
                    queue_age_str=age_str,
                    progress=0.0,
                )

                if bool(st.get("exclude_from_improvement")):
                    try:
                        _write_excluded_job_marker(job_dir)
                        _cleanup_job_dir(job_dir, {"status.json", EXCLUDED_JOB_MARKER_FILENAME})
                    except Exception as ce:
                        print(f">>> excluded queue-timeout cleanup failed: {ce}")

                _jobs_upsert(job_id, state="error", updated_at=_now_utc(), error=msg)
                print(f">>> queue timeout (pre-start): {job_id} | age={age_str}")
                continue

            with slot.lock:
                _ensure_worker_alive(idx)

            _status_write(
                job_dir,
                "running",
                step="worker_dispatch",
                progress=float(st.get("progress") or 0.0),
                owner_email=owner_id,
                owner_id=owner_id,
                worker_slot=idx,
                dispatched_at_utc=_now_utc().isoformat(),
            )

            slot.job_q.put(job_id)

            # Wait for completion, but watch for real activity every second.
            # Compatibility tests use a much shorter hard timeout
            # than full processing jobs.
            job_type = str(
                st.get("job_type")
                or "detection"
            ).strip().lower()

            is_compatibility_job = (
                job_type == "compatibility"
                or job_id.startswith(
                    "compatibility__"
                )
            )

            if is_compatibility_job:
                timeout_sec = max(
                    1,
                    int(COMPATIBILITY_TIMEOUT_SEC)
                )
            else:
                timeout_sec = max(
                    1,
                    int(WATCHDOG_TIMEOUT_MIN) * 60
                )

            started_wait = time.time()
            last_signal_at = time.time()
            last_sig = _job_activity_signature(job_dir)

            worker_ok = True
            worker_crashed = False
            status = None
            done_job_id = job_id
            err_msg = None

            while True:
                try:
                    status, done_job_id, err_msg = slot.done_q.get(timeout=1.0)
                    break

                except Empty:
                    now = time.time()

                    # Worker process died/crashed.
                    if slot.proc is not None and not slot.proc.is_alive():
                        worker_ok = False
                        worker_crashed = True
                        err_msg = "Worker process crashed during this job. Please try again."
                        break

                    # Hard cap: compatibility tests stop after
                    # 60 seconds; full jobs retain their normal watchdog.
                    if now - started_wait >= timeout_sec:
                        worker_ok = False
                        worker_crashed = False

                        if is_compatibility_job:
                            err_msg = (
                                COMPATIBILITY_TIMEOUT_ERROR_MSG
                            )
                        else:
                            err_msg = (
                                WATCHDOG_ERROR_MSG.format(
                                    mins=WATCHDOG_TIMEOUT_MIN
                                )
                            )

                        break

                    # External cancel.
                    if _is_canceled(job_dir):
                        worker_ok = False
                        worker_crashed = False
                        err_msg = "Job was canceled."
                        break

                    # Activity check.
                    cur_sig = _job_activity_signature(job_dir)

                    if cur_sig != last_sig:
                        last_sig = cur_sig
                        last_signal_at = now

                        # If image files are appearing, expose that as live progress.
                        try:
                            png_count = int(cur_sig[-2] or 0)
                            _write_live_finding_components_heartbeat(job_dir, owner_id, png_count)
                        except Exception:
                            pass

                        continue

                    # Soft cap: no status movement and no new image files.
                    if now - last_signal_at >= NO_SIGNAL_TIMEOUT_SEC:
                        worker_ok = False
                        worker_crashed = False
                        err_msg = (
                            f"No processing activity for {NO_SIGNAL_TIMEOUT_SEC} seconds. "
                            "Restarting this job."
                        )
                        break

            if not worker_ok:
                # If the job merely stopped showing activity, kill/requeue once instead
                # of making the customer manually retry.
                no_signal = str(
                    err_msg or ""
                ).startswith(
                    "No processing activity"
                )

                # Full jobs may be retried after a no-signal
                # worker restart. Quick compatibility tests
                # should stop instead of starting over.
                if (
                    no_signal
                    and not is_compatibility_job
                ):
                    print(f">>> no-signal timeout [{tag}]: {job_id} | {err_msg}")

                    with slot.lock:
                        _kill_persistent_worker(idx)
                        _spawn_persistent_worker(idx)

                    st_latest = _json_read_or_none(_status_paths(job_dir)["status"]) or st

                    if _requeue_existing_job(job_dir, st_latest, "worker_no_signal_timeout"):
                        continue

                    # If requeue was blocked by max attempts or missing PDF,
                    # _requeue_existing_job/_mark_job_interrupted already wrote status.
                    continue

                # Hard timeout / crash / cancel path.
                try:
                    with open(_cancel_path(job_dir), "w") as f:
                        f.write("1")
                except Exception:
                    pass

                if worker_crashed:
                    msg = "Worker process crashed during this job. Please try again."
                    print(f">>> worker crash [{tag}] during: {job_id}")
                else:
                    msg = err_msg or WATCHDOG_ERROR_MSG.format(mins=WATCHDOG_TIMEOUT_MIN)
                    print(f">>> watchdog/cancel [{tag}]: {job_id} | {msg}")

                latest_status = _json_read_or_none(_status_paths(job_dir)["status"]) or {}

                _status_write(
                    job_dir,
                    "error",
                    step="timeout_or_worker_error",
                    error=msg,
                    progress=_safe_float(latest_status.get("progress"), 0.0),
                    owner_email=owner_id,
                    owner_id=owner_id,
                )

                try:
                    st_after_error = _json_read_or_none(_status_paths(job_dir)["status"]) or {}
                    if bool(st_after_error.get("exclude_from_improvement")):
                        _cleanup_job_dir(job_dir, {"status.json", EXCLUDED_JOB_MARKER_FILENAME})
                    else:
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
    job_dir = _resolve_specs_job_dir_any(job_id)
    if job_dir is None:
        print(f">>> specs analysis could not resolve job_id: {job_id}")
        with _SPECS_LOCK:
            _SPECS_RUNNING.pop(job_id, None)
        return

    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}

    group_folder = _safe_folder_key(st.get("group_folder") or "personal", "personal")
    exclude_from_improvement = bool(st.get("exclude_from_improvement"))

    owner_email = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
    saved_pdf = Path(st.get("file_path") or "").resolve()
    specs_delete_after_utc = _excluded_specs_delete_after_utc() if exclude_from_improvement else None
    
    try:
        if _is_canceled(job_dir):
            shutil.rmtree(
                job_dir,
                ignore_errors=True,
            )
            return

        _status_write(
            job_dir,
            "running",
            group_folder=group_folder,
            created_at=st.get("created_at"),
            file_path=str(saved_pdf),
            job_dir_path=str(job_dir),
            owner_email=owner_email,
            owner_id=owner_email,
            node_id=NODE_ID,
            step="specs_analyzing",
            progress=15.0,
            exclude_from_improvement=exclude_from_improvement,
            retention_mode=("excluded_specs_short" if exclude_from_improvement else "standard"),
            delete_after_utc=specs_delete_after_utc,
            raw_pdf_deleted=False,
            job_type="specs_analysis",
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

        if _is_canceled(job_dir):
            shutil.rmtree(
                job_dir,
                ignore_errors=True,
            )

            print(
                f">>> canceled Specs job removed: {job_id}"
            )
            return

        result = dict(result)
        result["job_id"] = job_id
        result["job_dir"] = str(job_dir)
        result["saved_pdf"] = "" if exclude_from_improvement else str(saved_pdf)
        result["exclude_from_improvement"] = exclude_from_improvement
        result["retention_mode"] = "excluded_specs_short" if exclude_from_improvement else "standard"
        result["delete_after_utc"] = specs_delete_after_utc
        result["owner_email"] = owner_email
        result["owner_id"] = owner_email
        result["group_folder"] = group_folder
        result["job_type"] = "specs_analysis"

        _result_write(job_dir, result)

        _status_write(
            job_dir,
            "done",
            group_folder=group_folder,
            created_at=st.get("created_at"),
            file_path=("" if exclude_from_improvement else str(saved_pdf)),
            job_dir_path=str(job_dir),
            result_path=str(sp["result"]),
            owner_email=owner_email,
            owner_id=owner_email,
            node_id=NODE_ID,
            step="specs_complete",
            progress=100.0,
            exclude_from_improvement=exclude_from_improvement,
            retention_mode=("excluded_specs_short" if exclude_from_improvement else "standard"),
            delete_after_utc=specs_delete_after_utc,
            raw_pdf_deleted=exclude_from_improvement,
            job_type="specs_analysis",
        )

        if exclude_from_improvement:
            try:
                _write_excluded_job_marker(job_dir)
                _cleanup_excluded_specs_job_dir(job_dir)
            except Exception as ce:
                print(f">>> excluded specs cleanup failed: {ce}")

        print(f">>> specs analysis done: {job_id}")

    except Exception as e:
        if _is_canceled(job_dir):
            shutil.rmtree(
                job_dir,
                ignore_errors=True,
            )
            return

        tb = traceback.format_exc()
        print(f">>> specs analysis error [{job_id}]: {e}\n{tb}")

        _status_write(
            job_dir,
            "error",
            group_folder=group_folder,
            created_at=st.get("created_at"),
            file_path=("" if exclude_from_improvement else str(saved_pdf)),
            job_dir_path=str(job_dir),
            owner_email=owner_email,
            owner_id=owner_email,
            node_id=NODE_ID,
            step="specs_error",
            error=f"{type(e).__name__}: {e}",
            traceback=tb,
            progress=100.0,
            exclude_from_improvement=exclude_from_improvement,
            retention_mode=("excluded_specs_short" if exclude_from_improvement else "standard"),
            delete_after_utc=(_excluded_specs_delete_after_utc() if exclude_from_improvement else None),
            raw_pdf_deleted=exclude_from_improvement,
            job_type="specs_analysis",
        )

        if exclude_from_improvement:
            try:
                _write_excluded_job_marker(job_dir)
                _cleanup_excluded_specs_job_dir(job_dir)
            except Exception as ce:
                print(f">>> excluded specs cleanup failed: {ce}")

    finally:
        with _SPECS_LOCK:
            _SPECS_RUNNING.pop(job_id, None)

# ---------- Start worker pool + cleanup loop ----------
if not _IS_WORKER_SUBPROCESS:
    try:
        for i in range(MAX_WORKERS):
            _spawn_persistent_worker(i)

        for i in range(MAX_WORKERS):
            t = threading.Thread(
                target=_dequeue_loop,
                args=(i,),
                daemon=True,
            )
            t.start()
            _WORKERS.append(t)

        print(
            f">>> Worker pool started: {MAX_WORKERS} slots, "
            f"{MAX_WORKERS} dequeue threads, "
            f"per-user cap={MAX_INFLIGHT_PER_USER}"
        )

        # Important:
        # If the uplink process restarted, status.json files survived
        # but _JOB_Q did not. Requeue stranded jobs.
        _recover_orphaned_detection_jobs_on_startup()

    except Exception as e:
        print(
            f">>> Worker pool startup failed: {e}"
        )
        print(
            traceback.format_exc()
        )


# --------------------------------------------------
# Background retention cleanup
#
# This runs maintenance separately from customer
# requests so status/result calls stay lightweight.
# --------------------------------------------------
if not _IS_WORKER_SUBPROCESS:
    try:
        _retention_thread = threading.Thread(
            target=_excluded_cleanup_loop,
            name="job-retention-cleanup",
            daemon=True,
        )

        _retention_thread.start()

        print(
            f">>> Retention cleanup thread started | "
            f"interval={CLEANUP_SWEEP_INTERVAL_SEC}s"
        )

    except Exception as e:
        print(
            f">>> Could not start retention cleanup thread: "
            f"{type(e).__name__}: {e}"
        )


def _active_job_for_owner(owner_email: str, group_folder: str = "personal") -> dict | None:
    """
    Return the user's active queued/running job, if any.
    Excludes canceled/done/error jobs and deletes expired excluded jobs opportunistically.
    """
    owner_email = str(owner_email or "").strip().lower()
    if not owner_email:
        return None

    try:
        if not BASE_JOBS_DIR.is_dir():
            return None

        for job_dir in _iter_job_dirs_for_owner(owner_email, group_folder):
            if not job_dir.is_dir():
                continue

            sp = _status_paths(job_dir)
            st = _json_read_or_none(sp["status"]) or {}
            if not isinstance(st, dict) or not st:
                continue

            if _is_excluded_job_expired(st):
                try:
                    shutil.rmtree(job_dir, ignore_errors=True)
                except Exception:
                    pass
                continue

            job_owner = str(
                st.get("owner_email")
                or st.get("owner_id")
                or ""
            ).strip().lower()

            if job_owner != owner_email:
                continue

            job_type = str(
                st.get("job_type") or ""
            ).strip().lower()

            # Temporary compatibility tests must
            # never block a real/new submission.
            if (
                job_type == "compatibility"
                or job_dir.name.startswith(
                    "compatibility__"
                )
            ):
                continue

            state = str(st.get("state") or "").strip().lower()
            canceled = bool(st.get("canceled"))

            if canceled:
                continue

            # Repair stale active jobs before letting them block a new submission.
            # This prevents "active_job_exists" from being caused by a dead queued/running status.
            if state in ("queued", "running", "unknown"):
                st = _repair_stale_active_job_from_status_poll(job_dir, st) or st
                state = str(st.get("state") or "").strip().lower()

            if state in ("queued", "running", "unknown"):
                meta = _parse_job_note(st.get("job_note") or "")
                return {
                    "job_id": job_dir.name,
                    "job_name": meta.get("job_name") or job_dir.name,
                    "state": state or "unknown",
                    "step": st.get("step"),
                    "progress": st.get("progress"),
                    "created_at": st.get("created_at"),
                }

    except Exception as e:
        print(f">>> active job check failed for {owner_email}: {e}")

    return None

# ---------- API: submit / status / list / cancel ----------
@anvil.server.callable
def vm_ping():
    print(f">>> vm_ping called | NODE_ID={NODE_ID}")
    return {"ok": True, "node_id": NODE_ID}

@anvil.server.callable
def vm_get_active_job_for_user(owner_email: str, group_folder: str = "personal") -> dict:
    owner_email = str(owner_email or "").strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")

    if not owner_email:
        return {"ok": False, "active": False}

    active = _active_job_for_owner(owner_email, group_folder)
    if active:
        return {
            "ok": True,
            "active": True,
            "job": active
        }

    return {
        "ok": True,
        "active": False
    }

@anvil.server.callable
def vm_clear_temporary_analysis_jobs(
    owner_email: str,
    group_folder: str = "personal",
) -> dict:
    """
    Cancel/remove only temporary Compatibility
    and Specs jobs for one user.

    Normal BOM jobs are never touched.
    """
    owner_email = str(
        owner_email or ""
    ).strip().lower()

    group_folder = _safe_folder_key(
        group_folder or "personal",
        "personal",
    )

    if not owner_email:
        return {
            "ok": False,
            "error": "Missing owner email.",
        }

    removed = 0
    cancel_requested = 0

    # Compatibility jobs use the normal grouped
    # drawing-job location.
    for job_dir in list(
        _iter_job_dirs_for_owner(
            owner_email,
            group_folder,
        )
    ):
        try:
            status = _json_read_or_none(
                _status_paths(job_dir)["status"]
            ) or {}

            job_type = str(
                status.get("job_type") or ""
            ).strip().lower()

            if (
                job_type != "compatibility"
                and not job_dir.name.startswith(
                    "compatibility__"
                )
            ):
                continue

            state = str(
                status.get("state") or ""
            ).strip().lower()

            if state == "running":
                with open(
                    _cancel_path(job_dir),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("1")

                _status_write(
                    job_dir,
                    "canceled",
                    canceled=True,
                    step="compatibility_cancel_requested",
                )

                cancel_requested += 1

            else:
                shutil.rmtree(
                    job_dir,
                    ignore_errors=True,
                )

                removed += 1

        except Exception as e:
            print(
                ">>> compatibility temp cleanup "
                f"failed for {job_dir}: {e}"
            )

    # Specs jobs use their own grouped location.
    specs_root = _user_specs_root(
        owner_email,
        group_folder,
    )

    try:
        if specs_root.is_dir():
            for job_dir in list(
                specs_root.iterdir()
            ):
                if (
                    not job_dir.is_dir()
                    or not job_dir.name.startswith(
                        "specs_"
                    )
                ):
                    continue

                with _SPECS_LOCK:
                    is_running = bool(
                        _SPECS_RUNNING.get(
                            job_dir.name
                        )
                    )

                if is_running:
                    with open(
                        _cancel_path(job_dir),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write("1")

                    _status_write(
                        job_dir,
                        "canceled",
                        canceled=True,
                        step="specs_cancel_requested",
                    )

                    cancel_requested += 1

                else:
                    shutil.rmtree(
                        job_dir,
                        ignore_errors=True,
                    )

                    removed += 1

    except Exception as e:
        print(
            ">>> specs temp cleanup failed: "
            f"{type(e).__name__}: {e}"
        )

    return {
        "ok": True,
        "removed": removed,
        "cancel_requested": cancel_requested,
    }



@anvil.server.callable
def vm_submit_compatibility_test(
    media,
    owner_email=None,
    group_folder="personal",
    access_code_value="",
    plan_key="",
    company_name="",
):
    """
    Save a temporary PDF and queue a compatibility-only job.
    """
    if not owner_email or not str(
        owner_email
    ).strip():
        raise RuntimeError(
            "owner_email required"
        )

    owner_email = str(
        owner_email
    ).strip().lower()

    group_folder = _safe_folder_key(
        group_folder or "personal",
        "personal",
    )

    access_code_value = str(
        access_code_value or ""
    ).strip()

    plan_key = str(
        plan_key or ""
    ).strip().lower()

    company_name = str(
        company_name or ""
    ).strip()

    _cleanup_all_retention_policies()

    with _SUBMIT_LOCK:
        active = _active_job_for_owner(
            owner_email,
            group_folder,
        )

        if active:
            return {
                "ok": False,
                "state": "active_job_exists",
                "error": (
                    "You already have a job processing. "
                    "Please wait for it to finish before "
                    "running a compatibility test."
                ),
                "active_job": active,
            }

        job_dir = (
            _make_compatibility_job_dir(
                owner_email=owner_email,
                group_folder=group_folder,
            )
        )

        job_id = job_dir.name

        pdf_dir = (
            job_dir / "uploaded_pdfs"
        )

        saved_pdf = _save_media_to_disk(
            media,
            pdf_dir,
        )

        noticed_ts_ms = _epoch_ms()

        delete_after_utc = _utc_iso_z(
            datetime.now(timezone.utc)
            + timedelta(
                minutes=(
                    COMPATIBILITY_TEMP_RETENTION_MINUTES
                )
            )
        )

        _status_write(
            job_dir,
            "queued",

            job_type="compatibility",

            group_folder=group_folder,
            owner_folder=_owner_folder_key(
                owner_email
            ),

            access_code_value=(
                access_code_value
            ),
            plan_key=plan_key,
            company_name=company_name,

            created_at=(
                _now_utc().isoformat()
            ),

            file_path=str(saved_pdf),
            job_dir_path=str(job_dir),

            original_filename=str(
                getattr(
                    media,
                    "name",
                    "",
                )
                or "uploaded.pdf"
            ),

            noticed_ts_ms=noticed_ts_ms,

            owner_email=owner_email,
            owner_id=owner_email,

            node_id=NODE_ID,

            step="compatibility_received",
            progress=0.0,
            canceled=False,

            # Reuse the excluded-job cleanup system
            # as a fallback if the browser disappears.
            exclude_from_improvement=True,
            retention_mode=(
                "compatibility_temp"
            ),
            delete_after_utc=(
                delete_after_utc
            ),
            raw_pdf_deleted=False,
        )

        _enqueue_job(
            job_id,
            owner_email,
        )

        print(
            f">>> compatibility job queued: "
            f"{job_id}"
        )

        return {
            "ok": True,
            "job_id": job_id,
            "state": "queued",
            "job_type": "compatibility",
            "node_id": NODE_ID,
            "delete_after_utc": (
                delete_after_utc
            ),
        }

@anvil.server.callable
def vm_get_compatibility_status(
    job_id: str,
    owner_email: str,
    group_folder: str = "personal",
) -> dict:
    """
    Return compatibility progress or the finished report.
    """
    job_id = str(job_id or "").strip()

    owner_email = str(
        owner_email or ""
    ).strip().lower()

    group_folder = _safe_folder_key(
        group_folder or "personal",
        "personal",
    )

    if (
        not job_id
        or not owner_email
        or not job_id.startswith(
            "compatibility__"
        )
        or "/" in job_id
        or "\\" in job_id
        or ".." in job_id
    ):
        return {
            "state": "not_found",
            "error": (
                "Compatibility test not found."
            ),
        }

    job_dir = _resolve_job_dir_for_owner(
        job_id,
        owner_email,
        group_folder,
    )

    if job_dir is None:
        return {
            "state": "not_found",
            "error": (
                "Compatibility test not found."
            ),
        }

    sp = _status_paths(job_dir)

    status = _json_read_or_none(
        sp["status"]
    ) or {}

    if not isinstance(status, dict) or not status:
        return {
            "state": "not_found",
            "error": (
                "Compatibility test not found."
            ),
        }

    job_owner = str(
        status.get("owner_email")
        or status.get("owner_id")
        or ""
    ).strip().lower()

    job_type = str(
        status.get("job_type") or ""
    ).strip().lower()

    if (
        job_owner != owner_email
        or job_type != "compatibility"
    ):
        return {
            "state": "not_found",
            "error": (
                "Compatibility test not found."
            ),
        }

    if _is_excluded_job_expired(status):
        try:
            shutil.rmtree(
                job_dir,
                ignore_errors=True,
            )
        except Exception:
            pass

        return {
            "state": "not_found",
            "error": (
                "Compatibility test expired."
            ),
        }

    state = str(
        status.get("state") or "unknown"
    ).strip().lower()

    # Use the same stale-job protection as
    # normal drawing jobs.
    if state in {
        "queued",
        "running",
        "unknown",
    }:
        status = (
            _repair_stale_active_job_from_status_poll(
                job_dir,
                status,
            )
            or status
        )

        state = str(
            status.get("state") or "unknown"
        ).strip().lower()

    output = {
        "state": state,
        "job_id": job_id,
    }

    for key in (
        "step",
        "progress",
        "kept_pages",
        "dropped_pages",
        "compatibility_attempt",
        "compatibility_max_attempts",
        "cycle_time_ms",
        "cycle_time_str",
        "delete_after_utc",
    ):
        if key in status:
            output[key] = status[key]

    if state == "done":
        result = _json_read_or_none(
            sp["result"]
        ) or {}

        if not isinstance(result, dict) or not result:
            return {
                "state": "error",
                "error": (
                    "Compatibility result could "
                    "not be loaded."
                ),
            }

        output["progress"] = 100.0
        image_path_text = str(
            result.get(
                "compatibility_image_path"
            )
            or ""
        ).strip()

        if image_path_text:
            try:
                image_path = Path(
                    image_path_text
                ).resolve()

                # Security check: only return an image
                # that belongs to this temporary job.
                resolved_job_dir = Path(
                    job_dir
                ).resolve()

                if (
                    image_path.is_file()
                    and resolved_job_dir
                    in image_path.parents
                ):
                    suffix = (
                        image_path.suffix.lower()
                    )

                    if suffix in {
                        ".jpg",
                        ".jpeg"
                    }:
                        content_type = "image/jpeg"
                    else:
                        content_type = "image/png"

                    result[
                        "compatibility_image"
                    ] = BlobMedia(
                        content_type,
                        image_path.read_bytes(),
                        name=(
                            "compatibility-panel"
                            + suffix
                        )
                    )

            except Exception as image_error:
                print(
                    ">>> compatibility preview "
                    "could not be loaded: "
                    f"{type(image_error).__name__}: "
                    f"{image_error}"
                )
        output["result"] = result

        return output

    if state == "error":
        output["error"] = (
            status.get("error")
            or "Compatibility test failed."
        )

        return output

    if state in {"canceled", "cancelled"}:
        output["error"] = (
            "Compatibility test was canceled."
        )

        return output

    return output


@anvil.server.callable
def vm_delete_compatibility_job(
    job_id: str,
    owner_email: str,
    group_folder: str = "personal",
) -> bool:
    """
    Delete the entire temporary compatibility folder.

    This removes:
      - uploaded PDF
      - filtered PDF
      - panel crops
      - debug files
      - result.json
      - status.json
    """
    job_id = str(job_id or "").strip()

    owner_email = str(
        owner_email or ""
    ).strip().lower()

    group_folder = _safe_folder_key(
        group_folder or "personal",
        "personal",
    )

    if (
        not job_id
        or not owner_email
        or not job_id.startswith(
            "compatibility__"
        )
        or "/" in job_id
        or "\\" in job_id
        or ".." in job_id
    ):
        return False

    job_dir = _resolve_job_dir_for_owner(
        job_id,
        owner_email,
        group_folder,
    )

    if job_dir is None:
        return False

    status = _json_read_or_none(
        _status_paths(job_dir)["status"]
    ) or {}

    job_owner = str(
        status.get("owner_email")
        or status.get("owner_id")
        or ""
    ).strip().lower()

    job_type = str(
        status.get("job_type") or ""
    ).strip().lower()

    if (
        job_owner != owner_email
        or job_type != "compatibility"
    ):
        return False

    state = str(
        status.get("state") or ""
    ).strip().lower()

    # Never delete files while a worker could
    # still be using them.
    if state in {
        "uploaded",
        "queued",
        "running",
        "unknown",
    }:
        return False

    try:
        shutil.rmtree(
            job_dir,
            ignore_errors=False,
        )

        print(
            f">>> deleted compatibility "
            f"job folder: {job_id}"
        )

        return True

    except FileNotFoundError:
        return True

    except Exception as e:
        print(
            f">>> failed deleting compatibility "
            f"job {job_id}: "
            f"{type(e).__name__}: {e}"
        )

        return False

@anvil.server.callable
def vm_submit_for_detection(
    media,
    ui_overrides=None,
    job_note=None,
    owner_email=None,
    group_folder="personal",
    access_code_value="",
    plan_key="",
    company_name="",
    exclude_from_improvement=False
):
    """
    Create job folder, save PDF, record 'noticed' time, render images, persist normalized overrides, enqueue worker.
    Ownership is always the lowercased email address.
    """
    if not owner_email or not str(owner_email).strip():
        raise RuntimeError("owner_email required")

    owner_email = str(owner_email).strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")
    access_code_value = str(access_code_value or "").strip()
    plan_key = str(plan_key or "").strip().lower()
    company_name = str(company_name or "").strip()

    exclude_from_improvement = bool(exclude_from_improvement)

    # Opportunistic retention cleanup
    _cleanup_all_retention_policies()

    with _SUBMIT_LOCK:
        active = _active_job_for_owner(owner_email, group_folder)
        if active:
            return {
                "ok": False,
                "state": "active_job_exists",
                "error": "You already have a job processing. Please wait for it to finish or cancel it from My Jobs.",
                "active_job": active,
            }

        # Make room for this new standard job before creating it.
        # Excluded jobs follow the separate 24-hour retention policy.
        if not exclude_from_improvement:
            _cleanup_standard_retention_for_owner(
                owner_email,
                group_folder,
                reserve_slots=1
            )

        original_name = getattr(media, "name", "uploaded.pdf")
        job_dir = _make_job_dir(job_note, original_name, owner_email=owner_email, group_folder=group_folder)
        job_id = job_dir.name
        print(f">>> vm_submit_for_detection: job_dir={job_dir}, owner_email={owner_email!r}")

        # 1) Save the uploaded PDF
        pdf_dir = job_dir / "uploaded_pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        saved_pdf = _save_media_to_disk(media, pdf_dir)
        print(f">>> saved_pdf={saved_pdf}")

        # 2) Normalize overrides
        normalized_overrides = _normalize_ui_overrides(ui_overrides if isinstance(ui_overrides, dict) else {})

        # 3) Record noticed time immediately
        noticed_ts_ms = _epoch_ms()

        # 4) Persist initial status BEFORE enqueueing
        _status_write(
            job_dir,
            "queued",
            group_folder=group_folder,
            owner_folder=_owner_folder_key(owner_email),
            access_code_value=access_code_value,
            plan_key=plan_key,
            company_name=company_name,
            created_at=_now_utc().isoformat(),
            file_path=str(saved_pdf),
            job_dir_path=str(job_dir),
            ui_overrides=normalized_overrides,
            job_note=(job_note or ""),
            image_count=0,
            step="received",
            noticed_ts_ms=noticed_ts_ms,
            owner_email=owner_email,
            owner_id=owner_email,
            node_id=NODE_ID,
            canceled=False,
            progress=0.0,
            exclude_from_improvement=exclude_from_improvement,
            retention_mode=("excluded_24h" if exclude_from_improvement else "standard"),
            delete_after_utc=(_excluded_delete_after_utc() if exclude_from_improvement else None),
            raw_pdf_deleted=False
        )

        if exclude_from_improvement:
            _write_excluded_job_marker(job_dir)

        _enqueue_job(job_id, owner_email)

        return {
            "ok": True,
            "job_id": job_id,
            "state": "queued",
            "node_id": NODE_ID,
            "group_folder": group_folder,
            "exclude_from_improvement": exclude_from_improvement,
        }

@anvil.server.callable
def vm_upload_specs_pdf(
    media,
    owner_email=None,
    group_folder="personal",
    access_code_value="",
    plan_key="",
    company_name="",
    job_name="",
    exclude_from_improvement=False
):
    """
    Upload/save specs PDF into:
      /jobs/specs/groups/<group>/users/<owner>/<specs_job_id>

    Excluded specs:
      - raw PDF is deleted after analysis
      - generated artifacts remain only until the next short cleanup window
    """
    if not owner_email or not str(owner_email).strip():
        raise RuntimeError("owner_email required")

    owner_email = str(owner_email).strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")
    access_code_value = str(access_code_value or "").strip()
    plan_key = str(plan_key or "").strip().lower()
    company_name = str(company_name or "").strip()
    exclude_from_improvement = bool(exclude_from_improvement)

    job_dir = _make_specs_job_dir(
        media=media,
        owner_email=owner_email,
        group_folder=group_folder,
        job_name=job_name
    )
    job_id = job_dir.name

    pdf_dir = job_dir / "uploaded_pdfs"
    saved_pdf = _save_media_to_disk(media, pdf_dir)

    _status_write(
        job_dir,
        "uploaded",
        group_folder=group_folder,
        owner_folder=_owner_folder_key(owner_email),
        access_code_value=access_code_value,
        plan_key=plan_key,
        company_name=company_name,
        created_at=_now_utc().isoformat(),
        file_path=str(saved_pdf),
        job_dir_path=str(job_dir),
        owner_email=owner_email,
        owner_id=owner_email,
        node_id=NODE_ID,
        step="specs_uploaded",
        progress=5.0,
        exclude_from_improvement=exclude_from_improvement,
        retention_mode=("excluded_specs_short" if exclude_from_improvement else "standard"),
        delete_after_utc=None,
        raw_pdf_deleted=False,
        job_type="specs_analysis"
    )

    if exclude_from_improvement:
        _write_excluded_job_marker(job_dir)

    return {
        "ok": True,
        "job_id": job_id,
        "job_dir": str(job_dir),
        "saved_pdf": str(saved_pdf),
        "owner_email": owner_email,
        "owner_id": owner_email,
        "node_id": NODE_ID,
        "state": "uploaded",
        "group_folder": group_folder,
        "exclude_from_improvement": exclude_from_improvement
    }

@anvil.server.callable
def vm_start_specs_analysis(job_id: str, owner_email: str, group_folder: str = "personal"):
    """
    Start analysis only after the specs PDF has already been uploaded/saved.
    """
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")

    if not job_id or not owner_email:
        return {
            "ok": False,
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return {
            "ok": False,
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    job_dir = _resolve_specs_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_dir is None:
        return {
            "ok": False,
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    st = _json_read_or_none(_status_paths(job_dir)["status"]) or {}
    if not st:
        return {
            "ok": False,
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    job_owner = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
    if not job_owner or job_owner != owner_email:
        return {
            "ok": False,
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    with _SPECS_LOCK:
        if _SPECS_RUNNING.get(job_id):
            return {"ok": True, "job_id": job_id, "state": "running"}

        _SPECS_RUNNING[job_id] = True

    t = threading.Thread(target=_run_specs_analysis_job, args=(job_id,), daemon=True)
    t.start()

    return {"ok": True, "job_id": job_id, "state": "running"}

@anvil.server.callable
def vm_get_specs_status(job_id: str, owner_email: str, group_folder: str = "personal") -> dict:
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")

    if not job_id or not owner_email:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    job_dir = _resolve_specs_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_dir is None:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"])

    if not isinstance(st, dict) or not st:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    if _is_excluded_job_expired(st):
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f">>> deleted expired excluded specs job on status check: {job_dir}")
        except Exception:
            pass

        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    req_email = owner_email
    job_email = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()

    if not req_email or not job_email or req_email != job_email:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    state = str(st.get("state") or "unknown").strip().lower()

    if state == "done":
        res = _json_read_or_none(sp["result"]) or {}
        return {
            "state": "done",
            "result": res
        }

    if state == "error":
        return {
            "state": "error",
            "error": st.get("error") or "Specs analysis failed. Please try again."
        }

    out = {"state": state}

    for k in (
        "step",
        "progress",
        "exclude_from_improvement",
        "retention_mode",
        "delete_after_utc",
        "raw_pdf_deleted",
        "group_folder",
    ):
        if k in st:
            out[k] = st[k]

    return out

@anvil.server.callable
def vm_delete_specs_job(job_id: str, owner_email: str, group_folder: str = "personal") -> bool:
    """
    Delete a specs-only temp job folder.
    Supports grouped specs storage and legacy flat specs jobs.
    """
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()
    group_folder = _safe_folder_key(group_folder or "personal", "personal")

    if not job_id or not owner_email:
        return False

    if not job_id.startswith("specs_"):
        return False

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return False

    job_dir = _resolve_specs_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_dir is None or not job_dir.exists() or not job_dir.is_dir():
        return False

    st = _json_read_or_none(_status_paths(job_dir)["status"]) or {}
    job_owner = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()

    if not job_owner or job_owner != owner_email:
        return False

    try:
        shutil.rmtree(job_dir, ignore_errors=False)
        print(f">>> deleted specs temp job folder: {job_dir}")
        return True
    except Exception as e:
        print(f">>> failed deleting specs temp job folder {job_dir}: {e}")
        return False

def _natural_key(p: Path):
    """Generate a natural sort key so 'page2' sorts before 'page10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

@anvil.server.callable
def vm_list_magenta_overlay_images(job_id: str, owner_email: str, group_folder: str = "personal") -> list[str]:
    """
    Returns job-relative PNG paths for FULL-PAGE magenta overlays.
    Enforces ownership.
    """
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()

    if not job_id or not owner_email:
        return []

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return []

    job_root = _resolve_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_root is None:
        return []

    sp = _status_paths(job_root)
    st = _json_read_or_none(sp["status"]) or {}

    if _is_excluded_job_expired(st):
        try:
            shutil.rmtree(job_root, ignore_errors=True)
            print(f">>> deleted expired excluded job on magenta overlay list: {job_root}")
        except Exception:
            pass
        return []

    job_owner = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
    if not job_owner or job_owner != owner_email:
        return []

    pdf_images = job_root / "pdf_images"
    if not pdf_images.is_dir():
        return []

    candidate_dirs = [
        pdf_images / "magenta_overlays",
        pdf_images / "magenta_overlay",
        pdf_images / "page_overlays",
        pdf_images / "full_overlays",
        pdf_images / "overlays",
    ]

    found = []
    for d in candidate_dirs:
        if d.is_dir():
            found.extend(list(d.glob("*.png")))

    if not found:
        for p in pdf_images.rglob("*.png"):
            if "review_overlays" in p.parts:
                continue
            if "magenta" in p.name.lower():
                found.append(p)

    uniq = {}
    for p in found:
        try:
            rp = p.resolve().relative_to(job_root)
        except Exception:
            continue
        uniq[str(rp).replace("\\", "/")] = p

    rel_paths = list(uniq.keys())
    rel_paths.sort(key=lambda s: _natural_key(Path(s)))
    return rel_paths

@anvil.server.callable
def vm_list_overlay_images(job_id: str, owner_email: str, group_folder: str = "personal") -> list[str]:
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()

    if not job_id or not owner_email:
        return []

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return []

    job_root = _resolve_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_root is None:
        return []

    sp = _status_paths(job_root)
    st = _json_read_or_none(sp["status"]) or {}

    if _is_excluded_job_expired(st):
        try:
            shutil.rmtree(job_root, ignore_errors=True)
            print(f">>> deleted expired excluded job on overlay list: {job_root}")
        except Exception:
            pass
        return []

    job_owner = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
    if not job_owner or job_owner != owner_email:
        return []

    overlay_dir = job_root / "pdf_images" / "review_overlays"
    if not overlay_dir.is_dir():
        return []

    out = []
    for p in sorted(overlay_dir.glob("*.png")):
        try:
            out.append(str(p.resolve().relative_to(job_root)).replace("\\", "/"))
        except Exception:
            pass

    return out

@anvil.server.callable
def vm_fetch_image(
    job_id: str,
    owner_email: str,
    source_path: str,
    group_folder: str = "personal",
):
    """
    Return a lightweight cached WEB PREVIEW of a job image.

    IMPORTANT:
      - Original detection/OCR image is NEVER modified.
      - Large source images are resized/compressed for browser display.
      - Generated web previews are cached on disk.
      - Subsequent requests reuse the cached preview.

    This prevents huge PNG files from being pushed through
    Anvil Uplink every time the output page needs an image.
    """
    import hashlib
    import cv2

    started = time.perf_counter()

    job_id = str(
        job_id or ""
    ).strip()

    owner_email = str(
        owner_email or ""
    ).strip().lower()

    raw = str(
        source_path or ""
    ).strip().replace("\\", "/")

    if not job_id or not owner_email or not raw:
        raise RuntimeError(
            "Image unavailable."
        )

    if (
        "/" in job_id
        or "\\" in job_id
        or ".." in job_id
    ):
        raise RuntimeError(
            "Image unavailable."
        )

    # Never accept absolute client paths.
    p_in = Path(raw)

    if p_in.is_absolute():
        raise RuntimeError(
            "Image unavailable."
        )

    if ".." in p_in.parts:
        raise RuntimeError(
            "Image unavailable."
        )

    # --------------------------------------------------
    # Resolve correct job folder
    # --------------------------------------------------
    if job_id.startswith("specs_"):
        job_root = (
            _resolve_specs_job_dir_for_owner(
                job_id,
                owner_email,
                group_folder,
            )
        )
    else:
        job_root = (
            _resolve_job_dir_for_owner(
                job_id,
                owner_email,
                group_folder,
            )
        )

    if job_root is None:
        raise RuntimeError(
            "Image unavailable."
        )

    sp = _status_paths(
        job_root
    )

    st = (
        _json_read_or_none(
            sp["status"]
        )
        or {}
    )

    if _is_excluded_job_expired(st):
        try:
            shutil.rmtree(
                job_root,
                ignore_errors=True,
            )

            print(
                f">>> deleted expired excluded "
                f"job on image fetch: {job_root}"
            )

        except Exception:
            pass

        raise FileNotFoundError(
            "Job not found. "
            "Please resubmit your PDF."
        )

    job_owner = str(
        st.get("owner_email")
        or st.get("owner_id")
        or ""
    ).strip().lower()

    if (
        not job_owner
        or job_owner != owner_email
    ):
        raise RuntimeError(
            "Image unavailable."
        )

    # --------------------------------------------------
    # Resolve requested image
    # --------------------------------------------------
    p = (
        job_root / p_in
    ).resolve()

    try:
        p.relative_to(
            job_root
        )

    except ValueError:
        raise RuntimeError(
            "Image unavailable."
        )

    if not p.is_file():
        raise RuntimeError(
            "Image unavailable."
        )

    if p.suffix.lower() not in (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    ):
        raise RuntimeError(
            "Image unavailable."
        )

    stat = p.stat()

    original_size = int(
        stat.st_size
    )

    # --------------------------------------------------
    # Small images can already travel directly.
    #
    # No point spending CPU recompressing a 300 KB image.
    # --------------------------------------------------
    DIRECT_LIMIT = (
        300_000
    )

    if original_size <= DIRECT_LIMIT:
        data = p.read_bytes()

        ctype_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }

        ctype = ctype_map.get(
            p.suffix.lower(),
            "application/octet-stream",
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f">>> IMAGE DIRECT | "
            f"{p.name} | "
            f"{original_size / 1024 / 1024:.2f} MB | "
            f"{elapsed:.3f}s"
        )

        return BlobMedia(
            ctype,
            data,
            name=p.name,
        )

    # --------------------------------------------------
    # Large image:
    # create/reuse web-friendly cached JPEG.
    # --------------------------------------------------
    MAX_DIMENSION = 1800
    JPEG_QUALITY = 80

    cache_dir = (
        Path(job_root)
        / ".web_previews"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_identity = (
        f"{raw}|"
        f"{stat.st_size}|"
        f"{stat.st_mtime_ns}|"
        f"{MAX_DIMENSION}|"
        f"{JPEG_QUALITY}"
    )

    cache_hash = hashlib.sha1(
        cache_identity.encode(
            "utf-8"
        )
    ).hexdigest()

    preview_path = (
        cache_dir
        / f"{cache_hash}.jpg"
    )

    # --------------------------------------------------
    # Cached preview already exists.
    # --------------------------------------------------
    if preview_path.is_file():
        preview_bytes = (
            preview_path.read_bytes()
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f">>> IMAGE CACHE HIT | "
            f"{p.name} | "
            f"original="
            f"{original_size / 1024 / 1024:.2f} MB | "
            f"preview="
            f"{len(preview_bytes) / 1024 / 1024:.2f} MB | "
            f"{elapsed:.3f}s"
        )

        return BlobMedia(
            "image/jpeg",
            preview_bytes,
            name=(
                f"{p.stem}_web.jpg"
            ),
        )

    # --------------------------------------------------
    # Build cached preview.
    # --------------------------------------------------
    decode_start = (
        time.perf_counter()
    )

    img = cv2.imread(
        str(p),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise RuntimeError(
            "Image could not be decoded."
        )

    decode_sec = (
        time.perf_counter()
        - decode_start
    )

    h, w = img.shape[:2]

    longest = max(
        int(w),
        int(h),
    )

    if longest > MAX_DIMENSION:
        scale = (
            MAX_DIMENSION
            / float(longest)
        )

        new_w = max(
            1,
            int(round(w * scale)),
        )

        new_h = max(
            1,
            int(round(h * scale)),
        )

        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

    else:
        new_w = int(w)
        new_h = int(h)

    encode_start = (
        time.perf_counter()
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        img,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            JPEG_QUALITY,
        ],
    )

    if not ok:
        raise RuntimeError(
            "Could not create web preview."
        )

    preview_bytes = (
        encoded.tobytes()
    )

    encode_sec = (
        time.perf_counter()
        - encode_start
    )

    # Persist for future page loads.
    try:
        preview_path.write_bytes(
            preview_bytes
        )
    except Exception as e:
        print(
            f">>> Could not cache web preview | "
            f"{type(e).__name__}: {e}"
        )

    total_sec = (
        time.perf_counter()
        - started
    )

    print(
        "\n"
        ">>> WEB IMAGE CREATED\n"
        f"    source: {p.name}\n"
        f"    original dimensions: "
        f"{w}x{h}\n"
        f"    preview dimensions: "
        f"{new_w}x{new_h}\n"
        f"    original size: "
        f"{original_size / 1024 / 1024:.2f} MB\n"
        f"    preview size: "
        f"{len(preview_bytes) / 1024 / 1024:.2f} MB\n"
        f"    decode: "
        f"{decode_sec:.3f}s\n"
        f"    encode: "
        f"{encode_sec:.3f}s\n"
        f"    VM total: "
        f"{total_sec:.3f}s\n"
    )

    return BlobMedia(
        "image/jpeg",
        preview_bytes,
        name=f"{p.stem}_web.jpg",
    )

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
  """RPC callable: return the current watchdog timeout in minutes."""
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
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()

    job_dir = _resolve_job_dir_any(job_id)
    if job_dir is None:
        return (None, 0)

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

    # Scan grouped + legacy jobs by looking for status.json files.
    candidates = []

    try:
        for d in BASE_JOBS_DIR.iterdir():
            if d.is_dir() and d.name != "groups":
                candidates.append(d)
    except Exception:
        pass

    groups_root = BASE_JOBS_DIR / "groups"
    try:
        if groups_root.is_dir():
            for status_path in groups_root.rglob("status.json"):
                try:
                    candidates.append(status_path.parent)
                except Exception:
                    pass
    except Exception:
        pass

    seen = set()

    for d in candidates:
        try:
            d = _assert_under_base(d)
        except Exception:
            continue

        key = str(d)
        if key in seen:
            continue
        seen.add(key)

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

    rows.sort(key=lambda r: (r["noticed_ts_ms"], r["job_id"]))

    active_count = len(rows)

    for idx, row in enumerate(rows):
        if row["job_id"] == job_id:
            return (idx, active_count)

    return (None, active_count)

@anvil.server.callable
def vm_get_job_result(job_id: str, owner_email: str, group_folder: str = "personal") -> dict:
    """
    Return saved result.json for a completed job.
    This should never return {} silently.
    """
    try:
        job_id = str(job_id or "").strip()
        req_email = str(owner_email or "").strip().lower()

        if not job_id:
            return {"ok": False, "error": "Missing job_id."}

        if "/" in job_id or "\\" in job_id or ".." in job_id:
            return {"ok": False, "error": "Job not found. Please resubmit your PDF."}

        job_dir = _resolve_job_dir_for_owner(job_id, req_email, group_folder)
        if job_dir is None:
            return {
                "ok": False,
                "error": "Job not found. Please resubmit your PDF."
            }

        sp = _status_paths(job_dir)

        status = _json_read_or_none(sp["status"]) or {}

        if not isinstance(status, dict) or not status:
            return {
                "ok": False,
                "error": "Job not found. Please resubmit your PDF."
            }
        
        if _is_excluded_job_expired(status):
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
                print(f">>> deleted expired excluded job on result fetch: {job_dir}")
            except Exception:
                pass

            return {
                "ok": False,
                "error": "Job not found. Please resubmit your PDF."
            }
        
        job_email = str(
            status.get("owner_email")
            or status.get("owner_id")
            or ""
        ).strip().lower()

        if not req_email or not job_email or req_email != job_email:
            return {
                "ok": False,
                "error": "Job not found. Please resubmit your PDF."
            }

        # Prefer explicit result_path from status.json if present.
        result_path_raw = str(status.get("result_path") or "").strip()
        if result_path_raw:
            result_path = Path(result_path_raw)
        else:
            result_path = sp["result"]

        result = _json_read_or_none(result_path) or {}

        if not isinstance(result, dict) or not result:
            return {
                "ok": False,
                "error": "Could not load saved result."
            }

        # Always wrap result for Anvil client.
        return {
            "ok": True,
            "result": result
        }

    except Exception as e:
        return {
            "ok": False,
            "error": f"Result fetch failed: {type(e).__name__}: {e}"
        }

@anvil.server.callable
def vm_get_job_status(job_id: str, owner_email: str, group_folder: str = "personal") -> dict:
    """Status primarily from disk; does not return final result. Enforces ownership by email."""
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()

    if not job_id or not owner_email:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    job_dir = _resolve_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_dir is None:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    sp = _status_paths(job_dir)

    st = _json_read_or_none(sp["status"])
    if _is_excluded_job_expired(st):
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            print(f">>> deleted expired excluded job on status check: {job_dir}")
        except Exception:
            pass

        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }
    if not isinstance(st, dict) or not st:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF."
        }

    req_email = owner_email
    job_email = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()

    # Optional node hint
    job_node = str(st.get("node_id") or "").strip()
    node_hint = {"node_id": NODE_ID}
    if job_node and job_node != NODE_ID:
        node_hint["job_node_id"] = job_node

    if not req_email or not job_email or req_email != job_email:
        return {
            "state": "not_found",
            "error": "Job not found. Please resubmit your PDF.",
            **node_hint
        }

    state = (st.get("state") or "unknown").lower()
    if st.get("canceled") is True and state != "done":
        return {"state": "canceled", **node_hint}

    # Repair stale queued/running jobs before returning status to the client.
    # This prevents the UI from politely polling a dead status forever.
    if state in ("queued", "running", "unknown"):
        st = _repair_stale_active_job_from_status_poll(job_dir, st) or st
        state = (st.get("state") or "unknown").lower()

    # Build one status object for ALL states so fields do not disappear.
    out = {
        "state": state,
        "job_id": job_id,
        **node_hint
    }

    # Always pass through lightweight status fields the processing UI needs.
    for k in (
        "step",
        "progress",
        "image_count",
        "component_count",
        "items",
        "detected_images",
        "kept_pages",
        "dropped_pages",
        "noticed_ts_ms",
        "cycle_time_ms",
        "cycle_time_str",
        "queue_timeout",
        "queue_timeout_min",
        "queue_age_ms",
        "queue_age_str",
        "exclude_from_improvement",
        "retention_mode",
        "delete_after_utc",
        "raw_pdf_deleted",
    ):
        if k in st:
            out[k] = st[k]

    if state == "done":
        out["progress"] = 100.0
        out["result_ready"] = True
        return out

    if state == "error":
        out["error"] = st.get("error") or "Unknown error"
        return out

    if state in ("queued", "running"):
        try:
            queue_position, active_count = _get_queue_position(job_id, req_email)
            if queue_position is not None:
                out["queue_position"] = int(queue_position)
            out["active_count"] = int(active_count)
        except Exception:
            pass

    return out

@anvil.server.callable
def vm_list_jobs(owner_id: str, limit: int = 50, group_folder: str = "personal") -> list[dict]:
    print(f">>> vm_list_jobs called | owner_id={owner_id!r} | NODE_ID={NODE_ID}")

    owner_id = str(owner_id or "").strip().lower()
    if not owner_id:
        print(">>> vm_list_jobs: empty owner_id")
        return []

    # Enforce this user's standard retention before building the visible list.
    _cleanup_standard_retention_for_owner(owner_id, group_folder, reserve_slots=0)

    def _safe_iso(dt_s):
        if not isinstance(dt_s, str) or not dt_s.strip():
            return ""
        return dt_s.strip()

    rows = []
    try:
        job_dirs = list(_iter_job_dirs_for_owner(owner_id, group_folder))
        job_dirs = sorted(job_dirs, key=lambda p: p.name, reverse=True)

        for d in job_dirs:
            if not d.is_dir():
                continue

            # Hide temporary utility jobs from My Jobs.
            if (
                d.name.startswith("specs_")
                or d.name.startswith(
                    "compatibility__"
                )
            ):
                continue

            st = _json_read_or_none(_status_paths(d)["status"]) or {}

            if _is_excluded_job_expired(st):
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    print(f">>> deleted expired excluded job during list: {d}")
                except Exception:
                    pass
                continue

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

                # Manual edit marker for My Jobs page
                "manually_edited": bool(
                    st.get("manually_edited")
                    or st.get("manual_edit")
                    or st.get("edited")
                ),
                "last_edited_panel": st.get("last_edited_panel") or "",
                "last_edited_at_utc": st.get("last_edited_at_utc") or "",

                # Excluded job marker for My Jobs page
                "exclude_from_improvement": bool(st.get("exclude_from_improvement")),
                "retention_mode": st.get("retention_mode") or "standard",
                "delete_after_utc": st.get("delete_after_utc") or "",
                "raw_pdf_deleted": bool(st.get("raw_pdf_deleted")),
                "group_folder": st.get("group_folder") or group_folder,
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
def vm_delete_job(job_id: str, owner_id: str, group_folder: str = "personal") -> dict:
    """
    Permanently delete a non-active job owned by the user.

    Security:
    - job_id is path-sanitized
    - resolved path must stay under BASE_JOBS_DIR
    - status owner must match owner_id
    - active queued/running/unknown jobs are not hard-deleted here; cancel them instead
    """
    job_id = str(job_id or "").strip()
    owner_id = str(owner_id or "").strip().lower()

    if not job_id or not owner_id:
        return {"ok": False, "error": "Missing job id or owner."}

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return {"ok": False, "error": "Job not found."}

    job_dir = _resolve_job_dir_for_owner(job_id, owner_id, group_folder)
    if job_dir is None:
        return {"ok": False, "error": "Job not found."}

    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}

    if not isinstance(st, dict) or not st:
        return {"ok": False, "error": "Job not found."}

    job_owner = str(
        st.get("owner_email")
        or st.get("owner_id")
        or ""
    ).strip().lower()

    if not job_owner or job_owner != owner_id:
        return {"ok": False, "error": "Job not found."}

    state = str(st.get("state") or "").strip().lower()

    if state in ("queued", "running", "unknown"):
        return {
            "ok": False,
            "error": "This job is still processing. Please cancel it first.",
            "state": state or "unknown"
        }

    try:
        shutil.rmtree(job_dir)
        print(f">>> user deleted job folder: {job_id}")
    except FileNotFoundError:
        return {"ok": False, "error": "Job not found."}
    except Exception as e:
        return {"ok": False, "error": f"Could not delete job: {e}"}

    if job_dir.exists():
        return {"ok": False, "error": "Could not delete job."}

    return {
        "ok": True,
        "state": "deleted"
    }

@anvil.server.callable
def vm_cancel_job(job_id: str, owner_id: str, group_folder: str = "personal") -> dict:
    """
    Cancel a queued/running job, enforcing ownership.

    Security:
    - job_id is path-sanitized
    - owner_id must match owner_email/owner_id in status.json
    - completed/error/canceled jobs cannot be canceled
    """
    job_id = str(job_id or "").strip()
    owner_id = str(owner_id or "").strip().lower()

    if not job_id or not owner_id:
        return {"ok": False, "error": "Missing job id or owner."}

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        return {"ok": False, "error": "Job not found."}

    job_dir = _resolve_job_dir_for_owner(job_id, owner_id, group_folder)
    if job_dir is None:
        return {"ok": False, "error": "Job not found."}

    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}

    if not isinstance(st, dict) or not st:
        return {"ok": False, "error": "Job not found."}

    job_owner = str(
        st.get("owner_email")
        or st.get("owner_id")
        or ""
    ).strip().lower()

    if not job_owner or job_owner != owner_id:
        return {"ok": False, "error": "Job not found."}

    state = str(st.get("state") or "").strip().lower()

    if state in ("done", "error", "canceled", "cancelled"):
        return {
            "ok": False,
            "error": "This job is no longer cancelable.",
            "state": state or "unknown"
        }

    if state not in ("queued", "running", "unknown"):
        return {
            "ok": False,
            "error": "This job is not currently processing.",
            "state": state or "unknown"
        }

    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        with open(_cancel_path(job_dir), "w") as f:
            f.write("1")
    except Exception as e:
        return {"ok": False, "error": f"Could not mark job canceled: {e}"}

    now = _now_utc().isoformat()

    _status_write(
        job_dir,
        "canceled",
        canceled=True,
        canceled_at_utc=now,
        cancel_requested_by=owner_id,
        step="canceled",
        progress=0.0,
    )

    # If queued, no worker is actively using the files yet. Delete immediately.
    if state == "queued":
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            pass

        return {
            "ok": True,
            "state": "canceled",
            "deleted": True
        }

    # If running, do not delete files immediately while the worker may be inside OCR/render.
    # The .cancel marker is enough. The worker will hit a cancel checkpoint and delete the folder.
    return {
        "ok": True,
        "state": "canceled",
        "deleted": False
    }

# ---------- MAIN ----------
if not _IS_WORKER_SUBPROCESS:
    print(">>> Uplink ready; waiting for calls")
    anvil.server.wait_forever()