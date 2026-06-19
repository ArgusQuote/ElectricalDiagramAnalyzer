# Anvil Uplink VM (disk only) + Rules Engine defaults + cycle-time
# -------------------------------
import os, re, json, sys, threading, traceback, uuid
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

# ---------- CONFIG ----------
REPO_ROOT = Path("/home/paperspace/ElectricalDiagramAnalyzer").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Put jobs directly under the home directory
BASE_JOBS_DIR = Path.home() / "jobs"
BASE_JOBS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_JOB_RETENTION_HOURS = 24
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
from OcrLibrary.BreakerTableParserAPIv11 import BreakerTablePipeline, API_VERSION, reset_name_deduper
import RulesEngine.RulesEngine6 as RE2  # must expose process_job(payload)

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

def _utc_iso_z(dt=None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _excluded_delete_after_utc() -> str:
    return _utc_iso_z(datetime.now(timezone.utc) + timedelta(hours=EXCLUDED_JOB_RETENTION_HOURS))


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

    delete_after = str(status.get("delete_after_utc") or "").strip()
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
    Safe to run often.
    Handles both legacy flat jobs and grouped jobs.
    """
    try:
        if not BASE_JOBS_DIR.is_dir():
            return

        candidates = []

        # Legacy flat jobs directly under BASE_JOBS_DIR
        try:
            for job_dir in BASE_JOBS_DIR.iterdir():
                if not job_dir.is_dir():
                    continue
                if job_dir.name == "groups":
                    continue
                candidates.append(job_dir)
        except Exception:
            pass

        # Grouped jobs under BASE_JOBS_DIR/groups/<group>/users/<user>/<job>
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
        for job_dir in candidates:
            try:
                job_dir = _assert_under_base(job_dir)
            except Exception:
                continue

            key = str(job_dir)
            if key in seen:
                continue
            seen.add(key)

            sp = _status_paths(job_dir)
            st = _json_read_or_none(sp["status"]) or {}

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

def _edit_log_path(job_dir: Path) -> Path:
    return Path(job_dir) / "edits.json"


def _deep_copy_jsonable(obj):
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return obj


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

    attrs = cleaned.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    cleaned["attrs"] = attrs

    # User manually fixed the panel, so parser skip flags should not survive.
    cleaned.pop("_skipped", None)
    cleaned.pop("reason", None)

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

        # Preserve visual/source/status metadata unless the edited component explicitly supplied it.
        for key in (
            "source",
            "overlay_source",
            "overlaySource",
            "preview_source",
            "previewSource",
            "reviewOverlayPath",
            "review_overlay_path",
            "panelStatus",
            "panelNote",
            "specialHeaderType",
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
    rules_payload = _build_rules_payload(ui_overrides, components)

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
    panel_status = str((result_dict or {}).get("panelStatus") or "").strip()
    panel_note = str(hdr.get("panelNote") or "").strip()
    special_header_type = hdr.get("specialHeaderType") if isinstance(hdr.get("specialHeaderType"), dict) else None

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
        "panelStatus": panel_status,
        "panelNote": panel_note,
        "specialHeaderType": special_header_type,
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
            _process_job(job_id, pipeline)
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
                job_dir, "error", error=msg,
                queue_timeout=True, queue_timeout_min=QUEUE_TIMEOUT_MIN,
                queue_age_ms=age_ms, queue_age_str=age_str, progress=0.0,
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
                    job_dir, "error", error=msg, step="queue_timeout",
                    queue_timeout=True, queue_timeout_min=QUEUE_TIMEOUT_MIN,
                    queue_age_ms=age_ms, queue_age_str=age_str, progress=0.0,
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

# ---------- Start worker pool + cleanup loop ----------
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

    try:
        cleanup_thread = threading.Thread(
            target=_excluded_cleanup_loop,
            daemon=True,
            name="job-retention-cleanup",
        )
        cleanup_thread.start()
        print(
            f">>> job retention cleanup loop started | "
            f"interval={CLEANUP_SWEEP_INTERVAL_SEC}s | "
            f"standard_days={STANDARD_JOB_RETENTION_DAYS} | "
            f"standard_limit={STANDARD_JOB_HISTORY_LIMIT}"
        )
    except Exception as e:
        print(f">>> job retention cleanup loop startup failed: {e}")
        print(traceback.format_exc())

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

            state = str(st.get("state") or "").strip().lower()
            canceled = bool(st.get("canceled"))

            if canceled:
                continue

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
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()

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

    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)
    st = _json_read_or_none(sp["status"]) or {}

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
def vm_get_specs_status(job_id: str, owner_email: str) -> dict:
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

    job_dir = BASE_JOBS_DIR / job_id
    sp = _status_paths(job_dir)

    st = _json_read_or_none(sp["status"])
    if not isinstance(st, dict) or not st:
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
def vm_fetch_image(job_id: str, owner_email: str, source_path: str, group_folder: str = "personal"):
    """
    Return an image as BlobMedia.
    Accepts only job-relative paths inside this job folder.
    """
    job_id = str(job_id or "").strip()
    owner_email = str(owner_email or "").strip().lower()
    raw = str(source_path or "").strip().replace("\\", "/")

    if not job_id or not owner_email or not raw:
        raise RuntimeError("Image unavailable.")

    if "/" in job_id or "\\" in job_id or ".." in job_id:
        raise RuntimeError("Image unavailable.")

    # Do not accept absolute paths from the client.
    p_in = Path(raw)
    if p_in.is_absolute():
        raise RuntimeError("Image unavailable.")

    if ".." in p_in.parts:
        raise RuntimeError("Image unavailable.")

    job_root = _resolve_job_dir_for_owner(job_id, owner_email, group_folder)
    if job_root is None:
        raise RuntimeError("Image unavailable.")

    sp = _status_paths(job_root)
    st = _json_read_or_none(sp["status"]) or {}

    if _is_excluded_job_expired(st):
        try:
            shutil.rmtree(job_root, ignore_errors=True)
            print(f">>> deleted expired excluded job on image fetch: {job_root}")
        except Exception:
            pass
        raise FileNotFoundError("Job not found. Please resubmit your PDF.")

    job_owner = str(st.get("owner_email") or st.get("owner_id") or "").strip().lower()
    if not job_owner or job_owner != owner_email:
        raise RuntimeError("Image unavailable.")

    p = (job_root / p_in).resolve()

    try:
        p.relative_to(job_root)
    except ValueError:
        raise RuntimeError("Image unavailable.")

    if not p.is_file():
        raise RuntimeError("Image unavailable.")

    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise RuntimeError("Image unavailable.")

    ctype_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    ctype = ctype_map.get(p.suffix.lower(), "application/octet-stream")
    return BlobMedia(ctype, p.read_bytes(), name=p.name)

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
    _cleanup_expired_excluded_jobs()

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

    state = str(st.get("state") or "unknown").strip().lower()

    if st.get("canceled") is True and state != "done":
        return {
            "state": "canceled",
            **node_hint
        }

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

    _cleanup_all_retention_policies()

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

            # Hide specs jobs from My Jobs Page
            if d.name.startswith("specs_"):
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