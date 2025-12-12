# Anvil Uplink VM (disk only) + Rules Engine defaults + cycle-time
# -------------------------------
import os, re, json, sys, threading, traceback
import contextlib
import concurrent.futures
from multiprocessing import get_context
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime, timezone
import anvil.server
import platform
import os as _os
from anvil import BlobMedia

# ---------- CONFIG ----------
# Ensure your repo is on sys.path (for imports below)
REPO_ROOT = Path("/home/paperspace/ElectricalDiagramAnalyzer").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Put jobs directly under the home directory
BASE_JOBS_DIR = Path.home() / "jobs"
BASE_JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- PANEL FINDER CONFIG (PanelSearchToolV18) ----------
PANEL_FINDER_DEFAULTS = {
    # Same knobs you use in your dev env script
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
MAX_WORKERS = 3
MAX_INFLIGHT_PER_USER = 1

# ===== Determinism & Thread Caps (must run before heavy libs init) =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")

# ---------- WATCHDOG CONFIG ----------
WATCHDOG_TIMEOUT_MIN = int(os.environ.get("WATCHDOG_TIMEOUT_MIN", "15"))  # dial in prod
WATCHDOG_KILL_GRACE_SEC = int(os.environ.get("WATCHDOG_KILL_GRACE_SEC", "3"))
WATCHDOG_ERROR_MSG = (
  "This job took over {mins} minutes to process. "
  "Please trim the PDF to only relevant pages and try again."
)

def _set_runtime_determinism():
    # OpenCV: cap threads if available
    try:
        import cv2
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
    except Exception:
        pass

    # PyTorch/EasyOCR determinism if present
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

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
from PageFilter.PageFilterV2 import PageFilter
from VisualDetectionToolLibrary.PanelSearchToolV18 import PanelBoardSearch
from OcrLibrary.BreakerTableParserAPIv6 import BreakerTablePipeline, API_VERSION
import RulesEngine.RulesEngine3 as RE2  # must expose process_job(payload)

# ---------- CONNECT UPLINK ----------
ANVIL_UPLINK_KEY = os.environ.get("ANVIL_UPLINK_KEY", "")
if not ANVIL_UPLINK_KEY:
    raise RuntimeError("Set ANVIL_UPLINK_KEY in environment (ANVIL_UPLINK_KEY).")
anvil.server.connect(ANVIL_UPLINK_KEY)
print(">>> ENTRY OK")
# Identify this connected process (used for sticky routing diagnostics)
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

_warmup_ocr_once()

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
    return json.loads(json.dumps(_DEFAULT_OVERRIDES))

# ---------- PDF → images ----------
def render_pdf_to_images(saved_pdf: Path, img_dir: Path, dpi: int = 400) -> list[str]:
    """
    Run PageFilter first to keep only probable electrical/panel pages,
    then pass the (possibly filtered) PDF to PanelBoardSearch to produce crops.
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> rendering PDF → images: {saved_pdf} -> {img_dir} (dpi={dpi})")

    # --- 1) Filter pages (OCR first, footprints only if undecided) ---
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
    except Exception as e:
        print(f">>> PageFilter error: {e}")
        kept_pages, filtered_pdf = [], None

    # Choose which PDF to feed into the finder:
    # - if filter kept at least one page, use filtered_pdf
    # - else fall back to the original PDF (or return empty if you prefer)
    pdf_for_finder = filtered_pdf if (filtered_pdf and len(kept_pages) > 0) else str(saved_pdf)
    if pdf_for_finder == str(saved_pdf) and (filtered_pdf is not None) and len(kept_pages) == 0:
        print(">>> PageFilter kept 0 pages — falling back to original PDF")

    # --- 2) Run the panel finder (PanelSearchToolV18) on the chosen PDF ---
    local_finder = PanelBoardSearch(
        output_dir=str(img_dir),
        dpi=dpi,
        # All other knobs pulled from PANEL_FINDER_DEFAULTS so they match your dev env
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
    Applies your 4-strikes suppression rule.
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

    amperage   = _get_int_from_header("amperage", "main_amp", "mainAmperage")
    spaces_h   = _get_int_from_header("spaces")
    voltage    = _get_int_from_header("voltage")
    intRating  = _get_int_from_header("intRating", "interrupt_rating", "interruptRating", "kaic", "kaic_rating")
    main_amp   = _get_int_from_header("mainBreakerAmperage", "main_breaker_amperage", "main_breaker", "mainBreaker")
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
            "detected_breakers": det_brkrs,
        },
    }

    if notes:
        comp["notes"] = notes
        comp["attrs"]["breaker_data_suppressed"] = True

    return comp

# ---------- Worker-side call for one image ----------
def _btp_run_once(
    image_path: str,
    *,
    run_analyzer: bool = True,
    run_parser: bool = True,
    run_header: bool = True,
    debug: bool = True
) -> dict:
    """
    Construct a BreakerTablePipeline (v6) and run it for this image.
    (No global caching; per your request.)
    """
    pipe = BreakerTablePipeline(debug=debug)
    return pipe.run(
        image_path,
        run_analyzer=run_analyzer,
        run_parser=run_parser,
        run_header=run_header,
    )

def _run_job_target(job_id: str):
  """
  Subprocess entrypoint that runs the actual job logic.
  Keeping this thin ensures termination is clean.
  """
  try:
    _process_job(job_id)
  except Exception as e:
    # _process_job already writes status on exceptions, but as a backstop:
    job_dir = BASE_JOBS_DIR / job_id
    _status_write(job_dir, "error", error=f"{type(e).__name__}: {e}")

# ---------- Core job processing ----------
def _process_job(job_id: str):
    """
    Worker thread: read images → parse panel (Analyzer+Parser+Header) → run rules (with defaults) → write result.
    """
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

        # Ensure images are present
        pdf_dir = job_dir / "uploaded_pdfs"
        img_dir = job_dir / "pdf_images"
        imgs = sorted(str(p) for p in img_dir.glob("*.png"))

        if not imgs:
            pdfs = sorted(pdf_dir.glob("*.pdf"))
            if not pdfs:
                raise RuntimeError("No PDF found to render.")

            # Early heartbeat: rendering start
            _status_write(job_dir, "running", step="rendering", noticed_ts_ms=noticed_ts_ms, progress=2.0)

            # Render (PageFilter may be slow)
            try:
                imgs = render_pdf_to_images(pdfs[0], img_dir)
            finally:
                # Heartbeat right after render returns (even on exception path)
                _status_write(job_dir, "running", step="rendered", image_count=len(imgs), noticed_ts_ms=noticed_ts_ms, progress=9.0)

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
        _status_write(job_dir, "running", step="parsing", image_count=len(imgs), noticed_ts_ms=noticed_ts_ms, progress=10.0)

        debug_dir = job_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        total = len(imgs)
        done = 0

        # ---------- SEQUENTIAL IMAGE PARSING (deterministic per job) ----------
        print(f">>> sequential parsing for {total} images (per-job determinism)")
        components = [None] * total  # preserve order
        for idx, img_path in enumerate(imgs):
            if _is_canceled(job_dir):
                pct = 10.0 + (done / max(1, total)) * 80.0
                _status_write(job_dir, "canceled", step="parsing", image_count=total, noticed_ts_ms=noticed_ts_ms, progress=pct)
                _jobs_upsert(job_id, state="canceled", updated_at=_now_utc())
                print(f">>> worker canceled mid-parse: {job_id}")
                return

            try:
                # Run OCR/analysis sequentially
                raw = _btp_run_once(
                    img_path,
                    run_analyzer=True,
                    run_parser=True,
                    run_header=True,
                    debug=True
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
            pct = 10.0 + (done / max(1, total)) * 80.0
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

        rules_payload = _build_rules_payload(prev.get("ui_overrides") or _DEFAULT_OVERRIDES, components)
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
            "output_dir": str(job_dir / "pdf_images"),
            "images": imgs,
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

    except Exception as e:
        tb = traceback.format_exc()
        print(f">>> worker error [{job_id}]: {e}\n{tb}")
        _status_write(job_dir, "error", error=f"{type(e).__name__}: {e}")
        _jobs_upsert(job_id, state="error", updated_at=_now_utc(), error=f"{type(e).__name__}: {e}")

# ---------- Worker pool ----------
def _dequeue_loop(idx: int):
  threading.current_thread().name = f"pool-worker-{idx}"
  mp = get_context("spawn")  # safer than fork for libs like torch/opencv

  while not _STOP.is_set():
    try:
      job_id, owner_id = _JOB_Q.get(timeout=0.5)
    except Empty:
      continue

    # enforce per-user cap
    if not _enter_inflight(owner_id):
      threading.Timer(0.05, lambda: _enqueue_job(job_id, owner_id)).start()
      _JOB_Q.task_done()
      continue

    proc = None
    try:
      # Spawn child process for this job
      proc = mp.Process(target=_run_job_target, args=(job_id,), daemon=True)
      proc.start()

      # Watchdog wait
      timeout_sec = max(1, int(WATCHDOG_TIMEOUT_MIN) * 60)
      proc.join(timeout=timeout_sec)

      if proc.is_alive():
        # Timed out: mark canceled/error, terminate child, clean up
        job_dir = BASE_JOBS_DIR / job_id
        # Mark cancel file so future reads show canceled intent
        try:
          with open(_cancel_path(job_dir), "w") as f:
            f.write("1")
        except Exception:
          pass

        # Write an error status/result snapshot with message
        msg = WATCHDOG_ERROR_MSG.format(mins=WATCHDOG_TIMEOUT_MIN)
        _status_write(job_dir, "error", error=msg)
        _jobs_upsert(job_id, state="error", updated_at=_now_utc(), error=msg)

        # Try graceful term then hard kill
        try:
          proc.terminate()
        except Exception:
          pass
        proc.join(timeout=WATCHDOG_KILL_GRACE_SEC)
        if proc.is_alive():
          try:
            proc.kill()  # Python 3.9+ on POSIX
          except Exception:
            pass
          proc.join(timeout=1)

    finally:
      _leave_inflight(owner_id)
      _JOB_Q.task_done()

for i in range(MAX_WORKERS):
    t = threading.Thread(target=_dequeue_loop, args=(i,), daemon=True)
    t.start()
    _WORKERS.append(t)
print(f">>> Worker pool started: {MAX_WORKERS} threads, per-user cap={MAX_INFLIGHT_PER_USER}")

# ---------- API: submit / status / list / cancel ----------
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
def vm_list_overlay_images(job_id: str) -> list[str]:
    """Return absolute paths to all overlay PNGs for this job."""
    if not job_id:
        return []
    job_root = (BASE_JOBS_DIR / job_id).resolve()
    overlay_dir = (job_root / "pdf_images" / "magenta_overlays")
    if not overlay_dir.is_dir():
        return []
    return [str(p.resolve()) for p in sorted(overlay_dir.glob("*.png"))]

@anvil.server.callable
def vm_fetch_image(job_id: str, source_path: str):
    """
    Return the cropped PNG for a given panel as BlobMedia, enforcing that the file
    lives under this job folder.
    """
    if not job_id or not source_path:
        raise RuntimeError("job_id and source_path are required")

    job_root = (BASE_JOBS_DIR / job_id).resolve()
    p = Path(source_path).resolve()

    # Security: ensure requested file is inside this job folder
    if not str(p).startswith(str(job_root)):
        raise RuntimeError("Invalid image path for this job")

    if not p.is_file():
        raise RuntimeError(f"Image not found: {p}")

    ctype = "image/png" if p.suffix.lower() == ".png" else "application/octet-stream"
    data = p.read_bytes()
    return BlobMedia(ctype, data, name=p.name)

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

    if not job_email:
        # legacy/missing – backfill with the caller and continue
        st["owner_email"] = req_email
        st["owner_id"] = req_email
        _status_write(job_dir, st.get("state", "unknown"),
                      **{k: v for k, v in st.items() if k not in ("state", "ts")})
        job_email = req_email
        print(f">>> vm_get_job_status backfilled owner_email for {job_id}")

    if job_email != req_email:
        return {"state": "not_found", **node_hint}

    state = (st.get("state") or "unknown").lower()
    if st.get("canceled") is True and state != "done":
        return {"state": "canceled", **node_hint}

    if state == "done":
        res = _json_read_or_none(sp["result"]) or {}
        return {"state": "done", "result": res, **node_hint}
    if state == "error":
        return {"state": "error", "error": st.get("error") or "Unknown error", **node_hint}

    out = {"state": state, **node_hint}
    for k in ("step", "image_count", "chosen", "ui_overrides", "noticed_ts_ms", "cycle_time_str", "cycle_time_ms", "progress"):
        if k in st:
            out[k] = st[k]

    if "noticed_ts_ms" in st and isinstance(st["noticed_ts_ms"], int):
        elapsed_ms = max(0, _epoch_ms() - int(st["noticed_ts_ms"]))
        out["elapsed_ms"]  = elapsed_ms
        out["elapsed_str"] = _fmt_cycle_time(elapsed_ms)

    return out

@anvil.server.callable
def vm_list_jobs(owner_id: str, limit: int = 50) -> list[dict]:
    """List jobs owned by this user."""
    rows = []
    for d in sorted(BASE_JOBS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        st = _json_read_or_none(_status_paths(d)["status"])
        if not st:
            continue
        if str(st.get("owner_id") or "").strip().lower() != str(owner_id or "").strip().lower():
            continue
        rows.append({
            "job_id": d.name,
            "state": st.get("state"),
            "created_at": st.get("created_at"),
            "image_count": st.get("image_count"),
            "progress": st.get("progress", 0.0),
            "step": st.get("step"),
        })
        if len(rows) >= int(limit):
            break
    return rows

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
if __name__ == "__main__":
    print(">>> Uplink ready; waiting for calls")
    anvil.server.wait_forever()
