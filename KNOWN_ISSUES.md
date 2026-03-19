# Known Issues

## Open

### derekfirst.pdf hangs under multiprocessing concurrency

- **Status:** Resolved
- **Date:** 2025-03-19
- **Resolved:** 2026-03-19
- **Component:** `OcrLibrary/BreakerTableParser10.py` — EasyOCR `readtext()` via `_readtext_with_timeout()`
- **Symptom:** `derekfirst.pdf` completes successfully as a single job (~103s, 6 crops,
  103 breakers). However, when run under `multiprocessing` concurrency >= 2
  (as in `DevEnv/capacity_test.py`), the pipeline hangs indefinitely.
- **Root cause:** EasyOCR / PyTorch spawned full OpenMP / MKL / OpenBLAS thread
  pools in every worker process. With concurrency >= 2, the competing thread
  pools deadlocked at the C/OpenMP level. The `_readtext_with_timeout()` daemon
  thread could not interrupt C-level hangs.
- **Fix:** Extracted thread-pool caps and runtime determinism into a shared
  `WorkerSetup/worker_init.py` module (`cap_thread_pools()` sets
  `OMP_NUM_THREADS=1` etc.; `set_runtime_determinism()` sets
  `torch.set_num_threads(1)`, `cv2.setNumThreads(1)`, cuDNN determinism).
  `capacity_test.py` now calls `cap_thread_pools()` before heavy imports and
  `set_runtime_determinism()` at the start of each worker.
  `uplink_server.py` delegates to the same shared module.

### ML Table Detection not yet implemented

- **Status:** Open
- **Date:** 2025-03-01
- **Component:** `MLTableDetection/`, `TableDetectorML.py`
- **Problem:** The ML-based panel detection directory and module do not exist
  in the repo. ML-based panel detection is a planned future feature. Currently
  only heuristic detection via `PanelSearchToolV25` is available.
