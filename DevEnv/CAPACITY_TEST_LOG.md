# Capacity Test Log

Performance baselines from `capacity_test.py` — full pipeline
(PageFilter → PanelBoardSearch → BreakerTablePipeline) on a single PDF
at increasing concurrency levels.

---

## E.pdf — 2025-03-19

Single-page PDF, 5 panel crops detected per job.

| Concurrency | Success | Avg Job (s) | Batch Wall-Clock (s) | Slowdown vs 1x |
|:-----------:|:-------:|:-----------:|:--------------------:|:--------------:|
| 1           | 1/1     | 39.31       | 41.74                | — (baseline)   |
| 2           | 2/2     | 66.18       | 69.91                | 1.68x          |
| 3           | 3/3     | 110.28      | 113.78               | 2.81x          |
| 4           | 4/4     | 136.54      | 147.10               | 3.47x          |

**Notes:**
- All concurrency levels pass (0 failures) after fixing the Ghostscript
  temp-file race condition (`mkstemp` instead of deterministic `/tmp` path).
- Sweet spot is concurrency 2 for throughput-per-job.
- Concurrency 4 is stable but 3.47x slower per job due to CPU/memory contention.

---

## derekfirst.pdf — 2025-03-19

Multi-page PDF (~10 pages). **Pipeline hangs — known issue.**

**How far it gets before hanging:**
1. PageFilter: completes (filtered to electrical pages).
2. PanelBoardSearch: completes — finds **6 valid crops** (4 on page 7, 2 on
   page 8) plus 6 invalid tables rejected on pages 1, 4, 9.
3. BreakerTablePipeline: completes header+body parsing for **page 7 panels 1–4**
   successfully. Begins page 8 panel 1 — completes header analysis but **hangs
   during body/row parsing** (no `parser_body_combined` debug output produced).

**Root cause:** `_readtext_with_timeout()` initially used `ThreadPoolExecutor`
as a context manager, whose `__exit__` calls `shutdown(wait=True)` — blocking
until the hung EasyOCR thread finishes, defeating the timeout. Replaced with a
bare daemon thread + `thread.join(timeout)` which abandons the thread on timeout.

**After fix (single job):** Pipeline completes in ~103s. All 6 crops processed,
103 breakers detected. No OCR timeouts triggered in single-job run.

**Concurrency test (post-fix):** Still hangs at concurrency >= 2. The daemon
thread timeout does not resolve the multiprocessing case. The hang is likely
at the C/CUDA/OpenMP level where Python threads cannot interrupt it. See
`KNOWN_ISSUES.md` for details and possible next steps.
