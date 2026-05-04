"""Shared timeout wrapper for EasyOCR readtext() calls.

EasyOCR can hang indefinitely on pathological images or when the
underlying CUDA/PyTorch state degrades across reused reader instances.
This module provides a wall-clock timeout so no single OCR call can
block the pipeline forever.
"""
from __future__ import annotations

import threading

OCR_TIMEOUT_SEC = 30


def readtext_with_timeout(reader, image, timeout=OCR_TIMEOUT_SEC, **kwargs):
    """Run reader.readtext() with a wall-clock timeout.

    Wraps the call in a daemon thread.  If the thread does not finish
    within *timeout* seconds, returns ``[]`` and lets the daemon thread
    leak (Python cannot forcibly kill C-level work inside EasyOCR, but
    the daemon flag ensures it won't block process exit).
    """
    result = [None]
    exc_flag = [False]

    def _worker():
        try:
            result[0] = reader.readtext(image, **kwargs)
        except Exception:
            exc_flag[0] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        print(f"[OCR_TIMEOUT] readtext exceeded {timeout}s — skipping this region")
        return []

    if exc_flag[0]:
        return []
    return result[0] if result[0] is not None else []
