"""Thread-pool caps and runtime determinism for multiprocessing workers.

Call ``cap_thread_pools()`` **before** importing heavy libraries (PyTorch,
EasyOCR, NumPy with OpenBLAS/MKL) so that environment variables are visible
when those libraries initialise their thread pools.

Call ``set_runtime_determinism()`` inside each spawned worker process to
apply runtime settings (torch thread count, OpenCV thread count, cuDNN
determinism) that do not survive across ``multiprocessing.spawn``.
"""

import os


def cap_thread_pools() -> None:
    """Set environment variables that limit per-process thread pools.

    Must be called before ``import torch`` / ``import easyocr`` / ``import
    numpy`` so that OpenMP, OpenBLAS, MKL, etc. read the values during
    their own library init.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONHASHSEED", "0")


def set_runtime_determinism() -> None:
    """Apply runtime thread caps and determinism flags.

    Safe to call even if OpenCV or PyTorch are not installed -- missing
    libraries are silently skipped.
    """
    try:
        import cv2
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
    except ImportError:
        pass

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
