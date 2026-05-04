#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page filter V4 -- ML-based replacement.

Wraps ``PageClassifierML`` (MobileNetV2 binary classifier) behind the
same ``PageFilter`` class name and ``readPdf()`` return signature used
by earlier heuristic versions, so callers can swap imports without
changing their downstream code.

All heuristic logic (OCR, regex scoring, footprint detection) has been
removed.  Page keep/drop decisions are now made entirely by the trained
MobileNetV2 model checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PageFilter.PageClassifierML import PageClassifierML

logger = logging.getLogger(__name__)

_HEURISTIC_ONLY_PARAMS = frozenset({
    "dpi", "longest_cap_px", "proc_scale",
    "use_ocr", "ocr_gpu", "ocr_zoom", "crop_frac", "non_e_conf_min",
    "label_tall_factor", "crop_expand_right",
    "hard_hit_score", "min_hit_score",
    "w_strong", "w_riser", "w_diag", "w_base", "w_plural_bonus", "conf_mult",
    "pattern_e_loose", "pattern_non_e",
    "dilate_ink_px", "close_ink_px", "margin_shave_px",
    "min_whitespace_area_fr", "border_exclude_px",
    "rect_w_fr_range", "rect_h_fr_range", "min_rectangularity", "min_rect_count",
    "save_crop_pngs",
    "label_expand_up", "label_expand_left", "label_expand_left_retry", "label_zoom",
})


class PageFilter:
    """
    ML-powered page filter for electrical PDF drawings.

    Uses a fine-tuned MobileNetV2 to classify each page as
    "has panel schedule" or "no panel schedule", then assembles a
    filtered output PDF (optionally Ghostscript-rescaled to US Letter).

    Constructor accepts (and silently ignores) legacy heuristic
    keyword arguments so existing call sites don't break.
    """

    def __init__(
        self,
        output_dir: str,
        model_path: str,
        confidence_threshold: float = 0.5,
        render_dpi: int = 150,
        device: Optional[str] = None,
        verbose: bool = True,
        debug: bool = False,
        out_pdf_suffix: str = "_electrical_filtered.pdf",
        use_ghostscript_letter: bool = True,
        letter_orientation: str = "landscape",
        gs_use_cropbox: bool = True,
        gs_compat: str = "1.7",
        **kwargs: Any,
    ):
        ignored = sorted(set(kwargs) & _HEURISTIC_ONLY_PARAMS)
        if ignored and verbose:
            logger.info("Ignoring legacy heuristic params: %s", ", ".join(ignored))

        unknown = sorted(set(kwargs) - _HEURISTIC_ONLY_PARAMS)
        if unknown and verbose:
            logger.warning("Unknown params (ignored): %s", ", ".join(unknown))

        self._classifier = PageClassifierML(
            output_dir=output_dir,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            render_dpi=render_dpi,
            device=device,
            verbose=verbose,
            debug=debug,
            out_pdf_suffix=out_pdf_suffix,
            use_ghostscript_letter=use_ghostscript_letter,
            letter_orientation=letter_orientation,
            gs_use_cropbox=gs_use_cropbox,
            gs_compat=gs_compat,
        )

    def readPdf(
        self, pdf_path: str,
    ) -> tuple[list[int], list[int], Optional[str], Optional[str]]:
        """
        Classify pages and write a filtered output PDF.

        Returns:
            ``(kept_pages, dropped_pages, output_pdf_path, log_json_path)``
            where kept/dropped are 1-indexed page numbers.
        """
        return self._classifier.readPdf(pdf_path)
