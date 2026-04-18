#!/usr/bin/env python3
"""
ML-based page classifier using MobileNetV2.

Classifies PDF pages as "has panel schedule" or "no panel schedule"
without OCR, regex scoring, or footprint detection. Designed as a
faster, more accurate replacement for PageFilterV4.

Model:  MobileNetV2 (BSD-3, torchvision)
Speed:  ~5-20 ms/page on GPU, ~50-100 ms/page on CPU
License: BSD-3 (commercial-friendly)

Usage:
    from PageFilter.PageClassifierML import PageClassifierML

    classifier = PageClassifierML(
        output_dir="/path/to/output",
        model_path="/path/to/model.pt",
    )
    kept, dropped, details = classifier.classify_pdf("electrical.pdf")

    # Full filter (classify + write filtered PDF):
    kept, dropped, out_pdf, log_path = classifier.readPdf("electrical.pdf")
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pikepdf
import pypdfium2 as pdfium
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = ("no_panel", "has_panel")
POSITIVE_CLASS = 1
NUM_CLASSES = 2
DROPOUT_RATE = 0.3


class PageClassifierML:
    """
    Binary page classifier for electrical PDF drawings.

    Renders each page at low DPI, feeds it through a fine-tuned
    MobileNetV2, and returns a keep/drop decision per page.

    Provides a ``readPdf()`` API compatible with ``PageFilterV4``
    so it can be swapped into the existing pipeline.
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
    ):
        """
        Args:
            output_dir:  Where to write output files.
            model_path:  Path to a trained ``model.pt`` checkpoint.
            confidence_threshold: Minimum softmax probability to accept
                the positive ("has_panel") prediction.
            render_dpi:  Resolution for rendering PDF pages before
                classification.  Lower = faster; 150 is usually enough
                for page-level decisions.
            device:  ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
            verbose: Print per-page progress.
            debug:   Save a JSON log of every page decision.
            out_pdf_suffix: Suffix for the filtered output PDF.
            use_ghostscript_letter: Rescale output to US Letter via GS.
            letter_orientation: ``"landscape"`` or ``"portrait"``.
            gs_use_cropbox: Pass ``-dUseCropBox`` to Ghostscript.
            gs_compat: PDF compatibility level for Ghostscript output.
        """
        self.output_dir = os.path.expanduser(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.confidence_threshold = confidence_threshold
        self.render_dpi = render_dpi
        self.verbose = verbose
        self.debug = debug
        self.out_pdf_suffix = out_pdf_suffix

        self.debug_dir = os.path.join(self.output_dir, "filter_debug")
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Ghostscript post-processing (same knobs as PageFilterV4)
        self.use_ghostscript_letter = use_ghostscript_letter
        self.letter_orientation = (
            "landscape"
            if str(letter_orientation).lower().startswith("land")
            else "portrait"
        )
        self.gs_use_cropbox = gs_use_cropbox
        self.gs_compat = str(gs_compat)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[nn.Module] = None
        self.transform: Optional[transforms.Compose] = None
        self.config: dict[str, Any] = {}
        self._load_model(os.path.expanduser(model_path))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        """Load a trained page-classifier checkpoint."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint.get("config", {})

        input_size = self.config.get("input_size", 224)
        num_classes = self.config.get("num_classes", NUM_CLASSES)

        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(in_features, num_classes),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

        if self.verbose:
            metrics = checkpoint.get("metrics", {})
            val_acc = metrics.get("val_accuracy", "N/A")
            print(f"[INFO] Page classifier loaded: {model_path}")
            print(f"[INFO] MobileNetV2, input={input_size}x{input_size}, "
                  f"device={self.device}")
            if val_acc != "N/A":
                print(f"[INFO] Training val_accuracy: {val_acc:.3f}")

    # ------------------------------------------------------------------
    # Single-page inference
    # ------------------------------------------------------------------

    def classify_page_image(
        self, image: Image.Image,
    ) -> tuple[int, float]:
        """
        Classify one page image.

        Returns:
            ``(predicted_class, confidence)`` where 1 = has_panel.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)
            confidence, predicted = probs.max(1)

        return int(predicted.item()), float(confidence.item())

    # ------------------------------------------------------------------
    # PDF classification (no output PDF)
    # ------------------------------------------------------------------

    def classify_pdf(
        self, pdf_path: str,
    ) -> tuple[list[int], list[int], list[dict]]:
        """
        Classify every page in *pdf_path*.

        Returns:
            ``(kept_pages, dropped_pages, page_details)``
            where kept/dropped are **1-indexed** page numbers.
        """
        pdf_path = os.path.expanduser(pdf_path)
        if not Path(pdf_path).is_file():
            raise FileNotFoundError(pdf_path)

        doc = pdfium.PdfDocument(pdf_path)
        scale = self.render_dpi / 72.0

        kept: list[int] = []
        dropped: list[int] = []
        details: list[dict] = []

        if self.verbose:
            print(f"[INFO] Classifying {len(doc)} page(s) "
                  f"@ {self.render_dpi} DPI")

        for pidx in range(len(doc)):
            page = doc[pidx]
            page_no = pidx + 1

            pil_img = self._render_page(page, scale)
            pred_class, conf = self.classify_page_image(pil_img)
            has_panel = (
                pred_class == POSITIVE_CLASS
                and conf >= self.confidence_threshold
            )

            if has_panel:
                kept.append(page_no)
            else:
                dropped.append(page_no)

            detail = {
                "page": page_no,
                "prediction": CLASS_NAMES[pred_class],
                "confidence": round(conf, 4),
                "decision": "KEEP" if has_panel else "DROP",
            }
            details.append(detail)

            if self.verbose:
                status = "KEEP" if has_panel else "DROP"
                print(f"  Page {page_no:03d}: {status} "
                      f"({CLASS_NAMES[pred_class]}, conf={conf:.3f})")

        doc.close()

        if self.verbose:
            total = len(kept) + len(dropped)
            print(f"[DONE] Kept {len(kept)}/{total} pages")

        return kept, dropped, details

    # ------------------------------------------------------------------
    # readPdf  --  drop-in replacement for PageFilterV4.readPdf()
    # ------------------------------------------------------------------

    def readPdf(
        self, pdf_path: str,
    ) -> tuple[list[int], list[int], Optional[str], Optional[str]]:
        """
        Classify pages, write a filtered output PDF, and optionally
        rescale to US Letter via Ghostscript.

        Matches the ``PageFilterV4.readPdf()`` return signature::

            (kept_pages, dropped_pages, output_pdf_path, log_json_path)
        """
        pdf_path = os.path.expanduser(pdf_path)
        if not Path(pdf_path).is_file():
            raise FileNotFoundError(pdf_path)

        kept, dropped, details = self.classify_pdf(pdf_path)

        if not kept:
            if self.verbose:
                print("[WARN] No pages kept -- nothing saved")
            return kept, dropped, None, None

        # ---- build filtered PDF with pikepdf ----
        base = Path(pdf_path).stem
        out_pdf_path = os.path.join(
            self.output_dir, base + self.out_pdf_suffix)

        src_pdf = pikepdf.open(pdf_path)
        dst_pdf = pikepdf.new()
        for page_no in kept:
            dst_pdf.pages.append(src_pdf.pages[page_no - 1])

        raw_pdf = out_pdf_path
        if self.use_ghostscript_letter:
            raw_pdf = os.path.join(
                self.output_dir,
                base + self.out_pdf_suffix.replace(".pdf", "_raw.pdf"))

        dst_pdf.save(raw_pdf)
        dst_pdf.close()

        if self.verbose:
            print(f"[OK] Saved filtered PDF -> {raw_pdf}  "
                  f"(kept {len(kept)}/{len(kept) + len(dropped)})")

        # ---- optional Ghostscript letter-fit ----
        final_pdf = raw_pdf
        if self.use_ghostscript_letter:
            try:
                final_pdf = self._gs_fit_to_letter(raw_pdf, out_pdf_path)
                if self.verbose:
                    print(f"[OK] Ghostscript Letter "
                          f"({self.letter_orientation}) -> {final_pdf}")
                if not self.debug:
                    try:
                        os.remove(raw_pdf)
                    except OSError:
                        pass
            except Exception as exc:
                if self.verbose:
                    print(f"[WARN] Ghostscript failed: {exc}. "
                          f"Using raw output.")
                final_pdf = raw_pdf

        src_pdf.close()

        # ---- debug log ----
        log_path: Optional[str] = None
        if self.debug:
            log_path = os.path.join(
                self.debug_dir, f"{base}_classifier_log.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(details, f, indent=2)
            if self.verbose:
                print(f"[DEBUG] Log -> {log_path}")

        return kept, dropped, final_pdf, log_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_page(page, scale: float) -> Image.Image:
        """Render a pypdfium2 page to a PIL RGB image."""
        bitmap = page.render(scale=scale)
        pil_img = bitmap.to_pil()
        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            return bg
        if pil_img.mode != "RGB":
            return pil_img.convert("RGB")
        return pil_img

    def _gs_fit_to_letter(self, in_pdf: str, out_pdf: str) -> str:
        """Rescale *in_pdf* to US Letter via Ghostscript."""
        src = Path(in_pdf).expanduser().resolve()
        dst = Path(out_pdf).expanduser().resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        gs = shutil.which("gs")
        if not gs:
            raise RuntimeError(
                "Ghostscript not found. "
                "Install with: sudo apt-get install -y ghostscript")

        if self.letter_orientation == "landscape":
            w_pt, h_pt = 792, 612
        else:
            w_pt, h_pt = 612, 792

        args = [
            gs, "-q", "-dBATCH", "-dNOPAUSE",
            "-sDEVICE=pdfwrite",
            f"-dCompatibilityLevel={self.gs_compat}",
            "-dFIXEDMEDIA", "-dPDFFitPage",
            "-dAutoRotatePages=/None",
            f"-dDEVICEWIDTHPOINTS={w_pt}",
            f"-dDEVICEHEIGHTPOINTS={h_pt}",
        ]
        if self.gs_use_cropbox:
            args.append("-dUseCropBox")

        tmp_out = Path(tempfile.gettempdir()) / (dst.name + ".tmp")
        args.extend([f"-sOutputFile={tmp_out}", str(src)])

        cp = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"Ghostscript failed ({cp.returncode}). "
                f"Stderr:\n{cp.stderr.strip()}")

        tmp_out.replace(dst)
        return str(dst)
