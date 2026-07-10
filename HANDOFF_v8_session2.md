# TATR panel-detector v8 — session 2 handoff

Continue the TATR panel-detector v8 work (session 2). Read
`.cursor/rules/project/docs/known-issues.mdc` entries "ML eval-set
contamination…2026-07-07" and "v7 MILESTONE LOCKED" first. The previous session
did the root-cause analysis, built the data pipeline, and launched v8 training on
Paperspace. Pick up from monitoring training → evaluation → documentation.

## KEY ROOT-CAUSE FINDING (verified 2026-07-10, this is the crux)

v7's "whole-page blindness" is NOT poor generalization — it's a **training-label
under-coverage bug**. In the v6 dataset
(`~/Documents/TableAnnotations/v6/annotations_v6.json`), multi-page docs were
barely labeled: `derek2` had only page 9 of 11 labeled (9 boxes); `derekfirst`
only page 7 of 10 (4 boxes). The other ~19 derek pages were included with **0
annotations = trained as negatives** (those are exactly v6's "20 blank images").
The heuristic GT finds ~40 boxes on derek2 / ~23 on derekfirst. So the model was
explicitly taught "derek-style page = empty." Even single-page docs were
under-labeled (heuristic finds C:9 vs manual 5; I:8 vs 4; A:7 vs 4; M:15 vs 11).
Fix = **relabel ALL pages from the heuristic ground truth** (per the standing
decision that PanelSearchToolV25 is GT). Marco approved: relabel-all + regenerate
synth; single held-out fold `{derek2,derekfirst}` + 3 clean Diagrams PDFs first,
then expand to k-fold if it works; train on Paperspace (uplink was idle, "just
make sure anvil works when done").

## WHAT WAS BUILT (NOT yet committed — Marco commits himself)

- New script `MLTableDetection/build_heuristic_dataset.py` — renders each PDF page
  (≤2600px long side, never >400 DPI) and auto-labels with `PanelBoardSearch`
  (pure CV, no OCR). Boxes stored in top-left-origin PDF points; converts
  `point * render_scale` to pixels. Coord alignment visually verified on O p4
  (9/9 panels tight). Supports `--exclude`.
- Edited `MLTableDetection/synthesize_panels.py` — added
  `--rotate-deg/--scale-min/--scale-max/--brightness` CLI (backward-compatible;
  defaults = old hardcoded values).
- Datasets on laptop (`~/Documents/TableAnnotations/`): `v8_real_holdout/`
  (28 pages, 149 boxes, 20 docs excl derek2/derekfirst), `v8_synth_holdout/`
  (400 imgs, 1690 anns, dense grids: `--min-panels 2 --max-panels 8
  --scale-min 0.75 --scale-max 1.2`), merged `v8_holdout/` (428 imgs, 1839 anns;
  `annotations.json` symlink → `annotations_v6.json`). Same `v8_holdout/` copied
  to Paperspace `/home/paperspace/Documents/TableAnnotations/v8_holdout/`. 3
  Diagrams PDFs copied to Paperspace `~/Documents/Diagrams/`
  (`chucksmall.pdf`, `ELECTRICAL SET (Mark Up).pdf`, `Panels_Example.pdf`).

## TRAINING — COMPLETE 2026-07-10 ~07:29 (Paperspace)

Recipe = v4/v7 stable recipe + 28 epochs: `--epochs 28 --batch-size 2
--learning-rate 1e-5 --warmup-steps 50 --weight-decay 0.01 --val-split 0.1
--num-queries 15 --freeze-backbone-epochs 0`, output `models_v8/`. Finished
cleanly: `train_runtime` 2691 s, `train_loss` 5.5→0.674, exit 0, 5404 steps.
NOT a v6 collapse — `eval_map_50` rose monotonically 0.266 (ep1, already >
v7's best-ever 0.2555) → peak **0.3999 @ epoch 25**, eval_loss 1.37→~0.50.
- **Production weights: `models_v8/best/`** = the epoch-25 checkpoint
  (HF `best_model_checkpoint` = `checkpoint-4825`, `best_metric` 0.3999).
  `best/` has model.safetensors + config + preprocessor + training_args (no
  trainer_state.json in `best/`; the full 28-eval history lives in
  `checkpoint-5404/trainer_state.json`).
- Leftover `checkpoint-4825/5211/5404/` still on disk (~330 MB each) — prune
  after the held-out eval confirms `best/` is the one to keep (mirror the v7
  cleanup: verify training_args byte-identical, copy checkpoint-5404's
  trainer_state.json into best/ as the forensics record, then delete the
  three checkpoints + `runs/`).
- Env confirmed: transformers 4.55.4, torch 2.4.1+cu121, torchmetrics 1.9.0,
  base model cached, `compute_metrics` normalized-coords patch present.
- NOTE: 0.3999 is the internal 10%-mostly-synth val split — the REAL number
  is the held-out PDF eval in step 2, which is **not yet run** (see below).

## SESSION-2 STOP STATE (2026-07-10 ~02:36 laptop time — resume here)

- Step 1 DONE (training finished, `best/` verified as above).
- Step 2 (held-out eval) NOT done. It was launched then deliberately killed at
  a clean stopping point: it had only reached v7/derek2 (still in the CPU
  heuristic-render phase, no `box_comparison.json` written), so **zero eval
  results exist yet** — `baselines/v8_2026-07-10/` is empty. It was stopped to
  avoid an orphaned eval competing overnight with production anvil for the GPU.
- Verified at stop: no `evaluate_model.py`/`run_v8_eval.sh` procs left, GPU
  idle (1494 MiB = the 4 dormant anvil workers, 0% util), `anvil-uplink.service`
  **active**.
- **A ready-to-run eval runner is saved on Paperspace at
  `~/Documents/TableAnnotations/run_v8_eval.sh`** — it loops all 5 held-out
  PDFs × {v7,v8} into `baselines/v8_2026-07-10/{v7,v8}/<stem>/` at conf 0.5 /
  IoU 0.5 (verbose, logs to `~/Documents/TableAnnotations/v8_eval.log`).
  Tomorrow: confirm no customer job is running (`nvidia-smi`), then
  `cd ~/Documents/TableAnnotations && nohup bash run_v8_eval.sh > v8_eval.log 2>&1 &`
  and monitor. Set A = derek2, derekfirst (pdfToScan); Set B = chucksmall,
  ELECTRICAL_SET_MarkUp, Panels_Example (Diagrams). Then continue at step 3.

## REMAINING STEPS (in order)

1. DONE — training finished (5404/5404 steps, `models_v8/best/` written; peak val
   `eval_map_50` ≈ 0.398 @ epoch 26, no collapse, beat v7's best-ever 0.2555).
   START HERE at step 2 (held-out eval). Note: that val split is
   10%-mostly-synth — the REAL number is the held-out eval below.
2. **Re-baseline v7 AND eval v8 on the held-out sets** using
   `MLTableDetection/evaluate_model.py --pdf` (heuristic=GT, conf 0.5, IoU 0.5),
   on Paperspace. Set A (v7 saw these = unfair edge to v7):
   `~/Documents/pdfToScan/derek2.pdf`, `derekfirst.pdf`. Set B (fully
   independent, never trained): the 3 `~/Documents/Diagrams/` PDFs. Run each PDF
   for BOTH `--model .../models_v7/best` and `--model .../models_v8/best`.
   **Write ALL outputs under NEW dir
   `~/Documents/TableAnnotations/baselines/v8_2026-07-10/` — NEVER touch
   read-only `baselines/v7_2026-05-15/`.** Layout:
   `baselines/v8_2026-07-10/{v7,v8}/<pdf_stem>/`. Success criterion: if v8
   (never saw derek2/derekfirst) matches/beats v7 on set A, and wins on set B,
   the label-fix + generalization is proven. Focus per-PDF recall + FN.
3. Write `baselines/v8_2026-07-10/SUMMARY.md` (v7 format): per-PDF mAP@0.5 v7 vs
   v8, recall, FN counts, headline conclusion.
4. **Verify anvil still works**: `ssh paperspace-vm 'systemctl is-active
   anvil-uplink.service'` (should be `active`); GPU released after eval
   (`nvidia-smi`); kill any lingering eval procs. Do NOT enable/disable/restart
   the service unless broken.
5. Report to Marco. If v8 wins the held-out fold, propose (don't auto-run) k-fold
   expansion OR a final v8-production retrain on ALL 22 docs (heuristic-relabeled,
   incl derek). Draft a `known-issues.mdc` update (per `doc-conventions.mdc`:
   document committed/operational facts only, NOT uncommitted working-tree
   state). Tell Marco the new script + synth edits are uncommitted (suggest a
   commit msg in chat; don't commit without his OK).

## GOTCHAS

- Heuristic boxes are PDF points not pixels; `TableDetectorML` and
  `PanelBoardSearch` use identical top-left conversion so eval is apples-to-apples.
- derek2 pages are ARCH-E (36") → 14400px at 400 DPI; v7/v8 eval renders these
  live for detection (slow, several min/PDF — expected, seen on 07-07).
  `evaluate_model.py` already sets `Image.MAX_IMAGE_PIXELS=None`.
- venv: `source /home/paperspace/venv/bin/activate` (Paperspace) /
  `/home/marco/venv` (laptop).
- Don't run v7+v8 eval concurrently with a customer job (GPU OOM); watch
  `nvidia-smi`.

Example eval command:

```bash
ssh paperspace-vm 'cd ~/ElectricalDiagramAnalyzer && source ~/venv/bin/activate && \
python MLTableDetection/evaluate_model.py --model ~/Documents/TableAnnotations/models_v8/best \
  --pdf ~/Documents/pdfToScan/derek2.pdf \
  --output ~/Documents/TableAnnotations/baselines/v8_2026-07-10/v8/derek2 --conf 0.5 --iou 0.5'
```
