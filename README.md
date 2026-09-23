# Road damage detection with validated severity (RDD2022 India + Japan)

BCSE316L Design of Smart Cities, VIT Vellore: Devansh Rathore, Yash Chaubey, Divyanshu Singh.

**What it does:** it finds 4 kinds of road damage in phone/dash-cam photos, gives each one a severity from 1 to 5,
checks that severity against blind human ratings, and ranks ~100 m stretches of road by how urgently they need repair.

| code | damage |
|---|---|
| D00 | longitudinal crack (along the road) |
| D10 | transverse crack (across the road) |
| D20 | alligator crack (crazy-paving pattern) |
| D40 | pothole |

## Where each part runs

| Machine | Command | Takes |
|---|---|---|
| Laptop (RTX 4060) | `bash scripts/laptop.sh train` | about 4–5 h (YOLOv8s + SAM severity) |
| Kaggle (2× T4) | `notebooks/kaggle_frcnn.ipynb`, run with **Save & Run All** | about 5–7 h (Faster R-CNN) |
| Colab (T4) | `notebooks/colab_yolov8n.ipynb`, **Run all** | about 6 h (YOLOv8n × 3 seeds + 2 cross-country) |

All three can start at the same time. Every step is **safe to re-run**: finished work is skipped, interrupted
training resumes from its last checkpoint.

## Full order

1. Start all three runs (table above).
2. Laptop finishes → 3 people rate 150 defects, each on their own:
   `bash scripts/laptop.sh rate A`, then `rate B`, then `rate C` (~20 min each, progress saved after every click).
3. Download `results_kaggle.zip` (Kaggle version → Output) and `results_colab.zip` (downloaded by the notebook,
   also in Drive `MyDrive/road-defect/`) into `~/Downloads`.
4. `bash scripts/laptop.sh finish` merges everything, runs the severity statistics, writes `results/SUMMARY.md`,
   and uploads to Hugging Face if `HF_TOKEN` is set. It produces `~/Downloads/results_all.zip` to send to Claude.
5. Demo: `bash scripts/laptop.sh demo` (image + video tabs). Road video: `bash scripts/laptop.sh video clip.mp4`.

## Pipeline (each piece can also be run on its own)

| Step | Command | Output in `results/` |
|---|---|---|
| Download India + Japan (only those ~1.5 GB of the 13.3 GB official zip) | `python -m rdd.download` | `data/raw/` |
| Prep: XML→YOLO/COCO, drop other codes, dedupe, 70/15/15 stratified split (seed 42), thin backgrounds in train, oversample rare pairs | `python -m rdd.prep` | `dataset/` summary table, class charts, split lists |
| Train + evaluate a job | `python -m rdd.jobs <job>` | `weights/`, `training/<job>/`, `detection/<job>/` |
| Speed / size | `python -m rdd.benchmark` | `benchmark/` |
| Severity (horizon, SAM 2 masks, S = W·A·L) | `python -m rdd.severity` | `severity/` |
| Rating set / rating page | `python -m rdd.rating make` / `rate --rater A` | `rating/` |
| Severity validation (α, MAE, ρ, τ, CIs, 5-fold CV weights) | `python -m rdd.analyze` | `severity/validation.*` |
| Tables + SUMMARY.md | `python -m rdd.report` | `tables/`, `SUMMARY.md` |
| Hugging Face weights + Space | `python -m rdd.hf_release` | `release/` |

Jobs (in `config.yaml`): `yolov8s_main`, `yolov8n_seed0/1/2`, `xc_india`, `xc_japan`, `frcnn`.
Add `--smoke` for a 1-epoch plumbing check (its outputs are deleted afterwards).

## How the numbers are made (so you can explain them)

- **Test set is touched once, at the end.** Val picks the best epoch and the confidence cut-off (the one with the best
  mean F1 on val).
- **One evaluator for every model** (`rdd/evaluate.py`): COCO-style mAP@0.5 and mAP@0.5:0.95 (torchmetrics/pycocotools),
  per-class precision/recall/F1 at the val cut-off, and a confusion matrix built the same way for YOLO and Faster R-CNN.
  Ultralytics' own val numbers are saved too, but compare models using the common table.
- Every metric is reported for **combined, India-only and Japan-only** test images. Cross-country jobs train on one
  country and are tested on both.
- **Severity** `S = W_class × A × L`: A is the damaged area corrected for distance using a horizon row estimated per country
  from the training labels (`severity/horizon_fit.png` shows the fit); L is 1.2 in wheel paths, 1.0 centre, 0.8 edges;
  W starts at pothole 1.0 / alligator 0.7 / transverse 0.4 / longitudinal 0.3 and is tuned by 5-fold CV on the human
  ratings. S is cut into levels 1–5.
- **Validation**: 150 correctly-detected defects, balanced across classes and severity range, rated blind by 3 people.
  Reported: Krippendorff's α (do raters agree?), MAE, Spearman ρ, Kendall τ, each with bootstrap 95% CI, for class-only
  (baseline), box-area and SAM-mask-area severity.

## Config

Everything (paths, split, hyperparameters, time budgets, severity constants) lives in `config.yaml`. Useful knobs:
`jobs.<job>.time_hours` (training stops to fit this), `jobs.<job>.batch`, `severity.horizon_override` (set a number if
the horizon plot looks wrong), `video.segment_seconds`.

## Licence and credit

Code and weights: AGPL-3.0 (Ultralytics YOLO is AGPL-3.0). Data: RDD2022, Arya et al., *Geoscience Data Journal* (2024),
doi:10.1002/gdj3.260, figshare doi:10.6084/m9.figshare.21431547 (CC BY 4.0).
