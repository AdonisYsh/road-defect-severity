#!/usr/bin/env bash
# Laptop runner (RTX 4060). Every stage is safe to re-run: finished work is skipped.
#
#   bash scripts/laptop.sh train        data prep -> YOLOv8s -> eval -> SAM + severity -> rating set   (~4-5 h)
#   bash scripts/laptop.sh rate A       rating page for rater A (then B, then C)
#   bash scripts/laptop.sh finish       merge Kaggle/Colab zips from ~/Downloads -> benchmark -> severity stats
#                                       -> SUMMARY.md -> Hugging Face upload (if HF_TOKEN is set)
#   bash scripts/laptop.sh demo         Gradio demo at http://127.0.0.1:7860
#   bash scripts/laptop.sh video FILE   ranked 100 m segment list for a road video
set -euo pipefail

ROOT="$HOME/Projects/road-defect"
STAGE="${1:-}"
CURRENT="start"
trap 'echo; echo "FAIL: stopped at step: $CURRENT  (send Claude the last ~30 lines)"' ERR
cd "$ROOT"

say() { CURRENT="$1"; echo; echo "==> $1"; }

run() {
  if command -v direnv >/dev/null 2>&1 && [ -f .envrc ]; then
    direnv exec "$ROOT" "$@"
  else
    nix develop "$ROOT" --command bash -c 'source .venv/bin/activate && "$@"' _ "$@"
  fi
}

check() {
  local f
  for f in "$@"; do
    if [ ! -s "$f" ]; then echo "FAIL: expected output missing: $f"; exit 1; fi
  done
}

backup_results() {
  if [ -d results ]; then
    local b=".backup/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$b"
    tar -cf "$b/results.tar" results
    echo "backed up results/ -> $b/results.tar"
  fi
}

case "$STAGE" in
  train)
    say "1/7 GPU check"
    run python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
    say "2/7 dataset present (downloads only if missing)"
    run python -m rdd.download
    say "3/7 data prep (split, labels, charts)"
    if [ -s data/coco/test.json ] && [ -s results/dataset/dataset_summary.csv ]; then
      echo "already prepared; skipping (delete data/yolo to redo)"
    else
      run python -m rdd.prep
    fi
    say "4/7 train + evaluate YOLOv8s (the long one: watch the first epoch time)"
    run python -m rdd.jobs yolov8s_main
    say "5/7 benchmark YOLOv8s (RTX 4060 + CPU)"
    run python -m rdd.benchmark
    say "6/7 severity: horizon + SAM 2 masks on val/test detections"
    run python -m rdd.severity
    say "7/7 build the 150-defect rating set"
    [ -s results/rating/key_DO_NOT_SHOW_RATERS.csv ] && echo "rating set already exists; keeping it (ratings may be in progress)" \
      || run python -m rdd.rating make
    check results/weights/yolov8s_main.pt results/detection/yolov8s_main/metrics.csv \
          results/severity/detections_test.csv results/rating/key_DO_NOT_SHOW_RATERS.csv
    echo
    echo "PASS: laptop training + severity done"
    echo "Next: each rater runs  bash scripts/laptop.sh rate A   (then B, C)."
    ;;
  rate)
    R="${2:?give a rater letter, e.g. A}"
    say "rating page for rater $R (Ctrl-C when done; progress is saved after every click)"
    run python -m rdd.rating rate --rater "$R"
    ;;
  finish)
    backup_results
    say "1/5 merge cloud results from ~/Downloads (if present)"
    zips=()
    for z in "$HOME/Downloads/results_kaggle.zip" "$HOME/Downloads/results_colab.zip"; do
      [ -f "$z" ] && zips+=("$z") || echo "not found (skipping): $z"
    done
    [ ${#zips[@]} -gt 0 ] && run python -m rdd.report --merge "${zips[@]}"
    say "2/5 benchmark all models on the RTX 4060 + CPU"
    run python -m rdd.benchmark
    say "3/5 severity validation (needs >= 2 finished raters)"
    run python -m rdd.analyze
    say "4/5 summary tables + SUMMARY.md"
    run python -m rdd.report
    say "5/5 Hugging Face release"
    if [ -n "${HF_TOKEN:-}" ]; then run python -m rdd.hf_release; else run python -m rdd.hf_release --dry-run; echo "HF_TOKEN not set: built release/ but did not upload"; fi
    check results/SUMMARY.md results/severity/validation.csv results/severity/severity_config.json
    say "zip results for Claude"
    run python - <<'PY'
import zipfile, pathlib
out = pathlib.Path.home() / "Downloads" / "results_all.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in pathlib.Path("results").rglob("*"):
        if f.is_file() and not str(f).startswith(("results/weights", "results/rating/crops")):
            z.write(f)
print("wrote", out, round(out.stat().st_size / 1e6, 1), "MB")
PY
    echo
    echo "PASS: all results in results/SUMMARY.md — send ~/Downloads/results_all.zip to Claude"
    ;;
  demo)
    say "demo on http://127.0.0.1:7860"
    run python app/app.py
    ;;
  video)
    F="${2:?give a video file}"
    say "segment ranking for $F"
    run python -m rdd.video "$F"
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
