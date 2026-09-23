"""Hugging Face release: open weights + model card, and the Gradio demo Space.

  python -m rdd.hf_release              export ONNX, write model card, upload model repo + Space
  python -m rdd.hf_release --dry-run    build everything in release/ without uploading
Needs HF_TOKEN (a *write* token) in the environment.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from .common import REPO, cfg, classes, class_names, job_weights, log, path, results, root, step

CITATION = """@article{arya2024rdd2022,
  title   = {RDD2022: A multi-national image dataset for automatic road damage detection},
  author  = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
  journal = {Geoscience Data Journal},
  year    = {2024},
  doi     = {10.1002/gdj3.260}
}"""


def export_onnx(job):
    from ultralytics import YOLO

    w = job_weights(job)
    if not w.exists():
        return None
    onnx = w.with_suffix(".onnx")
    if not onnx.exists():
        f = YOLO(str(w)).export(format="onnx", imgsz=cfg()["yolo"]["imgsz"], simplify=True, dynamic=False)
        shutil.move(f, onnx) if Path(f) != onnx else None
    return onnx


def model_card(user, space):
    s = path("results")
    parts = [f"""---
license: {cfg()['hf']['license']}
library_name: ultralytics
pipeline_tag: object-detection
tags: [object-detection, yolov8, road-damage, pothole, rdd2022, smart-city]
---

# Road damage detection + validated severity (RDD2022 India + Japan)

YOLOv8 detectors for 4 road-damage types, trained on the official RDD2022 release (India + Japan), plus a
perspective-corrected severity score (1–5) validated against blind human raters.
Course project, BCSE316L Design of Smart Cities, VIT Vellore (Devansh Rathore, Yash Chaubey, Divyanshu Singh).

**Demo:** https://huggingface.co/spaces/{user}/{space}

| id | code | damage |
|---|---|---|
""" + "\n".join(f"| {i} | {c} | {n.replace('_', ' ')} |" for i, (c, n) in enumerate(zip(classes(), class_names()))),
             """
## Files
- `yolov8s_main.pt` / `.onnx` — main detector (YOLOv8s, 640 px)
- `yolov8n_seed0.pt` / `.onnx` — lightweight detector for phones/edge (YOLOv8n)
- `severity_config.json` — horizon rows, class weights and level cut-points for the severity score

```python
from ultralytics import YOLO
model = YOLO("yolov8s_main.pt")
model.predict("road.jpg", conf=0.25)
```

## Data and split
RDD2022 (Arya et al., 2024) — India (7,706 labelled images) + Japan (10,506). The official test labels are not public,
so all labelled images were split 70/15/15, stratified by country × rarest class, seed 42 (file lists in the project repo).
Non-target codes (D43, D44, D50, …) were dropped. All numbers below are on the held-out 15% test split.
"""]
    summ = s / "SUMMARY.md"
    if summ.exists():
        body = summ.read_text().split("\n", 1)[1]
        parts.append("## Results\n" + body.replace("## ", "### "))
    fails = s / "failures" / "failures.csv"
    if fails.exists():
        d = pd.read_csv(fails)
        why = d[d.why.notna() & (d.why.astype(str).str.strip() != "")]
        if len(why):
            parts.append("## Known failure cases\n" + "\n".join(f"- {r.kind}: {r.why}" for r in why.drop_duplicates("why").itertuples()))
    parts.append("""
## Intended use and limits
- **Decision support** for prioritising road inspection/repair. **Not** an automatic repair order.
- Daytime RGB phone/dash-cam images; trained on India + Japan roads. Expect lower accuracy on other countries,
  night, rain, or unusual camera angles.
- Severity uses a flat-road, fixed-horizon assumption (no camera calibration); pothole depth is not measured.
- Human ratings behind the severity validation are partly subjective; inter-rater agreement is reported.

## Licence
- Weights: **AGPL-3.0** (trained with Ultralytics YOLO, which is AGPL-3.0).
- Training data: RDD2022 by Arya et al., released on figshare (doi:10.6084/m9.figshare.21431547) under CC BY 4.0
  (the older GitHub README states CC BY-SA 4.0). Please credit the dataset:

```bibtex
""" + CITATION + "\n```\n")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    step("Hugging Face release")
    rel = root() / "release"
    shutil.rmtree(rel, ignore_errors=True)
    mdir, sdir = rel / "model", rel / "space"
    mdir.mkdir(parents=True)
    sdir.mkdir(parents=True)
    token = os.environ.get("HF_TOKEN")
    user = "YOUR_HF_USERNAME"
    if token and not a.dry_run:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user = api.whoami()["name"]
    h = cfg()["hf"]
    for job in ("yolov8s_main", "yolov8n_seed0"):
        w = job_weights(job)
        if w.exists():
            shutil.copy2(w, mdir / w.name)
            o = export_onnx(job)
            if o:
                shutil.copy2(o, mdir / o.name)
    sc = results("severity") / "severity_config.json"
    if sc.exists():
        shutil.copy2(sc, mdir / sc.name)
    (mdir / "README.md").write_text(model_card(user, h["space_repo"]))
    for f in ("results/SUMMARY.md",):
        if (root() / f).exists():
            shutil.copy2(root() / f, mdir / "SUMMARY.md")
    fl = path("results") / "failures"
    if fl.exists():
        (mdir / "failures").mkdir()
        for f in fl.glob("*_sheet.jpg"):
            shutil.copy2(f, mdir / "failures" / f.name)

    # Space: app + package + weights
    shutil.copy2(REPO / "app" / "app.py", sdir / "app.py")
    shutil.copytree(REPO / "rdd", sdir / "rdd", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "config.yaml", sdir / "config.yaml")
    shutil.copy2(job_weights(cfg()["severity"]["detector_job"]), sdir / "model.pt")
    if sc.exists():
        shutil.copy2(sc, sdir / "severity_config.json")
    shutil.copy2(REPO / "app" / "requirements.txt", sdir / "requirements.txt")
    (sdir / "packages.txt").write_text("libgl1\nlibglib2.0-0\n")
    (sdir / "README.md").write_text(f"""---
title: Road Damage Severity
emoji: 🛣️
colorFrom: gray
colorTo: red
sdk: gradio
app_file: app.py
license: agpl-3.0
pinned: false
---
Demo for [{user}/{h['model_repo']}](https://huggingface.co/{user}/{h['model_repo']}).
""")
    log(f"release folders ready in {rel}")
    if a.dry_run or not token:
        log("dry run (or HF_TOKEN not set): nothing uploaded")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    mid, sid = f"{user}/{h['model_repo']}", f"{user}/{h['space_repo']}"
    api.create_repo(mid, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(mdir), repo_id=mid, repo_type="model", commit_message="Upload weights + model card")
    api.create_repo(sid, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(sdir), repo_id=sid, repo_type="space", commit_message="Upload demo")
    log(f"model: https://huggingface.co/{mid}")
    log(f"demo:  https://huggingface.co/spaces/{sid}  (first build takes ~5 min)")


if __name__ == "__main__":
    main()
