"""Collect every result into tables + one SUMMARY.md, and move results between machines.

  python -m rdd.report                         build results/tables/*.csv and results/SUMMARY.md
  python -m rdd.report --pack                  zip results/ -> results_<env>.zip (Colab: also copied to Drive)
  python -m rdd.report --merge a.zip b.zip     unzip cloud results into this machine's results/, then rebuild
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from .common import cfg, classes, detect_env, log, path, results, root, step


def _cat(pattern):
    fs = sorted(path("results").glob(pattern))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def fmt(m, s=None):
    return f"{m:.3f}" if s is None or pd.isna(s) else f"{m:.3f} ± {s:.3f}"


def build():
    step("Build summary tables")
    t = results("tables")
    md = ["# Results summary", "", "All detection numbers are on OUR held-out TEST split (15%), evaluated with one common "
          "COCO-style evaluator for every model. Confidence cut-off chosen on val.", ""]
    det = _cat("detection/*/metrics.csv")
    pc = _cat("detection/*/per_class.csv")
    if len(det):
        det.to_csv(t / "detection_all.csv", index=False)
        pc.to_csv(t / "per_class_all.csv", index=False)
        seeds = det[det.job.str.startswith("yolov8n_seed")]
        md += ["## Detection — combined test set", "", "| model | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 | India mAP@0.5 | Japan mAP@0.5 |",
               "|---|---|---|---|---|---|---|---|"]
        rows = []
        for job, name in (("yolov8s_main", "YOLOv8s (baseline)"), ("frcnn", "Faster R-CNN R50-FPN v2")):
            d = det[det.job == job].set_index("test")
            if len(d):
                rows.append(dict(model=name, **{k: d.loc["all", k] for k in ("map50", "map50_95", "P", "R", "F1")},
                                 India=d.loc["India", "map50"], Japan=d.loc["Japan", "map50"]))
                md.append(f"| {name} | {d.loc['all','map50']:.3f} | {d.loc['all','map50_95']:.3f} | {d.loc['all','P']:.3f} | "
                          f"{d.loc['all','R']:.3f} | {d.loc['all','F1']:.3f} | {d.loc['India','map50']:.3f} | {d.loc['Japan','map50']:.3f} |")
        if len(seeds):
            g = seeds.groupby("test")
            mu, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
            n = seeds.job.nunique()
            md.append(f"| YOLOv8n (mean ± std, {n} seeds) | {fmt(mu.loc['all','map50'], sd.loc['all','map50'])} | "
                      f"{fmt(mu.loc['all','map50_95'], sd.loc['all','map50_95'])} | {mu.loc['all','P']:.3f} | {mu.loc['all','R']:.3f} | "
                      f"{mu.loc['all','F1']:.3f} | {fmt(mu.loc['India','map50'], sd.loc['India','map50'])} | "
                      f"{fmt(mu.loc['Japan','map50'], sd.loc['Japan','map50'])} |")
            pd.concat({"mean": mu, "std": sd}, axis=1).to_csv(t / "yolov8n_seeds_mean_std.csv")
        pd.DataFrame(rows).to_csv(t / "detection_main.csv", index=False)
        # per class
        md += ["", "## Per-class AP@0.5 (combined test)", "", "| model | " + " | ".join(classes()) + " |", "|---|" + "---|" * len(classes())]
        for job in pc.job.unique():
            d = pc[(pc.job == job) & (pc.test == "all")].set_index("cls")
            md.append(f"| {job} | " + " | ".join(f"{d.loc[c,'AP50']:.3f}" for c in classes()) + " |")
        # cross-country
        xc = det[det.job.isin(["xc_india", "xc_japan"])]
        if len(xc) or len(seeds):
            md += ["", "## Cross-country (YOLOv8n, mAP@0.5 on test)", "", "| trained on ↓ / tested on → | India | Japan |", "|---|---|---|"]
            mat = []
            for job, lab in (("xc_india", "India only"), ("xc_japan", "Japan only")):
                d = xc[xc.job == job].set_index("test")
                if len(d):
                    md.append(f"| {lab} | {d.loc['India','map50']:.3f} | {d.loc['Japan','map50']:.3f} |")
                    mat.append(dict(train=lab, India=d.loc["India", "map50"], Japan=d.loc["Japan", "map50"]))
            if len(seeds):
                md.append(f"| India + Japan (seed mean) | {mu.loc['India','map50']:.3f} | {mu.loc['Japan','map50']:.3f} |")
                mat.append(dict(train="India + Japan", India=mu.loc["India", "map50"], Japan=mu.loc["Japan", "map50"]))
            pd.DataFrame(mat).to_csv(t / "cross_country.csv", index=False)
    ul = _cat("detection/*/ultralytics_val.csv")
    if len(ul):
        ul.to_csv(t / "ultralytics_val_all.csv", index=False)
        md += ["", "(Ultralytics' own val numbers for the YOLO models are in tables/ultralytics_val_all.csv; they use a "
               "slightly different AP interpolation, so compare models with the table above.)"]
    bench = _cat("benchmark/*.csv")
    if len(bench):
        bench.to_csv(t / "benchmark_all.csv", index=False)
        md += ["", "## Speed and size (batch 1, 640 px, end-to-end incl. pre/post-processing)", "",
               "| model | device | img/s | params (M) | GFLOPs | file MB |", "|---|---|---|---|---|---|"]
        for r in bench.itertuples():
            md.append(f"| {r.job} | {r.device} | {r.fps:.1f} | {r.params_M:.1f} | {r.GFLOPs:.1f} | {r.size_MB:.1f} |")
    ds = path("results") / "dataset" / "dataset_summary.md"
    if ds.exists():
        md += ["", ds.read_text().replace("# Dataset summary", "## Dataset")]
    v = path("results") / "severity" / "validation.md"
    if v.exists():
        md += ["", v.read_text().replace("# Severity validation", "## Severity validation")]
    mb = path("results") / "severity" / "mask_vs_box_area.csv"
    if mb.exists():
        d = pd.read_csv(mb)
        md += ["", "## SAM mask area vs box area (test detections)", "", "| class | n | mean mask/box | median |", "|---|---|---|---|"]
        for r in d.itertuples():
            md.append(f"| {r.cls} | {int(r.count)} | {r.mean:.2f} | {r._4:.2f} |")
    (path("results") / "SUMMARY.md").write_text("\n".join(md) + "\n")
    log(f"-> {path('results') / 'SUMMARY.md'}")


def pack():
    env = detect_env()
    z = root() / f"results_{env}.zip"
    base = path("results")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in base.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(base.parent))
    log(f"packed -> {z} ({z.stat().st_size / 1e6:.1f} MB)")
    if env == "colab":
        drive = Path(cfg()["paths"]["colab_drive"])
        if drive.parent.exists():
            drive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(z, drive / z.name)
            log(f"copied to Google Drive: {drive / z.name}")
    return z


def merge(zips):
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if not m.startswith("results/"):
                    continue
                dst = path("results").parent / m
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as s, open(dst, "wb") as d:
                    shutil.copyfileobj(s, d)
        log(f"merged {z}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--merge", nargs="*")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge)
    build()
    if a.pack:
        pack()


if __name__ == "__main__":
    main()
