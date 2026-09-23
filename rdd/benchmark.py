"""Speed / size benchmark: FPS on this machine's GPU and CPU, params, GFLOPs, file size.

Writes results/benchmark/benchmark_<machine>.csv, so rows from laptop, Kaggle and Colab can be merged.
  python -m rdd.benchmark                      (every trained model present in results/weights)
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import cfg, detect_env, gpu_name, job_weights, log, results, step
from .evaluate import image_paths

JOBS = ["yolov8s_main", "yolov8n_seed0", "frcnn"]


def _time(fn, paths, warm=5):
    for p in paths[:warm]:
        fn(p)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for p in paths:
        fn(p)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return len(paths) / (time.perf_counter() - t)


def bench_yolo(job, dev, paths):
    from ultralytics import YOLO

    from ultralytics.utils.torch_utils import get_flops

    m = YOLO(str(job_weights(job)))
    params = sum(p.numel() for p in m.model.parameters())
    gflops = get_flops(m.model, cfg()["benchmark"]["imgsz"])
    fps = _time(lambda p: m.predict(p, imgsz=cfg()["benchmark"]["imgsz"], device=dev, verbose=False, half=dev != "cpu"), paths)
    return dict(params_M=params / 1e6, GFLOPs=gflops or np.nan, fps=fps)


def bench_frcnn(job, dev, paths):
    from .frcnn import build_model, read_image

    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(job_weights(job), map_location="cpu"))
    d = torch.device("cuda" if dev != "cpu" else "cpu")
    m.to(d).eval()
    params = sum(p.numel() for p in m.parameters())
    gflops = np.nan
    try:
        from torch.utils.flop_counter import FlopCounterMode

        x = torch.rand(3, 640, 640, device=d)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m([x])
        gflops = fc.get_total_flops() / 2e9  # multiply-adds -> GFLOPs (same convention as Ultralytics)
    except Exception as e:  # noqa: BLE001
        log(f"GFLOPs count failed for Faster R-CNN ({e})")

    @torch.no_grad()
    def run(p):
        with torch.autocast("cuda", enabled=d.type == "cuda"):
            m([read_image(p).to(d)])

    return dict(params_M=params / 1e6, GFLOPs=gflops, fps=_time(run, paths))


def main():
    step("Benchmark")
    b = cfg()["benchmark"]
    test = image_paths("test")
    rows = []
    devices = (["0"] if torch.cuda.is_available() else []) + ["cpu"]
    for job in JOBS:
        w = job_weights(job)
        if not w.exists():
            log(f"{job}: no weights at {w}, skipping")
            continue
        for dev in devices:
            n = b["gpu_images"] if dev != "cpu" else b["cpu_images"]
            paths = test[:n]
            fn = bench_frcnn if job == "frcnn" else bench_yolo
            r = fn(job, dev, paths)
            r.update(job=job, device=gpu_name() if dev != "cpu" else "CPU", env=detect_env(),
                     size_MB=Path(w).stat().st_size / 1e6, images=len(paths))
            log(f"{job} on {r['device']}: {r['fps']:.1f} img/s, {r['params_M']:.1f} M params, {r['GFLOPs']:.1f} GFLOPs")
            rows.append(r)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", gpu_name()).strip("_")
    out = results("benchmark") / f"benchmark_{detect_env()}_{slug}.csv"
    pd.DataFrame(rows)[["job", "env", "device", "fps", "params_M", "GFLOPs", "size_MB", "images"]].to_csv(out, index=False)
    log(f"-> {out}")


if __name__ == "__main__":
    main()
