"""Run training jobs from config.yaml, then evaluate them. Safe to re-run: finished jobs are skipped.

  python -m rdd.jobs yolov8s_main
  python -m rdd.jobs yolov8n_seed0 yolov8n_seed1 yolov8n_seed2 xc_india xc_japan
  python -m rdd.jobs frcnn            (uses every visible GPU via torchrun)
  add --smoke for a 1-epoch tiny run that only checks the plumbing
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .common import REPO, cfg, log, results, step


def run_job(job: str, smoke=False, force=False, device=None):
    j = cfg()["jobs"][job]
    if (results("detection", job) / "metrics.csv").exists() and not force and not smoke:
        log(f"{job}: already trained and evaluated (results/detection/{job}); skipping. Use --force to redo.")
        return
    if j["kind"] == "yolo":
        from .train_yolo import train

        kw = dict(epochs=1, fraction=0.05) if smoke else {}
        train(job, device=device, **kw)
    else:
        import torch

        n = torch.cuda.device_count()
        step(f"Train Faster R-CNN on {max(n, 1)} device(s)")
        env = dict(os.environ)
        if n > 1 and not smoke:
            cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n}", "-m", "rdd.frcnn"]
            subprocess.run(cmd, check=True, cwd=REPO, env=env)
        else:
            from .frcnn import train

            train(max_iters=20 if smoke else None, epochs=1 if smoke else None)
    from .evaluate import evaluate_job

    evaluate_job(job)
    if smoke and not os.environ.get("RDD_KEEP_SMOKE"):  # never let a smoke run pass for a finished real run
        import shutil

        from .common import job_weights, runs_dir

        for p in (runs_dir() / "detect" / job, runs_dir() / "frcnn" if j["kind"] == "frcnn" else None,
                  results("detection", job), results("training", job)):
            if p is not None:
                shutil.rmtree(p, ignore_errors=True)
        for p in [job_weights(job), *(runs_dir() / "preds").glob(f"{job}_*.pkl")]:
            p.unlink(missing_ok=True)
        log(f"{job}: smoke test OK (its outputs were deleted so the real run starts clean)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", choices=list(cfg()["jobs"]))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None, help="e.g. 0, 1 or 0,1")
    a = ap.parse_args()
    for job in a.jobs:
        run_job(job, a.smoke, a.force, a.device)
    log("jobs finished: " + ", ".join(a.jobs))


if __name__ == "__main__":
    main()
