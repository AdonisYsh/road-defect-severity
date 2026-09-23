"""Train one YOLO job from config.yaml (resumes automatically if a checkpoint exists)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import cfg, device_str, log, path, results, runs_dir, seed_all, step, weights_file


def train(job: str, epochs: int | None = None, time_hours: float | None = None, fraction: float = 1.0,
          device: str | None = None, batch: int | None = None) -> Path:
    j, y = cfg()["jobs"][job], cfg()["yolo"]
    from ultralytics import YOLO

    run_dir = runs_dir() / "detect" / job
    last, done = run_dir / "weights" / "last.pt", run_dir / "DONE"
    step(f"Train {job}: {j['model']} on data={j['data']} seed={j['seed']}")
    seed_all(j["seed"])
    if done.exists():
        log("already finished earlier; skipping training")
    elif last.exists():
        log(f"resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        t = time_hours if time_hours is not None else j.get("time_hours")
        args = dict(
            data=str(path("yolo") / f"{j['data']}.yaml"),
            epochs=epochs or j["epochs"],
            imgsz=y["imgsz"],
            batch=batch or j["batch"],
            seed=j["seed"],
            deterministic=True,
            patience=y["patience"],
            cos_lr=y["cos_lr"],
            close_mosaic=y["close_mosaic"],
            optimizer=y["optimizer"],
            cache=y["cache"],
            workers=min(y["workers"], os.cpu_count() or 2),
            device=device or device_str(),
            project=str(runs_dir() / "detect"),
            name=job,
            exist_ok=True,
            pretrained=True,
            amp=True,
            plots=True,
            fraction=fraction,
            **y["augment"],
        )
        if t and not epochs:
            args["time"] = float(t)
        log(f"args: {args}")
        YOLO(weights_file(j["model"])).train(**args)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"FAIL: {best} missing after training")
    done.write_text("ok\n")
    # collect
    wdir = results("weights")
    shutil.copy2(best, wdir / f"{job}.pt")
    tdir = results("training", job)
    for f in list(run_dir.glob("*.png")) + list(run_dir.glob("*.jpg")) + list(run_dir.glob("*.csv")) + list(run_dir.glob("*.yaml")):
        shutil.copy2(f, tdir / f.name)
    log(f"best checkpoint -> {wdir / f'{job}.pt'}")
    return best
