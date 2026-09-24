"""Train one YOLO job from config.yaml (resumes automatically if a checkpoint exists)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import cfg, device_str, log, path, results, runs_dir, seed_all, step, weights_file


def _finished(ckpt: Path) -> bool:
    """Ultralytics strips the optimizer from last.pt when training completes; such a file can't be resumed."""
    import torch

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return ck.get("optimizer") is None or ck.get("epoch", -1) == -1


def train(job: str, epochs: int | None = None, time_hours: float | None = None, fraction: float = 1.0,
          device: str | None = None, batch: int | None = None) -> Path:
    j, y = cfg()["jobs"][job], cfg()["yolo"]
    from ultralytics import YOLO

    run_dir = runs_dir() / "detect" / job
    last, done = run_dir / "weights" / "last.pt", run_dir / "DONE"
    step(f"Train {job}: {j['model']} on data={j['data']} seed={j['seed']}")
    seed_all(j["seed"])
    bs = batch or j["batch"]
    while not done.exists():
        try:
            if last.exists() and _finished(last):
                log("training had already reached its last epoch; using its best.pt")
                break
            if last.exists():
                log(f"resuming from {last} (batch {bs})")
                YOLO(str(last)).train(resume=True, batch=bs)
            else:
                t = time_hours if time_hours is not None else j.get("time_hours")
                args = dict(
                    data=str(path("yolo") / f"{j['data']}.yaml"),
                    epochs=epochs or j["epochs"],
                    imgsz=y["imgsz"],
                    batch=bs,
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
            break
        except Exception as e:  # noqa: BLE001
            if "nothing to resume" in str(e).lower() and (run_dir / "weights" / "best.pt").exists():
                log("training had already reached its last epoch; using its best.pt")
                break
            if "out of memory" not in str(e).lower() or bs <= 2:
                raise
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
            bs //= 2
            log(f"GPU ran out of memory -> retrying with batch {bs} (results stay comparable: "
                f"Ultralytics accumulates gradients to an effective batch of 64 either way)")
    if done.exists() and not (run_dir / "weights" / "best.pt").exists():
        raise SystemExit("FAIL: DONE marker but no best.pt")
    if done.exists():
        log("training finished (earlier or just now)")
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
