"""Shared helpers: config, environment detection, paths, logging, small maths."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
_CFG = None


def detect_env() -> str:
    if os.environ.get("RDD_ENV"):
        return os.environ["RDD_ENV"]
    if os.path.isdir("/kaggle/working"):
        return "kaggle"
    if "google.colab" in sys.modules or os.path.isdir("/content/sample_data") or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    return "local"


def cfg() -> dict:
    global _CFG
    if _CFG is None:
        with open(os.environ.get("RDD_CONFIG", REPO / "config.yaml")) as f:
            _CFG = yaml.safe_load(f)
    return _CFG


def root() -> Path:
    """Project root. The repo folder itself unless RDD_ROOT says otherwise."""
    if os.environ.get("RDD_ROOT"):
        return Path(os.environ["RDD_ROOT"]).expanduser().resolve()
    return REPO


def path(key: str) -> Path:
    p = Path(cfg()["paths"][key]).expanduser()
    return p if p.is_absolute() else root() / p


def runs_dir() -> Path:
    """Colab: checkpoints go to Google Drive (if mounted) so a disconnect loses nothing."""
    if detect_env() == "colab":
        drive = Path(cfg()["paths"]["colab_drive"])
        if drive.parent.exists():
            d = drive / "runs"
            d.mkdir(parents=True, exist_ok=True)
            return d
    d = path("runs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def results(*parts: str) -> Path:
    d = path("results").joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


CLASSES = None


def classes() -> list[str]:
    return cfg()["dataset"]["classes"]


def class_names() -> list[str]:
    return cfg()["dataset"]["class_names"]


def country_of(name: str) -> str:
    n = Path(name).name
    for c in cfg()["dataset"]["countries"]:
        if n.startswith(c + "_"):
            return c
    return n.split("_")[0]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def save_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    p.write_text(json.dumps(obj, indent=2, default=conv))


def load_json(p: Path):
    return json.loads(Path(p).read_text())


def device_str() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    import platform

    return f"CPU ({platform.processor() or platform.machine()})"


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy arrays -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def weights_file(name: str) -> str:
    """Prefer a local copy in weights/ (downloaded by the preflight), else let the library download it."""
    local = path("weights") / name
    return str(local) if local.exists() else name


def job_weights(job: str) -> Path:
    """Best checkpoint of a finished job, as collected into results/weights/."""
    ext = ".pth" if cfg()["jobs"][job]["kind"] == "frcnn" else ".pt"
    return path("results") / "weights" / f"{job}{ext}"


if __name__ == "__main__":
    print("env =", detect_env(), "| root =", root(), "| runs =", runs_dir(), "| device =", device_str())
