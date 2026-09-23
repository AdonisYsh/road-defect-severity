#!/usr/bin/env python3
# Road-defect project: preflight smoke test
# Runs unchanged on: laptop (inside the nix dev shell), Kaggle notebook cell, Colab notebook cell.
# Env switches: SKIP_DATA=1 (no dataset download), SKIP_GPU=1 (data checks only), PROJECT_ROOT=/path, FORCE_ENV=kaggle|colab|local

import os, re, sys, json, time, shutil, subprocess, platform, importlib.util, zipfile, urllib.request, urllib.error, urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# ---------- environment ----------
def detect_env():
    if os.environ.get("FORCE_ENV"):
        return os.environ["FORCE_ENV"]
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if Path("/kaggle/working").exists():
        return "kaggle"
    return "local"

ENV = detect_env()
DEFAULT_ROOT = {"kaggle": "/kaggle/working/road-defect", "colab": "/content/road-defect"}.get(ENV, os.getcwd())
ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_ROOT)).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data" / "raw"
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs_smoke"
REPORT_PATH = ROOT / "preflight_report.json"
SKIP_DATA = os.environ.get("SKIP_DATA") == "1"
SKIP_GPU = os.environ.get("SKIP_GPU") == "1"
DEVICE = os.environ.get("DEVICE", "0")  # testing only: DEVICE=cpu
TDEV = "cuda" if DEVICE != "cpu" else "cpu"
MIN_FREE_GB = {"local": 12, "kaggle": 5, "colab": 10}[ENV]

COUNTRIES = {
    "India": {"images": 7706, "D00": 1555, "D10": 68, "D20": 2021, "D40": 3187, "size": 720},
    "Japan": {"images": 10506, "D00": 4049, "D10": 3979, "D20": 6199, "D40": 2243, "size": 600},
}
TARGET = ["D00", "D10", "D20", "D40"]

REPORT = {"env": ENV, "root": str(ROOT), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}, "warnings": []}

# ---------- helpers ----------
def save_report():
    REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2, default=str))

CURRENT = {"step": "start"}

class PreflightFailed(Exception):
    pass

def step(title):
    CURRENT["step"] = title
    print(f"\n=== {title}", flush=True)

def ok(key, msg, **data):
    print(f"  OK    {msg}", flush=True)
    REPORT["checks"][key] = {"status": "ok", "msg": msg, **data}

def warn(key, msg, **data):
    print(f"  WARN  {msg}", flush=True)
    REPORT["checks"][key] = {"status": "warn", "msg": msg, **data}
    REPORT["warnings"].append(msg)

def fail(key, msg):
    print(f"  FAIL  {msg}", flush=True)
    REPORT["checks"][key] = {"status": "fail", "msg": msg}
    REPORT["result"] = "FAIL"
    save_report()
    print(f"\nFAIL: {msg}\nReport: {REPORT_PATH}", flush=True)
    raise PreflightFailed(msg)

def get_secret(name):
    if os.environ.get(name):
        return os.environ[name]
    if ENV == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            return None
    if ENV == "colab":
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            return None
    return None

# ---------- 1. basics ----------
def check_basics():
    step("1. Environment, Python, disk")
    ok("env", f"environment = {ENV}, project root = {ROOT}")
    ok("python", f"Python {platform.python_version()} on {platform.platform()}", version=platform.python_version())
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < MIN_FREE_GB:
        fail("disk", f"only {free_gb:.1f} GB free at {ROOT}, need >= {MIN_FREE_GB} GB")
    ok("disk", f"{free_gb:.1f} GB free", free_gb=round(free_gb, 1))
    try:
        urllib.request.urlopen("https://pypi.org", timeout=15)
        ok("internet", "internet reachable")
    except Exception as e:
        hint = {"kaggle": "right panel > Session options > Internet: ON (needs a phone-verified Kaggle account)",
                "colab": "restart the runtime (Runtime > Disconnect and delete runtime) and try again",
                "local": "check wifi / open the college captive-portal login page"}[ENV]
        fail("internet", f"no internet ({type(e).__name__}). Fix: {hint}")

# ---------- 2. packages ----------
PKGS = {
    "torch": "torch", "torchvision": "torchvision", "ultralytics": "ultralytics",
    "torchmetrics": "torchmetrics", "pycocotools": "pycocotools", "gradio": "gradio",
    "huggingface_hub": "huggingface_hub", "scipy": "scipy", "sklearn": "scikit-learn",
    "pandas": "pandas", "matplotlib": "matplotlib", "krippendorff": "krippendorff", "onnx": "onnx",
    "PIL": "pillow",
}

def check_packages():
    step("2. Python packages")
    missing = [pip for mod, pip in PKGS.items() if importlib.util.find_spec(mod) is None]
    if missing and ENV != "local":
        print(f"  installing: {' '.join(missing)}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        importlib.invalidate_caches()
        missing = [pip for mod, pip in PKGS.items() if importlib.util.find_spec(mod) is None]
    if missing:
        fail("packages", f"missing packages: {missing} (laptop: re-run setup_and_smoke.sh)")
    versions = {}
    for mod in ["torch", "torchvision", "ultralytics", "torchmetrics", "gradio", "huggingface_hub", "scipy", "sklearn", "pandas", "numpy"]:
        try:
            versions[mod] = importlib.import_module(mod).__version__
        except Exception:
            versions[mod] = "?"
    try:
        import cv2
        versions["opencv"] = cv2.__version__
    except Exception as e:
        fail("opencv", f"OpenCV import failed: {e} (laptop: libGL/glib missing from flake LD_LIBRARY_PATH)")
    ok("packages", "all packages import", versions=versions)
    REPORT["versions"] = versions
    for k, v in versions.items():
        print(f"        {k:16s} {v}")

# ---------- 3. GPU ----------
def check_gpu():
    step("3. GPU / CUDA")
    import torch
    if not torch.cuda.is_available():
        if torch.version.cuda is None and shutil.which("nvidia-smi") and ENV != "local":
            fail("cuda", f"a GPU is attached but torch {torch.__version__} is a CPU-only build. Fix: run "
                         "'!pip install -q --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126', "
                         "then restart the session/runtime and run this cell again")
        hint = {"kaggle": "right panel > Session options > Accelerator: GPU T4 x2 (the session restarts; run the cell again)",
                "colab": "Runtime > Change runtime type > T4 GPU > Save (the runtime restarts; run the cell again)",
                "local": "check /run/opengl-driver/lib/libcuda.so.1 exists; if the driver is older than the torch CUDA build, rm -rf .venv and re-run setup_and_smoke.sh"}[ENV]
        fail("cuda", f"torch.cuda.is_available() is False. Fix: {hint}")
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append({"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1), "capability": f"{p.major}.{p.minor}"})
    ok("cuda", f"{len(gpus)} GPU(s): " + ", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in gpus),
       gpus=gpus, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        REPORT["driver"] = drv
        print(f"        driver {drv}, torch CUDA {torch.version.cuda}")
    except Exception:
        warn("nvidia_smi", "nvidia-smi not callable (not fatal, torch sees the GPU)")
    x = torch.randn(4096, 4096, device=TDEV)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    tflops = 10 * 2 * 4096 ** 3 / (time.time() - t) / 1e12
    with torch.autocast("cuda", dtype=torch.float16):
        z = (x @ x).float().mean().item()
    if z != z:
        fail("gpu_math", "mixed-precision matmul returned NaN")
    ok("gpu_math", f"GPU matmul fp32 ~{tflops:.1f} TFLOPS, fp16 autocast OK", tflops_fp32=round(tflops, 1))

# ---------- 4. YOLO ----------
def check_yolo():
    step("4. YOLOv8: weights, inference, 1-epoch train, val metrics + confusion matrix")
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd(); os.chdir(WEIGHTS)
    try:
        y8s = YOLO("yolov8s.pt"); y8n = YOLO("yolov8n.pt")
    finally:
        os.chdir(cwd)
    img = str(ASSETS / "bus.jpg")
    r = y8s.predict(img, device=DEVICE, verbose=False)[0]
    if len(r.boxes) == 0:
        fail("yolo_predict", "YOLOv8s found 0 objects in the sample image")
    t = time.time()
    for _ in range(30):
        y8s.predict(img, device=DEVICE, verbose=False, imgsz=640)
    fps = 30 / (time.time() - t)
    ok("yolo_predict", f"YOLOv8s detects {len(r.boxes)} objects; ~{fps:.0f} img/s end-to-end", fps_yolov8s=round(fps, 1))
    REPORT["_bus_box"] = r.boxes.xyxy[0].tolist()

    res = YOLO(str(WEIGHTS / "yolov8n.pt")).train(
        data="coco8.yaml", epochs=1, imgsz=320, batch=8, device=DEVICE, workers=2,
        project=str(RUNS), name="coco8_train", exist_ok=True, plots=False, verbose=False)
    best = Path(res.save_dir) / "weights" / "best.pt"
    if not best.exists():
        fail("yolo_train", "training finished but best.pt was not written")
    ok("yolo_train", "1-epoch training on coco8 works (checkpoint saved)")

    m = YOLO(str(best)).val(data="coco8.yaml", imgsz=320, batch=8, device=DEVICE, plots=True,
                            project=str(RUNS), name="coco8_val", exist_ok=True, verbose=False)
    cm = Path(m.save_dir) / "confusion_matrix.png"
    if not cm.exists():
        fail("yolo_val", "val ran but confusion_matrix.png missing")
    ok("yolo_val", f"val metrics OK (mAP50={m.box.map50:.3f}), confusion matrix + curves saved", save_dir=str(m.save_dir))

# ---------- 5. SAM 2 ----------
def check_sam():
    step("5. SAM 2: box-prompted mask")
    from ultralytics import SAM
    from ultralytics.utils import ASSETS
    cwd = os.getcwd(); os.chdir(WEIGHTS)
    try:
        sam = SAM("sam2.1_b.pt")
    finally:
        os.chdir(cwd)
    box = REPORT.pop("_bus_box")
    res = sam(str(ASSETS / "bus.jpg"), bboxes=[box], device=DEVICE, verbose=False)[0]
    if res.masks is None or res.masks.data.shape[0] == 0:
        fail("sam", "SAM 2 returned no mask")
    area = int(res.masks.data[0].sum().item())
    box_area = int((box[2] - box[0]) * (box[3] - box[1]))
    ok("sam", f"SAM 2 mask OK: mask area {area} px vs box area {box_area} px")

# ---------- 6. Faster R-CNN ----------
def check_frcnn():
    step("6. Faster R-CNN (torchvision): inference + one training step with a 5-class head")
    import torch
    from torchvision.io import read_image
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from ultralytics.utils import ASSETS
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(TDEV).eval()
    img = (read_image(str(ASSETS / "bus.jpg"))[:3].float() / 255).to(TDEV)
    with torch.no_grad():
        out = model([img])[0]
    n = int((out["scores"] > 0.5).sum())
    if n == 0:
        fail("frcnn_predict", "Faster R-CNN found 0 objects in the sample image")
    ok("frcnn_predict", f"COCO Faster R-CNN detects {n} objects")
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 5).to(TDEV)
    model.train()
    target = [{"boxes": torch.tensor([[50.0, 50.0, 200.0, 200.0]], device=TDEV),
               "labels": torch.tensor([4], device=TDEV)}]
    loss = sum(model([img], target).values())
    loss_val = float(loss.detach())
    loss.backward()
    if not torch.isfinite(loss):
        fail("frcnn_train", "training loss is not finite")
    ok("frcnn_train", f"training step OK (loss {loss_val:.3f}, 4 classes + background)")
    del model; torch.cuda.empty_cache()

# ---------- 7. metrics + stats ----------
def check_metrics():
    step("7. Metric libraries (torchmetrics mAP, scipy, krippendorff)")
    import torch
    from torchmetrics.detection import MeanAveragePrecision
    mp = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    mp.update([{"boxes": torch.tensor([[10.0, 10, 50, 50]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}],
              [{"boxes": torch.tensor([[10.0, 10, 50, 50]]), "labels": torch.tensor([1])}])
    if abs(float(mp.compute()["map_50"]) - 1.0) > 1e-6:
        fail("torchmetrics", "torchmetrics mAP sanity check did not return 1.0")
    ok("torchmetrics", "torchmetrics mAP (pycocotools backend) OK")
    import numpy as np, krippendorff
    from scipy.stats import spearmanr, kendalltau
    a = np.array([1, 2, 3, 4, 5, 3, 2]); b = np.array([1, 2, 4, 4, 5, 3, 1])
    rho = spearmanr(a, b)[0]; tau = kendalltau(a, b)[0]
    alpha = krippendorff.alpha(reliability_data=np.vstack([a, b]), level_of_measurement="ordinal")
    ok("stats", f"spearman={rho:.2f} kendall={tau:.2f} krippendorff alpha={alpha:.2f} (dummy data)")

# ---------- 8. Hugging Face + Gradio ----------
def check_hf():
    step("8. Hugging Face token + Gradio")
    import gradio
    ok("gradio", f"gradio {gradio.__version__} imports")
    token = get_secret("HF_TOKEN")
    if not token:
        warn("hf", "HF_TOKEN not set: fine for now, needed only for the upload step "
                   "(laptop: export HF_TOKEN=...; Kaggle: Add-ons > Secrets; Colab: key icon > Secrets)")
        return
    try:
        from huggingface_hub import whoami
        info = whoami(token=token)
        role = info.get("auth", {}).get("accessToken", {}).get("role", "?")
        if role not in ("write", "fineGrained", "?"):
            warn("hf", f"logged in as {info['name']} but token role is '{role}': need a WRITE token to upload")
        else:
            ok("hf", f"Hugging Face login OK as {info['name']} (token role: {role})", hf_user=info["name"])
    except Exception as e:
        warn("hf", f"HF token rejected: {e}")

# ---------- 9. dataset ----------
FIGSHARE_URL = "https://ndownloader.figshare.com/files/38030910"   # official RDD2022 release (figshare 10.6084/m9.figshare.21431547)
FIGSHARE_BYTES = 13_264_172_619
FIGSHARE_NAME = "RDD2022_released_through_CRDDC2022.zip"
HF_REPO = "dronefreak/RDD2022"

def pick_zip_dir():
    if ENV == "local":
        return RAW
    cands = [Path(p) for p in ("/kaggle/tmp", "/tmp") if Path(p).exists()] + [RAW]
    return max(cands, key=lambda p: shutil.disk_usage(p).free)

def download(url, dest, expected=None, tries=6):
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if expected and have == expected:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0
                total = expected or (have + int(r.headers.get("Content-Length", 0)))
                mode = "ab" if have else "wb"
                done, t0, last = have, time.time(), -1
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(8 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        pct = int(100 * done / total) if total else 0
                        if pct // 5 != last:
                            last = pct // 5
                            speed = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"    {pct:3d}%  {done / 1e9:6.2f} / {total / 1e9:.2f} GB  ({speed:.0f} MB/s)", flush=True)
            if expected is None or tmp.stat().st_size == expected:
                break
        except Exception as e:
            print(f"    attempt {attempt}/{tries} interrupted ({type(e).__name__}: {e}); resuming in 10 s", flush=True)
            time.sleep(10)
    else:
        raise RuntimeError(f"download did not complete after {tries} attempts")
    size = tmp.stat().st_size
    if expected and size != expected:
        raise RuntimeError(f"size mismatch: got {size} bytes, expected {expected}")
    tmp.rename(dest)

def _norm(x):
    return re.sub(r"[^a-z]", "", x.lower().replace("rdd2022", ""))

def extract_country(zf, c, out):
    names = zf.namelist()
    key = _norm(c)
    zips = [n for n in names if n.lower().endswith(".zip") and _norm(Path(n).stem).startswith(key)]
    if zips:
        for zname in zips:
            print(f"  {c}: extracting nested {zname}", flush=True)
            inner_path = Path(zf.extract(zname, out.parent / "_nested"))
            with zipfile.ZipFile(inner_path) as inner:
                inner.extractall(out)
            inner_path.unlink()
        return f"nested-zip: {zips}"
    members = [n for n in names if not n.endswith("/") and any(_norm(part) == key for part in n.split("/")[:-1])]
    if not members:
        all_zips = [n for n in names if n.lower().endswith(".zip")]
        sample = names[:40]
        REPORT["official_zip_listing_sample"] = sample
        REPORT["official_zip_nested_zips"] = all_zips
        raise RuntimeError(f"{c}: could not find a {c} folder or zip inside the official zip. "
                           f"Nested zips: {all_zips[:15]} | first entries: {sample[:15]}")
    print(f"  {c}: extracting {len(members)} files", flush=True)
    for i, n in enumerate(members, 1):
        zf.extract(n, out)
        if i % 5000 == 0:
            print(f"        {i}/{len(members)}", flush=True)
    return "folder"

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def _resolve(url):
    opener = urllib.request.build_opener(_NoRedirect)
    for method in ("HEAD", "GET"):
        try:
            r = opener.open(urllib.request.Request(url, method=method, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"}), timeout=60)
            r.close()
            return url
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                return urllib.parse.urljoin(url, e.headers["Location"])
            if e.code in (403, 405) and method == "HEAD":
                continue
            raise
    return url

def _range_get(url, start, end):
    real = _resolve(url)
    req = urllib.request.Request(real, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end - 1}"})
    r = urllib.request.urlopen(req, timeout=60)
    if r.status != 206:
        r.close()
        raise RuntimeError(f"server ignored the byte-range request (HTTP {r.status})")
    return r

def _range_bytes(url, start, end):
    with _range_get(url, start, end) as r:
        return r.read()

def _range_to_file(url, start, end, dest, label, tries=8):
    total = end - start
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have >= total:
            break
        try:
            with _range_get(url, start + have, end) as r, open(tmp, "ab") as f:
                done, t0, last = have, time.time(), -1
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    pct = int(100 * done / total)
                    if pct // 10 != last:
                        last = pct // 10
                        speed = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                        print(f"    {label}: {pct:3d}%  {done / 1e9:5.2f} / {total / 1e9:.2f} GB  ({speed:.1f} MB/s)", flush=True)
        except Exception as e:
            print(f"    {label}: attempt {attempt}/{tries} interrupted ({type(e).__name__}: {e}); resuming in 10 s", flush=True)
            time.sleep(10)
    if not tmp.exists() or tmp.stat().st_size != total:
        raise RuntimeError(f"{label}: range download incomplete")
    tmp.rename(dest)

def _central_directory(url, total_size):
    import struct
    tail_len = min(total_size, 1 << 20)
    tail = _range_bytes(url, total_size - tail_len, total_size)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("zip end-of-central-directory record not found")
    _, _, _, _, n, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    j = tail.rfind(b"PK\x06\x07", 0, i)
    if j >= 0:
        (z64_eocd_off,) = struct.unpack("<Q", tail[j + 8:j + 16])
        rec = _range_bytes(url, z64_eocd_off, z64_eocd_off + 56)
        if rec[:4] != b"PK\x06\x06":
            raise RuntimeError("zip64 end record not found")
        n, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])
    cd = _range_bytes(url, cd_off, cd_off + cd_size)
    entries, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, _t, _d, _crc, csize, usize, nlen, xlen, clen, _dsk, _ia, _ea, off) = struct.unpack("<HHHIIIHHHHHII", cd[p + 10:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                vals, r = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    (usize,) = struct.unpack("<Q", vals[r:r + 8]); r += 8
                if csize == 0xFFFFFFFF:
                    (csize,) = struct.unpack("<Q", vals[r:r + 8]); r += 8
                if off == 0xFFFFFFFF:
                    (off,) = struct.unpack("<Q", vals[r:r + 8]); r += 8
            q += 4 + hlen
        entries.append({"name": name, "method": method, "csize": csize, "usize": usize, "offset": off})
        p += 46 + nlen + xlen + clen
    return entries, cd_off

def _extract_from_span(span_path, span_start, entry, out_path):
    import struct, zlib
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(span_path, "rb") as f:
        f.seek(entry["offset"] - span_start)
        hdr = f.read(30)
        if hdr[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {entry['name']}")
        nlen, xlen = struct.unpack("<HH", hdr[26:30])
        f.seek(nlen + xlen, 1)
        left = entry["csize"]
        d = zlib.decompressobj(-15) if entry["method"] == 8 else None
        if entry["method"] not in (0, 8):
            raise RuntimeError(f"unsupported compression method {entry['method']} for {entry['name']}")
        with open(out_path, "wb") as o:
            while left > 0:
                chunk = f.read(min(left, 8 << 20))
                if not chunk:
                    raise RuntimeError(f"span ended early while extracting {entry['name']}")
                left -= len(chunk)
                o.write(d.decompress(chunk) if d else chunk)
            if d:
                o.write(d.flush())
    if out_path.stat().st_size != entry["usize"]:
        raise RuntimeError(f"size mismatch after extracting {entry['name']}")

def fetch_official_partial():
    """Download only the India + Japan parts of the 13.3 GB official zip, using HTTP byte ranges."""
    print("  reading the official zip's table of contents remotely (a few MB)", flush=True)
    entries, cd_off = _central_directory(FIGSHARE_URL, FIGSHARE_BYTES)
    names = [e["name"] for e in entries]
    REPORT["official_zip_top_entries"] = sorted({n.split("/")[0] for n in names})[:40]
    REPORT["official_zip_second_level"] = sorted({"/".join(n.split("/")[:2]) for n in names})[:60]
    print(f"  zip contains {len(names)} entries; second level: {REPORT['official_zip_second_level'][:12]}", flush=True)
    by_off = sorted(entries, key=lambda e: e["offset"])
    next_off = {id(e): (by_off[k + 1]["offset"] if k + 1 < len(by_off) else cd_off) for k, e in enumerate(by_off)}
    work = RAW / "_partial"
    work.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for c in COUNTRIES:
        out = RAW / c
        if (out / ".extracted_ok").exists():
            continue
        key = _norm(c)
        picked = [e for e in entries if e["name"].lower().endswith(".zip") and _norm(Path(e["name"]).stem).startswith(key)]
        mode = "nested-zip"
        if not picked:
            picked = [e for e in entries if not e["name"].endswith("/") and any(_norm(part) == key for part in e["name"].split("/")[:-1])]
            mode = "folder"
        if not picked:
            raise RuntimeError(f"{c}: no {c} folder or zip found in the official zip listing")
        s = min(e["offset"] for e in picked)
        t = max(next_off[id(e)] for e in picked)
        total_bytes += t - s
        print(f"  {c}: {mode}, downloading {(t - s) / 1e9:.2f} GB of the 13.3 GB zip", flush=True)
        span = work / f"{c}.span"
        if not span.exists():
            _range_to_file(FIGSHARE_URL, s, t, span, c)
        for k, e in enumerate(picked, 1):
            if e["name"].endswith("/"):
                continue
            if mode == "nested-zip":
                inner = work / Path(e["name"]).name
                _extract_from_span(span, s, e, inner)
                print(f"  {c}: unpacking {inner.name}", flush=True)
                with zipfile.ZipFile(inner) as z:
                    z.extractall(out)
                inner.unlink()
            else:
                _extract_from_span(span, s, e, out / e["name"])
                if k % 5000 == 0:
                    print(f"        {k}/{len(picked)}", flush=True)
        span.unlink()
        (out / ".extracted_ok").write_text(f"official figshare zip (byte-range partial), {mode}, {time.strftime('%Y-%m-%d %H:%M:%S')}")
    shutil.rmtree(work, ignore_errors=True)
    for stale in [pick_zip_dir() / (FIGSHARE_NAME + ".part"), RAW / (FIGSHARE_NAME + ".part")]:
        if stale.exists():
            stale.unlink()
            print(f"  removed the old partial full-zip download {stale}", flush=True)
    REPORT["official_partial_download_gb"] = round(total_bytes / 1e9, 2)
    return True

def fetch_official():
    zdir = pick_zip_dir()
    zdir.mkdir(parents=True, exist_ok=True)
    z = zdir / FIGSHARE_NAME
    if not z.exists():
        need = (FIGSHARE_BYTES - (z.with_name(z.name + ".part").stat().st_size if z.with_name(z.name + ".part").exists() else 0)) / 1e9 + 1
        free = shutil.disk_usage(zdir).free / 1e9
        if free < need:
            raise RuntimeError(f"not enough disk for the 13.3 GB official zip at {zdir}: {free:.1f} GB free, need {need:.1f} GB")
        print(f"  downloading official RDD2022 zip (13.3 GB) from figshare into {zdir}", flush=True)
        download(FIGSHARE_URL, z, expected=FIGSHARE_BYTES)
    layout = {}
    with zipfile.ZipFile(z) as zf:
        _names = zf.namelist()
        REPORT["official_zip_top_entries"] = sorted({n.split("/")[0] for n in _names})[:40]
        REPORT["official_zip_second_level"] = sorted({"/".join(n.split("/")[:2]) for n in _names})[:60]
        print(f"  zip contains {len(_names)} entries; second level: {REPORT['official_zip_second_level'][:12]}", flush=True)
        for c in COUNTRIES:
            out = RAW / c
            if (out / ".extracted_ok").exists():
                continue
            layout[c] = extract_country(zf, c, out)
            (out / ".extracted_ok").write_text(f"official figshare zip, {layout[c]}, {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if ENV != "local":
        z.unlink()
        print("  deleted the big zip to free disk (cloud session)", flush=True)
    return layout

def fetch_hf():
    from huggingface_hub import snapshot_download
    out = RAW / "hf"
    pats = [f"*{c}_*" for c in COUNTRIES] + ["*.yaml", "*.md"]
    print(f"  downloading {', '.join(COUNTRIES)} files from Hugging Face {HF_REPO} (YOLO format)", flush=True)
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", allow_patterns=pats, local_dir=str(out), max_workers=16)
    (out / ".extracted_ok").write_text(f"hf {HF_REPO}, {time.strftime('%Y-%m-%d %H:%M:%S')}")

def check_dataset():
    step("9. RDD2022 India + Japan: download, integrity, counts vs paper")
    RAW.mkdir(parents=True, exist_ok=True)
    official_done = all((RAW / c / ".extracted_ok").exists() for c in COUNTRIES)
    hf_done = (RAW / "hf" / ".extracted_ok").exists()
    source = "official" if official_done else ("hf" if hf_done else None)
    if source is None:
        errors = []
        for label, fn, src in (("official zip, India + Japan parts only", fetch_official_partial, "official"),
                               ("official zip, full 13.3 GB", fetch_official, "official"),
                               (f"Hugging Face {HF_REPO}", fetch_hf, "hf")):
            try:
                print(f"  trying: {label}", flush=True)
                fn()
                source = src
                break
            except PreflightFailed:
                raise
            except Exception as e:
                errors.append(f"{label}: {type(e).__name__}: {e}")
                warn(f"download_{src}_{len(errors)}", f"{label} failed ({type(e).__name__}: {e}); trying the next source")
        if source is None:
            fail("dataset_download", "all sources failed. " + " | ".join(errors))
    else:
        print(f"  already downloaded ({source}), skipping")
    REPORT["dataset_source"] = {"official": f"figshare RDD2022 ({FIGSHARE_URL})", "hf": f"Hugging Face {HF_REPO}"}[source]
    summary = {}
    for c, meta in COUNTRIES.items():
        summary[c] = count_country(c, RAW / c, meta) if source == "official" else count_country_yolo(c, RAW / "hf", meta)
    REPORT["dataset"] = summary
    tot_imgs = sum(s["train_images"] for s in summary.values())
    tot = Counter()
    for s in summary.values():
        tot.update({k: s["class_counts"].get(k, 0) for k in TARGET})
    ok("dataset_total", f"[{source}] India + Japan: {tot_imgs} labelled images; " + ", ".join(f"{k}={tot[k]}" for k in TARGET))

def compare_to_paper(c, meta, n_imgs, classes, res, extra_issues=None):
    diffs = []
    if abs(n_imgs - meta["images"]) > 0.02 * meta["images"]:
        diffs.append(f"images {n_imgs} vs paper {meta['images']}")
    for k in TARGET:
        if abs(classes.get(k, 0) - meta[k]) > max(5, 0.02 * meta[k]):
            diffs.append(f"{k} {classes.get(k, 0)} vs paper {meta[k]}")
    if extra_issues:
        diffs.append(extra_issues)
    if diffs:
        warn(f"dataset_{c}", f"{c}: differs from paper/clean state: " + "; ".join(diffs), **res)
    else:
        ok(f"dataset_{c}", f"{c}: counts match the RDD2022 paper (within 2%)", **res)

def count_country_yolo(c, root, meta):
    import yaml
    names = TARGET
    ymls = sorted(root.rglob("*.yaml"))
    for y in ymls:
        try:
            d = yaml.safe_load(y.read_text())
            if isinstance(d, dict) and "names" in d:
                n = d["names"]
                names = [n[i] for i in sorted(n)] if isinstance(n, dict) else list(n)
                break
        except Exception:
            pass
    imgs = [p for p in root.rglob(f"{c}_*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    labels = [p for p in root.rglob(f"{c}_*.txt")]
    classes = Counter(); empty = 0; bad = 0
    for lf in labels:
        rows = [r.split() for r in lf.read_text().splitlines() if r.strip()]
        if not rows:
            empty += 1
        for r in rows:
            try:
                classes[str(names[int(r[0])])] += 1
            except (ValueError, IndexError):
                bad += 1
    code = {}
    for k in TARGET:
        for n in names:
            if k in str(n) or str(n).lower().startswith({"D00": "longitudinal", "D10": "transverse", "D20": "alligator", "D40": "pothole"}[k]):
                code[str(n)] = k
    classes = Counter({code.get(k, k): v for k, v in classes.items()})
    splits = Counter(p.parent.parent.name if p.parent.name.startswith("shard") else p.parent.name for p in imgs)
    res = {"source": "hf", "train_images": len(imgs), "label_files": len(labels), "class_names_in_yaml": [str(n) for n in names],
           "class_counts": dict(classes.most_common()), "images_with_no_damage": empty, "bad_label_rows": bad,
           "hf_split_folders": dict(splits.most_common(6)),
           "example_paths": [str(p.relative_to(root)) for p in (imgs[:1] + labels[:1])]}
    print(f"  {c}: {len(imgs)} images, {len(labels)} label files, {empty} with no damage")
    print(f"        classes: {dict(classes.most_common())}")
    compare_to_paper(c, meta, len(imgs), classes, res,
                     f"{bad} bad label rows" if bad else None)
    return res

def count_country(c, out, meta):
    def _in(p, word):
        return any(part.lower() == word for part in p.relative_to(out).parts[:-1])
    alljpg = [p for p in out.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    imgs = sorted(p for p in alljpg if _in(p, "train"))
    xmls = sorted(p for p in out.rglob("*.xml") if _in(p, "train"))
    test_imgs = sorted(p for p in alljpg if _in(p, "test"))
    if not imgs or not xmls:
        tree = sorted({str(p.relative_to(out).parent) for p in out.rglob("*") if p.is_file()})[:20]
        fail(f"layout_{c}", f"{c}: expected images + .xml labels under a 'train' folder, found dirs: {tree}")
    img_stems = {p.stem for p in imgs}; xml_stems = {p.stem for p in xmls}
    classes = Counter(); sizes = Counter()
    empty = 0; nontarget_only = 0; bad_xml = 0; boxes_bad = 0
    for x in xmls:
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError:
            bad_xml += 1; continue
        s = root.find("size")
        if s is not None:
            sizes[f"{s.findtext('width')}x{s.findtext('height')}"] += 1
        names = []
        for o in root.findall("object"):
            n = (o.findtext("name") or "").strip()
            names.append(n)
            bb = o.find("bndbox")
            if bb is not None:
                try:
                    x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                    if x2 <= x1 or y2 <= y1:
                        boxes_bad += 1
                except (TypeError, ValueError):
                    boxes_bad += 1
        classes.update(names)
        if not names:
            empty += 1
        elif not any(n in TARGET for n in names):
            nontarget_only += 1
    res = {
        "train_images": len(imgs), "train_xmls": len(xmls), "test_images_unlabelled": len(test_imgs),
        "images_without_xml": len(img_stems - xml_stems), "xml_without_image": len(xml_stems - img_stems),
        "class_counts": dict(classes.most_common()), "non_target_labels": {k: v for k, v in classes.items() if k not in TARGET},
        "images_with_no_damage": empty, "images_with_only_non_target": nontarget_only,
        "bad_xml": bad_xml, "invalid_boxes": boxes_bad, "image_sizes": dict(sizes.most_common(5)),
        "example_dirs": sorted({str(p.parent.relative_to(out)) for p in imgs[:1] + xmls[:1]}),
    }
    print(f"  {c}: {len(imgs)} train images ({len(test_imgs)} unlabelled test), "
          f"{empty} with no damage, {nontarget_only} with only non-target labels")
    print(f"        classes: {dict(classes.most_common())}")
    print(f"        sizes:   {dict(sizes.most_common(3))}")
    diffs = []
    if abs(len(imgs) - meta["images"]) > 0.02 * meta["images"]:
        diffs.append(f"images {len(imgs)} vs paper {meta['images']}")
    for k in TARGET:
        if abs(classes.get(k, 0) - meta[k]) > max(5, 0.02 * meta[k]):
            diffs.append(f"{k} {classes.get(k, 0)} vs paper {meta[k]}")
    if res["images_without_xml"] or res["xml_without_image"] or bad_xml or boxes_bad:
        diffs.append(f"{res['images_without_xml']} imgs w/o xml, {res['xml_without_image']} xml w/o img, "
                     f"{bad_xml} bad xml, {boxes_bad} invalid boxes (prep script will drop these)")
    if diffs:
        warn(f"dataset_{c}", f"{c}: differs from paper/clean state: " + "; ".join(diffs), **res)
    else:
        ok(f"dataset_{c}", f"{c}: counts match the RDD2022 paper (within 2%)", **res)
    return res

# ---------- main ----------
def main():
    print(f"Road-defect preflight | env={ENV} | root={ROOT}")
    check_basics()
    check_packages()
    if not SKIP_GPU:
        if DEVICE != "cpu":
            check_gpu()
        check_yolo(); check_sam(); check_frcnn()
    else:
        warn("gpu_skipped", "SKIP_GPU=1: GPU/model checks skipped")
    check_metrics() if not SKIP_GPU else None
    check_hf()
    if not SKIP_DATA:
        check_dataset()
    else:
        warn("data_skipped", "SKIP_DATA=1: dataset download skipped")
    REPORT["result"] = "PASS"
    save_report()
    n_warn = len(REPORT["warnings"])
    print("\n" + "=" * 60)
    print(f"PASS: all checks passed ({n_warn} warning(s)). Report: {REPORT_PATH}")
    for w in REPORT["warnings"]:
        print(f"  - {w}")
    print("=" * 60)

if __name__ == "__main__":
    _failed = False
    try:
        main()
    except PreflightFailed:
        _failed = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            fail("crash", f"unexpected error during '{CURRENT['step']}': {type(e).__name__}: {e}")
        except PreflightFailed:
            _failed = True
    if _failed and ENV == "local":
        sys.exit(1)
