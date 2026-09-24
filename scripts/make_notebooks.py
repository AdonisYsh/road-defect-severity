"""Generates notebooks/kaggle_frcnn.ipynb and notebooks/colab_yolov8n.ipynb."""
import json
from pathlib import Path

REPO_URL = "https://github.com/AdonisYsh/road-defect-severity.git"


def nb(cells, colab=False):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}}
    if colab:
        meta["accelerator"] = "GPU"
        meta["colab"] = {"gpuType": "T4", "provenance": []}
    return {"cells": out, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


SETUP = '''
import os, subprocess, shutil
REPO_URL = "{url}"   # <- your GitHub repo (must be public, or put a token in the URL)
ROOT = "{root}"
if os.path.isdir(os.path.join(ROOT, ".git")):
    subprocess.run(["git", "-C", ROOT, "pull", "--ff-only"], check=True)
else:
    tmp = ROOT + "_clone"
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, tmp], check=True)
    os.makedirs(ROOT, exist_ok=True)
    shutil.copytree(tmp, ROOT, dirs_exist_ok=True)   # keeps anything the preflight left (data/raw, weights/)
    shutil.rmtree(tmp)
os.chdir(ROOT)
subprocess.run("pip install -q -r requirements-cloud.txt", shell=True, check=True)
print("repo ready at", os.getcwd())
'''

kaggle = nb([
    ("markdown", """
# Kaggle — Faster R-CNN (2× T4)
**Settings (right panel):** Accelerator **GPU T4 ×2**, Internet **On**.

Best way to run: **Save Version → Save & Run All (Commit)**. It runs in the background (up to 12 h) even if you
close the browser. When it finishes, open the version → **Output** → download `results_kaggle.zip`.
"""),
    ("code", SETUP.format(url=REPO_URL, root="/kaggle/working/road-defect")),
    ("code", "!python -m rdd.download && python -m rdd.prep"),
    ("markdown", "Faster R-CNN on both GPUs (torchrun). Look at the time of the first 100 iterations in the log."),
    ("code", "!python -m rdd.jobs frcnn"),
    ("code", "!python -m rdd.benchmark"),
    ("code", """
!python -m rdd.report --pack
# keep the saved output small: drop the dataset copy, keep results_kaggle.zip
!rm -rf data runs/preds
!ls -lh results_kaggle.zip
"""),
])

colab = nb([
    ("markdown", """
# Colab — YOLOv8n × 3 seeds + 2 cross-country runs (T4)
**Runtime → Change runtime type → T4 GPU.** Then **Runtime → Run all**.

Checkpoints go to Google Drive (`MyDrive/road-defect/runs`). If Colab disconnects: reconnect and **Run all** again —
finished jobs are skipped and the running one resumes from its last epoch.
At the end `results_colab.zip` is also copied to `MyDrive/road-defect/`.
"""),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')"),
    ("code", SETUP.format(url=REPO_URL, root="/content/road-defect")),
    ("code", "!python -m rdd.download && python -m rdd.prep"),
    ("code", "!python -m rdd.jobs yolov8n_seed0 yolov8n_seed1 yolov8n_seed2"),
    ("code", "!python -m rdd.jobs xc_india xc_japan"),
    ("code", "!python -m rdd.benchmark"),
    ("code", "!python -m rdd.report --pack\nfrom google.colab import files\nfiles.download('results_colab.zip')"),
], colab=True)

d = Path(__file__).resolve().parent.parent / "notebooks"
d.mkdir(exist_ok=True)
(d / "kaggle_frcnn.ipynb").write_text(json.dumps(kaggle, indent=1))
(d / "colab_yolov8n.ipynb").write_text(json.dumps(colab, indent=1))
print("notebooks written")

# ---- Kaggle version of the Colab jobs: 2 GPUs, two job queues in parallel
PARALLEL = r'''
import os, subprocess, time, sys, pathlib, yaml
# ---- time budget per job (hours). Total wall time ~ max(queue sums) + ~15 min eval per job.
SEED_HOURS, XC_HOURS = 1.0, 0.8
cfg_p = pathlib.Path("config.yaml"); c = yaml.safe_load(cfg_p.read_text())
for j in ("yolov8n_seed0", "yolov8n_seed1", "yolov8n_seed2"):
    c["jobs"][j]["time_hours"] = SEED_HOURS
for j in ("xc_india", "xc_japan"):
    c["jobs"][j]["time_hours"] = XC_HOURS
c["yolo"]["workers"] = 2           # 4 CPUs shared by two trainings
cfg_p.write_text(yaml.safe_dump(c, sort_keys=False))
QUEUES = {"0": ["yolov8n_seed0", "yolov8n_seed1"],               # GPU 0: ~2.0 h
          "1": ["yolov8n_seed2", "xc_india", "xc_japan"]}         # GPU 1: ~2.6 h
procs = {}
for k, (gpu, jobs) in enumerate(QUEUES.items()):
    if k:
        time.sleep(180)   # stagger so the two runs don't build the label cache at the same moment
    log = open(f"log_gpu{gpu}.txt", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)   # each queue sees only its own GPU (as cuda:0)
    procs[gpu] = subprocess.Popen([sys.executable, "-m", "rdd.jobs", *jobs], env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
    print(f"GPU {gpu}: started {jobs}  (log: log_gpu{gpu}.txt)", flush=True)
while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    for gpu in procs:
        lines = [l for l in pathlib.Path(f"log_gpu{gpu}.txt").read_text(errors="ignore").splitlines() if l.strip()]
        print(f"[GPU {gpu}] {lines[-1][-160:] if lines else '...'}", flush=True)
for gpu, p in procs.items():
    print(f"GPU {gpu} finished with exit code {p.returncode}")
    print("".join(open(f"log_gpu{gpu}.txt", errors="ignore").readlines()[-15:]))
assert all(p.returncode == 0 for p in procs.values()), "a queue failed - see the log above"
'''

kaggle_n = nb([
    ("markdown", """
# Kaggle — YOLOv8n × 3 seeds + 2 cross-country runs (2× T4, in parallel)
Replaces the Colab notebook. **Settings:** Accelerator **GPU T4 ×2**, Internet **On**.
Run with **Save Version → Save & Run All (Commit)**. Takes about 3 hours.
At the end download **`results_colab.zip`** from Output (named that way so `laptop.sh finish` picks it up).
"""),
    ("code", SETUP.format(url=REPO_URL, root="/kaggle/working/road-defect")),
    ("code", "!python -m rdd.download && python -m rdd.prep"),
    ("code", PARALLEL),
    ("code", "!python -m rdd.benchmark"),
    ("code", """
!python -m rdd.report --pack
!mv results_kaggle.zip results_colab.zip
!rm -rf data runs/preds
!ls -lh results_colab.zip
"""),
])
(d / "kaggle_yolov8n.ipynb").write_text(json.dumps(kaggle_n, indent=1))
print("kaggle_yolov8n notebook written")
