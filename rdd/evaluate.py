"""Common evaluator used for EVERY model (YOLO and Faster R-CNN), so the numbers are comparable.

For a finished job it:
  1. predicts on val -> picks the confidence cut-off that maximises mean F1 (val only, never test)
  2. predicts on test -> mAP@0.5, mAP@0.5:0.95 (COCO-style, torchmetrics/pycocotools),
     per-class AP / P / R / F1 at the chosen cut-off, confusion matrix (raw + normalised), PR curves,
     separately for all / India / Japan
  3. YOLO only: also runs Ultralytics' own val(split='test') and keeps its plots
  4. the severity detector only: saves a failure gallery (missed / wrong class / false alarm)

Usage: python -m rdd.evaluate --job yolov8s_main
"""
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import (box_iou, cfg, classes, country_of, device_str, job_weights, load_json, log, path,
                     results, runs_dir, save_json, step)

NC = len(cfg()["dataset"]["classes"])


# ---------------------------------------------------------------- ground truth / predictions
def load_gt(split: str) -> dict[str, np.ndarray]:
    """name -> (N,5) [x1,y1,x2,y2,cls] from the COCO json written by prep."""
    js = load_json(path("coco") / f"{split}.json")
    by_id = {im["id"]: im["file_name"] for im in js["images"]}
    gt = {n: [] for n in by_id.values()}
    for a in js["annotations"]:
        x, y, w, h = a["bbox"]
        gt[by_id[a["image_id"]]].append([x, y, x + w, y + h, a["category_id"] - 1])
    return {k: np.array(v, dtype=float).reshape(-1, 5) for k, v in gt.items()}


def image_paths(split: str, scope: str = "all") -> list[str]:
    f = path("yolo") / (f"{split}.txt" if scope == "all" else f"{split}_{scope}.txt")
    return [l for l in f.read_text().split() if l]


def predict_yolo(weights, paths, tta=False, batch=16) -> dict[str, np.ndarray]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    for i in range(0, len(paths), 256):
        for r in model.predict(paths[i:i + 256], conf=0.001, iou=0.6, max_det=300, imgsz=cfg()["yolo"]["imgsz"],
                               augment=tta, device=device_str(), batch=batch, stream=True, verbose=False):
            b = r.boxes
            arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1) \
                if len(b) else np.zeros((0, 6))
            out[Path(r.path).name] = arr
        log(f"  predicted {min(i + 256, len(paths))}/{len(paths)}")
    return out


def get_predictions(job: str, split: str) -> dict[str, np.ndarray]:
    cache = runs_dir() / "preds" / f"{job}_{split}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    kind = cfg()["jobs"][job]["kind"]
    paths = image_paths(split)
    log(f"{job}: predicting {len(paths)} {split} images")
    if kind == "yolo":
        preds = predict_yolo(job_weights(job), paths, tta=cfg()["eval"]["tta"])
    else:
        from .frcnn import predict_frcnn

        preds = predict_frcnn(job_weights(job), paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(preds))
    return preds


# ---------------------------------------------------------------- matching
def match_same_class(pred, gt, iou_thr):
    """Greedy per-class matching (COCO style). Returns tp flags for preds (sorted by score) and n_gt per class."""
    tps = []  # (score, cls, tp)
    for c in range(NC):
        p = pred[pred[:, 5] == c]
        g = gt[gt[:, 4] == c]
        p = p[np.argsort(-p[:, 4])]
        used = np.zeros(len(g), bool)
        ious = box_iou(p[:, :4], g[:, :4])
        for i in range(len(p)):
            j = -1
            if len(g):
                cand = np.where(~used & (ious[i] >= iou_thr))[0]
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]
                    used[j] = True
            tps.append((p[i, 4], c, j >= 0))
    return tps


def prf_at(preds, gts, names, conf, iou_thr):
    tp = np.zeros(NC)
    fp = np.zeros(NC)
    ngt = np.zeros(NC)
    for n in names:
        p, g = preds.get(n, np.zeros((0, 6))), gts[n]
        p = p[p[:, 4] >= conf]
        for c in range(NC):
            ngt[c] += (g[:, 4] == c).sum()
        for s, c, t in match_same_class(p, g, iou_thr):
            tp[c] += t
            fp[c] += not t
    P = tp / np.clip(tp + fp, 1e-9, None)
    R = tp / np.clip(ngt, 1e-9, None)
    F1 = 2 * P * R / np.clip(P + R, 1e-9, None)
    return P, R, F1, ngt


def pick_conf(preds, gts, names, iou_thr):
    grid = np.round(np.arange(0.05, 0.91, 0.05), 2)
    f1s = [prf_at(preds, gts, names, c, iou_thr)[2].mean() for c in grid]
    best = float(grid[int(np.argmax(f1s))])
    return best, dict(zip(grid.tolist(), [float(f) for f in f1s]))


def confusion(preds, gts, names, conf, iou_thr):
    """Ultralytics layout: matrix[pred_class, true_class]; last row/col = background."""
    m = np.zeros((NC + 1, NC + 1), int)
    pairs = []  # (name, kind, gt_row or None, pred_row or None)
    for n in names:
        p, g = preds.get(n, np.zeros((0, 6))), gts[n]
        p = p[p[:, 4] >= conf]
        ious = box_iou(g[:, :4], p[:, :4])
        gm, pm = np.full(len(g), -1), np.full(len(p), -1)
        if ious.size:
            order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
            for gi, pi in order:
                if ious[gi, pi] < iou_thr:
                    break
                if gm[gi] < 0 and pm[pi] < 0:
                    gm[gi], pm[pi] = pi, gi
        for gi in range(len(g)):
            gc = int(g[gi, 4])
            if gm[gi] >= 0:
                pc = int(p[gm[gi], 5])
                m[pc, gc] += 1
                if pc != gc:
                    pairs.append((n, "wrong_class", g[gi], p[gm[gi]]))
            else:
                m[NC, gc] += 1
                pairs.append((n, "missed", g[gi], None))
        for pi in range(len(p)):
            if pm[pi] < 0:
                m[int(p[pi, 5]), NC] += 1
                pairs.append((n, "false_alarm", None, p[pi]))
    return m, pairs


def plot_confusion(m, out: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = classes() + ["background"]
    for norm in (False, True):
        mm = m.astype(float)
        if norm:
            mm = mm / np.clip(mm.sum(0, keepdims=True), 1e-9, None)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(mm, cmap="Blues")
        ax.set_xticks(range(NC + 1), labels, rotation=45, ha="right")
        ax.set_yticks(range(NC + 1), labels)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        for i in range(NC + 1):
            for j in range(NC + 1):
                if i == NC and j == NC:
                    continue
                v = mm[i, j]
                ax.text(j, i, f"{v:.2f}" if norm else f"{int(v)}", ha="center", va="center",
                        color="white" if v > mm.max() * 0.6 else "black", fontsize=8)
        ax.set_title(title + (" (normalised by true class)" if norm else ""))
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / ("confusion_matrix_normalized.png" if norm else "confusion_matrix.png"), dpi=150)
        plt.close(fig)


def coco_map(preds, gts, names, out: Path | None = None):
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    def fmt():
        P, T = [], []
        for n in names:
            p, g = preds.get(n, np.zeros((0, 6))), gts[n]
            P.append(dict(boxes=torch.tensor(p[:, :4], dtype=torch.float32), scores=torch.tensor(p[:, 4], dtype=torch.float32),
                          labels=torch.tensor(p[:, 5], dtype=torch.int64)))
            T.append(dict(boxes=torch.tensor(g[:, :4], dtype=torch.float32), labels=torch.tensor(g[:, 4], dtype=torch.int64)))
        return P, T

    P, T = fmt()
    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True, extended_summary=True)
    m.warn_on_many_detections = False
    m.update(P, T)
    r = m.compute()
    m50 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True, iou_thresholds=[0.5])
    m50.warn_on_many_detections = False
    m50.update(P, T)
    r50 = m50.compute()
    nn = lambda v: v if v >= 0 else float("nan")  # noqa: E731  (-1 = class absent from this test subset)
    per50 = {c: nn(v) for c, v in zip(r50["classes"].tolist(), r50["map_per_class"].tolist())}
    per = {c: nn(v) for c, v in zip(r["classes"].tolist(), r["map_per_class"].tolist())}
    res = dict(map50=float(r["map_50"]), map50_95=float(r["map"]), map75=float(r["map_75"]),
               ap50={classes()[c]: per50.get(c, float("nan")) for c in range(NC)},
               ap50_95={classes()[c]: per.get(c, float("nan")) for c in range(NC)})
    if out is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        prec = r["precision"]  # T,R,K,A,M
        rec = np.linspace(0, 1, prec.shape[1])
        fig, ax = plt.subplots(figsize=(6, 5))
        for k, c in enumerate(r["classes"].tolist()):
            pk = prec[0, :, k, 0, -1].numpy()
            pk = np.where(pk < 0, np.nan, pk)
            ax.plot(rec, pk, label=f"{classes()[c]} AP50={per50.get(c, 0):.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"PR curve @IoU 0.5 — mAP50={res['map50']:.3f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "PR_curve.png", dpi=150)
        plt.close(fig)
    return res


# ---------------------------------------------------------------- failure gallery
def failure_gallery(pairs, out: Path, n_per):
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    img_dir = path("yolo") / "images" / "test"
    rows = []
    for kind in ("missed", "wrong_class", "false_alarm"):
        items = [p for p in pairs if p[1] == kind]
        if kind == "false_alarm":
            items.sort(key=lambda p: -p[3][4])  # most confident false alarms first
        elif kind == "missed":  # potholes first, then others
            items.sort(key=lambda p: (p[2][4] != 3, rng.random()))
        else:
            rng.shuffle(items)
        d = out / kind
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        tiles = []
        for i, (n, _, g, p) in enumerate(items[:n_per]):
            im = cv2.imread(str(img_dir / n))
            if im is None:
                continue
            if g is not None:
                x1, y1, x2, y2 = map(int, g[:4])
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 0), 3)
                cv2.putText(im, f"GT {classes()[int(g[4])]}", (x1, max(15, y1 - 6)), 0, 0.6, (0, 200, 0), 2)
            if p is not None:
                x1, y1, x2, y2 = map(int, p[:4])
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 230), 2)
                cv2.putText(im, f"pred {classes()[int(p[5])]} {p[4]:.2f}", (x1, min(im.shape[0] - 5, y2 + 18)), 0, 0.6, (0, 0, 230), 2)
            f = d / f"{i:02d}_{Path(n).stem}.jpg"
            cv2.imwrite(str(f), im)
            tiles.append(cv2.resize(im, (320, 320)))
            rows.append(dict(kind=kind, image=n, file=str(f.relative_to(out)),
                             gt_class=classes()[int(g[4])] if g is not None else "",
                             pred_class=classes()[int(p[5])] if p is not None else "",
                             pred_conf=round(float(p[4]), 3) if p is not None else "", why=""))
        if tiles:
            while len(tiles) % 6:
                tiles.append(np.full_like(tiles[0], 255))
            grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
            cv2.imwrite(str(out / f"{kind}_sheet.jpg"), grid)
    pd.DataFrame(rows).to_csv(out / "failures.csv", index=False)
    log(f"failure gallery -> {out}  (fill the 'why' column: shadow? wet road? thin crack?)")


# ---------------------------------------------------------------- ultralytics' own val
def ultralytics_val(job: str, out: Path):
    from ultralytics import YOLO

    rows = []
    for scope in ["all"] + cfg()["dataset"]["countries"]:
        y = path("yolo") / "sets" / f"test_{scope}.yaml"
        m = YOLO(str(job_weights(job)))
        r = m.val(data=str(y), split="test", imgsz=cfg()["yolo"]["imgsz"], batch=16, device=device_str(),
                  project=str(runs_dir() / "val"), name=f"{job}_{scope}", exist_ok=True, plots=True, verbose=False)
        row = dict(job=job, test=scope, map50=float(r.box.map50), map50_95=float(r.box.map),
                   precision=float(r.box.mp), recall=float(r.box.mr),
                   speed_ms_per_img=sum(r.speed.values()))
        for i, c in enumerate(r.box.ap_class_index):
            p, rc, a50, a = r.box.class_result(i)
            row.update({f"{classes()[c]}_P": float(p), f"{classes()[c]}_R": float(rc),
                        f"{classes()[c]}_AP50": float(a50), f"{classes()[c]}_AP50_95": float(a)})
        rows.append(row)
        dst = out / f"ultralytics_{scope}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in Path(r.save_dir).glob("*.png"):
            shutil.copy2(f, dst / f.name)
    pd.DataFrame(rows).to_csv(out / "ultralytics_val.csv", index=False)
    return rows


# ---------------------------------------------------------------- main
def evaluate_job(job: str):
    step(f"Evaluate {job}")
    ev = cfg()["eval"]
    out = results("detection", job)
    val_gt, test_gt = load_gt("val"), load_gt("test")
    val_pred = get_predictions(job, "val")
    train_scope = cfg()["jobs"][job].get("data", "all")  # cross-country jobs pick conf on their own country's val
    val_names = [n for n in val_gt if train_scope == "all" or country_of(n) == train_scope]
    conf, sweep = pick_conf(val_pred, val_gt, val_names, ev["match_iou"])
    log(f"{job}: best confidence cut-off on val = {conf}")
    save_json({"conf": conf, "val_meanF1_by_conf": sweep}, out / "conf.json")

    test_pred = get_predictions(job, "test")
    rows, pcs = [], []
    for scope in ["all"] + cfg()["dataset"]["countries"]:
        names = [n for n in test_gt if scope == "all" or country_of(n) == scope]
        d = out / f"test_{scope}"
        d.mkdir(exist_ok=True)
        mp = coco_map(test_pred, test_gt, names, d)
        P, R, F1, ngt = prf_at(test_pred, test_gt, names, conf, ev["match_iou"])
        m, pairs = confusion(test_pred, test_gt, names, conf, ev["match_iou"])
        plot_confusion(m, d, f"{job} — test {scope} (conf≥{conf})")
        pd.DataFrame(m, index=[f"pred_{c}" for c in classes()] + ["pred_background"],
                     columns=[f"true_{c}" for c in classes()] + ["true_background"]).to_csv(d / "confusion_matrix.csv")
        rows.append(dict(job=job, test=scope, images=len(names), map50=mp["map50"], map50_95=mp["map50_95"],
                         map75=mp["map75"], conf=conf, P=P.mean(), R=R.mean(), F1=F1.mean()))
        for i, c in enumerate(classes()):
            pcs.append(dict(job=job, test=scope, cls=c, n_gt=int(ngt[i]), AP50=mp["ap50"][c], AP50_95=mp["ap50_95"][c],
                            P=P[i], R=R[i], F1=F1[i]))
        if scope == "all" and job == cfg()["severity"]["detector_job"]:
            failure_gallery(pairs, results("failures"), ev["failures_per_type"])
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pcs).to_csv(out / "per_class.csv", index=False)
    log(f"{job}: test mAP50={rows[0]['map50']:.3f}  mAP50-95={rows[0]['map50_95']:.3f}  "
        f"(India {rows[1]['map50']:.3f}, Japan {rows[2]['map50']:.3f})")
    if cfg()["jobs"][job]["kind"] == "yolo":
        ultralytics_val(job, out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    evaluate_job(ap.parse_args().job)


if __name__ == "__main__":
    main()
