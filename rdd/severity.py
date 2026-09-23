"""Severity score  S = W_class x A x L  ->  level 1..5.

A  perspective-corrected area:  (pixel area / W_true^2) x ((H - y_h) / (y - y_h))^2
   y_h = horizon row (fraction of image height) estimated per country from the training labels;
   y = bottom edge of the defect; W_true = true image width (India frames are un-squashed x960/720).
   Computed from the box, and from the SAM 2 mask (clipped to the box).
L  position weight: 1.2 wheel paths, 1.0 centre, 0.8 edges (horizontal position only).
W  class danger weight (initial pothole 1.0, alligator 0.7, transverse 0.4, longitudinal 0.3; tuned later).

  python -m rdd.severity            horizon + severity for all val/test detections + SAM masks
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import (box_iou, cfg, classes, country_of, job_weights, load_json, log, path, results, save_json,
                     step, weights_file)


class SeverityModel:
    def __init__(self, conf_file: Path | None = None, d: dict | None = None):
        s = cfg()["severity"]
        d = d or (load_json(conf_file) if conf_file and Path(conf_file).exists() else {})
        self.horizon = d.get("horizon", {c: 0.3 for c in cfg()["dataset"]["countries"]})
        w0 = d.get("class_weights", s["class_weights_init"])
        self.weights = {"box": d.get("class_weights_box", w0), "mask": d.get("class_weights_mask", w0)}
        self.bins = d.get("bins", {})  # method -> 4 cut points
        self.unsquash = s["unsquash"]
        self.lanes = s["lanes"]
        self.min_depth = s["min_depth_frac"]

    def to_dict(self):
        return dict(horizon=self.horizon, class_weights_box=self.weights["box"],
                    class_weights_mask=self.weights["mask"], bins=self.bins)

    def location(self, xc: float) -> float:
        """Bands mirrored around the centre: edge [0, 0.15) | wheel path [0.15, 0.40) | centre [0.40, 0.60] ..."""
        l = self.lanes
        x = min(xc, 1 - xc)  # distance from the nearest image side, 0..0.5
        if x < l["edge"]:
            return l["weights"]["edge"]
        if x < l["wheel_inner"]:
            return l["weights"]["wheel"]
        return l["weights"]["centre"]

    def area(self, pix_area: float, y_bottom: float, W: int, H: int, horizon: float, unsquash: float = 1.0) -> float:
        yh = horizon * H
        depth = max(y_bottom - yh, self.min_depth * H)
        wt = W * unsquash
        return (pix_area * unsquash) / (wt * wt) * ((H - yh) / depth) ** 2

    def score(self, cls: int, A: float, L: float, method: str = "box") -> float:
        return self.weights[method][classes()[cls]] * A * L

    def level(self, S: float, method: str) -> int:
        cuts = self.bins.get(method)
        if not cuts:
            return 0
        return int(1 + np.searchsorted(np.asarray(cuts), S, side="right"))

    def country_params(self, country: str | None):
        """Horizon + unsquash factor. Unknown source (e.g. a phone video): India horizon, no squash."""
        if country in self.horizon:
            return self.horizon[country], self.unsquash.get(country, 1.0)
        return self.horizon.get("India", 0.3), 1.0


# ---------------------------------------------------------------- horizon estimation
def estimate_horizon(force=False) -> dict:
    out = results("severity")
    f = out / "horizon.json"
    if f.exists() and not force:
        return load_json(f)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step("Estimate horizon row per country (from training labels)")
    s = cfg()["severity"]
    js = load_json(path("coco") / "train.json")
    ims = {im["id"]: im for im in js["images"] if "__r" not in im["file_name"]}
    want = {classes().index(c) + 1 for c in s["horizon_classes"]}
    res, fig_rows = {}, []
    fig, axes = plt.subplots(1, len(cfg()["dataset"]["countries"]), figsize=(11, 4.2))
    for ax, country in zip(np.atleast_1d(axes), cfg()["dataset"]["countries"]):
        ys, ws, tops = [], [], []
        for a in js["annotations"]:
            im = ims.get(a["image_id"])
            if im is None or country_of(im["file_name"]) != country:
                continue
            x, y, w, h = a["bbox"]
            tops.append(y / im["height"])
            if a["category_id"] in want:
                ys.append((y + h) / im["height"])
                ws.append(w / im["width"])
        ys, ws = np.array(ys), np.array(ws)
        top_q = float(np.quantile(tops, 0.01)) if tops else 0.3
        fit = None
        if len(ys) > 50:
            edges = np.quantile(ys, np.linspace(0, 1, 11))
            bx, by = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = (ys >= lo) & (ys <= hi)
                if k.sum() > 5:
                    bx.append(np.median(ys[k]))
                    by.append(np.median(ws[k]))
            slope, icpt = np.polyfit(bx, by, 1)
            if slope > 0:
                fit = float(-icpt / slope)
            ax.scatter(ys, ws, s=2, alpha=0.2)
            ax.plot(bx, by, "ro-", label="bin medians")
            if fit is not None:
                xx = np.linspace(max(fit, 0), 1, 20)
                ax.plot(xx, slope * xx + icpt, "k--", label=f"fit -> horizon {fit:.2f}")
        manual = s["horizon_override"].get(country)
        cands = [v for v in (fit, top_q - 0.02) if v is not None]
        yh = float(manual) if manual is not None else float(np.clip(min(cands), 0.0, 0.6))
        res[country] = yh
        fig_rows.append(dict(country=country, fit_from_widths=fit, top_1pct_of_boxes=top_q, override=manual, used=yh, n=len(ys)))
        ax.axvline(yh, color="g", label=f"used y_h = {yh:.2f}")
        ax.set_title(f"{country}: defect width vs bottom edge")
        ax.set_xlabel("bottom edge y / H")
        ax.set_ylabel("box width / W")
        ax.legend(fontsize=7)
        log(f"{country}: horizon y_h = {yh:.3f} H  (width fit {fit}, 1% box-top {top_q:.3f}, override {manual})")
    fig.tight_layout()
    fig.savefig(out / "horizon_fit.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(fig_rows).to_csv(out / "horizon.csv", index=False)
    save_json(res, f)
    return res


# ---------------------------------------------------------------- SAM masks
class MaskArea:
    def __init__(self):
        from ultralytics import SAM

        self.sam = SAM(weights_file(cfg()["severity"]["sam_weights"]))

    def __call__(self, img, boxes: np.ndarray) -> tuple[np.ndarray, list]:
        """Pixel area of the SAM mask inside each box, and the masks (bool HxW)."""
        if len(boxes) == 0:
            return np.zeros(0), []
        r = self.sam(img, bboxes=boxes[:, :4].tolist(), verbose=False)[0]
        masks = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else np.zeros((0, 1, 1), bool)
        if len(masks) != len(boxes):  # SAM dropped/merged some prompts -> one box at a time
            ms = []
            for b in boxes:
                rr = self.sam(img, bboxes=[b[:4].tolist()], verbose=False)[0]
                ms.append(rr.masks.data[0].cpu().numpy().astype(bool) if rr.masks is not None and len(rr.masks.data) else None)
            shape = next((m.shape for m in ms if m is not None), None)
            if shape is None:
                return np.full(len(boxes), np.nan), []
            masks = np.stack([m if m is not None else np.zeros(shape, bool) for m in ms])
        H, W = masks.shape[1:]
        areas = []
        for m, b in zip(masks, boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
            areas.append(float(m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)].sum()))
        return np.array(areas), list(masks)


# ---------------------------------------------------------------- run on val/test detections
def score_detections(sev: SeverityModel, split: str, preds, conf, gt, mask_area: MaskArea | None):
    img_dir = path("yolo") / "images" / split
    js = load_json(path("coco") / f"{split}.json")
    size = {im["file_name"]: (im["width"], im["height"]) for im in js["images"]}
    rows = []
    for k, (name, p) in enumerate(sorted(preds.items())):
        p = p[p[:, 4] >= conf]
        if not len(p):
            continue
        W, H = size[name]
        country = country_of(name)
        yh, us = sev.country_params(country)
        m_areas = np.full(len(p), np.nan)
        if mask_area is not None:
            m_areas, _ = mask_area(str(img_dir / name), p)
        g = gt.get(name, np.zeros((0, 5)))
        ious = box_iou(p[:, :4], g[:, :4])
        for i, d in enumerate(p):
            x1, y1, x2, y2, sc, c = d
            c = int(c)
            box_px = (x2 - x1) * (y2 - y1)
            L = sev.location(((x1 + x2) / 2) / W)
            A_box = sev.area(box_px, y2, W, H, yh, us)
            A_mask = sev.area(m_areas[i], y2, W, H, yh, us) if np.isfinite(m_areas[i]) and m_areas[i] > 0 else np.nan
            same = [j for j in range(len(g)) if int(g[j, 4]) == c and ious[i, j] >= 0.5]
            rows.append(dict(image=name, country=country, split=split, cls=classes()[c], cls_id=c, conf=float(sc),
                             x1=x1, y1=y1, x2=x2, y2=y2, W=W, H=H, box_area_px=box_px, mask_area_px=m_areas[i],
                             mask_box_ratio=m_areas[i] / box_px if box_px else np.nan,
                             A_box=A_box, A_mask=A_mask, L=L, W_class=sev.weights["box"][classes()[c]],
                             S_box=sev.score(c, A_box, L), S_mask=sev.score(c, A_mask, L, "mask") if np.isfinite(A_mask) else np.nan,
                             true_positive=bool(same)))
        if k % 250 == 0:
            log(f"  {split}: {k}/{len(preds)} images")
    return pd.DataFrame(rows)


def quantile_bins(values, props=None):
    """4 cut points splitting values into 5 levels. props = wanted share of each level (default 20% each)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    props = np.asarray(props if props is not None else [0.2] * 5, float)
    q = np.cumsum(props / props.sum())[:-1]
    return np.quantile(v, q).tolist() if len(v) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-sam", action="store_true")
    a = ap.parse_args()
    from .evaluate import get_predictions, load_gt

    s = cfg()["severity"]
    job = s["detector_job"]
    out = results("severity")
    horizon = estimate_horizon(force=a.force)
    sev = SeverityModel(d=dict(horizon=horizon))
    conf = load_json(results("detection", job) / "conf.json")["conf"]
    mask = None if a.no_sam else MaskArea()
    for split in ("val", "test"):
        f = out / f"detections_{split}.csv"
        if f.exists() and not a.force:
            log(f"{f.name} exists; skipping (use --force to redo)")
            continue
        step(f"Severity on {split} detections ({job}, conf>={conf}{'' if mask else ', no SAM'})")
        df = score_detections(sev, split, get_predictions(job, split), conf, load_gt(split), mask)
        df.to_csv(f, index=False)
        log(f"{len(df)} detections -> {f}")
    val = pd.read_csv(out / "detections_val.csv")
    sev.bins = {"box": quantile_bins(val["S_box"]), "mask": quantile_bins(val["S_mask"])}
    save_json(dict(sev.to_dict(), bins_init=sev.bins, conf=conf, detector=job, note="initial weights; bins = val quintiles"), out / "severity_config.json")
    plots()


def plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = results("severity")
    df = pd.read_csv(out / "detections_test.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [df.loc[df.cls == c, "mask_box_ratio"].dropna() for c in classes()]
    ax.boxplot(data, tick_labels=classes(), showfliers=False)
    ax.set_ylabel("SAM mask area / box area")
    ax.set_title("How much of each box is actually damage (test detections)")
    fig.tight_layout()
    fig.savefig(out / "mask_vs_box_area.png", dpi=150)
    plt.close(fig)
    g = df.groupby("cls")["mask_box_ratio"].describe()[["count", "mean", "50%", "25%", "75%"]]
    g.to_csv(out / "mask_vs_box_area.csv")
    log(f"mask/box area ratio per class:\n{g.round(3)}")


if __name__ == "__main__":
    main()
