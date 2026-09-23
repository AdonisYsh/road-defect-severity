"""Detection + severity for images and phone videos, and the ranked 100 m segment list.

Used by the Gradio demo and from the command line:
  python -m rdd.video road.mp4            -> results/video/<name>_annotated.mp4, _segments.csv, _tracks.csv
Each defect is tracked across frames (ByteTrack) and counted once, at its worst severity.
A segment = segment_seconds of video (12 s at 30 km/h ~ 100 m). Segment priority P = sum of severity levels.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import cfg, classes, class_names, device_str, job_weights, load_json, log, results
from .severity import MaskArea, SeverityModel

COLORS = {0: (255, 170, 0), 1: (0, 200, 255), 2: (200, 0, 255), 3: (0, 0, 255)}  # BGR
LEVEL_COL = {1: (80, 200, 80), 2: (80, 220, 200), 3: (0, 200, 255), 4: (0, 120, 255), 5: (0, 0, 230)}


def default_paths():
    w = job_weights(cfg()["severity"]["detector_job"])
    s = results("severity") / "severity_config.json"
    return w, s


class Analyzer:
    def __init__(self, weights=None, sev_conf=None):
        from ultralytics import YOLO

        w, s = default_paths()
        self.weights = str(weights or w)
        self.model = YOLO(self.weights)
        self.sev = SeverityModel(conf_file=Path(sev_conf or s))
        d = load_json(sev_conf or s) if Path(sev_conf or s).exists() else {}
        self.conf = float(d.get("conf", 0.25))
        self._sam = None

    @property
    def sam(self):
        if self._sam is None:
            self._sam = MaskArea()
        return self._sam

    def rows_for(self, frame, boxes, method, country=None):
        """boxes: (N,6) x1,y1,x2,y2,conf,cls -> list of dicts with area and severity."""
        H, W = frame.shape[:2]
        yh, us = self.sev.country_params(country)
        mask_px, masks = (self.sam(frame, boxes) if method == "mask" and len(boxes) else (None, []))
        out = []
        for i, (x1, y1, x2, y2, cf, c) in enumerate(boxes):
            c = int(c)
            px = (x2 - x1) * (y2 - y1)
            if mask_px is not None and np.isfinite(mask_px[i]) and mask_px[i] > 0:
                px = float(mask_px[i])
            A = self.sev.area(px, y2, W, H, yh, us)
            L = self.sev.location(((x1 + x2) / 2) / W)
            S = self.sev.score(c, A, L, method)
            out.append(dict(cls=classes()[c], type=class_names()[c], conf=round(float(cf), 3), area_px=int(px),
                            area_corrected=round(float(A), 5), position_weight=L, S=S, severity=self.sev.level(S, method),
                            box=(int(x1), int(y1), int(x2), int(y2)), mask=masks[i] if i < len(masks) else None, cls_id=c))
        return out

    @staticmethod
    def draw(frame, rows, track_ids=None):
        im = frame.copy()
        for k, r in enumerate(rows):
            x1, y1, x2, y2 = r["box"]
            col = LEVEL_COL.get(r["severity"], COLORS[r["cls_id"]])
            if r.get("mask") is not None:
                over = im.copy()
                over[r["mask"]] = col
                im = cv2.addWeighted(over, 0.35, im, 0.65, 0)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            tid = f"#{track_ids[k]} " if track_ids is not None and track_ids[k] is not None else ""
            label = f"{tid}{r['cls']} S{r['severity']} {r['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, 0, 0.55, 2)
            cv2.rectangle(im, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), col, -1)
            cv2.putText(im, label, (x1 + 2, y1 - 5), 0, 0.55, (255, 255, 255), 2)
        return im

    def image(self, bgr, method="box", conf=None, country=None):
        r = self.model.predict(bgr, conf=conf or self.conf, device=device_str(), verbose=False)[0]
        b = r.boxes
        boxes = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1) \
            if len(b) else np.zeros((0, 6))
        rows = self.rows_for(bgr, boxes, method, country)
        rows.sort(key=lambda r: -r["S"])
        table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("mask", "box", "cls_id", "S")} for r in rows])
        return self.draw(bgr, rows), table

    def video(self, src, method="box", conf=None, out_dir=None, progress=None):
        v = cfg()["video"]
        out_dir = Path(out_dir or results("video"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src).stem
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = v["vid_stride"]
        writer = _Writer(out_dir / f"{stem}_annotated.mp4", fps / stride)
        tracks = {}
        f = 0
        self.model.predictor = None  # fresh tracker state
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f % stride == 0:
                t = f / fps
                r = self.model.track(frame, persist=True, conf=conf or self.conf, tracker=v["tracker"],
                                     device=device_str(), verbose=False)[0]
                b = r.boxes
                if len(b):
                    boxes = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)
                    ids = b.id.int().cpu().tolist() if b.id is not None else [None] * len(b)
                else:
                    boxes, ids = np.zeros((0, 6)), []
                rows = self.rows_for(frame, boxes, method)
                for tid, row in zip(ids, rows):
                    if tid is None:
                        continue
                    tr = tracks.setdefault(tid, dict(track=tid, first_t=t, last_t=t, classes=Counter(), max_S=0, severity=0, max_conf=0))
                    tr["last_t"] = t
                    tr["classes"][row["cls"]] += 1
                    tr["max_conf"] = max(tr["max_conf"], row["conf"])
                    if row["S"] > tr["max_S"]:
                        tr["max_S"], tr["severity"] = row["S"], row["severity"]
                writer.write(self.draw(frame, rows, ids))
                if progress and total:
                    progress(min(1.0, f / total), desc=f"frame {f}/{total}")
            f += 1
        cap.release()
        writer.close()
        seg_s = v["segment_seconds"]
        metres = v["speed_kmh"] / 3.6 * seg_s
        tr = pd.DataFrame([dict(track=t["track"], cls=t["classes"].most_common(1)[0][0], first_seen_s=round(t["first_t"], 1),
                                last_seen_s=round(t["last_t"], 1), max_conf=t["max_conf"], severity=t["severity"], S=t["max_S"],
                                segment=int(t["first_t"] // seg_s))
                           for t in tracks.values() if sum(t["classes"].values()) >= 2])  # seen in >=2 sampled frames
        n_seg = max(1, math.ceil((f / fps) / seg_s))
        segs = []
        for s in range(n_seg):
            d = tr[tr.segment == s] if len(tr) else tr
            row = dict(segment=s + 1, start_s=s * seg_s, end_s=min((s + 1) * seg_s, round(f / fps, 1)),
                       approx_metres=f"{int(s * metres)}–{int((s + 1) * metres)}",
                       priority_P=int(d.severity.sum()) if len(d) else 0, defects=len(d),
                       worst_level=int(d.severity.max()) if len(d) else 0)
            for c in classes():
                row[c] = int((d.cls == c).sum()) if len(d) else 0
            segs.append(row)
        segs = pd.DataFrame(segs).sort_values(["priority_P", "worst_level"], ascending=False).reset_index(drop=True)
        segs.insert(0, "rank", range(1, len(segs) + 1))
        segs.to_csv(out_dir / f"{stem}_segments.csv", index=False)
        tr.to_csv(out_dir / f"{stem}_tracks.csv", index=False)
        log(f"video: {len(tr)} tracked defects in {n_seg} segments -> {out_dir}")
        return str(writer.path), segs, str(out_dir / f"{stem}_segments.csv")


class _Writer:
    """H.264 via imageio-ffmpeg (plays in browsers); falls back to OpenCV mp4v."""

    def __init__(self, p: Path, fps: float):
        self.path, self.fps, self.w = p, fps, None

    def write(self, bgr):
        if self.w is None:
            try:
                import imageio.v2 as iio

                self.w = iio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=7, macro_block_size=8)
                self.kind = "iio"
            except Exception:  # noqa: BLE001
                h, w = bgr.shape[:2]
                self.w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                self.kind = "cv2"
        if self.kind == "iio":
            self.w.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:
            self.w.write(bgr)

    def close(self):
        if self.w is not None:
            self.w.close() if self.kind == "iio" else self.w.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--method", choices=["box", "mask"], default="box")
    a = ap.parse_args()
    out, segs, csv = Analyzer().video(a.video, a.method)
    print(segs.head(10).to_string(index=False))
    print(f"\nannotated video: {out}\nranked segments: {csv}")


if __name__ == "__main__":
    main()
