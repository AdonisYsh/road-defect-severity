"""Blind human rating of 150 detected defects.

  python -m rdd.rating make            pick 150 class-balanced true detections, save crops + hidden key
  python -m rdd.rating rate --rater A  opens a small web page: one defect at a time, click 1-5 (resumable)

Raters never see the model's class, confidence or severity. Ratings -> results/rating/ratings_<rater>.csv
If the web page is a problem, the same CSVs can be filled by hand (id,rating).
"""
from __future__ import annotations

import argparse
import random

import cv2
import numpy as np
import pandas as pd

from .common import cfg, classes, log, path, results, step

RUBRIC = """**How urgently should this be repaired?** Look at the close-up *and* the full photo (the red box shows where it is).
Things near the top of the photo are further away, so they are bigger than they look.

| score | meaning |
|---|---|
| **1** | barely visible, hairline; no effect on driving |
| **2** | minor: small or thin; cosmetic for now |
| **3** | moderate: clearly visible; a driver would notice; repair in the next few months |
| **4** | serious: wide or spreading damage, or a pothole; uncomfortable or risky for tyres; repair soon |
| **5** | severe: large/deep pothole or broken surface; a safety hazard; fix immediately |

Rate on your own. Don't discuss with the other raters until all three are done."""


def make():
    step("Build the rating set")
    r = cfg()["rating"]
    out = results("rating")
    det = pd.read_csv(results("severity") / "detections_test.csv")
    tp = det[det.true_positive].copy()
    rng = np.random.default_rng(cfg()["project"]["seed"])
    n, k = r["n_crops"], len(classes())
    quota = {c: n // k + (1 if i < n % k else 0) for i, c in enumerate(classes())}
    picked = []
    for c in classes():
        d = tp[tp.cls == c].copy()
        if d.empty:
            continue
        d["q"] = pd.qcut(d.S_box.rank(method="first"), min(5, len(d)), labels=False)  # spread over the severity range
        per_bin = int(np.ceil(quota[c] / d.q.nunique()))
        take = pd.concat([g.sample(min(len(g), per_bin), random_state=int(rng.integers(1e9))) for _, g in d.groupby("q")])
        picked.append(take.head(quota[c]))
    sel = pd.concat(picked)
    if len(sel) < n:  # top up from whatever is left
        rest = tp.drop(sel.index)
        sel = pd.concat([sel, rest.sample(min(len(rest), n - len(sel)), random_state=1)])
    sel = sel.sample(frac=1, random_state=2).reset_index(drop=True)
    sel["id"] = [f"R{i + 1:03d}" for i in range(len(sel))]
    crops = out / "crops"
    crops.mkdir(exist_ok=True)
    pad = r["context_pad"]
    for _, row in sel.iterrows():
        im = cv2.imread(str(path("yolo") / "images" / "test" / row.image))
        H, W = im.shape[:2]
        if row.country == "India":  # show India frames un-squashed, as the camera saw them
            im = cv2.resize(im, (int(W * cfg()["severity"]["unsquash"]["India"]), H))
            sx = im.shape[1] / W
        else:
            sx = 1.0
        x1, x2, y1, y2 = row.x1 * sx, row.x2 * sx, row.y1, row.y2
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
        cx2, cy2 = int(min(im.shape[1], x2 + pad * bw)), int(min(H, y2 + pad * bh))
        crop = im[cy1:cy2, cx1:cx2]
        s = 480 / max(crop.shape[:2])
        cv2.imwrite(str(crops / f"{row.id}_crop.jpg"), cv2.resize(crop, None, fx=s, fy=s))
        ctx = im.copy()
        cv2.rectangle(ctx, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        s = 640 / max(ctx.shape[:2])
        cv2.imwrite(str(crops / f"{row.id}_context.jpg"), cv2.resize(ctx, None, fx=s, fy=s))
    sel.to_csv(out / "key_DO_NOT_SHOW_RATERS.csv", index=False)
    for rt in r["raters"]:
        f = out / f"ratings_{rt}.csv"
        if not f.exists():
            pd.DataFrame(dict(id=sel.id, rating="")).to_csv(f, index=False)
    (out / "RUBRIC.md").write_text(RUBRIC + "\n")
    log(f"{len(sel)} defects ready ({sel.cls.value_counts().to_dict()}) -> {out}")
    log("next: python -m rdd.rating rate --rater A   (then B, then C — each person on their own)")


def rate(rater: str, port: int):
    import gradio as gr

    out = results("rating")
    f = out / f"ratings_{rater}.csv"
    sheet = pd.read_csv(f, dtype={"rating": "string"})
    order = list(sheet.id)
    random.Random(rater).shuffle(order)  # each rater sees a different order

    def state_text():
        done = sheet.rating.notna() & (sheet.rating.astype(str).str.strip() != "")
        return int(done.sum())

    def next_id():
        done = set(sheet.loc[sheet.rating.notna() & (sheet.rating.astype(str).str.strip() != ""), "id"])
        for i in order:
            if i not in done:
                return i
        return None

    def show():
        i = next_id()
        if i is None:
            return None, None, f"### All {len(order)} done — thank you! You can close this tab.", None
        return (str(out / "crops" / f"{i}_crop.jpg"), str(out / "crops" / f"{i}_context.jpg"),
                f"### Rater {rater}: {state_text()} / {len(order)} rated — this is **{i}**", i)

    def save(score, cur):
        if cur is None or score is None:
            return show()
        sheet.loc[sheet.id == cur, "rating"] = str(int(score))
        sheet.to_csv(f, index=False)
        return show()

    with gr.Blocks(title=f"Road damage rating — rater {rater}") as demo:
        gr.Markdown(RUBRIC)
        head = gr.Markdown()
        cur = gr.State()
        with gr.Row():
            crop = gr.Image(label="close-up", height=380)
            ctx = gr.Image(label="full photo (red box)", height=380)
        with gr.Row():
            btns = [gr.Button(str(k), variant="primary" if k == 3 else "secondary") for k in range(1, 6)]
        for k, b in enumerate(btns, start=1):
            b.click(lambda c, k=k: save(k, c), inputs=cur, outputs=[crop, ctx, head, cur])
        demo.load(show, outputs=[crop, ctx, head, cur])
    demo.launch(server_port=port, inbrowser=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    r = sub.add_parser("rate")
    r.add_argument("--rater", required=True)
    r.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    make() if a.cmd == "make" else rate(a.rater, a.port)


if __name__ == "__main__":
    main()
