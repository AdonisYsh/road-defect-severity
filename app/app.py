"""Gradio demo: road damage -> severity 1-5 -> ranked repair list.

Laptop:   python app/app.py             (http://127.0.0.1:7860)
HF Space: this same file sits at the Space root next to rdd/, config.yaml, model.pt, severity_config.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if (p / "rdd").is_dir():
        sys.path.insert(0, str(p))
        break

import cv2  # noqa: E402
import gradio as gr  # noqa: E402

from rdd.video import Analyzer, default_paths  # noqa: E402


def pick(*cands):
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return None


w_default, s_default = default_paths()
WEIGHTS = pick(os.environ.get("RDD_WEIGHTS"), HERE / "model.pt", w_default)
SEVCONF = pick(os.environ.get("RDD_SEVERITY"), HERE / "severity_config.json", s_default)
if WEIGHTS is None:
    raise SystemExit("No trained weights found (results/weights/yolov8s_main.pt or model.pt next to app.py)")
AN = Analyzer(WEIGHTS, SEVCONF)
SOURCES = {"Phone / dash-cam (unknown camera)": None, "RDD2022 India photo": "India", "RDD2022 Japan photo": "Japan"}
METHODS = {"Box area (fast)": "box", "SAM 2 mask area (true pixels, slower)": "mask"}

INTRO = """# Road damage → severity → repair priority
Detects 4 damage types (D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole),
gives each a **severity 1–5** from its perspective-corrected size, type and position (wheel path counts more),
and for videos ranks each ~100 m stretch by **priority P = sum of severities**.
Decision support only — not an automatic repair order."""


def run_image(img, method, conf, source):
    if img is None:
        return None, None
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out, table = AN.image(bgr, METHODS[method], conf, SOURCES[source])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), table


def run_video(vid, method, conf, progress=gr.Progress()):
    if vid is None:
        return None, None, None
    out, segs, csv = AN.video(vid, METHODS[method], conf, out_dir=Path("outputs"), progress=progress)
    return out, segs, csv


with gr.Blocks(title="Road damage severity") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        method = gr.Radio(list(METHODS), value=list(METHODS)[0], label="Area method")
        conf = gr.Slider(0.05, 0.9, value=round(AN.conf, 2), step=0.05, label="Confidence cut-off (default picked on validation set)")
    with gr.Tab("Image"):
        with gr.Row():
            img_in = gr.Image(label="Road photo (upload or webcam)", sources=["upload", "webcam", "clipboard"], type="numpy")
            img_out = gr.Image(label="Detections + severity")
        source = gr.Dropdown(list(SOURCES), value=list(SOURCES)[0], label="Where is the photo from? (sets the horizon estimate)")
        img_btn = gr.Button("Analyse image", variant="primary")
        img_tab = gr.Dataframe(label="Defects, worst first")
        img_btn.click(run_image, [img_in, method, conf, source], [img_out, img_tab])
    with gr.Tab("Video → ranked 100 m segments"):
        gr.Markdown("Record from a bike/car at about 30 km/h in daylight. Each 12-second window ≈ 100 m.")
        with gr.Row():
            vid_in = gr.Video(label="Road video")
            vid_out = gr.Video(label="Annotated")
        vid_btn = gr.Button("Analyse video", variant="primary")
        seg_tab = gr.Dataframe(label="Segments ranked by repair priority")
        seg_csv = gr.File(label="Download ranked list (CSV)")
        vid_btn.click(run_video, [vid_in, method, conf], [vid_out, seg_tab, seg_csv])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1")
