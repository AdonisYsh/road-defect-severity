"""Faster R-CNN (ResNet50-FPN, torchvision) — training (single GPU or multi-GPU DDP) and prediction.

Train:   python -m rdd.frcnn                               (1 GPU)
         torchrun --nproc_per_node=2 -m rdd.frcnn          (Kaggle 2xT4)
Resumes from runs/frcnn/last.pth automatically. Stops early to fit frcnn.time_hours.
"""
from __future__ import annotations

import math
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image, read_file

from .common import cfg, classes, load_json, log, path, results, runs_dir, save_json, seed_all

NC = len(cfg()["dataset"]["classes"])


def build_model(pretrained=True):
    f = cfg()["frcnn"]
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    fn = fasterrcnn_resnet50_fpn_v2 if f["arch"] == "v2" else fasterrcnn_resnet50_fpn
    m = fn(weights="DEFAULT" if pretrained else None, weights_backbone=None,
           min_size=f["min_size"], max_size=f["max_size"], box_detections_per_img=300, box_score_thresh=0.001)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, NC + 1)
    return m


def read_image(p) -> torch.Tensor:
    return decode_image(read_file(str(p)), mode=ImageReadMode.RGB).float().div_(255)


class CocoDet(Dataset):
    def __init__(self, split: str, train: bool, limit: int | None = None):
        js = load_json(path("coco") / f"{split}.json")
        self.dir = path("yolo") / "images" / split
        self.ims = js["images"][:limit] if limit else js["images"]
        anns = {}
        for a in js["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        self.anns = anns
        self.train = train

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        im = self.ims[i]
        img = read_image(self.dir / im["file_name"])
        a = self.anns.get(im["id"], [])
        boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in (b["bbox"] for b in a)], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([b["category_id"] for b in a], dtype=torch.int64)
        if self.train:
            if torch.rand(1) < 0.5:
                img = img.flip(-1)
                W = img.shape[-1]
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
            if torch.rand(1) < 0.8:  # light photometric jitter
                img = (img * (0.75 + 0.5 * torch.rand(1)) + (torch.rand(1) - 0.5) * 0.15).clamp_(0, 1)
        return img, dict(boxes=boxes, labels=labels), im["file_name"]


def collate(b):
    return tuple(zip(*b))


@torch.no_grad()
def quick_val_map50(model, dev, limit=800):
    from torchmetrics.detection import MeanAveragePrecision

    ds = CocoDet("val", train=False, limit=limit)
    dl = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate)
    metric = MeanAveragePrecision(iou_thresholds=[0.5], sync_on_compute=False)  # rank 0 only: never sync across GPUs
    metric.warn_on_many_detections = False
    model.eval()
    for imgs, tg, _ in dl:
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            out = model([i.to(dev) for i in imgs])
        metric.update([{k: v.float().cpu() if k != "labels" else v.cpu() for k, v in o.items()} for o in out],
                      [{k: v for k, v in t.items()} for t in tg])
    model.train()
    return float(metric.compute()["map_50"])


def train(max_iters: int | None = None, epochs: int | None = None):
    f = cfg()["frcnn"]
    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if ddp:
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=60))
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    seed_all(cfg()["project"]["seed"] + rank)
    out = runs_dir() / "frcnn"
    out.mkdir(parents=True, exist_ok=True)
    last, best, done = out / "last.pth", out / "best.pth", out / "DONE"
    if done.exists():
        if rank == 0:
            log("Faster R-CNN already finished earlier; skipping training")
            collect()
        return

    model = build_model(pretrained=os.environ.get("RDD_NO_PRETRAINED") != "1").to(dev)
    ds = CocoDet("train", train=True)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True) if ddp else None
    nw = min(f["workers"], max(1, (os.cpu_count() or 2) // world))
    dl = DataLoader(ds, batch_size=f["batch_per_gpu"], shuffle=sampler is None, sampler=sampler,
                    num_workers=nw, collate_fn=collate, pin_memory=dev.type == "cuda",
                    persistent_workers=nw > 0, drop_last=True)
    lr = f["lr_per_16"] * f["batch_per_gpu"] * world / 16
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    n_epochs = epochs or f["epochs"]
    state = dict(epoch=0, planned=n_epochs, best_map=-1.0, history=[])
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        if rank == 0:
            log(f"resuming Faster R-CNN from epoch {state['epoch']}")
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
    core = model.module if ddp else model
    warmup = 500
    it_per_epoch = len(dl)
    budget = f["time_hours"] * 3600
    t_start = time.time()
    model.train()
    while state["epoch"] < state["planned"]:
        ep = state["epoch"]
        if sampler:
            sampler.set_epoch(ep)
        m1, m2 = int(state["planned"] * 2 / 3), int(state["planned"] * 8 / 9)
        base = lr * (0.1 if ep >= m1 else 1) * (0.1 if ep >= m2 else 1)
        t0, losses = time.time(), []
        for i, (imgs, tg, _) in enumerate(dl):
            gi = ep * it_per_epoch + i
            for g in opt.param_groups:
                g["lr"] = base * min(1.0, (gi + 1) / warmup)
            imgs = [x.to(dev, non_blocking=True) for x in imgs]
            tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = sum(model(imgs, tg).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            if rank == 0 and i % 100 == 0:
                log(f"epoch {ep + 1}/{state['planned']} iter {i}/{it_per_epoch} loss {np.mean(losses[-100:]):.3f} lr {opt.param_groups[0]['lr']:.4f}")
            if max_iters and i + 1 >= max_iters:
                break
        ep_time = time.time() - t0
        state["epoch"] = ep + 1
        # fit the time budget: after the first epoch, shrink the plan if needed (all ranks agree via rank 0)
        if ep == 0 or state.get("ep_time") is None:
            t = torch.tensor([ep_time * 1.1], device=dev)
            if ddp:
                torch.distributed.broadcast(t, 0)
            fit = max(2, int((budget - (time.time() - t_start) + ep_time) // float(t.item())))
            if fit < state["planned"]:
                if rank == 0:
                    log(f"one epoch takes {ep_time / 60:.1f} min -> shortening plan to {fit} epochs to fit {f['time_hours']} h")
                state["planned"] = fit
            state["ep_time"] = float(t.item())
        if rank == 0:
            vmap = quick_val_map50(core, dev, limit=800 if max_iters is None else 16)
            state["history"].append(dict(epoch=ep + 1, loss=float(np.mean(losses)), val_map50_subset=vmap, minutes=ep_time / 60))
            log(f"epoch {ep + 1}: mean loss {np.mean(losses):.3f}, val mAP50 {vmap:.3f}")
            if vmap > state["best_map"]:
                state["best_map"] = vmap
                torch.save(core.state_dict(), best)
            torch.save(dict(model=core.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), state=state), last)
        if ddp:
            torch.distributed.barrier()
    if rank == 0:
        done.write_text("ok\n")
        save_json(state["history"], out / "history.json")
        collect()
    if ddp:
        torch.distributed.destroy_process_group()


def collect():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = runs_dir() / "frcnn"
    shutil.copy2(out / "best.pth", results("weights") / "frcnn.pth")
    hist = load_json(out / "history.json") if (out / "history.json").exists() else []
    d = results("training", "frcnn")
    save_json(hist, d / "history.json")
    if hist:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot([h["epoch"] for h in hist], [h["loss"] for h in hist], "o-")
        ax[0].set_title("train loss")
        ax[1].plot([h["epoch"] for h in hist], [h["val_map50_subset"] for h in hist], "o-")
        ax[1].set_title("val mAP@0.5")
        for a in ax:
            a.set_xlabel("epoch")
        fig.tight_layout()
        fig.savefig(d / "results.png", dpi=150)
        plt.close(fig)
    log(f"Faster R-CNN best checkpoint -> {results('weights') / 'frcnn.pth'}")


@torch.no_grad()
def predict_frcnn(weights, paths, batch=8, score_min=0.001) -> dict[str, np.ndarray]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    m.to(dev).eval()
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [read_image(p).to(dev) for p in chunk]
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            res = m(imgs)
        for p, r in zip(chunk, res):
            k = r["scores"] >= score_min
            out[Path(p).name] = np.concatenate([r["boxes"][k].float().cpu().numpy(), r["scores"][k].float().cpu().numpy()[:, None],
                                                (r["labels"][k] - 1).cpu().numpy()[:, None]], 1)
        if (i // batch) % 50 == 0:
            log(f"  predicted {min(i + batch, len(paths))}/{len(paths)}")
    return out


if __name__ == "__main__":
    train()
