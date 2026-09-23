"""Data prep: VOC XML -> YOLO txt + COCO json, cleaning, stratified split, background thinning,
rare-class oversampling, charts and a dataset summary table.

Safe to re-run: data/yolo and data/coco are rebuilt from data/raw every time (hard links, so fast).
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from .common import cfg, classes, class_names, log, path, results, save_json, step

SPLITS = ("train", "val", "test")


def _scan_image(p: Path):
    try:
        data = p.read_bytes()
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w, h = im.size
        return hashlib.md5(data).hexdigest(), w, h
    except Exception:  # noqa: BLE001
        return None


def parse_country(country: str, stats: Counter, dropped: Counter):
    base = path("raw") / country / "train"
    xml_dir, img_dir = base / "annotations" / "xmls", base / "images"
    target = {c: i for i, c in enumerate(classes())}
    min_px = cfg()["dataset"]["min_box_px"]
    xmls = sorted(xml_dir.glob("*.xml"))
    imgs = {p.stem: p for p in img_dir.glob("*.jpg")}
    stats[f"{country}_xml"] = len(xmls)
    stats[f"{country}_images"] = len(imgs)
    xml_stems = {x.stem for x in xmls}
    stats[f"{country}_image_without_xml"] = len(set(imgs) - xml_stems)
    log(f"{country}: {len(xmls)} xml, {len(imgs)} images; checking images (corrupt check + hash)")
    with ThreadPoolExecutor(16) as ex:
        scans = dict(zip(xmls, ex.map(lambda x: _scan_image(imgs[x.stem]) if x.stem in imgs else None, xmls)))
    recs = []
    for x in xmls:
        if x.stem not in imgs:
            stats[f"{country}_xml_without_image"] += 1
            continue
        sc = scans[x]
        if sc is None:
            stats[f"{country}_corrupt_image"] += 1
            continue
        md5, W, H = sc
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError:
            stats[f"{country}_bad_xml"] += 1
            continue
        boxes = []
        for obj in root.iter("object"):
            name = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if name not in target:
                dropped[f"{country}:{name}"] += 1
                continue
            try:
                x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError, AttributeError):
                stats[f"{country}_invalid_box"] += 1
                continue
            x1, x2 = sorted((max(0, min(W, x1)), max(0, min(W, x2))))
            y1, y2 = sorted((max(0, min(H, y1)), max(0, min(H, y2))))
            if x2 - x1 < min_px or y2 - y1 < min_px:
                stats[f"{country}_invalid_box"] += 1
                continue
            boxes.append((target[name], x1, y1, x2, y2))
        name = imgs[x.stem].name
        if not name.startswith(country + "_"):
            name = f"{country}_{name}"
        recs.append(dict(name=name, country=country, src=imgs[x.stem], W=W, H=H, boxes=boxes, md5=md5))
    return recs


def stratify_key(rec, rarity_order):
    present = {b[0] for b in rec["boxes"]}
    for c in rarity_order:
        if c in present:
            return f"{rec['country']}_{classes()[c]}"
    return f"{rec['country']}_bg"


def split_records(recs, seed):
    from sklearn.model_selection import train_test_split

    box_counts = Counter(b[0] for r in recs for b in r["boxes"])
    rarity = sorted(range(len(classes())), key=lambda c: box_counts.get(c, 0))
    keys = [stratify_key(r, rarity) for r in recs]
    def merge_small(ks, n_min):  # tiny strata -> "<country>_misc" -> "misc"
        kc = Counter(ks)
        ks = [k if kc[k] >= n_min else k.split("_")[0] + "_misc" for k in ks]
        kc = Counter(ks)
        return [k if kc[k] >= n_min else "misc" for k in ks]

    keys = merge_small(keys, 20)
    tr, va, te = cfg()["dataset"]["split"]
    idx = np.arange(len(recs))
    i_tr, i_rest = train_test_split(idx, train_size=tr, random_state=seed, stratify=keys)
    rest_keys = merge_small([keys[i] for i in i_rest], 2)
    if min(Counter(rest_keys).values()) < 2:
        rest_keys = None
    i_va, i_te = train_test_split(i_rest, train_size=va / (va + te), random_state=seed, stratify=rest_keys)
    out = {}
    for s, ids in zip(SPLITS, (i_tr, i_va, i_te)):
        for i in ids:
            out[recs[i]["name"]] = s
    return out


def link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def yolo_lines(rec):
    W, H = rec["W"], rec["H"]
    return "".join(
        f"{c} {((x1 + x2) / 2) / W:.6f} {((y1 + y2) / 2) / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
        for c, x1, y1, x2, y2 in rec["boxes"]
    )


def main():
    step("Data prep")
    c = cfg()
    seed = c["project"]["seed"]
    rng = random.Random(seed)
    stats, dropped = Counter(), Counter()
    recs = []
    for country in c["dataset"]["countries"]:
        recs += parse_country(country, stats, dropped)

    # dedupe exact duplicates
    seen, uniq = set(), []
    for r in recs:
        if r["md5"] in seen:
            stats[f"{r['country']}_duplicate"] += 1
            continue
        seen.add(r["md5"])
        uniq.append(r)
    recs = uniq
    log(f"{len(recs)} clean labelled images after dropping corrupt/duplicate")

    assign = split_records(recs, seed)
    by_split = defaultdict(list)
    for r in recs:
        by_split[assign[r["name"]]].append(r)

    # thin background images (no target boxes) in TRAIN only; val/test keep the natural mix
    frac = c["dataset"]["background_fraction_train"]
    train = []
    for country in c["dataset"]["countries"]:
        pos = [r for r in by_split["train"] if r["country"] == country and r["boxes"]]
        bg = [r for r in by_split["train"] if r["country"] == country and not r["boxes"]]
        keep = min(len(bg), round(frac / (1 - frac) * len(pos)))
        rng.shuffle(bg)
        stats[f"{country}_train_background_dropped"] = len(bg) - keep
        train += pos + bg[:keep]
    by_split["train"] = sorted(train, key=lambda r: r["name"])

    # repeat-factor oversampling of rare (country, class) pairs, train only
    th, cap = c["dataset"]["oversample"]["threshold"], c["dataset"]["oversample"]["max_repeat"]
    n_tr = len(by_split["train"])
    img_freq = Counter()
    for r in by_split["train"]:
        for cl in {b[0] for b in r["boxes"]}:
            img_freq[(r["country"], cl)] += 1
    rf = {k: min(cap, max(1.0, math.sqrt(th / (v / n_tr)))) for k, v in img_freq.items()}
    repeats = {}
    for r in by_split["train"]:
        f = max([rf[(r["country"], b[0])] for b in r["boxes"]] or [1.0])
        repeats[r["name"]] = int(f) + (1 if rng.random() < f - int(f) else 0)
    save_json({f"{k[0]}:{classes()[k[1]]}": round(v, 3) for k, v in rf.items()}, results("dataset") / "repeat_factors.json")

    # write YOLO layout
    yolo = path("yolo")
    shutil.rmtree(yolo, ignore_errors=True)
    shutil.rmtree(path("coco"), ignore_errors=True)
    lists = defaultdict(list)
    coco = {}
    for s in SPLITS:
        (yolo / "images" / s).mkdir(parents=True, exist_ok=True)
        (yolo / "labels" / s).mkdir(parents=True, exist_ok=True)
        coco[s] = dict(images=[], annotations=[], categories=[{"id": i + 1, "name": n} for i, n in enumerate(classes())])
        for r in by_split[s]:
            n_copies = repeats.get(r["name"], 1) if s == "train" else 1
            stem = Path(r["name"]).stem
            for k in range(n_copies):
                name = r["name"] if k == 0 else f"{stem}__r{k}.jpg"
                img = yolo / "images" / s / name
                link(r["src"], img)
                (yolo / "labels" / s / (Path(name).stem + ".txt")).write_text(yolo_lines(r))
                lists[s].append(str(img))
                lists[f"{s}_{r['country']}"].append(str(img))
                iid = len(coco[s]["images"]) + 1
                coco[s]["images"].append(dict(id=iid, file_name=name, width=r["W"], height=r["H"]))
                for cl, x1, y1, x2, y2 in r["boxes"]:
                    coco[s]["annotations"].append(
                        dict(id=len(coco[s]["annotations"]) + 1, image_id=iid, category_id=cl + 1,
                             bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1), iscrowd=0)
                    )
    for k, v in lists.items():
        (yolo / f"{k}.txt").write_text("\n".join(v) + "\n")
    names = {i: n for i, n in enumerate(classes())}
    sets = yolo / "sets"
    sets.mkdir(exist_ok=True)
    for scope in ["all"] + c["dataset"]["countries"]:
        sfx = "" if scope == "all" else f"_{scope}"
        yaml.safe_dump(dict(path=str(yolo), train=f"train{sfx}.txt", val=f"val{sfx}.txt", test=f"test{sfx}.txt", names=names),
                       open(yolo / f"{scope}.yaml", "w"))
        for s in ("val", "test"):
            yaml.safe_dump(dict(path=str(yolo), train=f"{s}{sfx}.txt", val=f"{s}{sfx}.txt", test=f"{s}{sfx}.txt", names=names),
                           open(sets / f"{s}_{scope}.yaml", "w"))
    path("coco").mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        save_json(coco[s], path("coco") / f"{s}.json")

    # reproducibility: the exact split file lists
    sp = results("dataset", "splits")
    for s in SPLITS:
        (sp / f"{s}.txt").write_text("\n".join(sorted(r["name"] for r in by_split[s])) + "\n")

    summarise(recs, by_split, repeats, stats, dropped)
    log(f"train {len(lists['train'])} (incl. oversampled copies) / val {len(lists['val'])} / test {len(lists['test'])}")


def summarise(recs, by_split, repeats, stats, dropped):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = results("dataset")
    rows = []
    for s in SPLITS:
        for country in cfg()["dataset"]["countries"]:
            rs = [r for r in by_split[s] if r["country"] == country]
            cc = Counter(b[0] for r in rs for b in r["boxes"])
            row = dict(split=s, country=country, images=len(rs), background_images=sum(not r["boxes"] for r in rs))
            if s == "train":
                row["images_after_oversampling"] = sum(repeats.get(r["name"], 1) for r in rs)
            for i, cl in enumerate(classes()):
                row[cl] = cc.get(i, 0)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "dataset_summary.csv", index=False)
    tot = df.groupby("split")[["images", "background_images"] + classes()].sum().reindex(SPLITS)
    tot.to_csv(out / "dataset_split_totals.csv")
    save_json({"stats": dict(stats), "dropped_non_target_boxes": dict(dropped),
               "total_dropped_boxes": sum(dropped.values())}, out / "cleaning_report.json")
    md = ["# Dataset summary (RDD2022 India + Japan)", "",
          f"Labelled images after cleaning: {len(recs)}", "",
          "| split | country | images | background | " + " | ".join(classes()) + " |",
          "|---|---|---|---|" + "---|" * len(classes())]
    for r in rows:
        md.append(f"| {r['split']} | {r['country']} | {r['images']} | {r['background_images']} | "
                  + " | ".join(str(r[c]) for c in classes()) + " |")
    md += ["", f"Dropped non-target boxes: {sum(dropped.values())} ({', '.join(f'{k}={v}' for k, v in dropped.most_common())})",
           f"Other cleaning: {', '.join(f'{k}={v}' for k, v in sorted(stats.items()) if v and not k.endswith(('_xml', '_images')))}",
           "", "Background (no-damage) images are thinned in TRAIN only; val/test keep the natural mix."]
    (out / "dataset_summary.md").write_text("\n".join(md) + "\n")

    # charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    allb = pd.DataFrame([dict(country=r["country"], cls=classes()[b[0]]) for r in recs for b in r["boxes"]])
    piv = allb.groupby(["cls", "country"]).size().unstack(fill_value=0).reindex(classes())
    piv.plot.bar(ax=axes[0], rot=0)
    axes[0].set_title("Boxes per class (all labelled images)")
    axes[0].set_xticklabels([f"{c}\n{n}" for c, n in zip(classes(), class_names())], fontsize=8)
    for cont in axes[0].containers:
        axes[0].bar_label(cont, fontsize=7)
    tot[classes()].plot.bar(ax=axes[1], rot=0)
    axes[1].set_title("Boxes per class per split")
    fig.tight_layout()
    fig.savefig(out / "class_counts.png", dpi=150)
    plt.close(fig)
    log(f"dataset summary -> {out}")


if __name__ == "__main__":
    main()
