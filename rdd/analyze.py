"""Severity validation against the blind human ratings.

  python -m rdd.analyze

Reports, for class-only (baseline), box-area and mask-area severity:
  MAE (levels vs mean human score), Spearman rho, Kendall tau — each with bootstrap 95% CI,
  inter-rater Krippendorff alpha, and the paired improvement in rho over class-only.
Class weights are tuned with 5-fold CV (never scored on the fold they were tuned on).
Writes the final weights + level cut-points into results/severity/severity_config.json for the demo.
"""
from __future__ import annotations

import itertools
import warnings

warnings.filterwarnings("ignore", message=".*constant.*")

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from .common import cfg, classes, load_json, log, results, save_json, step
from .severity import quantile_bins


def spearman_fast(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def s_of(df, method, w):
    A = df[f"A_{method}"].to_numpy()
    if method == "mask":
        A = np.where(np.isfinite(A), A, df["A_box"].to_numpy())
    return np.array([w[c] for c in df.cls]) * A * df.L.to_numpy()


def tune(df, method, human):
    grid = cfg()["severity"]["weight_grid"]
    best, bw = -2, None
    for w0, w1, w2 in itertools.product(grid, grid, grid):
        w = {classes()[0]: w0, classes()[1]: w1, classes()[2]: w2, classes()[3]: 1.0}
        r = spearman_fast(s_of(df, method, w), human)
        if r > best:
            best, bw = r, w
    return bw


def rating_props(h):
    lv = np.clip(np.rint(h), 1, 5).astype(int)
    return [max((lv == k).mean(), 1e-3) for k in range(1, 6)]


def levels(S, cuts):
    return 1 + np.searchsorted(np.asarray(cuts), S, side="right")


def boot_ci(fn, n, B, rng):
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try:
            v = fn(i)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    step("Severity validation")
    s = cfg()["severity"]
    out = results("severity")
    rdir = results("rating")
    key = pd.read_csv(rdir / "key_DO_NOT_SHOW_RATERS.csv")
    R = {}
    for rt in cfg()["rating"]["raters"]:
        f = rdir / f"ratings_{rt}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["rating"] = pd.to_numeric(d.rating, errors="coerce")
            if d.rating.notna().sum() >= 0.9 * len(key):
                R[rt] = d.set_index("id").rating
            else:
                log(f"rater {rt}: only {d.rating.notna().sum()} ratings — left out")
    if len(R) < 2:
        raise SystemExit("FAIL: need at least 2 complete raters (results/rating/ratings_<rater>.csv)")
    rt = pd.DataFrame(R)
    key = key.set_index("id").join(rt, how="inner").dropna(subset=list(R))
    human = key[list(R)].mean(1).to_numpy()
    n = len(key)
    log(f"{n} defects rated by {len(R)} raters ({', '.join(R)})")

    import krippendorff

    rel = key[list(R)].T.to_numpy()
    alpha_ord = krippendorff.alpha(reliability_data=rel, level_of_measurement="ordinal")
    alpha_int = krippendorff.alpha(reliability_data=rel, level_of_measurement="interval")
    rng = np.random.default_rng(cfg()["project"]["seed"])
    a_ci = boot_ci(lambda i: krippendorff.alpha(reliability_data=rel[:, i], level_of_measurement="ordinal"), n, 300, rng)
    pair_rho = {f"{a}-{b}": spearmanr(key[a], key[b])[0] for a, b in itertools.combinations(R, 2)}
    log(f"inter-rater Krippendorff alpha (ordinal) = {alpha_ord:.3f}  95% CI {a_ci[0]:.3f}..{a_ci[1]:.3f}")

    # ---- out-of-fold predictions
    val = pd.read_csv(out / "detections_val.csv")
    sconf = load_json(out / "severity_config.json")
    folds = np.arange(n) % s["cv_folds"]
    rng.shuffle(folds)
    df = key.reset_index()
    pred = {m: dict(score=np.zeros(n), level=np.zeros(n)) for m in ("class_only", "box_init", "mask_init", "box_cv", "mask_cv")}
    fold_w = {"box": [], "mask": []}
    for k in range(s["cv_folds"]):
        tr, te = folds != k, folds == k
        cm = pd.Series(human[tr]).groupby(df.cls[tr].to_numpy()).mean()
        cls_score = df.cls[te].map(cm).fillna(human[tr].mean()).to_numpy()
        pred["class_only"]["score"][te] = cls_score
        pred["class_only"]["level"][te] = np.clip(np.rint(cls_score), 1, 5)
        props = rating_props(human[tr])
        for m in ("box", "mask"):
            w = tune(df[tr], m, human[tr])
            fold_w[m].append(w)
            cuts = quantile_bins(s_of(df[tr], m, w), props)
            S = s_of(df[te], m, w)
            pred[f"{m}_cv"]["score"][te] = S
            pred[f"{m}_cv"]["level"][te] = levels(S, cuts)
    w_init = s["class_weights_init"]
    for m in ("box", "mask"):
        S = s_of(df, m, w_init)
        pred[f"{m}_init"]["score"] = S
        pred[f"{m}_init"]["level"] = levels(S, sconf.get("bins_init", sconf["bins"])[m])

    rows = []
    B = s["bootstrap"]
    base = pred["class_only"]["score"]
    for m, p in pred.items():
        sc, lv = p["score"], p["level"]
        row = dict(method=m, n=n,
                   MAE=float(np.abs(lv - human).mean()),
                   spearman=float(spearmanr(sc, human)[0]),
                   kendall=float(kendalltau(sc, human)[0]))
        row["MAE_lo"], row["MAE_hi"] = boot_ci(lambda i: np.abs(lv[i] - human[i]).mean(), n, B, rng)
        row["spearman_lo"], row["spearman_hi"] = boot_ci(lambda i: spearmanr(sc[i], human[i])[0], n, B, rng)
        row["kendall_lo"], row["kendall_hi"] = boot_ci(lambda i: kendalltau(sc[i], human[i])[0], n, B, rng)
        if m != "class_only":
            d = lambda i: spearmanr(sc[i], human[i])[0] - spearmanr(base[i], human[i])[0]  # noqa: E731
            row["rho_gain_vs_class_only"] = row["spearman"] - rows[0]["spearman"]
            row["gain_lo"], row["gain_hi"] = boot_ci(d, n, B, rng)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "validation.csv", index=False)

    # ---- final weights (all ratings) + demo cut-points (val detections, matched to the rating distribution)
    props = rating_props(human)
    final = {}
    for m in ("box", "mask"):
        w = tune(df, m, human)
        final[m] = w
        vv = val.copy()
        sconf["bins"][m] = quantile_bins(s_of(vv, m, w), props)
    sconf.update(class_weights_box=final["box"], class_weights_mask=final["mask"],
                 note="weights tuned on all human ratings; bins = val-detection quantiles matched to the rating distribution")
    save_json(sconf, out / "severity_config.json")
    save_json(dict(alpha_ordinal=alpha_ord, alpha_ordinal_ci=a_ci, alpha_interval=alpha_int, pairwise_spearman=pair_rho,
                   raters=list(R), n=n, fold_weights=fold_w, final_weights=final,
                   rating_distribution=dict(zip(range(1, 6), props))), out / "validation_extra.json")
    df_out = df.assign(human_mean=human, **{f"pred_{m}_level": p["level"] for m, p in pred.items()},
                       **{f"pred_{m}_score": p["score"] for m, p in pred.items()})
    df_out.to_csv(out / "validation_items.csv", index=False)
    report(res, alpha_ord, a_ci, alpha_int, pair_rho, final, n, list(R))
    plots(df_out, res)


def report(res, a, a_ci, a_int, pair, final, n, raters):
    md = [f"# Severity validation ({n} defects, raters {', '.join(raters)})", "",
          f"Inter-rater agreement: Krippendorff α (ordinal) = **{a:.3f}** (95% CI {a_ci[0]:.3f}–{a_ci[1]:.3f}); interval α = {a_int:.3f}",
          "Pairwise Spearman between raters: " + ", ".join(f"{k} {v:.2f}" for k, v in pair.items()), "",
          "| method | MAE (levels) | Spearman ρ | Kendall τ | Δρ vs class-only |", "|---|---|---|---|---|"]
    for r in res.itertuples():
        gain = "—" if r.method == "class_only" else f"{r.rho_gain_vs_class_only:+.3f} ({r.gain_lo:+.3f}…{r.gain_hi:+.3f})"
        md.append(f"| {r.method} | {r.MAE:.2f} ({r.MAE_lo:.2f}–{r.MAE_hi:.2f}) | {r.spearman:.3f} ({r.spearman_lo:.3f}–{r.spearman_hi:.3f}) | "
                  f"{r.kendall:.3f} ({r.kendall_lo:.3f}–{r.kendall_hi:.3f}) | {gain} |")
    md += ["", "*_init = hand-set weights (no tuning); *_cv = weights tuned by 5-fold CV, scored only on held-out folds.*",
           "Brackets = bootstrap 95% CI. A Δρ interval that excludes 0 means the gain over class-only is not chance.", "",
           f"Final weights (box): {final['box']}", f"Final weights (mask): {final['mask']}"]
    (results("severity") / "validation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def plots(df, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = results("severity")
    ms = ["class_only", "box_cv", "mask_cv"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    jit = np.random.default_rng(0).normal(0, 0.08, len(df))
    for ax, m in zip(axes, ms):
        ax.scatter(df[f"pred_{m}_level"] + jit, df.human_mean, s=12, alpha=0.6)
        ax.plot([1, 5], [1, 5], "k--", lw=1)
        r = res.set_index("method").loc[m]
        ax.set_title(f"{m}: ρ={r.spearman:.2f}, MAE={r.MAE:.2f}")
        ax.set_xlabel("predicted level")
    axes[0].set_ylabel("mean human rating")
    fig.tight_layout()
    fig.savefig(out / "validation_scatter.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    r = res.set_index("method")
    ax.bar(r.index, r.spearman, yerr=[r.spearman - r.spearman_lo, r.spearman_hi - r.spearman], capsize=4)
    ax.set_ylabel("Spearman ρ with human raters (95% CI)")
    ax.set_title("Does the severity score rank defects like humans do?")
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout()
    fig.savefig(out / "validation_rho.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
