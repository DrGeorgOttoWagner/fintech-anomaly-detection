"""Publication-style static figures (PNG + SVG) rendered from computed analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NEG = "#5B7083"
POS = "#C8344F"
ACCENT = "#0F766E"
SERIES = ["#0F766E", "#C8344F", "#3B6FB6", "#D08C1A", "#6B4FA0", "#2E8B57", "#8C564B", "#5B7083", "#B8860B", "#1F77B4"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "legend.frameon": False,
})


class FigureWriter:
    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.items: list[dict] = []

    def save(self, fig, name: str, caption: str) -> dict:
        fig.tight_layout()
        png, svg = self.dir / f"{name}.png", self.dir / f"{name}.svg"
        fig.savefig(png)
        fig.savefig(svg)
        plt.close(fig)
        item = {"name": name, "png": f"figures/{png.name}", "svg": f"figures/{svg.name}", "caption": caption, "number": len(self.items) + 1}
        self.items.append(item)
        return item


def class_distribution(fw: FigureWriter, cb: dict) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    names = [cb["negative_name"], cb["positive_name"]]
    counts = [cb["n_negative"], cb["n_positive"]]
    bars = axes[0].bar(names, counts, color=[NEG, POS])
    for b, c in zip(bars, counts):
        axes[0].annotate(f"{c:,}\n({c / cb['n']:.3%})", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Records")
    axes[0].set_title("Class counts (linear scale)")
    axes[0].set_ylim(0, max(counts) * 1.25)
    axes[1].bar(names, counts, color=[NEG, POS])
    axes[1].set_yscale("log")
    axes[1].set_title("Class counts (log scale)")
    return fw.save(fig, "class_distribution", f"Class distribution: {cb['n_positive']:,} {cb['positive_name']} vs. {cb['n_negative']:,} {cb['negative_name']} records.")


def value_histograms(fw: FigureWriter, lin: dict, log: dict | None, col: str, pos: str, neg: str) -> dict:
    n = 2 if log else 1
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.1))
    axes = np.atleast_1d(axes)
    for ax, h, title in ((axes[0], lin["histogram_clipped"], f"{col} (≤ 99th percentile, density)"),) + (((axes[1], log["histogram"], f"log(1 + {col}), density"),) if log else ()):
        edges = np.asarray(h["edges"])
        width = np.diff(edges)
        for name, color in ((neg, NEG), (pos, POS)):
            s = h["series"].get(name)
            if s:
                ax.bar(edges[:-1], s["density"], width=width, align="edge", alpha=0.55, color=color, label=f"{name} (n={s['n']:,})")
        ax.set_title(title)
        ax.legend(fontsize=7)
    return fw.save(fig, f"hist_{_slug(col)}", f"Class-conditional distribution of {col} (densities normalised within each class; computed on all records).")


def boxplots(fw: FigureWriter, box: dict, col: str, pos: str, neg: str, log: bool = True) -> dict:
    fig, ax = plt.subplots(figsize=(6, 2.6))
    stats = []
    for name in (neg, pos):
        b = box[name]
        stats.append({"label": f"{name}\n(n={b['count']:,})", "med": b["median"], "q1": b["q1"], "q3": b["q3"], "whislo": b["whisker_low"], "whishi": b["whisker_high"], "fliers": []})
    bp = ax.bxp(stats, vert=False, showfliers=False, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], (NEG, POS)):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    if log and all(box[k]["min"] >= 0 for k in (neg, pos)):
        ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(col + (" (symmetric log scale)" if log else ""))
    ax.set_title(f"{col} by class — box = IQR, whiskers = 1.5×IQR (outliers not drawn)")
    return fw.save(fig, f"box_{_slug(col)}", f"Box plots of {col} by class; {box[pos]['n_outliers']:,} {pos} and {box[neg]['n_outliers']:,} {neg} values lie beyond the whiskers.")


def time_series(fw: FigureWriter, t: dict, pos: str) -> dict:
    s = t["series"]
    x = np.array([(r["start"] - t["start"]) / 3600 for r in s]) if t["kind"] != "ordinal" else np.array([r["start"] for r in s])
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.bar(x, [r["n"] for r in s], width=(t["bin_size"] / 3600 if t["kind"] != "ordinal" else t["bin_size"]) * 0.9, align="edge", color=NEG, alpha=0.5, label="All records")
    ax1.set_ylabel("Records per bin")
    ax2 = ax1.twinx()
    ax2.plot(x, [r.get("n_positive", 0) for r in s], color=POS, lw=1.6, marker="o", ms=2.5, label=f"{pos} records")
    ax2.set_ylabel(f"{pos} per bin", color=POS)
    ax2.grid(False)
    ax1.set_xlabel("Hours since first record" if t["kind"] != "ordinal" else t["column"])
    ax1.set_title(f"Records and {pos.lower()} events over time ({t['bin_size'] / 3600:g} h bins)" if t["kind"] != "ordinal" else "Records over time")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    return fw.save(fig, "time_series", f"Volume of all records (bars) and of {pos.lower()} records (line) per time bin of '{t['column']}' ({t['unit']}).")


def time_rate(fw: FigureWriter, t: dict, pos: str) -> dict:
    s = t["series"]
    x = np.array([(r["start"] - t["start"]) / 3600 for r in s])
    rate = np.array([r["positive_rate"] if r["positive_rate"] is not None else np.nan for r in s]) * 100
    fig, axes = plt.subplots(1, 2 if t.get("hour_of_cycle") else 1, figsize=(9, 3))
    axes = np.atleast_1d(axes)
    axes[0].plot(x, rate, color=POS, lw=1.5)
    axes[0].set_xlabel("Hours since first record")
    axes[0].set_ylabel(f"{pos} rate (%)")
    axes[0].set_title(f"{pos} rate per time bin")
    if t.get("hour_of_cycle"):
        hc = t["hour_of_cycle"]
        axes[1].bar([h["hour"] for h in hc], [h["n"] for h in hc], color=NEG, alpha=0.5)
        ax2 = axes[1].twinx()
        ax2.plot([h["hour"] for h in hc], [100 * (h["positive_rate"] or 0) for h in hc], color=POS, marker="o", ms=3)
        ax2.set_ylabel(f"{pos} rate (%)", color=POS)
        ax2.grid(False)
        axes[1].set_xlabel("Hour offset from first record, modulo 24 (not clock time)")
        axes[1].set_ylabel("Records")
        axes[1].set_title("Hour-of-cycle profile")
    return fw.save(fig, "time_rate", f"{pos} rate over elapsed time and by hour offset modulo 24 (offsets relative to the first record, not calendar hours).")


def correlation_heatmap(fw: FigureWriter, corr: dict) -> dict:
    m = np.array(corr["matrix"], dtype=float)
    labels = corr["labels"]
    size = max(5, min(11, 0.28 * len(labels) + 2))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label=f"{corr['method'].title()} r")
    ax.set_title(f"{corr['method'].title()} correlation matrix")
    return fw.save(fig, f"correlation_{corr['method']}", f"{corr['method'].title()} correlations between numeric columns and the binary target ({corr['computed_on']}).")


def relevance_bars(fw: FigureWriter, rel: dict, top: int = 15) -> dict:
    rows = rel["features"][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(rows) + 1.2))
    ax.barh([r["feature"] for r in rows], [r["ks_statistic"] for r in rows], color=ACCENT)
    ax.set_xlabel("Kolmogorov–Smirnov statistic (class separation)")
    ax.set_title("Univariate class separation (top features)")
    return fw.save(fig, "feature_relevance", "Univariate separation between classes per feature (KS statistic). Associations only — not causal effects.")


def class_conditional_grid(fw: FigureWriter, dists: list[dict], pos: str, neg: str) -> dict:
    n = len(dists)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.2 * rows), squeeze=False)
    for ax, d in zip(axes.flat, dists):
        h = d["histogram"]
        edges = np.asarray(h["edges"])
        for name, color in ((neg, NEG), (pos, POS)):
            s = h["series"].get(name)
            if s:
                ax.bar(edges[:-1], s["density"], width=np.diff(edges), align="edge", alpha=0.55, color=color, label=name)
        ax.set_title(d["column"], fontsize=9)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    return fw.save(fig, "class_conditional", "Class-conditional densities of the most separating features (display clipped to the 0.5–99.5th percentiles).")


def pca_variance(fw: FigureWriter, pca: dict) -> dict:
    comps = pca["components"]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.bar([c["component"] for c in comps], [c["variance"] for c in comps], color=ACCENT, alpha=0.8)
    ax.set_ylabel("Variance")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set_title("Variance of precomputed PCA components")
    return fw.save(fig, "pca_variance", f"Variance per precomputed component (max |r| between components = {pca['max_abs_inter_component_corr']:.3f}).")


def model_curves(fw: FigureWriter, runs: list[dict], partition: str, prevalence: float | None) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, r in enumerate(runs):
        c = r["detail"]["curves"].get(partition) or {}
        color = SERIES[i % len(SERIES)]
        label = f"{r['algorithm_name']} [{r['imbalance']}]"
        m = r["metrics"][partition]
        if "pr" in c:
            axes[0].plot(c["pr"]["recall"], c["pr"]["precision"], color=color, lw=1.4, label=f"{label} AP={m['pr_auc']:.3f}")
        if "roc" in c:
            axes[1].plot(c["roc"]["fpr"], c["roc"]["tpr"], color=color, lw=1.4, label=f"{label} AUC={m['roc_auc']:.3f}")
    if prevalence:
        axes[0].axhline(prevalence, color="grey", ls=":", lw=1, label=f"random (prevalence {prevalence:.4f})")
    axes[1].plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"Precision–Recall ({partition})", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC ({partition})", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(fontsize=6, loc="lower left")
    axes[1].legend(fontsize=6, loc="lower right")
    return fw.save(fig, f"curves_{partition}", f"Precision–recall and ROC curves on the {partition} partition (original class prevalence).")


def confusion_grid(fw: FigureWriter, runs: list[dict], partition: str, pos: str, neg: str) -> dict:
    n = len(runs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.3 * rows), squeeze=False)
    for ax, r in zip(axes.flat, runs):
        cm = r["metrics"][partition]["confusion"]
        m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(np.log1p(m), cmap="Greens")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8, color="black")
        ax.set_xticks([0, 1], [f"pred {neg}", f"pred {pos}"], fontsize=6)
        ax.set_yticks([0, 1], [f"true {neg}", f"true {pos}"], fontsize=6)
        ax.grid(False)
        ax.set_title(f"{r['algorithm_name']}\n[{r['imbalance']}]", fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return fw.save(fig, f"confusion_{partition}", f"Confusion matrices on the {partition} partition at the threshold frozen on validation.")


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower() or "col"
