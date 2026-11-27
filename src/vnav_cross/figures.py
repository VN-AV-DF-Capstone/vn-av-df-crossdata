"""Figures (English, PNG, 150 dpi). Colors follow the model, never its rank (fixed order below,
Okabe-Ito palette, color-blind safe); every chart with several series has a legend."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#F0E442", "#000000", "#882255", "#999999"]
ORDER = ["trihead_v5", "trihead_v5_no_pair", "avh_align_plus", "bl_ens_av", "avh_align_sup", "xlsr_frozen", "avh_align",
         "speechforensics", "avh_align_official", "lipinc_v2"]
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"


def color(key):
    name = key.rsplit("_seed", 1)[0]
    return PALETTE[ORDER.index(name) % len(PALETTE)] if name in ORDER else PALETTE[-1]


def style(ax, title=None, xlabel=None, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def heatmap(matrix, rows, cols, path, title, fmt="{:.3f}", vmin=0.4, vmax=1.0, cmap="Blues"):
    """Sequential one-hue map (AUC); a cell with no value stays blank and says 'n/a'."""
    data = np.array([[np.nan if v is None else v for v in r] for r in matrix], float)
    fig, ax = plt.subplots(figsize=(max(6.0, 2.6 + 1.25 * len(cols)), 1.0 + 0.42 * len(rows)))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = data[i, j]
            text = "n/a" if np.isnan(v) else fmt.format(v)
            shade = 0 if np.isnan(v) else (v - vmin) / (vmax - vmin)
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if shade > 0.6 else INK)
    ax.set_title(title, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return save(fig, path)


def roc(curves, path, title):
    """curves: [(key, label, fpr, tpr, auc)]."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
    for key, label, fpr, tpr, value in curves:
        dashed = key.rsplit("_seed", 1)[-1] not in ("42", key)  # seed 42 solid, other seeds dashed
        ax.plot(fpr, tpr, color=color(key), linewidth=1.8, linestyle="--" if dashed else "-",
                label=f"{label} ({value:.3f})")
    ax.axvline(0.05, color=MUTED, linewidth=0.8, linestyle=":")
    style(ax, title, "False positive rate", "True positive rate")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    return save(fig, path)


def grouped_bars(groups, series, values, path, title, ylabel, ylim=(0.4, 1.0), ref=None, ref_label=None,
                 errors=None):
    """values[key][group] (None = missing); series = [(key, label)] (the model key picks the color)
    or [(key, label, color)] for series that are not models (branches, parts)."""
    fig, ax = plt.subplots(figsize=(max(5.5, 1.4 + 0.9 * len(groups) * max(1, len(series)) / 3), 3.8))
    width = 0.8 / max(1, len(series))
    x = np.arange(len(groups))
    for i, entry in enumerate(series):
        key, label = entry[0], entry[1]
        shade = entry[2] if len(entry) > 2 else color(key)
        ys = [values[key].get(g) for g in groups]
        pos = x - 0.4 + width * (i + 0.5)
        heights = [np.nan if y is None else y for y in ys]
        yerr = None
        if errors:
            lo = [0 if errors[key].get(g) is None or y is None else y - errors[key][g][0] for g, y in zip(groups, ys)]
            hi = [0 if errors[key].get(g) is None or y is None else errors[key][g][1] - y for g, y in zip(groups, ys)]
            yerr = [lo, hi]
        ax.bar(pos, heights, width * 0.92, color=shade, label=label, yerr=yerr,
               error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": MUTED})
    if ref is not None:
        ax.axhline(ref, color=MUTED, linewidth=0.9, linestyle="--", label=ref_label)
    ax.set_xticks(x, groups, rotation=15, ha="right", fontsize=8)
    present = [v for s in series for v in values[s[0]].values() if v is not None]
    low = ylim[0]
    if present and min(present) < low:  # never hide a bar below the default floor (e.g. AUC 0.38)
        low = np.floor((min(present) - 0.05) * 10) / 10
    ax.set_ylim(low, ylim[1])
    style(ax, title, None, ylabel)
    ax.legend(fontsize=7, frameon=False, ncol=min(4, len(series) + (ref is not None)),
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return save(fig, path)


def dumbbell(rows, path, title, left="original", right="trimmed"):
    """rows: [(label, key, a, b)]: one line per model and dataset, a -> b."""
    fig, ax = plt.subplots(figsize=(8.0, 0.9 + 0.32 * len(rows)))
    for i, (label, key, a, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color=GRID, linewidth=2)
        ax.scatter([a], [i], color=color(key), s=36, zorder=3, marker="o")
        ax.scatter([b], [i], color=color(key), s=36, zorder=3, marker="D", facecolors="white")
        ax.text(max(a, b) + 0.005, i, f"{b - a:+.3f}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    style(ax, title, f"AUC (circle = {left}, diamond = {right})")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    return save(fig, path)


def forest(rows, path, title, xlabel):
    """rows: [(label, key, delta, lo, hi)]; vertical line at 0."""
    fig, ax = plt.subplots(figsize=(8.0, 0.9 + 0.3 * len(rows)))
    for i, (label, key, d, lo, hi) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=color(key), linewidth=2)
        ax.scatter([d], [i], color=color(key), s=30, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    style(ax, title, xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    return save(fig, path)


def histograms(series, path, title, threshold=None, xlabel="score"):
    """series: [(label, values, color)] (reals vs fakes of one model and dataset)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    values = np.concatenate([np.asarray(v, float) for _, v, _ in series if len(v)])
    bins = np.linspace(np.nanmin(values), np.nanmax(values), 40) if len(values) else 10
    for label, v, c in series:
        ax.hist(v, bins=bins, alpha=0.55, color=c, label=f"{label} (n={len(v)})", density=True)
    if threshold is not None:
        ax.axvline(threshold, color=INK, linewidth=1, linestyle="--", label="VN-AV-DF threshold (5% val FPR)")
    style(ax, title, xlabel, "density")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, path)


MODULE_GROUPS = (
    ("audio-visual sync", ("sync", "phi", "cost", "lag_score")),
    ("lip / head motion", ("trace", "motion")),
    ("audio (on frozen XLS-R)", ("audio",)),
    ("temporal fusion", ("mix", "temporal")),
    ("output heads", ("visual_head", "audio_head", "mismatch_head", "type_head", "head", "out")),
)


def module_group(module):
    return next((g for g, names in MODULE_GROUPS if module in names), "other")


def stacked_params(models, path, title):
    """models: [(label, [(module, count)])] → horizontal stacked bars by module group (millions)."""
    groups = [g for g, _ in MODULE_GROUPS] + ["other"]
    shades = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(8.0, 1.6 + 0.45 * len(models)))
    used = []
    for i, (label, mods) in enumerate(models):
        totals = {}
        for module, count in mods:
            totals[module_group(module)] = totals.get(module_group(module), 0) + count
        left = 0.0
        for g in groups:
            if totals.get(g):
                ax.barh(i, totals[g] / 1e6, left=left, color=shades[g], edgecolor="white", linewidth=1)
                left += totals[g] / 1e6
                if g not in used:
                    used.append(g)
        ax.text(left + 0.02, i, f"{left:.2f} M", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(models)), [m[0] for m in models], fontsize=8)
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=shades[g]) for g in used]
    style(ax, title, "trainable parameters (millions; frozen AV-HuBERT / XLS-R excluded)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(handles, used, fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    return save(fig, path)


def resolution(per_dataset, path, gate=360):
    """per_dataset: {label: [short side px]}."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels = list(per_dataset)
    data = [per_dataset[k] for k in labels]
    try:
        ax.boxplot(data, orientation="horizontal", widths=0.5, showfliers=False)
    except TypeError:  # matplotlib < 3.10
        ax.boxplot(data, vert=False, widths=0.5, showfliers=False)
    ax.set_yticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax.axvline(gate, color="#D55E00", linewidth=1.2, linestyle="--", label=f"M3-R resolution gate ({gate} px)")
    style(ax, "Input short side per dataset", "short side of the source video (px)")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, path)
