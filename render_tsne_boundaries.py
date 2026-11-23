"""Render saved t-SNE coordinates with descriptive density outlines.

Run: python render_tsne_boundaries.py
Reuses the existing sample and group labels; no API calls or refitting.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs/embeddings"
NAMES = ["Drafts and discussion", "Meetings and scheduling", "FYI and short replies",
         "Energy trading and operations", "Attachments and document requests",
         "Corporate/legal mail and signatures"]
COLORS = ["#2468bb", "#dc7824", "#0a927d", "#8b50b6", "#b3920b", "#cf4668"]


def main():
    data = pd.read_parquet(OUT / "tsne_sample_coordinates.parquet")
    xy = data[["tsne_1", "tsne_2"]].to_numpy()
    labels = data.cluster.to_numpy()
    assert np.isfinite(xy).all() and set(labels) == set(range(6))
    spread = np.ptp(xy, axis=0)
    low, high = xy.min(axis=0) - spread * .05, xy.max(axis=0) + spread * .05
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 260),
                         np.linspace(low[1], high[1], 260))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    density = []
    for cluster in range(6):
        points = xy[labels == cluster]
        kde = gaussian_kde(points.T)
        # Density threshold based on observed points, rather than an arbitrary
        # region containing every outlier. Approximately 70% exceed this level.
        level = float(np.quantile(kde(points.T), .30))
        field = kde(grid).reshape(xx.shape)
        density.append((field, level))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "figure.facecolor": "white", "axes.facecolor": "#fafbfc"})
    fig, ax = plt.subplots(figsize=(15, 10))
    draw_order = np.random.default_rng(6450).permutation(len(xy))
    ax.scatter(xy[draw_order, 0], xy[draw_order, 1],
               c=np.asarray(COLORS)[labels[draw_order]], s=8, alpha=.48, linewidths=0,
               rasterized=True, zorder=2)
    for cluster, (field, level) in enumerate(density):
        ax.contourf(xx, yy, field, levels=[level, field.max() + 1],
                    colors=[COLORS[cluster]], alpha=.055, zorder=1)
        line = ax.contour(xx, yy, field, levels=[level], colors=[COLORS[cluster]],
                          linewidths=2.5, zorder=4)
        line.set_path_effects([pe.Stroke(linewidth=4.5, foreground="white", alpha=.8), pe.Normal()])
    counts = pd.Series(labels).value_counts()
    handles = [Line2D([], [], marker="o", markersize=8, linewidth=2, color=COLORS[c],
                      label=f"{c}  {NAMES[c]}  (n={counts[c]:,})") for c in range(6)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              frameon=False, title="Original 256D groups\nCounts in plotted sample", labelspacing=1.3)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("Enron email embeddings: t-SNE with group outlines", fontsize=20, x=.06, ha="left", y=.97)
    fig.text(.06, .92, f"{len(data):,} randomly sampled emails | Existing six-group assignments retained", fontsize=12, color="#475569")
    fig.text(.06, .065, "Outlines trace each group's main density region (approximately 70% of its sampled points).", fontsize=11)
    fig.text(.06, .04, "Overlapping outlines reflect mixed groups; they are not decision boundaries. t-SNE island spacing does not measure topic distance.", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.06, right=.69, top=.88, bottom=.12)
    fig.savefig(OUT / "tsne_cluster_boundaries.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Separate panels make overlapping group footprints easier to inspect.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    for cluster, ax in enumerate(axes.flat):
        field, level = density[cluster]
        ax.scatter(xy[:, 0], xy[:, 1], s=3, color="#d1d6df", alpha=.22, linewidths=0)
        selected = labels == cluster
        ax.contourf(xx, yy, field, levels=[level, field.max() + 1],
                    colors=[COLORS[cluster]], alpha=.10)
        ax.scatter(xy[selected, 0], xy[selected, 1], s=7, color=COLORS[cluster], alpha=.6, linewidths=0)
        ax.contour(xx, yy, field, levels=[level], colors=[COLORS[cluster]], linewidths=2)
        ax.set_title(f"{cluster}: {NAMES[cluster]}\n{selected.sum():,} sampled emails", fontsize=12)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("t-SNE group footprints on the same map", fontsize=20, y=.99)
    fig.text(.5, .025, "Each panel highlights one existing group. Outlines show approximately 70% density cores; gray points are all sampled emails.", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, .95), h_pad=2)
    fig.savefig(OUT / "tsne_cluster_panels.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("Saved tsne_cluster_boundaries.png and tsne_cluster_panels.png")


if __name__ == "__main__":
    main()
