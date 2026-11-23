"""Save KDE distributions of all three semantic cosine similarity scores.

Run: python plot_semantic_score_kde.py
Uses every saved score; no sampling or API calls.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs/semantic_search"
PATTERNS = [
    ("who", "WHO: Risk ownership", "#2468bb"),
    ("how_much", "HOW MUCH: Protection adequacy", "#c16618"),
    ("when", "WHEN: Protection timing", "#008877"),
]


def main():
    scores = pd.read_parquet(OUT / "all_email_scores.parquet")
    columns = [f"score_{pattern}" for pattern, _, _ in PATTERNS]
    values = scores[columns].to_numpy()
    if len(scores) != 152885 or scores.email_id.duplicated().any():
        raise ValueError("Expected 152,885 unique email rows")
    if not np.isfinite(values).all() or (np.abs(values) > 1).any():
        raise ValueError("Invalid cosine similarity scores")
    padding = .04 * np.ptp(values)
    grid = np.linspace(max(-1, values.min() - padding), min(1, values.max() + padding), 700)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(12, 7))
    statistics = []
    for pattern, label, color in PATTERNS:
        sample = scores[f"score_{pattern}"].to_numpy(dtype=np.float64)
        kde = gaussian_kde(sample, bw_method="scott")
        density = kde(grid)
        ax.fill_between(grid, density, color=color, alpha=.08)
        ax.plot(grid, density, color=color, linewidth=2.5, label=label)
        statistics.append({"pattern": pattern, "emails": len(sample),
                           "mean": float(sample.mean()), "median": float(np.median(sample)),
                           "std": float(sample.std(ddof=1)), "min": float(sample.min()),
                           "p95": float(np.quantile(sample, .95)), "p99": float(np.quantile(sample, .99)),
                           "max": float(sample.max()), "kde_scott_factor": float(kde.factor)})
    ax.set(xlabel="Cosine similarity to pattern description", ylabel="Probability density",
           xlim=(grid.min(), grid.max()), ylim=(0, None))
    ax.grid(axis="y", color="#e2e8f0", linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    fig.suptitle("Enron email similarity scores by pattern", x=.09, ha="left", fontsize=19, y=.97)
    fig.text(.09, .90, f"All {len(scores):,} primary emails per curve | Gaussian kernel density estimates", color="#475569", fontsize=12)
    summary = "    |    ".join(f"{row['pattern'].upper().replace('_', ' ')} median {row['median']:.3f}" for row in statistics)
    fig.text(.09, .065, summary, fontsize=11)
    fig.text(.09, .025, "Scott bandwidth; each curve integrates to approximately 1. Similarity scores are not probabilities of wrongdoing.", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.09, right=.97, bottom=.18, top=.85)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"cosine_similarity_kde.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(statistics).to_csv(OUT / "cosine_similarity_distribution_stats.csv", index=False)
    (OUT / "cosine_similarity_distribution_stats.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print(pd.DataFrame(statistics).to_string(index=False))
    print(f"Saved {OUT / 'cosine_similarity_kde.png'}")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
