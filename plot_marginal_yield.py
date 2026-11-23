"""Plot observed marginal lead yield and the cost of each candidate depth.

Run after marginal_yield.py evaluate: python plot_marginal_yield.py
Uses recorded initial analyst judgments; does not infer labels from scores.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs/semantic_search/marginal_yield"
PATTERNS = [("who", "WHO: Risk ownership", "#2468bb"),
            ("how_much", "HOW MUCH: Protection adequacy", "#c16618"),
            ("when", "WHEN: Protection timing", "#008877")]


def main():
    data = pd.read_csv(OUT / "pattern_marginal_yield.csv")
    pooled = pd.read_csv(OUT / "pooled_marginal_yield.csv")
    sizes = pd.read_csv(OUT / "pool_sizes.csv")
    if not data.complete.all() or not pooled.complete.all():
        raise ValueError("Unreviewed sampled slots cannot be shown as zero yield")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, .9], hspace=.55, wspace=.3)
    for i, (pattern, title, color) in enumerate(PATTERNS):
        ax = fig.add_subplot(grid[0, i])
        band = data[data.pattern == pattern]
        ax.plot(band.k, band.observed_marginal_yield, color=color, linewidth=2.5, marker="o", markersize=8)
        ax.axhline(.01, color="#64748b", linestyle="--", linewidth=1.4)
        for r in band.itertuples():
            ax.annotate(f"{r.new_distinct_leads} / {r.reviewed_unique_emails}",
                        (r.k, r.observed_marginal_yield), xytext=(0, 13), textcoords="offset points",
                        ha="center", fontsize=11, color=color)
        ax.set(title=title, xlabel="Top K per pattern", ylim=(-.005, .067), xlim=(65, 535), xticks=[100, 200, 300, 500])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_yticks([0, .01, .025, .05])
        ax.grid(axis="y", color="#e2e8f0")
        if i == 0:
            ax.set_ylabel("New distinct leads / additional emails reviewed")
        if pattern == "how_much":
            ax.annotate("Late-band yield rises\nto 5% at K=500", xy=(500, .05), xytext=(275, .059),
                        color=color, fontsize=10, ha="center",
                        arrowprops=dict(arrowstyle="->", color=color))
    ax = fig.add_subplot(grid[1, :2])
    ax.plot(pooled.k, pooled.observed_marginal_yield, color="#6e529b", marker="o", linewidth=2.5)
    ax.axhline(.01, color="#64748b", linestyle="--", linewidth=1.4, label="Illustrative 1% reference")
    for r in pooled.itertuples():
        ax.annotate(f"{r.new_distinct_leads} / {r.additional_unique_emails_reviewed}",
                    (r.k, r.observed_marginal_yield), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=10)
    ax.set(title="Combined yield after deduplicating emails and cases", xlabel="Top K per pattern",
           ylabel="New leads / new unique emails reviewed", ylim=(-.0015, .014), xticks=[100, 200, 300, 500], xlim=(65, 535))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#e2e8f0")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax = fig.add_subplot(grid[1, 2])
    bars = ax.bar(range(len(sizes)), sizes.unique_emails, color="#607b92", width=.6)
    for bar, number in zip(bars, sizes.unique_emails):
        ax.text(bar.get_x() + bar.get_width()/2, number + 25, f"{number:,}", ha="center", fontsize=11)
    ax.set(title="Size of the full candidate pool", xlabel="Top K per pattern", ylabel="Unique retrieved emails",
           xticks=range(len(sizes)), xticklabels=sizes.k_per_pattern, ylim=(0, 1320))
    fig.suptitle("Selecting retrieval depth with marginal lead yield", x=.07, ha="left", fontsize=21, y=.98)
    fig.text(.07, .925, "Initial analyst pilot: 40 random emails per pattern per rank band | 480 assessment slots, 431 unique emails", fontsize=12, color="#475569")
    fig.text(.07, .075, "Point labels = new distinct leads / emails actually reviewed. K=100 is the baseline; later points summarize (previous K, K].", fontsize=10)
    fig.text(.07, .047, "Dashed line = illustrative 1% reference, not a calibrated optimum. Pooled yield can hide a late increase in one pattern.", fontsize=10)
    fig.text(.07, .019, "Leads are initial investigative candidates, not wrongdoing findings. Unresolved items remain; zero sampled discoveries does not establish high recall.", fontsize=10, color="#475569")
    fig.subplots_adjust(top=.855, bottom=.17, left=.07, right=.97)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"marginal_yield_vs_top_k.{suffix}", dpi=250, bbox_inches="tight")
    plt.close(fig)

    interactive = make_subplots(rows=1, cols=3, shared_yaxes=True,
                                subplot_titles=[title for _, title, _ in PATTERNS])
    for i, (pattern, title, color) in enumerate(PATTERNS, 1):
        band = data[data.pattern == pattern]
        extra = [[int(r.new_distinct_leads), int(r.reviewed_unique_emails), int(r.unclear_emails),
                  str(r.new_lead_ids) if pd.notna(r.new_lead_ids) else "None"] for r in band.itertuples()]
        interactive.add_trace(go.Scatter(x=band.k, y=band.observed_marginal_yield, mode="lines+markers+text",
                                        text=[f"{r.new_distinct_leads}/{r.reviewed_unique_emails}" for r in band.itertuples()],
                                        textposition="top center", name=title, line=dict(color=color, width=3),
                                        marker=dict(size=10), customdata=extra,
                                        hovertemplate="K=%{x}<br>Observed yield: %{y:.2%}<br>New distinct leads: %{customdata[0]}<br>Reviewed: %{customdata[1]}<br>Unclear: %{customdata[2]}<br>%{customdata[3]}<extra></extra>"), row=1, col=i)
        interactive.add_hline(y=.01, line_dash="dash", line_color="#64748b", row=1, col=i)
        interactive.update_xaxes(title_text="Top K per pattern", tickvals=[100, 200, 300, 500], row=1, col=i)
        interactive.update_yaxes(tickformat=".0%", range=[-.005, .067], row=1, col=i)
    interactive.update_layout(template="plotly_white", height=620, width=1400, showlegend=False,
                              title="Observed marginal lead yield vs Top K · Initial analyst pilot")
    chart = interactive.to_html(full_html=False, include_plotlyjs=True)
    table = sizes.to_html(index=False)
    body = """<!doctype html><html><head><meta charset='utf-8'><title>Marginal yield vs Top K</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#17283b;max-width:1450px;margin:30px auto;padding:0 24px}table{border-collapse:collapse}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}.note{background:#f3f6fa;padding:20px}a{color:#175ea7}</style></head><body>
<h1>Choose retrieval depth from marginal lead yield</h1><p>40 random emails per pattern in each band: 1–100, 101–200, 201–300, 301–500. Overlapping samples were read once; pattern assessments and distinct cases are tracked separately.</p>"""
    body += chart + "<h2>Candidate-pool size at each K</h2>" + table
    body += """<div class='note'><b>Interpretation:</b> A common stopping depth has not been demonstrated through K=500. The HOW MUCH curve rises to 2 new leads / 40 reviewed in the final band. Stopping at K=300 would miss these sampled discoveries, despite the preceding low-yield intervals. K=500 is a useful provisional candidate-pool depth (1,141 unique emails), not a demonstrated optimum.<p>WHO has two observed low-yield increments ending at 500. WHEN has no qualifying discoveries in these samples, but unresolved dating messages remain. Low yield may reflect query mismatch; it is not evidence that the pattern is absent.</p><p>The 1% line is an illustrative operational reference. With 40 reviewed emails, one new case means 2.5%; zero findings cannot certify the true yield is below 1%. These are initial analyst screening judgments, with full-message follow-up for escalated candidates; they are not validated labels or recall estimates.</p></div>
<p><a href='review_audit.html'>Inspect reviewed emails, rationales and lead cases</a> · <a href='pattern_marginal_yield.csv'>Pattern data</a> · <a href='pooled_marginal_yield.csv'>Deduplicated combined data</a></p></body></html>"""
    (OUT / "marginal_yield_explorer.html").write_text(body, encoding="utf-8")
    print(f"Saved {OUT / 'marginal_yield_vs_top_k.png'} and interactive HTML")


if __name__ == "__main__":
    main()
