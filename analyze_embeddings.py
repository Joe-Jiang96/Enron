"""Reproducible PCA, sampled t-SNE, and original-space clustering of Enron emails.

Run: python analyze_embeddings.py
Extra dependencies: scikit-learn matplotlib pandas scipy threadpoolctl
No API calls. Outputs go to analysis_outputs/embeddings/.
"""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import plotly.graph_objects as go
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 6450
GROUP_NAMES = ["Drafts and discussion", "Meetings and scheduling", "FYI and short replies",
               "Energy trading and operations", "Attachments and document requests",
               "Corporate/legal mail and signatures"]


def main(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("Loading and validating embeddings...", flush=True)
    table = pq.read_table(args.embeddings)
    ids = table["email_id"].to_pylist()
    x = np.asarray(table["embeddings"].combine_chunks().values).reshape(len(ids), -1).copy()
    if x.shape != (152885, 256) or not np.isfinite(x).all() or len(set(ids)) != len(ids):
        raise ValueError(f"Unexpected data shape/values/IDs: {x.shape}")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero vectors found")
    x = normalize(x, copy=False)
    texts = table["body_retrieval"].to_pylist()
    del table
    meta = pq.read_table(args.emails, columns=["email_id", "subject", "date_utc", "sender", "custodian"]).to_pandas()
    if meta.email_id.duplicated().any():
        raise ValueError("Duplicate source IDs")
    meta = meta.set_index("email_id").loc[ids].reset_index()
    meta["sender"] = meta.sender.map(lambda values: ", ".join(values))
    spe = pd.read_csv(ROOT / "data/spe_subset_emails.csv", usecols=["email_id", "in_new_text"])
    spe_ids = set(spe.email_id)
    spe_authored_ids = set(spe.loc[spe.in_new_text, "email_id"])
    meta["spe_match"] = meta.email_id.isin(spe_ids)
    meta["spe_in_new_text"] = meta.email_id.isin(spe_authored_ids)
    meta["text_chars"] = [len(text) for text in texts]
    meta["year"] = meta.date_utc.dt.tz_convert("America/Chicago").dt.year
    text_counts = pd.Series(texts).value_counts()
    text_quality = {"distinct_retrieval_texts": len(text_counts),
                    "emails_in_repeated_text_groups": int(text_counts[text_counts > 1].sum()),
                    "emails_under_50_characters": int((meta.text_chars < 50).sum()),
                    "most_common_texts": [{"text": text, "emails": int(count)}
                                          for text, count in text_counts.head(12).items()]}

    print("Fitting PCA on all emails...", flush=True)
    pca = PCA(n_components=50, svd_solver="randomized", random_state=SEED)
    pcs = pca.fit_transform(x)
    variance = pca.explained_variance_ratio_
    meta["pca_1"], meta["pca_2"] = pcs[:, 0], pcs[:, 1]
    print(f"PCA: first 2 PCs {variance[:2].sum():.1%}, first 50 PCs {variance.sum():.1%}", flush=True)

    train_idx = rng.choice(len(ids), min(40000, len(ids)), replace=False)
    eval_idx = rng.choice(len(ids), min(2500, len(ids)), replace=False)
    candidates = []
    for k in (6, 10, 14, 18):
        model = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=5, random_state=SEED)
        model.fit(x[train_idx])
        score = silhouette_score(x[eval_idx], model.predict(x[eval_idx]), metric="cosine")
        candidates.append({"k": k, "cosine_silhouette": float(score)})
        print(f"Clustering candidate k={k}: silhouette {score:.4f}", flush=True)
    best_k = max(candidates, key=lambda item: item["cosine_silhouette"])["k"]
    model = MiniBatchKMeans(n_clusters=best_k, batch_size=2048, n_init=10, random_state=SEED)
    labels = model.fit_predict(x)
    # Order group numbers by descending size to simplify tables and chart legends.
    order = pd.Series(labels).value_counts().index.to_list()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[label] for label in labels])
    centers = model.cluster_centers_[order]
    meta["cluster"] = labels
    final_silhouette = float(silhouette_score(x[eval_idx], labels[eval_idx], metric="cosine"))
    pd.DataFrame(candidates).to_csv(out / "cluster_selection.csv", index=False)

    # Descriptive keywords are derived from the full saved text; embeddings for
    # very long emails used only its first 8,191 tokens (see embedding script).
    print("Profiling groups using keywords, dates, metadata, and representative emails...", flush=True)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=18000,
                                min_df=30, max_df=.65, ngram_range=(1, 2), sublinear_tf=True,
                                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")
    tfidf = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()
    global_mean = np.asarray(tfidf.mean(axis=0)).ravel()
    summaries, representatives = [], []
    for cluster in range(best_k):
        indices = np.flatnonzero(labels == cluster)
        group = meta.iloc[indices]
        mean = np.asarray(tfidf[indices].mean(axis=0)).ravel()
        distinctive = mean - global_mean
        terms = vocabulary[np.argsort(distinctive)[-12:][::-1]].tolist()
        distances = np.sum((x[indices] - centers[cluster]) ** 2, axis=1)
        # Different text examples prevent near-identical mailbox/thread copies
        # from occupying every representative slot.
        seen, examples = set(), []
        for local in np.argsort(distances):
            i = int(indices[local])
            key = texts[i].strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            example = meta.iloc[i].to_dict()
            example["body_excerpt"] = texts[i][:1800]
            example["distance_to_centroid"] = float(distances[local])
            representatives.append(example)
            examples.append({"email_id": ids[i], "subject": str(meta.iloc[i].subject), "text": texts[i][:600]})
            if len(examples) == 5:
                break
        n_spe = int(group.spe_match.sum())
        row = {"cluster": cluster, "emails": len(indices), "share": len(indices) / len(ids),
               "keywords": ", ".join(terms), "spe_matches": n_spe,
               "spe_rate": n_spe / len(indices),
               "spe_enrichment": (n_spe / len(indices)) / meta.spe_match.mean(),
               "spe_in_new_text": int(group.spe_in_new_text.sum()),
               "median_text_chars": float(group.text_chars.median()),
               "under_50_chars_share": float((group.text_chars < 50).mean()),
               "phone_or_fax_share": float(pd.Series([texts[i] for i in indices]).str.contains(r"(?i)\b(?:fax|phone)\b").mean()),
               "top_senders": group.sender.value_counts().head(3).to_dict(),
               "top_custodians": group.custodian.value_counts().head(3).to_dict(),
               "year_counts": group.year.value_counts().sort_index().to_dict(),
               "examples": examples}
        summaries.append(row)
        print(f"Group {cluster}: {len(indices):,} emails; {n_spe} SPE; {', '.join(terms[:6])}", flush=True)
    pd.DataFrame([{key: value for key, value in row.items() if key != "examples"}
                  for row in summaries]).to_csv(out / "cluster_profiles.csv", index=False)
    pd.DataFrame(representatives).to_csv(out / "representative_emails.csv", index=False)
    meta.to_parquet(out / "email_coordinates_and_clusters.parquet", index=False)

    sample = np.sort(rng.choice(len(ids), min(args.sample_size, len(ids)), replace=False))
    print(f"Running t-SNE on {len(sample):,} randomly sampled emails...", flush=True)
    tsne = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto",
                max_iter=1250, random_state=SEED, verbose=1, n_jobs=4)
    xy = tsne.fit_transform(pcs[sample])
    sampled = meta.iloc[sample].copy()
    sampled["tsne_1"], sampled["tsne_2"] = xy[:, 0], xy[:, 1]
    sampled.to_parquet(out / "tsne_sample_coordinates.parquet", index=False)
    # Neighborhood diagnostic on a random subset; no claim of global fidelity.
    check_idx = rng.choice(len(sample), min(2000, len(sample)), replace=False)
    trust = float(trustworthiness(x[sample[check_idx]], xy[check_idx], n_neighbors=15, metric="cosine"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "#fafbfc"})
    colors = plt.get_cmap("tab20")(np.arange(best_k))
    names = GROUP_NAMES if best_k == 6 else [f"Group {c}" for c in range(best_k)]
    legend = [Line2D([0], [0], marker="o", linestyle="", color=colors[c],
                     label=f"{c}: {names[c]} ({summaries[c]['emails']:,})") for c in range(best_k)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    shuffle = rng.permutation(len(ids))
    axes[0].scatter(pcs[shuffle, 0], pcs[shuffle, 1], c=colors[labels[shuffle]], s=1, alpha=.15, rasterized=True)
    axes[0].set(title=f"PCA · all {len(ids):,} emails", xlabel=f"PC1 ({variance[0]:.1%} variance)",
                ylabel=f"PC2 ({variance[1]:.1%} variance)")
    axes[1].scatter(xy[:, 0], xy[:, 1], c=colors[labels[sample]], s=5, alpha=.6, linewidths=0)
    axes[1].set(title=f"t-SNE · random sample of {len(sample):,}", xlabel="t-SNE 1", ylabel="t-SNE 2")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.legend(handles=legend, title="Exploratory groups · full population counts", loc="lower center",
               bbox_to_anchor=(.5, .045), ncol=min(best_k, 3), frameon=False, fontsize=9)
    fig.suptitle("Enron email embeddings: broad structure and local neighborhoods", fontsize=17, y=1.01)
    fig.text(.5, .01, "Colors come from clustering the original 256-dimensional vectors. t-SNE island spacing is not a measure of topic distance.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .23, 1, 1))
    fig.savefig(out / "pca_tsne_clusters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, coordinates, frame, title in [(axes[0], pcs[:, :2], meta, "PCA · all emails"),
                                          (axes[1], xy, sampled, "t-SNE · random sample")]:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], color="#bcc5ce", s=3, alpha=.18)
        quoted = (frame.spe_match & ~frame.spe_in_new_text).to_numpy()
        authored = frame.spe_in_new_text.to_numpy()
        ax.scatter(coordinates[quoted, 0], coordinates[quoted, 1], s=22, color="#e49325", alpha=.8,
                   label=f"SPE only in quoted text ({quoted.sum():,})", linewidths=0)
        ax.scatter(coordinates[authored, 0], coordinates[authored, 1], s=25, color="#b52549", alpha=.85,
                   label=f"SPE in new text ({authored.sum():,})", linewidths=0)
        ax.set(title=title)
        ax.legend(loc="best", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Where known SPE-name matches fall in the embedding space", fontsize=16)
    fig.text(.5, .01, "SPE labels come from full-text name rules; embeddings represent retrieval text. A name match is not a wrongdoing label.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(out / "spe_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(np.arange(1, 51), np.cumsum(variance) * 100, color="#256b9b", linewidth=2)
    axes[0].set(title="PCA information retained", xlabel="Number of principal components", ylabel="Cumulative explained variance (%)")
    axes[1].barh(range(best_k), [r["emails"] for r in summaries], color=colors)
    axes[1].invert_yaxis()
    axes[1].set(title="Group sizes", xlabel="Emails", ylabel="Group")
    axes[2].barh(range(best_k), [r["spe_rate"] * 100 for r in summaries], color=colors)
    axes[2].axvline(meta.spe_match.mean() * 100, color="#333333", linestyle="--", label="Overall SPE-match rate")
    axes[2].invert_yaxis()
    axes[2].set(title="SPE-name prevalence by group", xlabel="SPE-name matches (%)", ylabel="Group")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    interactive = go.Figure()
    for cluster in range(best_k):
        select = np.flatnonzero(labels[sample] == cluster)
        rows = sampled.iloc[select]
        hover = [[html.escape(str(row.email_id)), html.escape(str(row.subject))[:180],
                  html.escape(texts[int(sample[select[j]])][:450]).replace("\n", "<br>"),
                  "New-text SPE match" if row.spe_in_new_text else "Quoted-only SPE match" if row.spe_match else "No SPE-name match"]
                 for j, row in enumerate(rows.itertuples())]
        color = "rgb(%d,%d,%d)" % tuple((colors[cluster][:3] * 255).astype(int))
        interactive.add_trace(go.Scattergl(
            x=xy[select, 0], y=xy[select, 1], mode="markers", name=f"{cluster}: {names[cluster]}",
            marker=dict(size=5, color=color, opacity=.65), customdata=hover,
            hovertemplate="%{customdata[1]}<br>ID: %{customdata[0]}<br>%{customdata[3]}<br><br>%{customdata[2]}<extra></extra>"))
    interactive.update_layout(title=f"Enron embeddings: t-SNE of {len(sample):,} random emails",
                              template="plotly_white", height=850,
                              xaxis=dict(title="t-SNE 1", showticklabels=False),
                              yaxis=dict(title="t-SNE 2", showticklabels=False),
                              legend=dict(orientation="h", y=-.05),
                              margin=dict(b=140))
    interactive.write_html(out / "tsne_explorer.html", include_plotlyjs=True)
    report = {"rows": len(ids), "dimensions": x.shape[1], "seed": SEED,
              "norm_min": float(norms.min()), "norm_max": float(norms.max()),
              "pca_variance_first_2": float(variance[:2].sum()), "pca_variance_first_50": float(variance.sum()),
              "clustering": "MiniBatchKMeans on L2-normalized original 256D vectors",
              "selection_candidates": candidates, "chosen_k": best_k, "final_cosine_silhouette": final_silhouette,
              "tsne_sample": len(sample), "tsne_perplexity": 40, "tsne_kl_divergence": float(tsne.kl_divergence_),
              "trustworthiness_subset_2000_cosine_k15": trust,
              "spe_matches": int(meta.spe_match.sum()), "spe_in_new_text": int(meta.spe_in_new_text.sum()),
              "text_quality": text_quality,
              "clusters": summaries}
    (out / "analysis_summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Done. Original-space silhouette={final_silhouette:.4f}; sampled trustworthiness={trust:.4f}. Outputs: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=ROOT / "embedding/primary_email_embeddings.parquet")
    parser.add_argument("--emails", type=Path, default=ROOT / "data/emails_analysis.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs/embeddings")
    parser.add_argument("--sample-size", type=int, default=12000)
    with threadpool_limits(limits=4):
        main(parser.parse_args())
