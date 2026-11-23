"""Build a deduplicated review table for the top 20 emails per semantic pattern.

Run: python review_semantic_matches.py
Existing scores/embeddings are reused. No API calls or source changes.
Optional --labels JSON contains {email_id: {pattern: {label, rationale}}}.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PATTERNS = ("who", "how_much", "when")


def render_html(review, out, top_n):
    def esc(value):
        return html.escape(str(value))
    rows, cards = [], []
    for row in review.to_dict("records"):
        eid = row["email_id"]
        ranks = " / ".join(f"{p.upper()}: {row['rank_' + p]}" for p in PATTERNS)
        scores = " / ".join(f"{p.upper()}: {row['score_' + p]:.4f}" for p in PATTERNS)
        assessments = "".join(f"<p><b>{p.upper()}</b>: {esc(row['review_' + p] or 'Unreviewed')} — {esc(row['rationale_' + p])}</p>"
                              for p in row["selected_by"].split(", "))
        rows.append(f"<tr><td><a href='#{eid}'>{esc(row['subject'])}</a></td><td>{esc(row['date_houston'])}</td>"
                    f"<td>{esc(row['selected_by'])}</td><td>{esc(ranks)}</td></tr>")
        cards.append(f"<article id='{eid}'><h2>{esc(row['subject'])}</h2><p><b>Email ID:</b> {eid}</p>"
                     f"<p><b>Date (Houston):</b> {esc(row['date_houston'])}<br><b>From:</b> {esc(row['sender'])}"
                     f"<br><b>To:</b> {esc(row['recipients'])}<br><b>CC:</b> {esc(row['cc'])}</p>"
                     f"<p>{esc(ranks)}<br>{esc(scores)}</p>{assessments}"
                     f"<h3>Retrieval text represented by embedding</h3><pre>{esc(row['body_retrieval'])}</pre>"
                     f"<details><summary>Full cleaned message, including quoted context</summary><pre>{esc(row['body_clean'])}</pre></details>"
                     "<p><a href='#index'>Back to index</a></p></article>")
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Semantic search review</title>
<style>body{font:15px/1.55 system-ui,sans-serif;color:#172033;max-width:1150px;margin:40px auto;padding:0 24px;background:#f5f7fa}
article{padding:28px;background:white;border:1px solid #dce2e9;border-radius:8px;margin:24px 0;scroll-margin-top:20px}
table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dce2e9}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 ui-monospace,monospace;background:#f5f7fa;padding:18px}
a{color:#215aa3}summary{cursor:pointer;font-weight:600}h1,h2,h3{line-height:1.25}p{overflow-wrap:anywhere}
</style></head><body>"""
    document += f"<h1 id='index'>Top {top_n} semantic matches per pattern</h1><p>{len(review)} unique emails after merging WHO, HOW MUCH and WHEN results.</p>"
    document += "<p>Relevance labels assess retrieval usefulness, not wrongdoing. Full cleaned text is additional context and may contain forwarded third-party statements. Verify attribution, dates and original evidence before drawing conclusions.</p>"
    document += "<table><thead><tr><th>Subject</th><th>Date (Houston)</th><th>Selected for</th><th>Global ranks</th></tr></thead><tbody>"
    document += "".join(rows) + "</tbody></table>" + "".join(cards) + "</body></html>"
    (out / f"top{top_n}_review.html").write_text(document, encoding="utf-8")


def main(args):
    scores = pd.read_parquet(args.scores)
    if len(scores) != 152885 or scores.email_id.duplicated().any():
        raise ValueError("Expected 152,885 unique score rows")
    if not np.isfinite(scores[[f"score_{p}" for p in PATTERNS]]).all().all():
        raise ValueError("Invalid scores")
    if not 1 <= args.top_n <= len(scores):
        raise ValueError("Invalid --top-n")
    selected = set()
    for pattern in PATTERNS:
        order = scores.sort_values([f"score_{pattern}", "email_id"], ascending=[False, True]).index
        ranks = pd.Series(np.arange(1, len(scores) + 1), index=order)
        scores[f"rank_{pattern}"] = ranks
        selected.update(scores.loc[order[:args.top_n], "email_id"])
    review = scores[scores.email_id.isin(selected)].copy()
    review["selected_by"] = review.apply(lambda r: ", ".join(p for p in PATTERNS if r[f"rank_{p}"] <= args.top_n), axis=1)
    review["best_rank"] = review[[f"rank_{p}" for p in PATTERNS]].min(axis=1)
    review = review.sort_values(["best_rank", "email_id"]).reset_index(drop=True)
    meta_columns = ["email_id", "canonical_email_id", "message_id", "subject", "subject_norm", "date_utc",
                    "sender", "recipients", "cc", "bcc", "custodian", "body_clean", "body_new_candidate"]
    meta = pq.read_table(args.emails, columns=meta_columns, filters=[("email_id", "in", sorted(selected))]).to_pandas()
    bodies = pq.read_table(args.embeddings, columns=["email_id", "body_retrieval"],
                           filters=[("email_id", "in", sorted(selected))]).to_pandas()
    if set(meta.email_id) != selected or set(bodies.email_id) != selected:
        raise ValueError("Missing candidate metadata/text")
    review = review.merge(meta, on="email_id", how="left", validate="one_to_one").merge(bodies, on="email_id", how="left", validate="one_to_one")
    review["date_houston"] = review.date_utc.dt.tz_convert("America/Chicago").astype(str)
    review["before_reference_cutoff"] = review.date_utc < pd.Timestamp("2001-10-16", tz="America/Chicago")
    for column in ("sender", "recipients", "cc", "bcc"):
        review[column] = review[column].map(lambda values: ", ".join(values))
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else {}
    if set(labels) - selected:
        raise ValueError("Review labels refer to emails outside this top-N pool")
    for pattern in PATTERNS:
        review[f"review_{pattern}"] = [labels.get(eid, {}).get(pattern, {}).get("label", "") for eid in review.email_id]
        review[f"rationale_{pattern}"] = [labels.get(eid, {}).get(pattern, {}).get("rationale", "") for eid in review.email_id]
        if not review[f"review_{pattern}"].isin(["", "relevant", "unclear", "irrelevant"]).all():
            raise ValueError("Invalid relevance label")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    review.to_parquet(out / f"top{args.top_n}_review.parquet", index=False)
    review.to_csv(out / f"top{args.top_n}_review.csv", index=False, encoding="utf-8-sig")
    per_pattern = []
    for pattern in PATTERNS:
        subset = review[review[f"rank_{pattern}"] <= args.top_n].sort_values(f"rank_{pattern}").copy()
        subset.insert(0, "pattern", pattern)
        per_pattern.append(subset)
        print(pattern.upper(), subset[["email_id", "subject", f"score_{pattern}", f"rank_{pattern}"]].to_string(index=False), sep="\n")
    pd.concat(per_pattern).to_csv(out / f"top{args.top_n}_by_pattern.csv", index=False, encoding="utf-8-sig")
    render_html(review, out, args.top_n)
    summary = {"top_n": args.top_n, "selected_slots": len(PATTERNS) * args.top_n,
               "unique_emails": len(review), "multi_pattern_emails": int(review.selected_by.str.contains(",").sum()),
               "before_reference_cutoff": int(review.before_reference_cutoff.sum()),
               "score_file_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
               "review_labels": str(args.labels) if args.labels else None,
               "initial_review_counts": {p: review.loc[review[f"rank_{p}"] <= args.top_n,
                                                        f"review_{p}"].replace("", "unreviewed").value_counts().to_dict()
                                         for p in PATTERNS},
               "labels_mean": "Retrieval relevance to a pattern; not an evidence finding or wrongdoing conclusion"}
    (out / f"top{args.top_n}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(review)} unique emails from {len(PATTERNS) * args.top_n} ranked slots to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "analysis_outputs/semantic_search/all_email_scores.parquet")
    parser.add_argument("--emails", type=Path, default=ROOT / "data/emails_analysis.parquet")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "embedding/primary_email_embeddings.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs/semantic_search/review")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--labels", type=Path)
    main(parser.parse_args())
