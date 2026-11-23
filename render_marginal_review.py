"""Render an auditable review ledger behind the marginal-yield chart."""
import html
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "analysis_outputs/semantic_search/marginal_yield"


def main():
    emails = pd.read_parquet(OUT / "review_emails.parquet")
    jobs = pd.read_csv(OUT / "reviewed_jobs.csv").fillna("")
    leads = json.loads((OUT / "lead_registry.json").read_text(encoding="utf-8"))
    def esc(value):
        return html.escape(str(value))
    doc = """<!doctype html><html><head><meta charset='utf-8'><title>Marginal-yield review audit</title>
<style>body{font:15px/1.6 system-ui,sans-serif;color:#17283b;max-width:1250px;margin:35px auto;padding:0 24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:16px;font:14px/1.5 monospace}details{padding:14px 0;border-bottom:1px solid #ddd}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}a{color:#175ea7}</style></head><body>
<h1>Marginal-yield pilot: review audit</h1>
<p>Initial reviewer: Codex. Forty randomly selected emails per pattern per rank band; 480 pattern-band assessments across 431 unique emails. Sampling seed: 6450. Earlier top-20 assessments were not forced into this random sample and were not automatically promoted to credible leads.</p>
<p>Review scope: retrieval-text triage, using abbreviated excerpts for long messages and targeted full-message follow-up for escalated cases. Full source text is provided below so judgments can be checked. This is initial analyst screening, not a fully adjudicated labeled dataset. Context and unresolved items are not established negatives. No wrongdoing findings are asserted.</p>
<p>A credible investigative lead requires a named transaction/product and a concrete retained-risk, protection-capacity, or representation inconsistency worth tracing. It need not satisfy the complete wrongdoing test. Routine contract terms, mere entity names, secondary news and lawful risk mitigation alone do not qualify. Legitimate explanations remain attached to every case. Distinct-case grouping is conservative and judgment-dependent.</p>
<p><a href='marginal_yield_explorer.html'>Back to chart</a></p><h2>Distinct-case registry</h2>"""
    lookup = emails.set_index("review_id")
    for key, lead in leads.items():
        doc += f"<h3>{esc(lead['title'])}</h3><p><b>{esc(key)}</b> · {esc(lead['status'])}</p>"
        doc += f"<p>{esc(lead.get('alternative', lead.get('limitation', '')))}</p>"
        if lead.get("deduplication"):
            doc += f"<p>{esc(lead['deduplication'])}</p>"
        for rid in lead.get("review_ids", []):
            doc += f"<a href='#{rid}'>{rid}: {esc(lookup.loc[rid].subject)}</a><br>"
    doc += "<h2>Screened emails and pattern judgments</h2>"
    for row in emails.itertuples():
        assessments = jobs[jobs.review_id == row.review_id]
        doc += f"<details id='{row.review_id}'><summary><b>{row.review_id}</b> · {esc(row.subject or '(No subject)')} · first sampled stage {row.first_sample_stage}</summary>"
        doc += f"<p>ID: {row.email_id}<br>UTC date: {esc(row.date_utc)}<br>From: {esc(row.sender)}<br>To: {esc(row.recipients)}<br>CC: {esc(row.cc)}</p>"
        doc += "<table><tr><th>Pattern/band</th><th>Rank</th><th>Assessment</th><th>Lead</th><th>Rationale</th></tr>"
        for item in assessments.itertuples():
            doc += f"<tr><td>{esc(item.pattern)} ({item.lower_k}, {item.upper_k}]</td><td>{item.rank}</td><td>{esc(item.verdict)}</td><td>{esc(item.lead_id)}</td><td>{esc(item.rationale)}</td></tr>"
        doc += f"</table><h3>Saved retrieval text</h3><pre>{esc(row.body_retrieval)}</pre><h3>Full cleaned message with quoted context</h3><pre>{esc(row.body_clean)}</pre></details>"
    doc += "</body></html>"
    (OUT / "review_audit.html").write_text(doc, encoding="utf-8")
    print(f"Saved audit for {len(emails)} unique emails and {len(jobs)} assessments.")


if __name__ == "__main__":
    main()
