"""Render the inspected Mahonia case notes, source messages and evidence matrix.

Run after reconstruct_mahonia.py: python render_mahonia_report.py
The authored case_notes.json records analyst judgments separately from discovery.
No API calls. No original files are altered.
"""
import hashlib
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs/semantic_search/mahonia"


def main():
    notes = json.loads((OUT / "case_notes.json").read_text(encoding="utf-8"))
    discovery = pd.read_parquet(OUT / "discovery_emails.parquet")
    if discovery.email_id.duplicated().any():
        raise ValueError("Duplicate discovery IDs")
    lookup = discovery.set_index("email_id")
    curated = [event[0] for event in notes["events"]]
    contextual = [event[0] for event in notes["context"]]
    if not set(curated + contextual) <= set(lookup.index):
        raise ValueError("Annotated IDs missing from discovery")
    discovery["case_scope"] = discovery.email_id.map(
        lambda eid: "working_sequence" if eid in curated else "selected_context" if eid in contextual else "discovery_only")
    scores = pd.read_parquet(ROOT / "analysis_outputs/semantic_search/all_email_scores.parquet")
    discovery = discovery.merge(scores, on="email_id", how="left", validate="one_to_one")
    discovery["in_primary_embedding_population"] = discovery.score_who.notna()
    discovery.to_parquet(OUT / "case_messages.parquet", index=False)
    timeline = []
    for i, (eid, title, observation, qualification) in enumerate(notes["events"], 1):
        row = lookup.loc[eid]
        timeline.append({"event": i, "email_id": eid, "date_utc": row.date_utc,
                         "date_houston": row.date_utc.tz_convert("America/Chicago").isoformat(),
                         "subject": row.subject, "sender": ", ".join(row.sender),
                         "recipients": ", ".join(row.recipients), "cc": ", ".join(row.cc),
                         "event_title": title, "observation": observation, "qualification": qualification})
    pd.DataFrame(timeline).to_csv(OUT / "thread_timeline.csv", index=False, encoding="utf-8-sig")
    evidence = pd.DataFrame(notes["evidence"], columns=["element", "status", "observation", "email_ids", "alternative_or_missing_evidence"])
    evidence.to_csv(OUT / "evidence_matrix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(notes["date_cautions"]).to_csv(OUT / "date_cautions.csv", index=False)
    edges = pd.read_csv(OUT / "quoted_text_links.csv")
    core_edges = edges[edges.source_email_id.isin(curated) & edges.target_email_id.isin(curated)]
    core_edges.to_csv(OUT / "working_sequence_quote_links.csv", index=False)

    def esc(value):
        return html.escape(str(value))

    def link(eid, label=None):
        return f"<a href='#mail-{eid}'>{esc(label or lookup.loc[eid].subject or '(No subject)')}</a>"

    def refs(values):
        return "<br>".join(link(eid.strip()) for eid in values.split(";"))

    document = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Mahonia correspondence reconstruction</title>
<style>body{font:16px/1.6 system-ui,sans-serif;color:#18283b;background:#f4f6f8;margin:0}main{max-width:1240px;margin:32px auto;padding:0 24px}
h1,h2,h3{line-height:1.3}h1{font-size:32px}h2{margin-top:40px}h3{margin-bottom:8px}a{color:#175ea7}
section,.callout{background:white;padding:22px;border:1px solid #dce3ea;border-radius:8px;margin:18px 0}
.callout{border-left:5px solid #2468bb}.caution{background:#fff9ed}.small{font-size:13px;color:#526275}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce3ea;padding:11px;overflow-wrap:anywhere}th{background:#eef2f6}
details{border-top:1px solid #dce3ea;padding:14px 0;scroll-margin-top:20px}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 ui-monospace,monospace;background:#f5f7fa;padding:18px}
.scroll{overflow-x:auto}.tag{font-size:12px;padding:3px 7px;background:#e9eef5;border-radius:4px}nav a{margin-right:18px}
</style></head><body><main>"""
    document += f"<h1>{esc(notes['title'])}</h1><p>Initial analyst reconstruction from cleaned corpus text. No API calls.</p>"
    document += "<nav><a href='#timeline'>Timeline</a><a href='#links'>Quote links</a><a href='#evidence'>Evidence test</a><a href='#context'>Separate context</a><a href='#sources'>Source messages</a></nav>"
    document += f"<div class='callout'><b>Assessment</b><p>{esc(notes['conclusion'])}</p></div>"
    document += f"<p>{esc(notes['scope'])}</p><p>{esc(notes['review_basis'])}</p>"
    document += "<div class='callout caution'><b>Link and date limits</b><p>Same subject or shared participants do not establish a formal thread. Quotation links below demonstrate repeated message text, not authenticated transport headers. Stored and quoted clock times disagree; the timeline uses dates rather than precise hours.</p></div>"
    document += "<h2 id='timeline'>Working sequence: October 2000–January 2001</h2><section><div class='scroll'><table><tr><th>Date</th><th>Message/event</th><th>Observed content</th><th>Link strength and qualification</th></tr>"
    for event in timeline:
        document += f"<tr><td>{event['date_houston'][:10]}</td><td><b>{esc(event['event_title'])}</b><br>{link(event['email_id'])}<br><span class='small'>{esc(event['sender'])} → {esc(event['recipients'])}</span></td><td>{esc(event['observation'])}</td><td>{esc(event['qualification'])}</td></tr>"
    document += "</table></div></section>"
    document += f"<h2 id='links'>Text-supported quotation links within this sequence</h2><p>{len(core_edges)} links, including transitive quotations. A later reply can reproduce more than one earlier message.</p><section><table><tr><th>Source message</th><th>Reproduced in</th><th>Matched characters</th></tr>"
    for row in core_edges.itertuples():
        source_date = lookup.loc[row.source_email_id].date_utc.date()
        target_date = lookup.loc[row.target_email_id].date_utc.date()
        document += f"<tr><td>{source_date}: {link(row.source_email_id)}</td><td>{target_date}: {link(row.target_email_id)}</td><td>{row.matched_characters_normalized:,}</td></tr>"
    document += "</table></section><p class='small'>Matching normalizes whitespace and case and removes recognized signature names before comparing the earlier message's own text with the later quoted region. Subject-only or participant-only links are not counted here.</p>"
    document += "<h2 id='evidence'>Evidence test</h2><section><div class='scroll'><table><tr><th>Element</th><th>Assessment</th><th>Observation and sources</th><th>Alternatives or missing evidence</th></tr>"
    for row in evidence.itertuples():
        document += f"<tr><td>{esc(row.element)}</td><td>{esc(row.status)}</td><td>{esc(row.observation)}<p>{refs(row.email_ids)}</p></td><td>{esc(row.alternative_or_missing_evidence)}</td></tr>"
    document += "</table></div></section><h2 id='context'>Separate and earlier transaction context</h2><section>"
    for eid, title, observation in notes["context"]:
        document += f"<h3>{esc(title)}</h3><p>{link(eid)} · {lookup.loc[eid].date_utc.date()}</p><p>{esc(observation)}</p>"
    document += "</section><h2>Clock-time cautions</h2><section>"
    for row in notes["date_cautions"]:
        document += f"<p>{link(row['email_id'])}<br>{esc(row['source'])}<br>{esc(row['quoted'])}<br><b>{esc(row['implication'])}</b></p>"
    document += "</section><h2>What would resolve the remaining evidence gaps?</h2><section><ul>"
    gaps = ["The actual executed Chase/Mahonia CSA amendments, Exhibit A, custody agreement, collateral schedules and haircut values.",
            "The original AA correspondence and final accounting decision, including who approved which treatment and when.",
            "Actual collateral calls, securities movements and accounting entries, matched to the correct Mahonia series.",
            "An attributable representation or instruction that conflicts with the known economics, if one exists; preserve compliant alternatives as well."]
    document += "".join(f"<li>{esc(gap)}</li>" for gap in gaps) + "</ul></section>"
    document += "<h2>Limitations</h2><section><ul>" + "".join(f"<li>{esc(text)}</li>" for text in notes["limitations"]) + "</ul></section>"
    document += f"<h2 id='sources'>All {len(discovery)} discovered messages</h2><p>Working sequence and selected context have analyst notes above. Discovery-only entries have not been individually validated as belonging to this case.</p><section>"
    for row in discovery.sort_values(["date_utc", "email_id"]).itertuples():
        date = row.date_utc.tz_convert("America/Chicago").isoformat() if pd.notna(row.date_utc) else "Unknown"
        document += f"<details id='mail-{row.email_id}'><summary>{date[:10]} · {esc(row.subject or '(No subject)')} <span class='tag'>{esc(row.case_scope)}</span></summary>"
        document += f"<p><b>Email ID:</b> {row.email_id}<br><b>Message ID:</b> {esc(row.message_id)}<br><b>Stored UTC:</b> {esc(row.date_utc)}<br><b>Houston conversion:</b> {esc(date)}<br><b>From:</b> {esc(', '.join(row.sender))}<br><b>To:</b> {esc(', '.join(row.recipients))}<br><b>CC:</b> {esc(', '.join(row.cc))}<br><b>Custodian:</b> {esc(row.custodian)}<br><b>Discovery:</b> {esc(', '.join(row.discovery_reasons))}</p>"
        document += f"<h3>New-text candidate</h3><pre>{esc(row.body_new_candidate or '(Empty new text; quoted/forwarded material only)')}</pre><h3>Full cleaned message, including quoted context</h3><pre>{esc(row.body_clean)}</pre><p><a href='#timeline'>Back to timeline</a></p></details>"
    document += "</section></main></body></html>"
    (OUT / "mahonia_reconstruction.html").write_text(document, encoding="utf-8")
    text = [notes["title"], "=" * 40, notes["conclusion"], notes["scope"], notes["review_basis"], "", "WORKING SEQUENCE"]
    text += [f"{row['date_houston'][:10]} | {row['email_id']} | {row['event_title']}\n{row['observation']}\nQualification: {row['qualification']}\n" for row in timeline]
    text += ["EVIDENCE TEST"] + [" | ".join(row) for row in notes["evidence"]]
    text += ["", "SEPARATE CONTEXT"] + [" | ".join(row) for row in notes["context"]]
    text += ["", "LIMITATIONS"] + notes["limitations"]
    (OUT / "mahonia_reconstruction.txt").write_text("\n\n".join(text), encoding="utf-8")
    with (ROOT / "data/emails_analysis.parquet").open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"discovered_emails": len(discovery), "explicit_mahonia_emails": int(discovery.discovery_reasons.map(lambda r: "explicit_mahonia" in r).sum()),
                "working_sequence_emails": len(curated), "selected_context_emails": len(contextual),
                "all_quote_links": len(edges), "working_sequence_quote_links": len(core_edges),
                "outside_primary_embedding_population": int((~discovery.in_primary_embedding_population).sum()),
                "cleaned_source_sha256": source_hash,
                "case_notes_sha256": hashlib.sha256((OUT / "case_notes.json").read_bytes()).hexdigest()}
    (OUT / "case_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved {OUT / 'mahonia_reconstruction.html'}")


if __name__ == "__main__":
    main()
