"""Collect Mahonia correspondence and traceable context from the full corpus.

Run: python reconstruct_mahonia.py
No API calls. Includes emails outside the primary embedding population.
Automated matches are discovery leads, not authenticated thread edges.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SEED_ID = "97e8e939090808cef30067cfb8940da2"
ANCHORS = {"soma.ghosh@enron.com", "sara.shackleton@enron.com", "tana.jones@enron.com",
           "cris.sherman@enron.com", "clement.abrams@enron.com", "tim.proffitt@enron.com"}
MAHONIA = re.compile(r"\bmahonia\b", re.I)
RELATED_SUBJECT = re.compile(
    r"re[- ]?hypothec|use of posted collateral|domestic custod|\bchase\b.*\bsecurit"
    r"|\bsecurit\w*\b.*\bchase\b|amendment of chase/erms isda|amendment to security agreement.*\bchase\b", re.I)
COLLATERAL = re.compile(r"rehypothec|custod|collateral|securit", re.I)
CHASE = re.compile(r"\bchase\b", re.I)
START = datetime(2000, 10, 1, tzinfo=timezone.utc)
END = datetime(2000, 12, 16, tzinfo=timezone.utc)


def main(args):
    source = pq.ParquetFile(args.emails)
    columns = [c for c in source.schema_arrow.names if c != "minhash_signature"]
    records, all_locations = [], []
    for batch in source.iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            all_locations.append({key: row[key] for key in ["email_id", "canonical_email_id", "custodian", "folder", "date_utc"]})
            subject, full, new = row["subject"], row["body_clean"], row["body_new_candidate"]
            participants = set(row["sender"] + row["recipients"] + row["cc"] + row["bcc"])
            reasons = []
            if MAHONIA.search(subject + "\n" + full):
                reasons.append("explicit_mahonia")
            if RELATED_SUBJECT.search(subject):
                reasons.append("related_subject")
            if (row["date_utc"] and START <= row["date_utc"] < END and participants & ANCHORS
                    and COLLATERAL.search(subject + "\n" + full) and CHASE.search(subject + "\n" + full)):
                reasons.append("near_seed_chase_collateral_participant")
            if row["email_id"] == SEED_ID:
                reasons.append("seed")
            if not reasons:
                continue
            row["discovery_reasons"] = reasons
            row["mahonia_in_new_text"] = bool(MAHONIA.search(new))
            row["mahonia_in_subject"] = bool(MAHONIA.search(subject))
            row["new_text_sha256"] = hashlib.sha256(new.encode()).hexdigest()
            records.append(row)
    if SEED_ID not in {r["email_id"] for r in records}:
        raise ValueError("Seed email missing")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    discovery = pa.Table.from_pylist(records)
    pq.write_table(discovery, out / "discovery_emails.parquet", compression="zstd")
    frame = discovery.to_pandas()
    representatives = frame[~frame.is_duplicate_copy].sort_values(["date_utc", "email_id"])
    selected_canonical = set(frame.canonical_email_id)
    locations = [r for r in all_locations if r["canonical_email_id"] in selected_canonical]
    pq.write_table(pa.Table.from_pylist(locations), out / "mailbox_locations.parquet", compression="zstd")
    summary_columns = ["email_id", "canonical_email_id", "date_utc", "subject", "sender", "recipients", "cc", "custodian",
                       "discovery_reasons", "mahonia_in_new_text", "mahonia_in_subject", "new_text_sha256"]
    summary = representatives[summary_columns].copy()
    summary["date_houston"] = summary.date_utc.dt.tz_convert("America/Chicago").astype(str)
    for key in ["sender", "recipients", "cc", "discovery_reasons"]:
        summary[key] = summary[key].map(lambda values: ", ".join(values))
    summary["new_text_excerpt"] = representatives.body_new_candidate.str.slice(0, 600)
    summary.to_csv(out / "discovery_index.csv", index=False, encoding="utf-8-sig")
    # A quotation edge means the earlier email's own text is reproduced in a
    # later cleaned body. This does not authenticate a formal reply header.
    normalize = lambda text: re.sub(r"\s+", " ", text).strip().casefold()
    signature_name = re.compile(r"(?im)^\s*(?:teresa g\.? bushman|sara shackleton|soma ghosh)\s*$")
    links = []
    for target in representatives.to_dict("records"):
        full, new = normalize(target["body_clean"]), normalize(target["body_new_candidate"])
        if not full.startswith(new):
            continue
        quoted = full[len(new):]
        for parent in representatives.to_dict("records"):
            # A repeated free-form signature is not evidence of a message link.
            own_text = signature_name.split(parent["body_new_candidate"], maxsplit=1)[0]
            own = normalize(own_text)
            if parent["email_id"] == target["email_id"] or len(own) < 120 or own not in quoted:
                continue
            links.append({"source_email_id": parent["email_id"], "target_email_id": target["email_id"],
                          "link_type": "own_text_reproduced_in_quoted_region",
                          "matched_characters_normalized": len(own),
                          "source_message_id": parent["message_id"], "target_message_id": target["message_id"],
                          "caution": "Text-supported link; no In-Reply-To/References headers available"})
    pd.DataFrame(links).to_csv(out / "quoted_text_links.csv", index=False)
    report = {"seed_id": SEED_ID, "source_rows": source.metadata.num_rows, "matched_rows": len(frame),
              "representative_emails": len(representatives),
              "reasons_in_representatives": dict(Counter(reason for reasons in representatives.discovery_reasons for reason in reasons)),
              "explicit_mahonia_representatives": int(representatives.discovery_reasons.map(lambda x: "explicit_mahonia" in x).sum()),
              "quotation_links": len(links),
              "date_context_window_utc": [START.isoformat(), END.isoformat()],
              "scope": "Full cleaned corpus, not limited by primary baseline or October 2001 cutoff",
              "limitations": ["Raw originals and In-Reply-To/References headers are not available in this checkout.",
                               "Subject, participant and keyword matches are discovery links, not verified thread membership.",
                               "Date/time inconsistencies, near-duplicate text and missing attachments need review."]}
    (out / "discovery_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(summary[["email_id", "date_houston", "subject", "discovery_reasons"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emails", type=Path, default=ROOT / "data/emails_analysis.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs/semantic_search/mahonia")
    main(parser.parse_args())
