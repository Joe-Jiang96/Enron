"""Prepare and evaluate a reproducible marginal-yield review of semantic ranks.

python marginal_yield.py prepare
python marginal_yield.py evaluate
No API calls. Analyst assessments are stored separately from rankings.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs/semantic_search/marginal_yield"
PATTERNS = ("who", "how_much", "when")
DEPTHS = (100, 200, 300, 500)
SEED = 6450
SAMPLE_SIZE = 40


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(ROOT / "analysis_outputs/semantic_search/all_email_scores.parquet")
    assert len(scores) == 152885 and scores.email_id.is_unique
    jobs, pools = [], []
    rng = np.random.default_rng(SEED)
    for pattern in PATTERNS:
        ranked = scores.sort_values([f"score_{pattern}", "email_id"], ascending=[False, True]).copy()
        ranked[f"rank_{pattern}"] = np.arange(1, len(ranked) + 1)
        scores = scores.merge(ranked[["email_id", f"rank_{pattern}"]], on="email_id", validate="one_to_one")
        lower = 0
        for upper in DEPTHS:
            band = ranked.iloc[lower:upper]
            selected = rng.choice(len(band), size=min(SAMPLE_SIZE, len(band)), replace=False)
            for i in np.sort(selected):
                row = band.iloc[int(i)]
                jobs.append({"pattern": pattern, "lower_k": lower, "upper_k": upper,
                             "email_id": row.email_id, "rank": int(row[f"rank_{pattern}"]),
                             "score": float(row[f"score_{pattern}"]), "band_population": len(band)})
            lower = upper
    job_frame = pd.DataFrame(jobs)
    selected_ids = set(job_frame.email_id)
    for upper in DEPTHS:
        ids = set(scores.loc[(scores[[f"rank_{p}" for p in PATTERNS]] <= upper).any(axis=1), "email_id"])
        previous = set() if not pools else pools[-1]["ids"]
        pools.append({"k_per_pattern": upper, "maximum_slots": 3 * upper,
                      "unique_emails": len(ids), "additional_unique_emails": len(ids - previous), "ids": ids})
    pd.DataFrame([{key: value for key, value in row.items() if key != "ids"} for row in pools]).to_csv(OUT / "pool_sizes.csv", index=False)
    largest = scores[(scores[[f"rank_{p}" for p in PATTERNS]] <= max(DEPTHS)).any(axis=1)]
    largest.to_parquet(OUT / "ranked_pool_through_k500.parquet", index=False)
    selected = scores[scores.email_id.isin(selected_ids)].copy()
    first = job_frame.groupby("email_id").upper_k.min()
    selected["first_sample_stage"] = selected.email_id.map(first)
    selected["best_rank"] = selected[[f"rank_{p}" for p in PATTERNS]].min(axis=1)
    selected = selected.sort_values(["first_sample_stage", "best_rank", "email_id"]).reset_index(drop=True)
    selected.insert(0, "review_id", [f"R{i:03d}" for i in range(1, len(selected) + 1)])
    meta = pq.read_table(ROOT / "data/emails_analysis.parquet",
                         columns=["email_id", "subject", "subject_norm", "date_utc", "sender", "recipients", "cc", "body_clean", "body_new_candidate"],
                         filters=[("email_id", "in", sorted(selected_ids))]).to_pandas()
    text = pq.read_table(ROOT / "embedding/primary_email_embeddings.parquet", columns=["email_id", "body_retrieval"],
                         filters=[("email_id", "in", sorted(selected_ids))]).to_pandas()
    selected = selected.merge(meta, on="email_id", validate="one_to_one").merge(text, on="email_id", validate="one_to_one")
    for column in ("sender", "recipients", "cc"):
        selected[column] = selected[column].map(lambda values: ", ".join(values))
    job_frame = job_frame.merge(selected[["email_id", "review_id"]], on="email_id", validate="many_to_one")
    # Prevent a changed sampling plan from silently reassigning saved assessments.
    plan = {"depths": DEPTHS, "sample_size": SAMPLE_SIZE, "seed": SEED,
            "jobs_sha256": hashlib.sha256(job_frame.to_csv(index=False).encode()).hexdigest(),
            "scores_sha256": hashlib.sha256((ROOT / "analysis_outputs/semantic_search/all_email_scores.parquet").read_bytes()).hexdigest()}
    manifest = OUT / "review_plan.json"
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError("Existing review plan differs; preserve assessments before making a new plan")
    manifest.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    selected.to_parquet(OUT / "review_emails.parquet", index=False)
    job_frame.to_csv(OUT / "review_jobs.csv", index=False)
    print(f"Prepared {len(jobs)} pattern-band slots, {len(selected)} unique sampled emails, {selected.body_retrieval.nunique()} distinct retrieval texts.")
    print(pd.read_csv(OUT / "pool_sizes.csv").to_string(index=False))


def evaluate():
    jobs = pd.read_csv(OUT / "review_jobs.csv")
    emails = pd.read_parquet(OUT / "review_emails.parquet")
    annotations = json.loads((OUT / "assessments.json").read_text(encoding="utf-8"))
    leads = json.loads((OUT / "lead_registry.json").read_text(encoding="utf-8"))
    valid_codes = {"credible_lead", "context", "unclear", "irrelevant"}
    rows = []
    for job in jobs.to_dict("records"):
        item = annotations.get(job["review_id"], {}).get(job["pattern"])
        if item:
            if item["verdict"] not in valid_codes or not item.get("rationale"):
                raise ValueError("Invalid assessment")
            lead = item.get("lead_id", "")
            if item["verdict"] == "credible_lead" and (not lead or lead not in leads):
                raise ValueError("Credible leads require a registered distinct case ID")
            if lead and lead not in leads:
                raise ValueError("Unknown lead ID")
        rows.append({**job, "verdict": item["verdict"] if item else "unreviewed",
                     "lead_id": item.get("lead_id", "") if item else "",
                     "rationale": item["rationale"] if item else ""})
    reviewed = pd.DataFrame(rows)
    reviewed.to_csv(OUT / "reviewed_jobs.csv", index=False, encoding="utf-8-sig")
    known = {key for key, value in leads.items() if value.get("known_before_pilot")}
    pattern_rows = []
    for pattern in PATTERNS:
        seen = set(known)
        streak = 0
        for k in DEPTHS:
            band = reviewed[(reviewed.pattern == pattern) & (reviewed.upper_k == k)]
            complete = band.verdict.ne("unreviewed").all()
            accepted = set(band.loc[band.verdict == "credible_lead", "lead_id"]) - {""}
            new = accepted - seen
            seen |= accepted
            n = int(band.loc[band.verdict != "unreviewed", "email_id"].nunique())
            rate = len(new) / n if n else None
            if k == DEPTHS[0]:
                streak = 0  # Baseline band does not count toward two increments.
            elif complete and rate < .01:
                streak += 1
            else:
                streak = 0
            pattern_rows.append({"pattern": pattern, "k": k, "sampled_emails": len(band),
                                 "reviewed_unique_emails": n, "complete": bool(complete),
                                 "new_distinct_leads": len(new) if complete else None,
                                 "new_lead_ids": ", ".join(sorted(new)),
                                 "observed_marginal_yield": rate if complete else None,
                                 "unclear_emails": int((band.verdict == "unclear").sum()),
                                 "two_low_increments_observed": bool(streak >= 2)})
    pd.DataFrame(pattern_rows).to_csv(OUT / "pattern_marginal_yield.csv", index=False)
    # Combined counts use unique emails at their first scheduled review stage;
    # overlapping query slots must never be summed as extra email reviews.
    stage_first = jobs.groupby("email_id").upper_k.min()
    all_seen = set(known)
    pooled = []
    for k in DEPTHS:
        new_ids = set(stage_first[stage_first == k].index)
        applicable = reviewed[reviewed.email_id.isin(new_ids)]
        complete = applicable.verdict.ne("unreviewed").all()
        accepted = set(applicable.loc[applicable.verdict == "credible_lead", "lead_id"]) - {""}
        new_leads = accepted - all_seen
        all_seen |= accepted
        pooled.append({"k": k, "additional_unique_emails_reviewed": len(new_ids) if complete else None,
                       "new_distinct_leads": len(new_leads) if complete else None,
                       "new_lead_ids": ", ".join(sorted(new_leads)),
                       "observed_marginal_yield": len(new_leads) / len(new_ids) if complete and new_ids else None,
                       "complete": bool(complete)})
    pd.DataFrame(pooled).to_csv(OUT / "pooled_marginal_yield.csv", index=False)
    full = emails.merge(reviewed.groupby("email_id")[["pattern", "verdict", "lead_id", "rationale"]].apply(lambda g: "; ".join(
        f"{r.pattern}: {r.verdict} [{r.lead_id}] {r.rationale}" for r in g.itertuples())).rename("assessment"),
                        on="email_id", validate="one_to_one")
    full.to_parquet(OUT / "reviewed_emails.parquet", index=False)
    print(pd.DataFrame(pattern_rows).to_string(index=False))
    print("\nPOOLED\n", pd.DataFrame(pooled).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else evaluate()
