# Enron EDA decision memo

Snapshot of cleaning run `20261113T083630222199Z`, produced by `Enron_EDA.ipynb` (sections 1, 2, 3, 5 and 7). Tables and charts are in `eda_outputs/20261113T083630222199Z/`.
Not done in this pass: monthly SPE rates with an event timeline (item 4) and contrastive term analysis (item 6).

## What the data looks like
- **249,243 unique emails** (2,491 further mailbox copies) from **158 mailboxes** (after merging labels that differ only in case; 258 labels in the raw data). Median mailbox: 938 emails; the 10 largest hold 32%.
- **Dates are unreliable at the edges.** Dates run from 1979 to 2044. 458 unique emails fall outside 1998 to 2002. Exclude or manually check them in any time-based step.
- **Most emails carry quoted text.** 43.7% have a quote marker, and among those 95.7% are under half new text. 7.5% of emails have no new text at all (bare forwards).
- 21.4% of emails are from senders outside Enron (news, vendors) and 4.7% go to more than 20 recipients.

## SPE term list: which names can be trusted
Precision is the share of sampled emails (10 per stratum) that use the name in the Enron SPE or structured-finance sense, from one reviewer working on short excerpts. Intervals are wide; see `term_precision.csv`.

| term | class | emails | with context | without context |
|---|---|---|---|---|
| raptor | stands alone | 344 | 100% (10/10) | 90% (9/10) |
| ljm | stands alone | 315 | 100% (10/10) | 100% (10/10) |
| jedi | stands alone | 166 | 80% (8/10) | 90% (9/10) |
| whitewing | stands alone | 87 | 100% (10/10) | 100% (10/10) |
| chewco | stands alone | 75 | 100% (10/10) | 100% (10/10) |
| hawaii_125 | stands alone | 52 | 100% (10/10) | 100% (10/10) |
| condor | stands alone | 49 | 100% (10/10) | 100% (10/10) |
| trutta | stands alone | 21 | 100% (6/6) | 100% (10/10) |
| project_velocity | stands alone | 3 | 100% (3/3) | n/a |
| marlin | needs context | 218 | 100% (10/10) | 30% (3/10) |
| osprey | needs context | 154 | 90% (9/10) | 60% (6/10) |
| rhythms | needs context | 55 | 100% (10/10) | 0% (0/9) |
| talon | needs context | 48 | 90% (9/10) | 50% (5/10) |
| southampton | needs context | 35 | 67% (2/3) | 10% (1/10) |
| bobcat | needs context | 26 | 100% (10/10) | 60% (6/10) |
| porcupine | needs context | 16 | 100% (6/6) | 50% (5/10) |
| timberwolf | needs context | 12 | 100% (2/2) | 70% (7/10) |
| hawaii | dropped | 759 | 20% (2/10) | 20% (2/10) |
| cuiaba | dropped | 208 | 0% (0/10) | 20% (2/10) |

## Decisions for the retrieval steps
| Decision | Evidence | Recommendation |
|---|---|---|
| **Which names stand alone** | Precision table above | Use the 9 stand-alone names as direct filters. Require a financial-structure word (proposal Table 2 groups B and C) within about 100 words for talon, porcupine, osprey, marlin, rhythms, timberwolf, bobcat, southampton. Drop hawaii and cuiaba as stand-alone terms. |
| **Cost of those rules** | `term_class_cost.csv` | About 250 real mentions are left out (rough estimate). Almost all of the gap is plain "Hawaii": an estimated 139 real mentions (range 56-290) sit in emails with no "Hawaii 125" wording. Excerpts show the same structure under other names (McGarrett, Danno, Bali Trust); these are leads to source and test, not yet terms. |
| **Search text view** | 440 of 911 SPE emails (48%) mention a term only in quoted or forwarded text | Search the full text for recall, but rank and attribute using the new text, and show the whole thread to reviewers. A quoted-only match shows what the sender received, not what they wrote. |
| **How big the SPE pool is** | 911 unique emails (0.37% of the corpus); 472 distinct subjects | The name-based pool is smaller than the proposal's 1,000-per-channel design. Treat it as one candidate source and keep the semantic and lexical channels for emails that never name a vehicle. |
| **Review pool** | Sender inside Enron, term in the sender's own text, not a mass mailing: 394 emails (234 before the cutoff) | A workable size for human screening before any model-based reranking. |
| **Cutoff (2001-10-16T00:00:00 Houston)** | 529 before, 382 on or after (41.9% of the pool, against 31.2% of the whole corpus); 3 within 24 hours | The cutoff removes a large share of SPE discussion, including internal legal and finance email after the announcement. The group should decide whether to keep it for ranking and still read later internal emails as context. |
| **Duplicates** | 1.0% of rows are copies; 12 SPE emails have a copy in another mailbox | Rank unique emails only; use `canonical_email_id` to list every mailbox that held a chosen email. |

## Who is in the data (section 5)
- **9 of the 15 people named in the Powers report or the proposal examples have no mailbox here**: Andrew Fastow, Richard Causey, Ben Glisan, Michael Kopper, Jeffrey McMahon, Jordan Mintz, Sherron Watkins, Ryan Siurek, Steve Pruett. They appear only through other people's mail, mostly as recipients. 6 do: Jeffrey Skilling, Kenneth Lay, Richard Buy, James Derrick, Vince Kaminski, John Lavorato.
- For people without a mailbox, the corpus can show what they were sent, not what they wrote or did with it. Evidence of their awareness will come from replies and from emails written by others. The definition's tiers (all three elements in one thread, versus exposure and awareness with a presentation concern) should reflect this.
- **Who handles SPE emails most:** sara.shackleton (210), mary.cook (99), mark.taylor (41), carol.clair (41), louise.kitchen (39), lisa.mellencamp (37), stuart.zisman (36), gordon.mckillop (35). 14 of the top 25 addresses have no mailbox in the data, so their view is partial and some are legal or finance staff reviewing deals. Handling many emails is not evidence of wrongdoing.
- **Mailbox holders who are not on the email:** 18.8% of SPE emails sit in a mailbox whose owner is neither sender nor addressee (20.5% under the stricter address map). Most frequent: martin-t (37), kitchen-l (35), whalley-g (15), bass-e (9), whalley-l (6). Possible causes include distribution lists, BCC, address aliases and forwarding; each case needs a look before it counts as evidence of receipt.
- **Seed emails:** 2 of 2 found, 2 in the SPE pool. They shaped the term list, so this checks the pipeline, not its accuracy.

## Limits and open items
- Precision labels come from one reviewer, ten emails per stratum, about 200 characters of context. Ask the group to spot-check, especially for the context-required terms.
- Marlin, Trutta and Project Velocity are not in the Powers report (the proposal is their source). Timberwolf, Bobcat and Southampton are candidate additions; they need page citations before the paper uses them.
- Which addresses belong to which mailbox is inferred. Sent-folder evidence is strong; name matches are weaker. Address aliases are not merged, except two for Kaminski. Custodian labels with typos (for example `lavorado-j` next to `lavorato-j`) are still separate.
- Counts measure how often words appear, not wrongdoing. The network measures who handled emails, not who was responsible.
- Next steps from the original plan: monthly rates with an event timeline, and contrastive term analysis to find vocabulary that does not name a vehicle.

## Structural baseline (Section 4)

Search set for TF-IDF and semantic search: unique emails (B1) from an Enron sender (B2), to at most 20 addresses and not from a broadcast account (B3), with non-empty `body_retrieval` (B4), dated in [1999-01-01, 2002-01-01) Houston time (B5). No topic terms are used.

| after rule | emails left | unique emails failing this rule alone |
|---|---|---|
| all records | 251734 |  |
| B1 unique | 249243 |  |
| B2 internal sender | 195781 | 53462 |
| B3 not a broadcast | 185142 | 13474 |
| B4 has retrieval text | 168411 | 18657 |
| B5 dated, in window | 152885 | 27667 |

- Primary baseline: **152,885** emails. Fallback set (passes all rules but B4; search on `body_clean`): **15,346**.
- Recall check: **623 of 911** SPE-subset emails are in the primary baseline. Where the others go:

| status | SPE emails |
|---|---|
| in primary baseline | 623 |
| fallback (no retrieval text) | 121 |
| B2 internal sender | 67 |
| B5 dated, in window | 58 |
| B3 not a broadcast | 37 |
| B4 has retrieval text | 5 |

- Excluded SPE emails are sampled in `baseline_excluded_spe_sample.csv` for the group to read. If they matter (e.g. Andersen or Vinson & Elkins writing in), relax B2 or B3, or search them separately.
