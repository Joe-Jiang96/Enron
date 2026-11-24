# Enron
Enron Project Group 6

## Results and findings

| Analysis | Main outputs |
|---|---|
| Embedding exploration | [Findings](analysis_outputs/embeddings/findings.txt), [PCA and t-SNE](analysis_outputs/embeddings/pca_tsne_clusters.png), [t-SNE boundaries](analysis_outputs/embeddings/tsne_cluster_boundaries.png), [interactive map](analysis_outputs/embeddings/tsne_explorer.html) |
| Semantic similarity | [Score distributions](analysis_outputs/semantic_search/cosine_similarity_kde.png), [all email scores](analysis_outputs/semantic_search/all_email_scores.parquet), [query descriptions](semantic_queries.json) |
| Initial top-20 review | [Readable review](analysis_outputs/semantic_search/review/top20_review.html), [review table](analysis_outputs/semantic_search/review/top20_review.csv), [review notes](analysis_outputs/semantic_search/review/review_notes.txt) |
| Mahonia reconstruction | [Report](analysis_outputs/semantic_search/mahonia/mahonia_reconstruction.html), [timeline](analysis_outputs/semantic_search/mahonia/thread_timeline.csv), [evidence matrix](analysis_outputs/semantic_search/mahonia/evidence_matrix.csv) |
| Selecting Top K | [Marginal-yield chart](analysis_outputs/semantic_search/marginal_yield/marginal_yield_vs_top_k.png), [interactive chart](analysis_outputs/semantic_search/marginal_yield/marginal_yield_explorer.html), [method and findings](analysis_outputs/semantic_search/marginal_yield/method_and_findings.txt), [review audit](analysis_outputs/semantic_search/marginal_yield/review_audit.html) |
| Hybrid ranking and email precision | [Precision/elbow chart](analysis_outputs/hybrid_precision/precision_elbow.png), [review report](analysis_outputs/hybrid_precision/precision_review.html), [combined ranking](analysis_outputs/hybrid_precision/combined_deduplicated.csv), [method and findings](analysis_outputs/hybrid_precision/method_and_findings.txt) |
| Definition-focused ranking revision | [Comparison chart](analysis_outputs/hybrid_precision_revision/revised_precision.png), [review report](analysis_outputs/hybrid_precision_revision/revision_report.html), [revised ranking](analysis_outputs/hybrid_precision_revision/revised_ranking.csv), [method and findings](analysis_outputs/hybrid_precision_revision/method_and_findings.txt) |

Download HTML reports and open them in a browser; GitHub displays their source.
The conclusions are exploratory and the review labels are initial analyst
judgments. Similarity and lead-discovery scores do not establish wrongdoing.

## Repository and local data

The repository includes scripts, notebooks, query definitions, compact datasets,
final analysis reports/charts, review judgments and case notes. The following
large files remain local and are excluded by `.gitignore`:

- `data/emails_analysis.parquet`: cleaned email corpus, required to rerun text
  joins and case discovery. Generate it with the cleaning notebook or copy a
  matching cleaned run into `data/`.
- `embedding/primary_email_embeddings.parquet`: completed paid embeddings,
  required to rerun vector analysis and scoring.
- `analysis_outputs/embeddings/email_coordinates_and_clusters.parquet`: the
  complete derived coordinate/group table. Sampled t-SNE coordinates are kept
  in the repository for redrawing the sample charts.
- `analysis_outputs/semantic_search/all_email_scores.csv`: convenient local
  CSV export; the smaller Parquet version is included in the repository.

Embedding checkpoints can be removed after the completed embedding file is
validated. Other ignored duplicate tables and SVG exports are reproducible;
PNG charts and readable reports are retained. The `analysis_outputs/` folder
itself is intentionally not ignored. Preserve its authored assessments,
`case_notes.json`, lead registry and sampling-plan files with the analysis.

## Primary email embeddings

Install dependencies and validate the input data without making API calls:

```powershell
pip install -r requirements-embeddings.txt
python embed_primary_emails.py --dry-run
```

Set `OPENAI_API_KEY` in the environment used to launch Python, then run:

```powershell
python embed_primary_emails.py
```

The script selects exactly 152,885 `primary` IDs from
`data/structural_baseline.parquet` and joins them to `data/emails_analysis.parquet`.
It embeds `body_retrieval` with `text-embedding-3-small`, using 256 dimensions.
If the source is an older table without `body_retrieval`, it reconstructs the
field from `body_new_candidate` using the cleaning notebook's rules and reports
this explicitly. Missing IDs, duplicate IDs or empty retrieval text stop the run.

Each email is truncated to its first 8,191 `cl100k_base` tokens when necessary.
Requests contain up to 128 emails and 100,000 tokens. The SDK retries transient
errors. Rerun the same command after an interruption to reuse completed batches
in `embedding/.checkpoints/`. An interrupted request that was not saved may be
billed again when retried. Run only one instance per output folder.

The final file is `embedding/primary_email_embeddings.parquet`, with exactly three
columns: `email_id` (string), `embeddings` (a list of 256 float32 values), and
`body_retrieval` (the full cleaned text before token truncation). Existing
two-column checkpoints are reused; an existing two-column completed output is
enriched with text without new API calls.
Checkpoint metadata prevents reuse when inputs or settings change. API calls
incur usage charges; `--dry-run` makes no API calls. The script does not load
`.env` files automatically. Keep keys out of source code and version control.

## Explore the embedding space

```powershell
pip install -r requirements-embedding-analysis.txt
python analyze_embeddings.py
```

This performs PCA on all 152,885 vectors, clusters normalized vectors in their
original 256 dimensions, and plots t-SNE for a seeded random sample of 12,000
emails after PCA reduction to 50 dimensions. It compares 6, 10, 14 and 18 groups
using sampled cosine silhouette scores; groups are exploratory, not validated
topic labels. No API calls are made.

Outputs in `analysis_outputs/embeddings/` include PNG charts, a standalone
`tsne_explorer.html` with email subjects/text on hover, group profiles,
representative emails, per-email coordinates and group assignments, and JSON
diagnostics. The HTML embeds excerpts from the emails, so treat it as corpus
data when sharing. The saved text is untruncated, whereas long embedding inputs
were truncated. t-SNE inter-island distances and apparent group sizes should
not be interpreted as original-space distances or population proportions.

## Three-pattern semantic retrieval

The exact WHO, HOW MUCH and WHEN descriptions are stored in `semantic_queries.json`.
Run with the same Python environment as embedding analysis:

```powershell
python semantic_search_patterns.py --dry-run
python semantic_search_patterns.py
```

The first search sends the three descriptions in one standard OpenAI request,
using `text-embedding-3-small` at 256 dimensions. Query vectors are cached for
later runs. Existing email embeddings are reused directly and L2-normalized
before cosine scoring. There is one score per email per query. No chunk
embeddings, t-SNE coordinates, or cluster labels are used for retrieval.

Outputs under `analysis_outputs/semantic_search/`:

- `all_email_scores.parquet` and `all_email_scores.csv`: all 152,885 email IDs,
  with `score_who`, `score_how_much`, and `score_when` (exactly four columns).
- `query_embeddings.parquet` and `search_summary.json`: cached queries,
  configuration, provenance and limitations.

The existing primary population spans 1999-2001 Houston dates; the search adds
no date cutoff. Every email receives three scores; no candidate selection or
context reconstruction is performed in this step.
Similarity scores are not probabilities or findings of wrongdoing. Review
original messages and linked context for risk, knowledge, contradictory action,
and legitimate alternative explanations after retrieval.

## Review the strongest semantic matches

```powershell
python review_semantic_matches.py
```

This ranks each pattern independently and selects its top 20 emails, then
merges selected IDs while preserving all three original scores and global
ranks. Outputs in `analysis_outputs/semantic_search/review/` include a review
CSV/Parquet, a 60-slot per-pattern CSV, an HTML reading view, and provenance.
The table includes metadata, retrieval text, full cleaned context, and blank
pattern-specific relevance/rationale fields. `--top-n` changes the review size;
`--labels path/to/labels.json` adds existing assessments.

Initial Codex screening of the current run is in `top20_initial_labels.json` in
that review folder. To include those assessments in the outputs, run:

```powershell
python review_semantic_matches.py --labels analysis_outputs/semantic_search/review/top20_initial_labels.json
```

These labels mean relevant lead, unclear, or irrelevant to a retrieval pattern;
they are not human-validated findings or probabilities of wrongdoing. Retain
source email IDs and inspect original evidence before attributing statements.

## Mahonia correspondence reconstruction

```powershell
python reconstruct_mahonia.py
python render_mahonia_report.py
```

Discovery scans the full cleaned corpus for Mahonia, related custody/
rehypothecation subjects, and near-seed Chase/collateral/participant context.
It retains copies and mailbox locations, and identifies whitespace-normalized
own-text quotations while excluding recognized signature-only matches.
Discovery matches and reproduced quotations do not authenticate formal threads.

Outputs are under `analysis_outputs/semantic_search/mahonia/`:
`discovery_emails.parquet`, `discovery_index.csv`, `quoted_text_links.csv`,
`case_messages.parquet`, `thread_timeline.csv`, `evidence_matrix.csv`, date
cautions, and the readable `mahonia_reconstruction.html`/`.txt` report.
The report renderer uses the separately authored `case_notes.json` in that
folder; preserve it with the case artifacts. It contains the inspected working
sequence, separate transaction context, judgments and qualifications rather
than automatically inferred wrongdoing labels. Source hashes are recorded in
`case_manifest.json`. Raw originals, attachments and reply-reference headers
are absent from this checkout, so the report explicitly preserves those gaps.

## Marginal-yield pilot for selecting retrieval depth

The final round-robin analysis uses the saved ranking, review sample, judgment
ledger and reassessment records in `analysis_outputs/round_robin_extended/`.

```powershell
python extend_round_robin.py prepare
python extend_round_robin.py evaluate
python -m unittest test_extend_round_robin.py
```

The semantic max-score and TF-IDF top 1,000 alternate before deduplication
by email ID, leaving 1,920 unique emails from 2,000 slots. Combined ranks
1–1900 are sampled in 100-email bands with 20 draws per first-occurrence
source. The first ten bands use seed 6454, and the later bands use seed 6455.
Both sources are fully reviewed in the final partial band, ranks 1901–1920.

`prepare` verifies the fixed draws against the saved sampling manifest.
`evaluate` applies the judgment ledger and review records to regenerate the
780 verdicts in `sample_verdicts_compact.csv` and the band estimates in
`precision_by_band_and_source.csv`. The chart displays ranks through 1800.
The geometric elbow uses the full 1,920-email ranking.

```powershell
python marginal_yield.py prepare
python marginal_yield.py evaluate
python plot_marginal_yield.py
python render_marginal_review.py
```

The fixed pilot samples 40 emails per pattern from rank bands 1-100, 101-200,
201-300 and 301-500, with seed 6450. Outputs and separately authored assessments
are in `analysis_outputs/semantic_search/marginal_yield/`; preserve
`assessments.json`, `screening_notes.json` and `lead_registry.json` with the case
artifacts. Evaluation requires those assessments and never treats missing
reviews as zero yield. The chart shows per-pattern discovery yield, unique
combined yield and full candidate-pool size. The HTML explorer links to the
sample audit. `method_and_findings.txt` documents scope, case grouping and
uncertainty. The current pilot does not demonstrate a shared optimal stopping
K through 500; HOW MUCH has a late yield increase. No API calls are made.
