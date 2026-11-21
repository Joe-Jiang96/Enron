# Enron SPE Investigation (INTA 6450, Group 6)

Finds Enron emails from before October 16, 2001 that connect **risk Enron kept** with **misleading financial treatment** of special-purpose entity (SPE) deals such as Raptor, LJM and JEDI. The method combines lexical and semantic retrieval, rank fusion, LLM reranking (JEV), and human verification. The aim is defensible investigative leads, not findings of guilt. See the [proposal](docs/proposal_tjiang307_annotated.pdf) for the full definition and method.

## Repository layout

```
notebooks/   Analysis notebooks, numbered in run order
data/        Small committed data files (email ID lists, SPE subset, hand labels)
docs/        Proposal, data dictionary, EDA decision memo
```

| File | What it is |
|---|---|
| [notebooks/01_data_cleaning.ipynb](notebooks/01_data_cleaning.ipynb) | Turns the 251,734 raw JSON emails into one cleaned table (`emails_analysis.parquet`) |
| [notebooks/02_eda.ipynb](notebooks/02_eda.ipynb) | Corpus profile, SPE term precision, SPE subset, search baseline, network; writes the decision memo |
| [data/structural_baseline.parquet](data/structural_baseline.parquet) | IDs of the emails to search: 152,885 `primary` + 15,346 `fallback` |
| [data/spe_subset_emails.csv](data/spe_subset_emails.csv) | 911 emails that name an SPE; used to check retrieval recall |
| [docs/data_dictionary.md](docs/data_dictionary.md) | Every column and quality flag in the cleaned table |
| [docs/eda_decision_memo.md](docs/eda_decision_memo.md) | EDA findings and recommendations for retrieval |

## Setup

1. `pip install -r requirements.txt` (Python 3.9 or newer)
2. Download and extract the [course email archive](https://s3.amazonaws.com/inta.gatech/inta6450-emails.zip).
3. In the first code cell of each notebook, set:
   - `DATA_DIR`: folder holding the extracted `*.json` files
   - `OUTPUT_ROOT`: folder for large generated files, **outside this repo**
4. Run `01_data_cleaning.ipynb`, then `02_eda.ipynb`. Check that `quality_report.json` reports `source_manifest_sha256 = adcf847d…c11b`; this proves your source files match everyone else's.

Large generated files (`enron_raw.parquet`, `enron_cleaning_outputs/`, `eda_outputs/`) go to `OUTPUT_ROOT` and are never committed.

## Pipeline status

| Step | Status |
|---|---|
| 1. Cleaning | Done |
| 2. EDA and search baseline | Done (monthly timeline and contrastive terms not yet) |
| 3a. Lexical retrieval: dictionary, 200-word window rules, TF-IDF | In progress |
| 3b. Semantic retrieval: embeddings | Not started |
| 4. Rank fusion (RRF) and context packets | Not started |
| 5. JEV reranking and human verification | Not started |

## Conventions

- Join everything on `email_id` (the JSON filename). Rank only rows with `is_duplicate_copy == False`.
- Times are Houston local time. Use `cutoff_utc('2001-10-16T00:00:00')` for the cutoff, not midnight UTC.
- Development happens on the `dev-jt` branch.
