# Enron cleaning data dictionary

## Source and execution

Run [Enron_Data_Cleaning.ipynb](./Enron_Data_Cleaning.ipynb) top to bottom from the project directory. Input: `inta6450-emails\dataset\*.json`, extracted from the [course archive](https://s3.amazonaws.com/inta.gatech/inta6450-emails.zip). The initial directory inspection found **251,734 JSON files**; the notebook reports the count on each run. The instructor notebook contains an older saved server count of 237,964, so local/server equivalence is **not established**. Never assume local filename IDs equal Elasticsearch IDs.

Requires Python plus `numpy`, `pyarrow` and `tzdata` (`pip install numpy pyarrow tzdata`). `tzdata` supplies the timezone database that Windows lacks; without it, Houston-time conversion fails and timezone-aware Parquet dates cannot be read into Python.

**Two stages:**

1. **Stage 0** copies every source file's exact bytes and SHA-256 hash into `enron_raw.parquet` (project folder). It runs once and is reused unless the source file list changes or `REBUILD_RAW = True`. The reuse check compares file names only, not contents.
2. **Stage 1** cleans from `enron_raw.parquet` and writes a new timestamped folder under `enron_cleaning_outputs` with three files: `emails_analysis.parquet`, `quality_report.json` and `cleaning_review_sample.jsonl`. Originals and previous runs are not overwritten.

`MAX_FILES=None` cleans the full corpus; a positive value creates a pilot. **No date cutoff is applied during cleaning.** It is applied downstream with `cutoff_utc()`, so the group can change it without rerunning.

## Files and their roles

| File | Role |
|---|---|
| `enron_raw.parquet` | **Evidence archive.** Exact original files. Quote and verify final emails from here with `load_raw(email_ids)`. |
| `emails_analysis.parquet` | **Main table** (19 columns, one row per email, duplicates included). Input for EDA, TF-IDF, Elasticsearch, near-duplicate clustering and context review. |
| `quality_report.json` | Configuration, package versions, corpus/pilot status, source fingerprint, funnel counts, quote-marker counts, flag counts, parse errors and outstanding reviews. Nonempty `parse_errors` makes the run fail after the report is written. Completion does not certify cleaning accuracy. |
| `cleaning_review_sample.jsonl` | Seeded random sample of 40 emails with original text (`body_raw`), new text, marker type, flags and blank review fields. Tests the cleaning, not wrongdoing. |

## Settings that affect results

| Setting | Default | Meaning |
|---|---|---|
| `DUP_MAX_SHIFT_HOURS` | 26 | Largest date shift (in whole half-hours) still treated as the same email stored in another mailbox. |
| `SHINGLE_SIZE`, `MINHASH_PERMUTATIONS` | 5, 64 | MinHash uses 5-word shingles and a 64-value signature. |
| `RANDOM_SEED` | 6450 | Fixes the review sample and MinHash functions. |

## Evidence archive: enron_raw.parquet

| Field | Type | Definition |
|---|---|---|
| `email_id` | string | Filename stem; joins to the main table. |
| `source_file` | string | Project-relative path of the original JSON file. |
| `source_sha256` | string | SHA-256 of the original bytes. |
| `raw_bytes` | binary | Exact original file content. `json.loads(raw_bytes)` gives the original `text`, `headers` (including `date`, `x-origin`, `x-folder`, display names), structured `date`, `from`, `to`, `cc`, `bcc`, `subject` and `messageId`. |

## Main table: emails_analysis.parquet

Zstandard-compressed Parquet. Load selected columns, e.g. `pandas.read_parquet(path, columns=['email_id', 'date_utc', 'body_new_candidate'])`.

| Column | Type | Null? | Definition / use |
|---|---|---|---|
| `email_id` | string | never | Local filename stem; links to `enron_raw.parquet`. Not a server ID. |
| `message_id` | string | possible | Original Message-ID (structured field, header fallback). |
| `canonical_email_id` | string | never | The first copy (in file order) of this email. Equals `email_id` unless this row is a copy. Group by it to find every mailbox that held the email. |
| `is_duplicate_copy` | boolean | never | `canonical_email_id != email_id`. Filter `== False` to rank each email once. |
| `custodian` | string | possible | Mailbox the copy was collected from (original `x-origin`, e.g. `Smith-M`). The mailbox owner, not necessarily the sender or an addressee. |
| `folder` | string | possible | Folder within that mailbox (original `x-folder`, e.g. `...\Sent Items`, `...\Deleted Items`). |
| `date_utc` | timestamp (UTC) | possible | Structured date preferred, header date fallback. Null if neither parses; no timezone is guessed. Convert to Houston time with `.dt.tz_convert('America/Chicago')` when needed. |
| `sender` | list of strings | never | Lowercased, sorted, unique From addresses. Empty list if none. |
| `recipients` | list of strings | never | Same rule for To. An empty list means no usable addresses, not proof of no recipient. |
| `cc`, `bcc` | list of strings | never | Same rule for CC and BCC. |
| `subject` | string | never | Original subject (header fallback); `''` if missing or invalid. |
| `subject_norm` | string | never | Subject with whitespace normalized, leading RE/FW/FWD prefixes removed, and case-folded. A thread-grouping hint only. |
| `body_clean` | string | never | Full email text with Unicode NFC, one line-ending style, and repeated spaces/tabs collapsed. Case, punctuation, numbers, signatures and quoted text are kept. Use for recall-oriented keyword search, MinHash and reading context. |
| `body_new_candidate` | string | never | `body_clean` cut at the first recognized quote marker, with `>`-prefixed lines removed. Use for TF-IDF/BM25/embedding weighting and for attributing statements to the sender. A heuristic: quotes pasted without headers, signatures and disclaimers can remain. `''` for forwards without a comment. |
| `body_retrieval` | string | never | `body_new_candidate` with HTML tags/entities converted to text, base64 runs of 100+ characters removed, and a trailing confidentiality notice (within the last 1,500 characters) or a `-- ` signature block (up to 10 lines) cut. Free-form signatures are kept. **Use for TF-IDF and embeddings** (structural baseline rule B4). `''` when the new text is empty. |
| `quote_marker_type` | string | possible | Rule that set the cut: `outlook_original_message`, `forwarded_by`, `forwarded_message`, `on_wrote`, `outlook_header_block`, `notes_header_block`. Null if no marker was found, in which case `body_new_candidate` equals `body_clean` minus any `>` lines. |
| `minhash_signature` | list of 64 uint64 | possible | MinHash of lowercase letter/digit word 5-shingles of `body_clean`, for downstream LSH near-duplicate clustering. Null for fewer than 5 words. |
| `quality_flags` | list of strings | never | Inspection flags below. Flags never exclude records. |

Quoted text is not stored separately; it is the part of `body_clean` after the cut, and can be regenerated with `split_quoted`.

### Downstream handling of empty lists

Address columns hold empty lists, not nulls, when no addresses exist. Check with `len(x) == 0` rather than `isnull()`. `DataFrame.explode()` turns an empty list into a single NaN row, so drop those before building sender→recipient edges for a network graph. Text columns are always strings (possibly `''`), so tokenizers never receive `None`. pyarrow does not enforce the "never null" columns on its own; the notebook checks every row before writing.

## Quality flags

| Flag | Meaning |
|---|---|
| `structured_date_missing` / `_invalid` / `_invalid_type` / `_timezone_missing` | Structured date problem. |
| `header_date_missing` / `_invalid` / `_invalid_type` / `_timezone_missing` | Header date problem. |
| `date_from_header` | Structured date unusable; `date_utc` comes from the header. |
| `date_unresolved` | Neither date parses; `date_utc` is null. |
| `date_disagreement` | Both dates parse but differ. The structured date is used. |
| `date_outside_inspection_range` | Year outside 1990–2005. An inspection heuristic, not a claim of corruption. |
| `duplicate_timezone_shift` | Copy whose date differs from the canonical copy by whole half-hours. Its true time is uncertain. |
| `missing_sender` / `missing_recipients` | No usable From address, or no To/CC/BCC addresses. |
| `{from,to,cc,bcc}_invalid_type` / `_invalid_entry` / `_header_invalid_type` | Source address field has an unexpected shape. |
| `subject_invalid_type` | Subject was not a string. |
| `empty_body` | Cleaned body is empty. |
| `short_new_text` | New text under 20 characters, including legitimate brief replies. |
| `quote_split_at_start` | Marker at the very top, so the new text is empty (typical of forwards without comment). Check that the split is right. |
| `inline_quotes_removed` | One or more `>`-prefixed lines were removed from the new text. |
| `attachment_mentioned` | Attachment wording or a file extension appears; attachment files are not in the corpus. |
| `too_short_for_minhash` | Fewer than 5 words; no MinHash signature. |
| `unicode_replacement_character` | Original text contains U+FFFD; no speculative encoding repair is attempted. |

## Downstream helpers (notebook Section 7)

- `load_raw(email_ids)` returns `{email_id: original JSON}` from the evidence archive.
- `cutoff_utc('2001-10-16T00:00:00')` converts Houston local time to UTC with daylight saving applied (midnight Oct 16, 2001 CDT = 05:00 UTC; using midnight UTC would wrongly treat 7 pm–midnight Oct 15 Houston emails as post-cutoff). Filter with `df[df.date_utc < cutoff]`. Review null-date rows and rows within about 24 hours of the cutoff manually, since mailbox copies can carry shifted dates.

The report's `source_manifest_sha256` hashes sorted `source_file + TAB + source_sha256 + NEWLINE` entries in UTF-8; paths use this Windows project's relative path representation.

## Boundaries and acceptance criteria

1. Full run: attempted = valid + quarantined; valid = representatives + duplicate copies; Parquet rows = valid records. Resolve parse errors and rerun before final analysis. A run with no report was interrupted and must not be used.
2. Manually inspect the random sample **and** about 10 examples per `quote_marker_type`, plus `quote_split_at_start` and `duplicate_timezone_shift` rows, comparing with `load_raw`. Record extraction accuracy.
3. Near-duplicate clustering is **not** done here. Band `minhash_signature` for LSH (e.g. 16 bands × 4 rows ≈ 0.5 Jaccard threshold; 8 × 8 ≈ 0.77) and validate the threshold on inspected pairs. Signatures cover the full body, so a short reply that quotes an entire earlier message will look similar to it; combine with sender and date when grouping threads.
4. Confirm named seed emails and custodian coverage using originals. Do not infer missing mailboxes from sender counts alone.
5. Agree on the cutoff (Houston time) and on the text view. Full text preserves forwarded evidence; new text reduces repeated quoted material in TF-IDF but can miss or misattribute content. No stopword removal, stemming or alias merging is applied here.
6. Rank only `is_duplicate_copy == False` rows. For each final email, collect all rows sharing its `canonical_email_id` to show every mailbox that held it. Hosted Elasticsearch searches its own original index, not this cleaned text; link its results explicitly and do not modify the shared server.
7. Preserve the funnel from corpus to the final 5–8 emails, including contradictory or contextual content.

These outputs support the paper's source citation, methodology, email traceability and limitations. They do not establish wrongdoing or substitute for email-by-email analysis.
