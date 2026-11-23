"""Embed primary Enron emails with text-embedding-3-small (256 dimensions).

Install: pip install -r requirements-embeddings.txt
Preview: python embed_primary_emails.py --dry-run
Run:     python embed_primary_emails.py
The API key is read from OPENAI_API_KEY. Completed batches resume automatically.
"""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
MODEL = "text-embedding-3-small"
DIMENSIONS = 256
MAX_TOKENS = 8191  # Stay below the model's 8,192-token input limit.
REQUEST_TOKENS = 100_000  # Also bound total tokens per API request.
SCHEMA = pa.schema([
    pa.field("email_id", pa.string(), nullable=False),
    pa.field("embeddings", pa.list_(pa.float32(), DIMENSIONS), nullable=False),
])
OUTPUT_SCHEMA = SCHEMA.append(pa.field("body_retrieval", pa.string(), nullable=False))

# Same retrieval-text rules as Enron_Data_Cleaning.ipynb. Older analysis files
# lack body_retrieval, so reconstruct it without modifying the source Parquet.
HTML_TAG = re.compile(r"(?i)</?(?:html|head|body|div|p|br|table|tbody|td|tr|th|font|span|img|b|i|u|a|style|meta)\b[^>]*>")
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")
DISCLAIMER = re.compile(
    r"(?i)(?:confidentiality notice|this e-?mail is the property of enron"
    r"|this (?:e-?mail|message)[^\n]{0,80}(?:confidential|privileged|intended (?:only|solely)))"
)
SIG_DELIMITER = re.compile(r"(?m)^-- ?$")


def make_body_retrieval(new_text):
    text = html.unescape(HTML_TAG.sub(" ", new_text))
    text = BASE64_RUN.sub(" ", text)
    match = DISCLAIMER.search(text)
    if match and match.start() > 0 and len(text) - match.start() <= 1500:
        text = text[:match.start()]
    sig = list(SIG_DELIMITER.finditer(text))
    if sig and sig[-1].start() > 0 and text[sig[-1].start():].count("\n") <= 10:
        text = text[:sig[-1].start()]
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")).strip()


def load_inputs(baseline_path, emails_path, expected_count):
    baseline = pq.read_table(baseline_path, columns=["email_id", "set"]).to_pydict()
    ids = [eid for eid, label in zip(baseline["email_id"], baseline["set"]) if label == "primary"]
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ValueError(f"Expected {expected_count:,} unique primary IDs; found {len(ids):,} rows.")
    if any(not isinstance(eid, str) or not eid for eid in ids):
        raise ValueError("Primary email IDs must be nonempty strings.")
    source = pq.ParquetFile(emails_path)
    derive = "body_retrieval" not in source.schema_arrow.names
    column = "body_new_candidate" if derive else "body_retrieval"
    if column not in source.schema_arrow.names:
        raise ValueError("Email table needs body_retrieval or body_new_candidate.")
    if derive:
        print("body_retrieval is missing; reconstructing with the cleaning notebook's rules.", flush=True)
    wanted, texts = set(ids), {}
    for batch in source.iter_batches(columns=["email_id", column], batch_size=4096):
        for row in batch.to_pylist():
            eid = row["email_id"]
            if eid not in wanted:
                continue
            if eid in texts:
                raise ValueError(f"Duplicate source email_id: {eid}")
            text = row[column]
            if not isinstance(text, str):
                raise ValueError(f"Missing text for {eid}")
            texts[eid] = make_body_retrieval(text) if derive else text
            if not texts[eid].strip():
                raise ValueError(f"Primary email {eid} has empty retrieval text; check baseline/source versions.")
    missing = wanted - texts.keys()
    if missing:
        raise ValueError(f"{len(missing):,} primary IDs missing from email table; example: {next(iter(missing))}")
    return ids, texts


def prepare(ids, texts, batch_size):
    encoder = tiktoken.get_encoding("cl100k_base")
    batches, current, current_tokens = [], [], 0
    truncated = total_tokens = 0
    digest = hashlib.sha256()
    for eid in ids:
        # Treat strings resembling special tokens as ordinary email text.
        tokens = encoder.encode(texts[eid], disallowed_special=())
        truncated += len(tokens) > MAX_TOKENS
        tokens = tokens[:MAX_TOKENS]
        if not tokens:
            raise ValueError(f"No tokens for primary email {eid}")
        digest.update(json.dumps([eid, tokens], separators=(",", ":")).encode())
        digest.update(b"\n")
        if current and (len(current) >= batch_size or current_tokens + len(tokens) > REQUEST_TOKENS):
            batches.append(current)
            current, current_tokens = [], 0
        current.append((eid, tokens))
        current_tokens += len(tokens)
        total_tokens += len(tokens)
    if current:
        batches.append(current)
    config = {"model": MODEL, "dimensions": DIMENSIONS, "max_tokens": MAX_TOKENS,
              "batch_size": batch_size, "request_tokens": REQUEST_TOKENS, "input_sha256": digest.hexdigest()}
    print(f"Primary emails: {len(ids):,}; truncated: {truncated:,}; input tokens: {total_tokens:,}; batches: {len(batches):,}", flush=True)
    return batches, {b"enron_embedding_config": json.dumps(config, sort_keys=True).encode()}


def validate_table(table, expected_ids, metadata):
    valid_schema = any(table.schema.equals(schema, check_metadata=False)
                       for schema in (SCHEMA, OUTPUT_SCHEMA))
    if not valid_schema or table.schema.metadata != metadata:
        raise ValueError("Embedding schema/configuration mismatch; use a different output folder for a new run.")
    if table["email_id"].to_pylist() != expected_ids:
        raise ValueError("Embedding IDs do not match the current input/order.")
    vectors = np.asarray(table["embeddings"].to_pylist(), dtype=np.float32)
    if vectors.shape != (len(expected_ids), DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding dimensions or non-finite values.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Zero embedding encountered.")


def run(args):
    if not 1 <= args.batch_size <= 2048:
        raise ValueError("--batch-size must be between 1 and 2048.")
    ids, texts = load_inputs(args.baseline, args.emails, args.expected_count)
    batches, metadata = prepare(ids, texts, args.batch_size)
    if args.dry_run:
        print("Dry run complete. No API calls or output files created.")
        return
    destination = args.output_dir / "primary_email_embeddings.parquet"
    if destination.exists():
        table = pq.read_table(destination)
        validate_table(table, ids, metadata)
        expected_texts = [texts[eid] for eid in ids]
        if "body_retrieval" not in table.column_names:
            # Enrich an older completed output without requesting embeddings again.
            table = table.append_column(OUTPUT_SCHEMA.field("body_retrieval"),
                                        pa.array(expected_texts, type=pa.string()))
            temporary = destination.with_suffix(".tmp")
            pq.write_table(table, temporary, compression="zstd")
            temporary.replace(destination)
            print("Added body_retrieval to existing output without API calls.")
        elif table["body_retrieval"].to_pylist() != expected_texts:
            raise ValueError("Saved body_retrieval differs from current source; use a new output folder.")
        print(f"Already complete: {destination}")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in your environment before running.")
    parts = args.output_dir / ".checkpoints"
    parts.mkdir(parents=True, exist_ok=True)
    manifest = parts / "config.json"
    config_bytes = metadata[b"enron_embedding_config"]
    if manifest.exists() and manifest.read_bytes() != config_bytes:
        raise ValueError("Checkpoint configuration/input changed. Choose a new --output-dir.")
    manifest.write_bytes(config_bytes)
    # Check every existing part before spending money on any new API requests.
    for index, batch in enumerate(batches):
        part = parts / f"batch_{index:06d}.parquet"
        if part.exists():
            validate_table(pq.read_table(part), [eid for eid, _ in batch], metadata)
    client = OpenAI(max_retries=8, timeout=120.0)
    for index, batch in enumerate(batches):
        part = parts / f"batch_{index:06d}.parquet"
        if part.exists():
            continue
        response = client.embeddings.create(
            model=MODEL, input=[tokens for _, tokens in batch],
            encoding_format="float", dimensions=DIMENSIONS,
        )
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(len(batch))):
            raise ValueError("API returned missing or duplicate embedding indices.")
        table = pa.Table.from_arrays([
            pa.array([eid for eid, _ in batch], type=pa.string()),
            pa.array([item.embedding for item in data], type=SCHEMA.field("embeddings").type),
        ], schema=SCHEMA.with_metadata(metadata))
        validate_table(table, [eid for eid, _ in batch], metadata)
        temporary = part.with_suffix(".tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(part)
        print(f"Batch {index + 1:,}/{len(batches):,} saved ({len(batch)} emails).", flush=True)
    temporary = destination.with_suffix(".tmp")
    with pq.ParquetWriter(temporary, OUTPUT_SCHEMA.with_metadata(metadata), compression="zstd") as writer:
        for index, batch in enumerate(batches):
            table = pq.read_table(parts / f"batch_{index:06d}.parquet")
            validate_table(table, [eid for eid, _ in batch], metadata)
            # Save full retrieval text, even when the API input was truncated.
            if "body_retrieval" in table.column_names:
                table = table.drop(["body_retrieval"])
            table = table.append_column(OUTPUT_SCHEMA.field("body_retrieval"),
                                        pa.array([texts[eid] for eid, _ in batch], type=pa.string()))
            writer.write_table(table)
    if pq.ParquetFile(temporary).metadata.num_rows != len(ids):
        raise ValueError("Final row count mismatch.")
    temporary.replace(destination)
    print(f"Saved {len(ids):,} rows, 3 columns: {destination}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/structural_baseline.parquet")
    parser.add_argument("--emails", type=Path, default=ROOT / "data/emails_analysis.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "embedding")
    parser.add_argument("--expected-count", type=int, default=152885)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/count tokens without calling OpenAI.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
