"""Score existing email embeddings against three separately embedded descriptions.

python semantic_search_patterns.py --dry-run
python semantic_search_patterns.py
No email embeddings are regenerated. Only three descriptions are sent to OpenAI.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
MODEL = "text-embedding-3-small"
DIMENSIONS = 256


def unit_vectors(values):
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSIONS or not np.isfinite(vectors).all():
        raise ValueError("Expected finite 256-dimensional vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot score zero vectors")
    return vectors / norms


def cosine_scores(emails, queries):
    return np.clip(unit_vectors(emails) @ unit_vectors(queries).T, -1, 1)


def load_query_vectors(queries, out):
    fingerprint = hashlib.sha256(json.dumps(queries, sort_keys=True).encode()).hexdigest()
    config = {b"query_config": json.dumps({"model": MODEL, "dimensions": DIMENSIONS,
                                           "queries_sha256": fingerprint}, sort_keys=True).encode()}
    path = out / "query_embeddings.parquet"
    if path.exists():
        table = pq.read_table(path)
        if table.schema.metadata != config or table["query_id"].to_pylist() != [q["query_id"] for q in queries]:
            raise ValueError("Cached queries/settings changed; choose a new --output-dir.")
        print("Reusing cached query embeddings.", flush=True)
        return unit_vectors(table["embeddings"].to_pylist())
    print("Embedding the three query descriptions in one request...", flush=True)
    with OpenAI(max_retries=5, timeout=120) as client:
        result = client.embeddings.create(model=MODEL, dimensions=DIMENSIONS, encoding_format="float",
                                          input=[q["description"] for q in queries])
    items = sorted(result.data, key=lambda item: item.index)
    if [item.index for item in items] != list(range(len(queries))):
        raise ValueError("Unexpected query embedding response indices")
    vectors = unit_vectors([item.embedding for item in items])
    table = pa.table({"query_id": [q["query_id"] for q in queries],
                      "description": [q["description"] for q in queries],
                      "embeddings": pa.array(vectors.tolist(), type=pa.list_(pa.float32(), DIMENSIONS))})
    temporary = path.with_suffix(".tmp")
    pq.write_table(table.replace_schema_metadata(config), temporary, compression="zstd")
    temporary.replace(path)
    return vectors


def main(args):
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    if [q["query_id"] for q in queries] != ["who", "how_much", "when"]:
        raise ValueError("Expected query IDs who, how_much, when in that order")
    encoder = tiktoken.get_encoding("cl100k_base")
    counts = [len(encoder.encode(q["description"], disallowed_special=())) for q in queries]
    if any(count == 0 or count > 8191 for count in counts):
        raise ValueError("Query descriptions must have 1 to 8191 tokens")
    source = pq.ParquetFile(args.embeddings)
    config = json.loads(source.schema_arrow.metadata[b"enron_embedding_config"])
    if config["model"] != MODEL or config["dimensions"] != DIMENSIONS:
        raise ValueError("Query and email embedding model/dimensions must match")
    table = source.read(columns=["email_id", "embeddings"])
    ids = table["email_id"].to_pylist()
    if len(ids) != 152885 or len(set(ids)) != len(ids):
        raise ValueError("Expected 152,885 unique email IDs")
    emails = unit_vectors(np.asarray(table["embeddings"].combine_chunks().values).reshape(len(ids), -1))
    print(f"Validated {len(ids):,} email vectors. Query token counts: {counts}.", flush=True)
    if args.dry_run:
        print("Dry run complete; no API calls or output files created.")
        return
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    query_vectors = load_query_vectors(queries, out)
    scores = cosine_scores(emails, query_vectors)
    columns = {"email_id": ids}
    for j, query in enumerate(queries):
        columns[f"score_{query['query_id']}"] = scores[:, j]
    score_table = pa.table(columns)
    pq.write_table(score_table, out / "all_email_scores.parquet", compression="zstd")
    with (out / "all_email_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=score_table.column_names)
        writer.writeheader()
        writer.writerows(score_table.to_pylist())
    summary = {"emails_scored": len(ids), "model": MODEL, "dimensions": DIMENSIONS,
               "query_tokens": counts,
               "email_embedding_config": config, "queries": queries,
               "scoring": "Cosine similarity of L2-normalized query/email vectors; one truncated-text vector per email, no chunk maximum",
               "date_filter": "None beyond existing primary baseline (1999-2001 Houston dates)",
               "limitations": ["Scores are similarities, not wrongdoing probabilities.",
                               "Full saved retrieval text may exceed the 8191 tokens used for the vector.",
                               "Thread reconstruction and risk/knowledge/contradictory-action verification remain human review tasks."]}
    (out / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(ids):,} emails with three cosine scores each to {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=ROOT / "embedding/primary_email_embeddings.parquet")
    parser.add_argument("--queries", type=Path, default=ROOT / "semantic_queries.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis_outputs/semantic_search")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
