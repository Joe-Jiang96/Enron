"""Offline integration checks; no API key or network needed."""

import argparse
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import embed_primary_emails as script


class EmbeddingWorkflowTests(unittest.TestCase):
    def test_resume_truncation_and_id_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, emails = root / "baseline.parquet", root / "emails.parquet"
            pq.write_table(pa.table({"email_id": ["a", "b", "c", "excluded"],
                                    "set": ["primary", "primary", "primary", "fallback"]}), baseline)
            # Source order differs from baseline order; c needs truncation.
            pq.write_table(pa.table({"email_id": ["c", "excluded", "b", "a"],
                                    "body_retrieval": ["long", "skip", "second", "first"]}), emails)
            args = argparse.Namespace(baseline=baseline, emails=emails, expected_count=3,
                                      batch_size=2, output_dir=root / "output", dry_run=False)
            tokenizer = SimpleNamespace(encode=lambda text, **kw: {
                "first": [1], "second": [2], "long": [3] * 9000,
            }[text])
            requests = []

            def create(**kwargs):
                requests.append(kwargs)
                if len(requests) == 2:
                    raise RuntimeError("Simulated interruption")
                return SimpleNamespace(data=[
                    SimpleNamespace(index=i, embedding=[float(tokens[0])] * 256)
                    for i, tokens in reversed(list(enumerate(kwargs["input"])))
                ])

            client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
            with patch.object(script.tiktoken, "get_encoding", return_value=tokenizer), \
                    patch.object(script, "OpenAI", return_value=client), \
                    patch.dict(script.os.environ, {"OPENAI_API_KEY": "fake-test-key"}):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    script.run(args)
                self.assertFalse((args.output_dir / "primary_email_embeddings.parquet").exists())
                script.run(args)
                self.assertEqual(len(requests), 3)  # First saved batch was not requested again.
                self.assertEqual(len(requests[-1]["input"][0]), script.MAX_TOKENS)
                self.assertEqual(requests[-1]["dimensions"], 256)
                result = pq.read_table(args.output_dir / "primary_email_embeddings.parquet")
                self.assertEqual(result.column_names, ["email_id", "embeddings", "body_retrieval"])
                self.assertEqual(result["body_retrieval"].to_pylist(), ["first", "second", "long"])
                self.assertEqual(result["email_id"].to_pylist(), ["a", "b", "c"])
                self.assertEqual([v[0] for v in result["embeddings"].to_pylist()], [1, 2, 3])
                script.run(args)
                self.assertEqual(len(requests), 3)  # Completed run makes no new requests.
                # Older completed two-column outputs gain text without rebilling.
                destination = args.output_dir / "primary_email_embeddings.parquet"
                pq.write_table(result.drop(["body_retrieval"]), destination)
                script.run(args)
                self.assertEqual(len(requests), 3)
                self.assertEqual(pq.read_table(destination)["body_retrieval"].to_pylist(),
                                 ["first", "second", "long"])

    def test_missing_or_empty_primary_text_fails_before_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, emails = root / "baseline.parquet", root / "emails.parquet"
            pq.write_table(pa.table({"email_id": ["a"], "set": ["primary"]}), baseline)
            pq.write_table(pa.table({"email_id": ["other"], "body_retrieval": ["text"]}), emails)
            with self.assertRaisesRegex(ValueError, "missing from email table"):
                script.load_inputs(baseline, emails, 1)
            pq.write_table(pa.table({"email_id": ["a"], "body_retrieval": ["  "]}), emails)
            with self.assertRaisesRegex(ValueError, "empty retrieval text"):
                script.load_inputs(baseline, emails, 1)


if __name__ == "__main__":
    unittest.main()
