"""Offline checks for cosine similarity scoring."""
import unittest
import numpy as np

from semantic_search_patterns import cosine_scores, unit_vectors


class SemanticScoringTests(unittest.TestCase):
    def test_cosine_scores_and_ties(self):
        emails = np.zeros((4, 256), dtype=np.float32)
        emails[:, :2] = [[2, 0], [1, 0], [0, 5], [-1, 0]]
        queries = np.zeros((3, 256), dtype=np.float32)
        queries[:, :2] = [[3, 0], [0, 2], [1, 1]]
        scores = cosine_scores(emails, queries)
        np.testing.assert_allclose(scores[:, 0], [1, 1, 0, -1], atol=1e-6)
        np.testing.assert_allclose(scores[:, 1], [0, 0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(scores[:, 2], np.array([1, 1, 1, -1]) / np.sqrt(2), atol=1e-6)

    def test_invalid_vectors_are_rejected(self):
        for values in (np.zeros((1, 256)), np.ones((1, 128)), np.full((1, 256), np.nan)):
            with self.assertRaises(ValueError):
                unit_vectors(values)


if __name__ == "__main__":
    unittest.main()
