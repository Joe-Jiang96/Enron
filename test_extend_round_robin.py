"""Regression checks for preserved reviews and the partial final census."""
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from extend_round_robin import (
    ORIGINAL_SEED,
    OUT,
    SEED,
    apply_semantic_reassessment,
    apply_semantic_top200_review,
    statistics,
)


class ExtendedRoundRobinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=pd.read_parquet(OUT/'sample_relevancy_verdicts.parquet')
        cls.c=pd.read_parquet(OUT/'combined_deduplicated.parquet')
        cls.stats,cls.summary=statistics(cls.d)

    def test_original_samples_and_labels_preserved(self):
        original_ranks = self.c[self.c.combined_rank.le(1000)]
        rng = np.random.default_rng(ORIGINAL_SEED)
        expected_ids = []
        for _, group in original_ranks.groupby(['band_end', 'first_source'], sort=True):
            positions = np.sort(rng.choice(len(group), 20, replace=False))
            expected_ids.extend(group.iloc[positions].email_id)
        retained = self.d[self.d.combined_rank.le(1000)].reset_index(drop=True)
        self.assertEqual(retained.email_id.tolist(), expected_ids)

        ledger_path = OUT / 'assessments.tsv'
        ledger = pd.read_csv(ledger_path, sep='\t', keep_default_na=False)
        ledger_lines = ledger_path.read_bytes().splitlines(keepends=True)
        original_labels = ledger_lines[0] + b''.join(line for line in ledger_lines[1:] if line.startswith(b'R'))
        plan = json.loads((OUT / 'review_plan.json').read_text(encoding='utf-8'))
        self.assertEqual(plan['original_labels_sha256'], hashlib.sha256(original_labels).hexdigest())
        sample = pd.read_parquet(OUT / 'review_sample.parquet').query('combined_rank <= 1000')
        original = sample.merge(ledger, on='review_id', validate='one_to_one')
        original, _ = apply_semantic_reassessment(original)
        original, _ = apply_semantic_top200_review(original)
        pd.testing.assert_frame_equal(retained, original)

    def test_extended_fixed_draws_and_census(self):
        rng=np.random.default_rng(SEED);expected=[]
        for key,g in self.c[self.c.combined_rank.gt(1000)].groupby(['band_end','first_source'],sort=True):
            n=min(20,len(g));expected.extend(g.iloc[np.sort(rng.choice(len(g),n,replace=False))].email_id.tolist())
            selected=self.d[self.d.band_end.eq(key[0]) & self.d.first_source.eq(key[1])]
            self.assertEqual(len(selected),n)
            self.assertTrue(selected.stratum_population.eq(len(g)).all())
            np.testing.assert_allclose(selected.sampling_weight,len(g)/n)
        self.assertEqual(self.d[self.d.combined_rank.gt(1000)].email_id.tolist(),expected)
        self.assertEqual(len(expected),380)
        self.assertEqual(self.d.email_id.nunique(),780)
        self.assertEqual(self.d.combined_rank.max(),1920)

    def test_last_band_uses_twenty_not_one_hundred(self):
        end=self.stats[self.stats.k.eq(1920)].set_index('source')
        self.assertEqual(end.loc['combined','band'],'1901-1920')
        self.assertEqual(end.loc['combined','population'],20)
        self.assertEqual(end.loc['combined','sampled'],20)
        census = self.d.loc[self.d.combined_rank.gt(1900)]
        self.assertEqual(end.loc['combined','precision'],census.relevant.sum()/20)
        for source, group in census.groupby('first_source'):
            self.assertEqual(len(group),10)
            self.assertEqual(end.loc[source,'precision'],group.relevant.sum()/10)
        np.testing.assert_allclose(end.precision,end.precision_ci_low)
        np.testing.assert_allclose(end.precision,end.precision_ci_high)

    def test_band_estimates_match_export(self):
        exported = pd.read_parquet(OUT / 'precision_by_band_and_source.parquet')
        pd.testing.assert_frame_equal(self.stats, exported)

    def test_totals_and_label_consistency(self):
        d=self.d;keys=d.body_clean.map(lambda s:' '.join(s.split()))
        self.assertTrue(d.groupby(keys).relevant.nunique().le(1).all())
        self.assertTrue(d.relevant.isin([0,1]).all())
        self.assertTrue(d.loc[d.relevant.eq(1),'tier'].isin(['direct','context','indirect']).all())
        self.assertAlmostEqual(float((d.relevant*d.sampling_weight).sum()),self.summary['combined']['estimated_relevant_through_endpoint'])
        self.assertAlmostEqual(self.summary['combined']['estimated_relevant_through_endpoint'],
            self.summary['semantic']['estimated_relevant_through_endpoint']+self.summary['tfidf']['estimated_relevant_through_endpoint'])
        self.assertTrue(self.stats.precision.between(self.stats.precision_ci_low,self.stats.precision_ci_high).all())


if __name__=='__main__':unittest.main()
