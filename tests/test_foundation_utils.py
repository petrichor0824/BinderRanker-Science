from __future__ import annotations

import math
import unittest

from binderranker.ranking import rank_rows
from binderranker.scoring.normalization import quantile, robust_norm_value
from binderranker.selection import select_top


class FoundationUtilityTests(unittest.TestCase):
    def test_frozen_quantile_and_degenerate_normalization(self) -> None:
        self.assertEqual(1.75, quantile([1.0, 2.0, 3.0, 4.0], 0.25))
        self.assertEqual(0.5, robust_norm_value(2.0, 1.0, 1.0))
        self.assertTrue(math.isnan(quantile([], 0.5)))

    def test_deterministic_rank_and_budget_selection(self) -> None:
        ranked = rank_rows(
            [
                {"pdb_name": "b", "score": 0.8},
                {"pdb_name": "a", "score": 0.8},
                {"pdb_name": "c", "score": 0.2},
            ],
            score_column="score",
        )
        self.assertEqual(["a", "b", "c"], [row["pdb_name"] for row in ranked])
        self.assertEqual([1, 2], [row["rank"] for row in select_top(ranked, top_k=2)])
        self.assertEqual(2, len(select_top(ranked, top_fraction=0.5)))


if __name__ == "__main__":
    unittest.main()

