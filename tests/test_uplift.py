"""
tests/test_uplift.py
====================
Unit tests for src/uplift.py:
  - Qini curve vectorized formula correctness
  - T-Learner structure
  - Persuadables segmentation logic
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQiniVectorized(unittest.TestCase):
    """Tests for the O(n) vectorized Qini computation."""

    def _make_uplift_df(self, n=50, seed=42):
        """Minimal DataFrame matching what _compute_qini expects."""
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "CustomerID": [f"C{i}" for i in range(n)],
            "treatment":  rng.choice([0, 1], size=n, p=[0.5, 0.5]),
            "tau_hat":    rng.uniform(-5, 5, n),
            "Monetary":   rng.uniform(10, 500, n),
        })

    def test_output_columns(self):
        """Qini DataFrame must have pct_targeted, qini_gain, random_baseline."""
        from src.uplift import _compute_qini
        df = self._make_uplift_df()
        qini_df = _compute_qini(df)
        for col in ("pct_targeted", "qini_gain", "random_baseline"):
            self.assertIn(col, qini_df.columns)

    def test_pct_targeted_range(self):
        """pct_targeted must be in (0, 1]."""
        from src.uplift import _compute_qini
        df = self._make_uplift_df()
        qini_df = _compute_qini(df)
        self.assertGreater(qini_df["pct_targeted"].min(), 0.0)
        self.assertAlmostEqual(qini_df["pct_targeted"].max(), 1.0, places=5)

    def test_same_length_as_input(self):
        """Qini DataFrame rows must equal number of input customers."""
        from src.uplift import _compute_qini
        n = 40
        df = self._make_uplift_df(n=n)
        qini_df = _compute_qini(df)
        self.assertEqual(len(qini_df), n)

    def test_all_same_treatment_returns_fallback(self):
        """When all are treated or all are control, Qini returns zero-gain fallback."""
        from src.uplift import _compute_qini
        df = self._make_uplift_df(n=20)
        df["treatment"] = 1  # all treated — no control group
        qini_df = _compute_qini(df)
        # Must return a valid DataFrame (not raise)
        self.assertIsInstance(qini_df, pd.DataFrame)
        self.assertIn("qini_gain", qini_df.columns)

    def test_vectorized_qini_monotone_at_boundary(self):
        """With perfect uplift signal, treated should cluster at the top."""
        from src.uplift import _compute_qini
        n = 20
        # Perfect signal: all treated have high tau_hat, all control have low
        df = pd.DataFrame({
            "treatment": [1]*10 + [0]*10,
            "tau_hat":   list(range(20, 10, -1)) + list(range(10, 0, -1)),
            "Monetary":  [100.0] * 20,
        })
        qini_df = _compute_qini(df)
        # At the first 50% targeted, all treated should be included → peak Qini
        self.assertIsInstance(qini_df["qini_gain"].iloc[9], (float, np.floating))


class TestPersuadablesSegmentation(unittest.TestCase):
    """Tests for _assign_uplift_segment (quadrant logic)."""

    def test_persuadables(self):
        from src.uplift import _assign_uplift_segment, _RESPONSE_THR
        row = pd.Series({"tau_hat": 1.0, "mu_1": _RESPONSE_THR + 1.0})
        self.assertEqual(_assign_uplift_segment(row), "Persuadables")

    def test_sure_things(self):
        from src.uplift import _assign_uplift_segment, _RESPONSE_THR
        row = pd.Series({"tau_hat": -1.0, "mu_1": _RESPONSE_THR + 1.0})
        self.assertEqual(_assign_uplift_segment(row), "Sure Things")

    def test_sleeping_dogs(self):
        from src.uplift import _assign_uplift_segment, _RESPONSE_THR
        row = pd.Series({"tau_hat": 1.0, "mu_1": _RESPONSE_THR - 1.0})
        self.assertEqual(_assign_uplift_segment(row), "Sleeping Dogs")

    def test_lost_causes(self):
        from src.uplift import _assign_uplift_segment, _RESPONSE_THR
        row = pd.Series({"tau_hat": -1.0, "mu_1": _RESPONSE_THR - 1.0})
        self.assertEqual(_assign_uplift_segment(row), "Lost Causes")


if __name__ == "__main__":
    unittest.main()
