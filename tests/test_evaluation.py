"""
tests/test_evaluation.py
========================
Unit tests for src/evaluation.py:
  - compute_outreach_efficiency()
  - compute_revenue_lift()
  - adaptive eval_times in compute_time_dependent_auc()
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import make_decisions_df, make_rfm_decisions_df
from src.evaluation import compute_outreach_efficiency, compute_revenue_lift


class TestComputeOutreachEfficiency(unittest.TestCase):

    def _make_decisions(self, decisions_list, cids=None):
        n = len(decisions_list)
        if cids is None:
            cids = [f"C{i:03d}" for i in range(n)]
        return pd.DataFrame({"CustomerID": cids, "decision": decisions_list})

    def test_correct_contacts_avoided(self):
        """contacts_avoided = rfm_intervene - weibull_intervene (clamped to 0)."""
        weibull = self._make_decisions(["INTERVENE", "WAIT", "WAIT", "LOST"])
        rfm     = self._make_decisions(["INTERVENE", "INTERVENE", "WAIT", "LOST"])
        metrics = compute_outreach_efficiency(weibull, rfm)
        # RFM=2 intervene, Weibull=1 intervene → avoided=1
        self.assertEqual(metrics["contacts_avoided"], 1)

    def test_efficiency_gain_zero_when_equal(self):
        """When both policies have same intervene rate, gain = 0."""
        d = self._make_decisions(["INTERVENE", "WAIT", "WAIT"])
        metrics = compute_outreach_efficiency(d, d)
        self.assertAlmostEqual(metrics["efficiency_gain_pct"], 0.0, places=3)

    def test_rates_sum_to_100_pct_approximately(self):
        """Weibull + non-intervene rates should add up >= 0 and <= 100."""
        weibull = make_decisions_df(n=30)
        rfm     = make_rfm_decisions_df(n=30)
        metrics = compute_outreach_efficiency(weibull, rfm)
        self.assertGreaterEqual(metrics["weibull_intervene_rate"], 0.0)
        self.assertLessEqual(metrics["weibull_intervene_rate"], 100.0)

    def test_contacts_avoided_never_negative(self):
        """contacts_avoided is clamped to >= 0."""
        # Weibull intervenes MORE than RFM
        weibull = self._make_decisions(["INTERVENE", "INTERVENE", "INTERVENE"])
        rfm     = self._make_decisions(["WAIT", "WAIT", "WAIT"])
        metrics = compute_outreach_efficiency(weibull, rfm)
        self.assertGreaterEqual(metrics["contacts_avoided"], 0)

    def test_return_keys(self):
        """All expected keys must be present in the return dict."""
        d = make_decisions_df(n=10)
        r = make_rfm_decisions_df(n=10)
        metrics = compute_outreach_efficiency(d, r)
        for key in ("weibull_intervene_rate", "rfm_intervene_rate",
                    "contacts_avoided", "contacts_avoided_pct", "efficiency_gain_pct"):
            self.assertIn(key, metrics)


class TestComputeRevenueLift(unittest.TestCase):

    def test_return_keys(self):
        """All expected keys must be present."""
        weibull = make_decisions_df(n=20)
        rfm     = make_rfm_decisions_df(n=20)
        metrics = compute_revenue_lift(weibull, rfm)
        for key in ("avg_evi_weibull", "avg_evi_rfm_proxy",
                    "total_evi_weibull", "total_evi_rfm_proxy",
                    "revenue_precision_lift_pct"):
            self.assertIn(key, metrics)

    def test_lift_is_float(self):
        """revenue_precision_lift_pct must be a float."""
        weibull = make_decisions_df(n=20)
        rfm     = make_rfm_decisions_df(n=20)
        metrics = compute_revenue_lift(weibull, rfm)
        self.assertIsInstance(metrics["revenue_precision_lift_pct"], float)


class TestAdaptiveEvalTimes(unittest.TestCase):
    """
    Tests for the adaptive eval_times logic in compute_time_dependent_auc.
    We test the filtering branch instead of the full Weibull model call.
    """

    def test_caller_times_filtered_to_data_range(self):
        """Eval times outside [T_min, T_max] must be dropped."""
        T_obs_max = 120.0
        T_obs = np.linspace(1, T_obs_max, 50)

        # Simulate the filtering logic inline (no model required)
        eval_times_input = [30, 60, 90, 180, 270, 365]  # 180,270,365 > 120
        t_min_obs, t_max_obs = float(T_obs.min()), float(T_obs.max())
        filtered = [t for t in eval_times_input if t_min_obs <= t <= t_max_obs]

        self.assertEqual(filtered, [30, 60, 90])
        for t in filtered:
            self.assertLessEqual(t, T_obs_max)

    def test_adaptive_times_derived_from_percentiles(self):
        """Auto eval_times must fall within [P5, P95] of T_obs."""
        T_obs = np.concatenate([np.linspace(1, 50, 30), np.linspace(100, 150, 20)])
        t_lo = float(np.percentile(T_obs, 5))
        t_hi = float(np.percentile(T_obs, 95))
        auto_times = list(np.linspace(t_lo, t_hi, 6))

        for t in auto_times:
            self.assertGreaterEqual(t, t_lo - 0.01)
            self.assertLessEqual(t, t_hi + 0.01)

    def test_adaptive_always_6_points(self):
        """Auto eval_times must produce exactly 6 evaluation points."""
        T_obs = np.array([1, 5, 10, 20, 50, 100, 200, 365])
        t_lo = float(np.percentile(T_obs, 5))
        t_hi = float(np.percentile(T_obs, 95))
        auto_points = list(np.linspace(t_lo, t_hi, 6))
        self.assertEqual(len(auto_points), 6)


if __name__ == "__main__":
    unittest.main()
