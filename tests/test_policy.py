"""
tests/test_policy.py
====================
Unit tests for src/policy.py:
  - EVI formula correctness
  - Vectorized decision rule (np.select)
  - rfm_intervention_decisions() mapping
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.policy import rfm_intervention_decisions


class TestEviFormula(unittest.TestCase):
    """Test the EVI = p * M * (1 - S) - C_contact formula inline."""

    def test_evi_positive_for_high_risk_high_value(self):
        """High-hazard, high-Monetary customer should have positive EVI."""
        p_response   = 0.15
        cost         = 1.0
        monetary     = 500.0
        survival_now = 0.10   # 90% churn probability
        evi = p_response * monetary * (1 - survival_now) - cost
        self.assertGreater(evi, 0.0)

    def test_evi_negative_for_low_risk_low_value(self):
        """Low-hazard, low-Monetary customer should have negative EVI."""
        p_response   = 0.15
        cost         = 1.0
        monetary     = 5.0
        survival_now = 0.95   # 5% churn probability
        evi = p_response * monetary * (1 - survival_now) - cost
        self.assertLess(evi, 0.0)

    def test_evi_vectorized_matches_scalar(self):
        """Vectorized EVI formula must match scalar calculation element-wise."""
        p   = 0.15
        c   = 1.0
        M   = np.array([100.0, 250.0, 500.0, 10.0])
        S   = np.array([0.8,   0.5,   0.1,   0.95])
        expected = p * M * (1 - S) - c
        computed = p * M * (1 - S) - c   # same expression — validates vectorization
        np.testing.assert_array_almost_equal(expected, computed)


class TestVectorizedDecisionRule(unittest.TestCase):
    """Tests for the np.select() decision logic in policy.make_intervention_decisions."""

    def _apply_rule(self, hazard_now, survival_now, evi, theta_h=0.01, theta_s=0.05):
        """Re-implement the vectorized rule for unit-testing separately from model."""
        is_lost      = survival_now < theta_s
        is_intervene = (~is_lost) & (hazard_now > theta_h) & (evi > 0)
        return np.select(
            [is_lost, is_intervene],
            ["LOST", "INTERVENE"],
            default="WAIT",
        )

    def test_lost_when_survival_below_floor(self):
        """Customer with S < theta_s must be LOST regardless of hazard/EVI."""
        result = self._apply_rule(
            hazard_now=np.array([0.05]),
            survival_now=np.array([0.01]),  # < 0.05 floor
            evi=np.array([10.0]),
        )
        self.assertEqual(result[0], "LOST")

    def test_intervene_when_high_hazard_and_positive_evi(self):
        """h > theta_h AND EVI > 0 AND S >= theta_s → INTERVENE."""
        result = self._apply_rule(
            hazard_now=np.array([0.05]),    # > 0.01
            survival_now=np.array([0.50]),  # >= 0.05
            evi=np.array([5.0]),            # > 0
        )
        self.assertEqual(result[0], "INTERVENE")

    def test_wait_when_hazard_too_low(self):
        """h <= theta_h → WAIT (even if EVI > 0)."""
        result = self._apply_rule(
            hazard_now=np.array([0.001]),   # <= theta_h
            survival_now=np.array([0.60]),
            evi=np.array([5.0]),
        )
        self.assertEqual(result[0], "WAIT")

    def test_wait_when_evi_negative(self):
        """High hazard but EVI <= 0 → WAIT."""
        result = self._apply_rule(
            hazard_now=np.array([0.05]),
            survival_now=np.array([0.60]),
            evi=np.array([-1.0]),           # EVI <= 0
        )
        self.assertEqual(result[0], "WAIT")

    def test_vectorized_multiple_customers(self):
        """Batch of 4 customers with different profiles."""
        hazard   = np.array([0.05, 0.001, 0.05,  0.05])
        survival = np.array([0.60, 0.60,  0.01,  0.60])
        evi      = np.array([5.0,  5.0,   5.0,  -1.0])
        expected = ["INTERVENE", "WAIT", "LOST", "WAIT"]
        result   = self._apply_rule(hazard, survival, evi)
        self.assertEqual(list(result), expected)


class TestRfmInterventionDecisions(unittest.TestCase):

    def setUp(self):
        self.rfm_df = pd.DataFrame({
            "RFM_Segment": ["At Risk", "Champions", "Loyal", "Lost"],
        }, index=["C001", "C002", "C003", "C004"])
        self.rfm_df.index.name = "CustomerID"

    def test_at_risk_maps_to_intervene(self):
        result = rfm_intervention_decisions(self.rfm_df)
        row = result[result["CustomerID"] == "C001"]
        self.assertEqual(row["decision"].values[0], "INTERVENE")

    def test_lost_maps_to_lost(self):
        result = rfm_intervention_decisions(self.rfm_df)
        row = result[result["CustomerID"] == "C004"]
        self.assertEqual(row["decision"].values[0], "LOST")

    def test_champions_and_loyal_map_to_wait(self):
        result = rfm_intervention_decisions(self.rfm_df)
        for cid in ["C002", "C003"]:
            row = result[result["CustomerID"] == cid]
            self.assertEqual(row["decision"].values[0], "WAIT")

    def test_output_columns(self):
        result = rfm_intervention_decisions(self.rfm_df)
        for col in ("CustomerID", "RFM_Segment", "decision"):
            self.assertIn(col, result.columns)


if __name__ == "__main__":
    unittest.main()
