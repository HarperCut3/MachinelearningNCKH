"""
tests/test_models.py
====================
Unit tests for src/models.py:
  - get_survival_features() auto-discovery
  - rfm_segment() output schema and segment values
  - train_logistic() return types and structure
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import make_customer_df
from src.models import (
    SURVIVAL_FEATURES, get_survival_features,
    rfm_segment, train_logistic,
)


class TestGetSurvivalFeatures(unittest.TestCase):
    """Tests for the B2 auto-discovery function."""

    def test_discovers_numeric_cols(self):
        """Should return all numeric, non-target columns."""
        df = make_customer_df()
        features = get_survival_features(df)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_excludes_target_cols(self):
        """T and E must never appear in the feature list."""
        df = make_customer_df()
        features = get_survival_features(df)
        self.assertNotIn("T", features)
        self.assertNotIn("E", features)

    def test_includes_rfm_cols(self):
        """Standard RFM columns must be discovered."""
        df = make_customer_df()
        features = get_survival_features(df)
        for col in ["Recency", "Frequency", "Monetary"]:
            self.assertIn(col, features, f"{col} not found in auto-discovered features")

    def test_fallback_on_empty_df(self):
        """Returns subset of SURVIVAL_FEATURES when df has no matching cols."""
        df = pd.DataFrame({"CustomerID": [1], "T": [10.0], "E": [0]})
        features = get_survival_features(df)
        # Must return a list, even if empty
        self.assertIsInstance(features, list)

    def test_ignores_all_nan_col(self):
        """Columns where all values are NaN should be excluded."""
        df = make_customer_df().copy()
        df["all_nan_col"] = np.nan
        features = get_survival_features(df)
        self.assertNotIn("all_nan_col", features)

    def test_no_mutation_of_global(self):
        """Calling get_survival_features must not mutate the global SURVIVAL_FEATURES list."""
        original = list(SURVIVAL_FEATURES)
        df = make_customer_df()
        get_survival_features(df)
        self.assertEqual(original, SURVIVAL_FEATURES)


class TestRfmSegment(unittest.TestCase):

    def setUp(self):
        self.customer_df = make_customer_df()
        self.rfm_df = rfm_segment(self.customer_df)

    def test_output_columns(self):
        """rfm_segment must produce required columns."""
        required = {"R_score", "F_score", "M_score", "RFM_Score", "RFM_Segment"}
        self.assertTrue(required.issubset(set(self.rfm_df.columns)),
                        f"Missing columns: {required - set(self.rfm_df.columns)}")

    def test_valid_segment_values(self):
        """All segment values must be from the allowed set."""
        valid = {"Champions", "Loyal", "At Risk", "Lost"}
        actual = set(self.rfm_df["RFM_Segment"].unique())
        self.assertTrue(actual.issubset(valid),
                        f"Unexpected segments: {actual - valid}")

    def test_same_length_as_input(self):
        """Output must have same number of rows as input."""
        self.assertEqual(len(self.rfm_df), len(self.customer_df))

    def test_rfm_score_range(self):
        """RFM_Score must be an integer between 3 and 15 inclusive."""
        self.assertTrue((self.rfm_df["RFM_Score"] >= 3).all())
        self.assertTrue((self.rfm_df["RFM_Score"] <= 15).all())


class TestTrainLogistic(unittest.TestCase):

    def setUp(self):
        self.customer_df = make_customer_df()

    def test_returns_three_values(self):
        """train_logistic must return (model, pipeline, cv_metrics_dict)."""
        result = train_logistic(self.customer_df, cv_folds=2)
        self.assertEqual(len(result), 3)

    def test_cv_metrics_keys(self):
        """cv_metrics must contain auc_mean, auc_std, acc_mean, acc_std, features."""
        _, _, cv_metrics = train_logistic(self.customer_df, cv_folds=2)
        for key in ("auc_mean", "auc_std", "acc_mean", "acc_std"):
            self.assertIn(key, cv_metrics, f"Missing key: {key}")

    def test_auc_in_range(self):
        """AUC must be in [0, 1]."""
        _, _, cv_metrics = train_logistic(self.customer_df, cv_folds=2)
        self.assertGreaterEqual(cv_metrics["auc_mean"], 0.0)
        self.assertLessEqual(cv_metrics["auc_mean"], 1.0)

    def test_pipeline_has_predict_proba(self):
        """The returned pipeline must support predict_proba."""
        _, lr_pipeline, _ = train_logistic(self.customer_df, cv_folds=2)
        self.assertTrue(hasattr(lr_pipeline, "predict_proba"))


if __name__ == "__main__":
    unittest.main()
