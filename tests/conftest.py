"""
tests/conftest.py
=================
Shared synthetic data fixtures for the test suite.
All fixtures are unittest-compatible (no pytest dependency required).

Usage in unittest:
    from tests.conftest import make_customer_df, make_transactions_df, make_snapshot
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def make_snapshot() -> pd.Timestamp:
    return pd.Timestamp("2023-01-01")


def make_transactions_df(
    n_customers: int = 50,
    n_rows: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data matching the standard schema.

    Output columns: CustomerID, InvoiceNo, InvoiceDate, TotalSpend, Quantity
    """
    rng = np.random.default_rng(seed)
    snap = make_snapshot()

    customer_ids = [f"C{i:03d}" for i in range(1, n_customers + 1)]
    rows = []
    for i, cid in enumerate(customer_ids):
        # Each customer gets between 1 and 10 transactions
        n_tx = rng.integers(1, 11)
        for j in range(n_tx):
            days_ago = rng.integers(1, 400)
            rows.append({
                "CustomerID":  cid,
                "InvoiceNo":   f"INV-{cid}-{j}",
                "InvoiceDate": snap - timedelta(days=int(days_ago)),
                "TotalSpend":  round(float(rng.uniform(5.0, 500.0)), 2),
                "Quantity":    int(rng.integers(1, 10)),
            })

    df = pd.DataFrame(rows)
    # Ensure InvoiceDate is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def make_customer_df(tau: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Generate a customer-level feature DataFrame (output of build_customer_features).
    This mirrors the schema produced by feature_engine.py.
    """
    from src.feature_engine import build_customer_features
    df_tx = make_transactions_df(seed=seed)
    snap  = make_snapshot()
    return build_customer_features(df_tx, snap, tau=tau)


def make_decisions_df(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic Weibull decision table (output of policy.make_intervention_decisions).
    """
    rng = np.random.default_rng(seed)
    decisions = rng.choice(["INTERVENE", "WAIT", "LOST"], size=n, p=[0.3, 0.5, 0.2])
    return pd.DataFrame({
        "CustomerID":  [f"C{i:03d}" for i in range(n)],
        "hazard_now":  rng.uniform(0.0, 0.05, n),
        "survival":    rng.uniform(0.1, 1.0, n),
        "evi":         rng.uniform(-2.0, 10.0, n),
        "decision":    decisions,
        "Monetary":    rng.uniform(20.0, 500.0, n),
        "optimal_window_days": rng.uniform(10.0, 200.0, n),
    })


def make_rfm_decisions_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic RFM decision table (output of policy.rfm_intervention_decisions).
    """
    rng = np.random.default_rng(seed)
    segments = rng.choice(["Champions", "Loyal", "At Risk", "Lost"], size=n)
    seg_map = {"At Risk": "INTERVENE", "Lost": "LOST", "Loyal": "WAIT", "Champions": "WAIT"}
    return pd.DataFrame({
        "CustomerID":   [f"C{i:03d}" for i in range(n)],
        "RFM_Segment":  segments,
        "decision":     [seg_map[s] for s in segments],
    })
