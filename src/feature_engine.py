"""
src/feature_engine.py
=====================
Transforms clean transaction-level data into a customer-level feature matrix
suitable for survival analysis.

Features produced:
  RFM (Static):
    - Recency          : Days from last purchase to snapshot date
    - Frequency        : Number of unique invoices
    - Monetary         : Total spend (GBP)

  Temporal (Dynamic):
    - InterPurchaseTime: Mean inter-purchase gap (days)
    - GapStability     : Std dev of inter-purchase gaps (days)
    - SinglePurchase   : Binary flag — customer made only 1 purchase

  Survival Target:
    - T                : Observation window (days from first to last purchase)
    - E                : Event indicator (1 = churned, 0 = censored)

Churn Definition:
    E_i = 1  if  Recency_i > tau  (customer has been inactive > tau days)
    E_i = 0  if  Recency_i <= tau (customer is still considered active)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_customer_features(
    df: pd.DataFrame,
    snapshot: pd.Timestamp,
    tau: int = 90,
) -> pd.DataFrame:
    """
    Aggregate transaction-level data to customer-level RFM + Survival features.

    Parameters
    ----------
    df : pd.DataFrame
        Clean transaction DataFrame from data_loader.load_and_clean().
    snapshot : pd.Timestamp
        Reference date for Recency calculation (max date + 1 day).
    tau : int, optional
        Inactivity threshold in days to define churn event (default: 90).

    Returns
    -------
    pd.DataFrame
        Customer-level DataFrame indexed by CustomerID with columns:
        [Recency, Frequency, Monetary, InterPurchaseTime, GapStability,
         SinglePurchase, T, E]
    """
    logger.info(f"Building customer features | snapshot={snapshot.date()} | tau={tau} days")

    # ── Group by customer ────────────────────────────────────────────────────
    grp = df.groupby("CustomerID")

    # ── RFM Features ─────────────────────────────────────────────────────────
    recency    = (snapshot - grp["InvoiceDate"].max()).dt.days.rename("Recency")
    frequency  = grp["InvoiceNo"].nunique().rename("Frequency")
    monetary   = grp["TotalSpend"].sum().rename("Monetary")

    # ── Temporal Features ─────────────────────────────────────────────────────
    def _inter_purchase_stats(dates: pd.Series):
        """Compute mean and std of inter-purchase gaps for a customer."""
        sorted_dates = dates.sort_values().drop_duplicates()
        if len(sorted_dates) < 2:
            return pd.Series({"InterPurchaseTime": np.nan, "GapStability": 0.0})
        gaps = sorted_dates.diff().dropna().dt.days
        return pd.Series({
            "InterPurchaseTime": gaps.mean(),
            "GapStability":      gaps.std(ddof=1) if len(gaps) > 1 else 0.0,
        })

    temporal_stats = grp["InvoiceDate"].apply(_inter_purchase_stats).unstack()

    # ── SinglePurchase Flag ───────────────────────────────────────────────────
    single_purchase = (frequency == 1).astype(int).rename("SinglePurchase")

    # ── Survival Target: T (observation window) ───────────────────────────────
    # T = days from first purchase to last purchase
    # For single-purchase customers: T = 0 (they have no repeat history)
    first_purchase = grp["InvoiceDate"].min()
    last_purchase  = grp["InvoiceDate"].max()
    T = (last_purchase - first_purchase).dt.days.rename("T")

    # Ensure T >= 1 to avoid zero-duration issues in survival models
    T = T.clip(lower=1)

    # ── Survival Target: E (event indicator) ─────────────────────────────────
    # E = 1 if customer has been inactive for more than tau days (churned)
    # E = 0 if customer is still within the active window (censored)
    E = (recency > tau).astype(int).rename("E")

    # ── Assemble customer DataFrame ───────────────────────────────────────────
    customer_df = pd.concat(
        [recency, frequency, monetary, temporal_stats, single_purchase, T, E],
        axis=1
    )

    # ── Impute InterPurchaseTime for single-purchase customers ────────────────
    # Use median of multi-purchase customers as a conservative estimate
    median_ipt = customer_df.loc[
        customer_df["SinglePurchase"] == 0, "InterPurchaseTime"
    ].median()
    customer_df["InterPurchaseTime"] = customer_df["InterPurchaseTime"].fillna(median_ipt)
    customer_df["GapStability"]      = customer_df["GapStability"].fillna(0.0)

    # ── Log summary statistics ────────────────────────────────────────────────
    n_customers  = len(customer_df)
    n_churned    = customer_df["E"].sum()
    n_censored   = n_customers - n_churned
    churn_rate   = n_churned / n_customers * 100

    logger.info(
        f"Customer features built: {n_customers:,} customers | "
        f"Churned (E=1): {n_churned:,} ({churn_rate:.1f}%) | "
        f"Censored (E=0): {n_censored:,} ({100 - churn_rate:.1f}%)"
    )
    logger.info(
        f"T stats — mean: {customer_df['T'].mean():.1f}d | "
        f"median: {customer_df['T'].median():.1f}d | "
        f"max: {customer_df['T'].max():.1f}d"
    )

    return customer_df


def sensitivity_analysis_tau(
    df: pd.DataFrame,
    snapshot: pd.Timestamp,
    tau_values: list = None,
) -> dict:
    """
    Run feature engineering across multiple tau thresholds.
    Used to assess robustness of the churn definition.

    Parameters
    ----------
    df : pd.DataFrame
        Clean transaction DataFrame.
    snapshot : pd.Timestamp
        Snapshot date.
    tau_values : list of int, optional
        List of inactivity thresholds to test (default: [60, 90, 120]).

    Returns
    -------
    dict
        Mapping {tau: customer_df} for each threshold.
    """
    if tau_values is None:
        tau_values = [60, 90, 120]

    results = {}
    for tau in tau_values:
        logger.info(f"--- Sensitivity: tau = {tau} days ---")
        results[tau] = build_customer_features(df, snapshot, tau=tau)
    return results
