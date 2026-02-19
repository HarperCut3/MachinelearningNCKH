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
    - InterPurchaseTime: Mean inter-purchase gap (days); 0.0 for single-visit
    - GapDeviation     : Std dev of inter-purchase gaps (days); 0.0 if < 2 gaps
    - SinglePurchase   : Binary flag — customer made only 1 purchase

  Survival Target:
    - T  : Observation window in days.
           Repeat purchasers : days from first to last purchase (active span).
           Single purchasers : Recency (days from only purchase to snapshot).
           Rationale: T must represent how long the customer has been *observed*,
           not just their inter-purchase span.  Using T=clip(1) for single-buyers
           creates an artificial spike at T=1 that biases rho < 1.
    - E  : Event indicator (1 = churned, 0 = censored)

Churn Definition:
    E_i = 1  if  Recency_i > tau  (customer has been inactive > tau days)
    E_i = 0  if  Recency_i <= tau (customer is still considered active)

Performance Note (Phase 7):
    InterPurchaseTime and GapDeviation are computed with fully vectorized
    pandas operations — no .apply() on groups.  The approach:
      1. Sort df by [CustomerID, InvoiceDate] once.
      2. Compute per-row date diff via groupby(...).diff().dt.days.
      3. Aggregate [mean, std] with a single groupby().agg() call.
    This is O(N log N) vs the previous O(N * k) .apply() loop.
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
        [Recency, Frequency, Monetary, InterPurchaseTime, GapDeviation,
         SinglePurchase, T, E]
    """
    logger.info(f"Building customer features | snapshot={snapshot.date()} | tau={tau} days")

    # ── Group by customer ────────────────────────────────────────────────────
    grp = df.groupby("CustomerID")

    # ── RFM Features ─────────────────────────────────────────────────────────
    recency   = (snapshot - grp["InvoiceDate"].max()).dt.days.rename("Recency")
    frequency = grp["InvoiceNo"].nunique().rename("Frequency")
    monetary  = grp["TotalSpend"].sum().rename("Monetary")

    # ── Temporal Features: Vectorized (Phase 7 performance fix) ──────────────
    # Step 1: Sort the full DataFrame once by CustomerID + InvoiceDate.
    #         We only need unique (CustomerID, InvoiceDate) pairs — multiple
    #         invoices on the same day produce a gap of 0 which is uninformative
    #         and inflates means. Drop duplicates for gap calculation.
    df_sorted = (
        df[["CustomerID", "InvoiceDate"]]
        .drop_duplicates()
        .sort_values(["CustomerID", "InvoiceDate"])
    )

    # Step 2: Compute per-row gap in days via groupby diff.
    #         diff() is NaN for the *first* date of each customer — that is
    #         correct; the first visit has no preceding gap.
    df_sorted["gap_days"] = (
        df_sorted.groupby("CustomerID")["InvoiceDate"]
        .diff()
        .dt.days
    )

    # Step 3: Aggregate mean and std of gaps in one pass.
    #         ddof=1 (sample std) matches the previous implementation.
    #         Customers with only one unique date → all gaps are NaN → mean/std
    #         are NaN, which we impute below.
    gap_agg = (
        df_sorted.groupby("CustomerID")["gap_days"]
        .agg(
            InterPurchaseTime=("mean"),
            GapDeviation=("std"),
        )
    )

    # ── SinglePurchase Flag ───────────────────────────────────────────────────
    single_purchase = (frequency == 1).astype(int).rename("SinglePurchase")

    # ── Survival Target: T (observation window) ───────────────────────────────
    # Repeat purchasers  : T = days from first to last purchase (active span)
    # Single purchasers  : T = Recency (days from only purchase to snapshot)
    # Rationale: T.clip(1) for single-buyers creates a spike at T=1 that
    #   biases rho < 1, inverting the Weibull hazard direction.
    first_purchase = grp["InvoiceDate"].min()
    last_purchase  = grp["InvoiceDate"].max()

    T_repeat = (last_purchase - first_purchase).dt.days  # 0 for single buyers
    T_single = recency                                    # observation window
    T = T_repeat.where(single_purchase == 0, other=T_single).rename("T")
    T = T.clip(lower=1)  # safety floor; T=0 causes log(0) in Weibull

    # ── Survival Target: E (event indicator) ─────────────────────────────────
    # E = 1 if customer has been inactive for more than tau days (churned)
    # E = 0 if customer is still within the active window (censored)
    E = (recency > tau).astype(int).rename("E")

    # ── Assemble customer DataFrame ───────────────────────────────────────────
    customer_df = pd.concat(
        [recency, frequency, monetary, gap_agg, single_purchase, T, E],
        axis=1,
    )

    # ── Anti-Leakage Imputation ───────────────────────────────────────────────
    # Impute InterPurchaseTime and GapDeviation with 0.0 for single-purchase
    # customers (NOT cross-customer median — that is leakage).
    # Zero is semantically correct: single-buyers have no inter-purchase gaps.
    # The SinglePurchase=1 flag lets the model learn a separate effect.
    n_single = int(customer_df["InterPurchaseTime"].isna().sum())
    customer_df["InterPurchaseTime"] = customer_df["InterPurchaseTime"].fillna(0.0)
    customer_df["GapDeviation"]      = customer_df["GapDeviation"].fillna(0.0)
    if n_single > 0:
        logger.info(
            f"Imputed InterPurchaseTime=0.0 and GapDeviation=0.0 for "
            f"{n_single:,} single-purchase customers (anti-leakage guard)."
        )

    # ── Log summary statistics ────────────────────────────────────────────────
    n_customers = len(customer_df)
    n_churned   = customer_df["E"].sum()
    n_censored  = n_customers - n_churned
    churn_rate  = n_churned / n_customers * 100

    logger.info(
        f"Customer features built: {n_customers:,} customers | "
        f"Churned (E=1): {n_churned:,} ({churn_rate:.1f}%) | "
        f"Censored (E=0): {n_censored:,} ({100 - churn_rate:.1f}%)"
    )
    logger.info(
        f"T stats — mean: {customer_df['T'].mean():.1f}d | "
        f"median: {customer_df['T'].median():.1f}d | "
        f"max: {customer_df['T'].max():.1f}d | "
        f"single-purchase (T=Recency): {n_single:,} customers"
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
        List of inactivity thresholds to test.
        If None, computes adaptively from the dataset duration:
        [duration//5, duration//3, duration//2].
        Rationale: fixed values like {60, 90, 120} are meaningless for
        a dataset shorter than 120 days (e.g. Ta Feng = 120d total).

    Returns
    -------
    dict
        Mapping {tau: customer_df} for each threshold.
    """
    if tau_values is None:
        duration = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days
        tau_values = sorted(set([
            max(duration // 5, 1),
            max(duration // 3, 1),
            max(duration // 2, 1),
        ]))
        logger.info(
            f"[SensitivityTau] Adaptive tau values (dataset duration={duration}d): "
            f"{tau_values}"
        )

    results = {}
    for tau in tau_values:
        logger.info(f"--- Sensitivity: tau = {tau} days ---")
        results[tau] = build_customer_features(df, snapshot, tau=tau)
    return results
