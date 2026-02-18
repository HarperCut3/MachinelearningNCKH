"""
src/policy.py
=============
Decision-Centric Intervention Policy Engine.

For each customer, computes:
  1. Instantaneous hazard h(t_now | x)  — risk of churning right now
  2. Survival probability S(t_now | x)  — probability still active
  3. Expected Value of Intervention (EVI) — economic signal

Decision Rule:
  IF h(t_now) > θ_h  AND  EVI > 0  → INTERVENE
  ELIF S(t_now) < θ_s               → LOST (do not contact)
  ELSE                               → WAIT

EVI Formula:
  EVI(t*, i) = p_response * Monetary_i * [1 - S(t* | x_i)] - C_contact

where:
  p_response  = campaign response rate (default: 0.15)
  C_contact   = cost per outreach in GBP (default: 1.0)
  t*          = current observation time (days since first purchase)
"""

import logging
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

logger = logging.getLogger(__name__)

# ── Default Policy Thresholds ─────────────────────────────────────────────────
DEFAULT_HAZARD_THRESHOLD   = 0.01   # h(t) per day — above this = high risk
DEFAULT_SURVIVAL_FLOOR     = 0.05   # S(t) below this = customer is lost
DEFAULT_RESPONSE_RATE      = 0.15   # 15% campaign response rate
DEFAULT_COST_PER_CONTACT   = 1.0    # £1.00 per email/contact


def _compute_hazard_from_survival(
    survival_fn: pd.DataFrame,
    t_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Numerically differentiate survival function to obtain hazard rates.

    h(t) ≈ -ΔS(t) / (S(t) * Δt)

    Parameters
    ----------
    survival_fn : pd.DataFrame
        Survival function matrix, shape (len(t_grid), n_customers).
        Columns = customer indices, rows = time points.
    t_grid : np.ndarray
        Time points corresponding to survival_fn rows.

    Returns
    -------
    pd.DataFrame
        Hazard rate matrix, same shape as survival_fn.
    """
    S = survival_fn.values  # shape: (T, N)
    dt = np.diff(t_grid, prepend=t_grid[0])[:, None]  # (T, 1)

    # Numerical derivative: h(t) = -dS/dt / S(t)
    dS = np.diff(S, axis=0, prepend=S[[0], :])
    hazard = -dS / (S + 1e-10) / (dt + 1e-10)
    hazard = np.clip(hazard, 0, None)  # hazard must be non-negative

    return pd.DataFrame(hazard, index=survival_fn.index, columns=survival_fn.columns)


def compute_intervention_signals(
    waf: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    customer_df: pd.DataFrame,
    t_now: float = None,
    t_grid_steps: int = 200,
) -> pd.DataFrame:
    """
    Compute hazard, survival, and EVI signals for all customers.

    Parameters
    ----------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model.
    df_scaled : pd.DataFrame
        Scaled feature DataFrame (same as used for fitting), with T and E.
    customer_df : pd.DataFrame
        Original (unscaled) customer DataFrame — used for Monetary values.
    t_now : float, optional
        Current time point in days. Defaults to median T in dataset.
    t_grid_steps : int
        Number of time steps for survival function evaluation (default: 200).

    Returns
    -------
    pd.DataFrame
        Per-customer signals with columns:
        [hazard_now, survival_now, evi, optimal_window_days]
    """
    t_max = df_scaled["T"].max()
    t_grid = np.linspace(1, t_max, t_grid_steps)

    if t_now is None:
        t_now = float(df_scaled["T"].median())
        logger.info(f"t_now not specified — using median T = {t_now:.1f} days")

    # ── Survival function S(t | x) for all customers ─────────────────────────
    logger.info(f"Computing survival functions over {t_grid_steps} time steps...")
    survival_fn = waf.predict_survival_function(df_scaled, times=t_grid)
    # survival_fn: shape (t_grid_steps, n_customers), columns = customer index

    # ── Hazard function h(t | x) ──────────────────────────────────────────────
    hazard_fn = _compute_hazard_from_survival(survival_fn, t_grid)

    # ── Extract values at t_now ───────────────────────────────────────────────
    t_idx = np.argmin(np.abs(t_grid - t_now))

    hazard_now  = hazard_fn.iloc[t_idx]    # Series, index = customer index
    survival_now = survival_fn.iloc[t_idx]  # Series, index = customer index

    # ── Optimal intervention window: time at which h(t) is maximized ─────────
    optimal_t_idx = hazard_fn.idxmax(axis=0)  # index label of max hazard row
    # Map index label back to t_grid value
    idx_to_t = dict(zip(range(len(t_grid)), t_grid))
    optimal_window_days = optimal_t_idx.map(
        lambda row_label: t_grid[list(hazard_fn.index).index(row_label)]
        if row_label in hazard_fn.index else np.nan
    )

    # ── Assemble signals DataFrame ────────────────────────────────────────────
    signals = pd.DataFrame({
        "hazard_now":          hazard_now.values,
        "survival_now":        survival_now.values,
        "optimal_window_days": t_grid[hazard_fn.values.argmax(axis=0)],
    }, index=df_scaled.index)

    # ── Merge Monetary from original customer_df ──────────────────────────────
    signals["Monetary"] = customer_df["Monetary"].values

    return signals


def make_intervention_decisions(
    waf: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    customer_df: pd.DataFrame,
    t_now: float = None,
    theta_h: float = DEFAULT_HAZARD_THRESHOLD,
    theta_s: float = DEFAULT_SURVIVAL_FLOOR,
    p_response: float = DEFAULT_RESPONSE_RATE,
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
) -> pd.DataFrame:
    """
    Apply the full decision policy to all customers.

    Decision Rule:
      IF h(t_now) > θ_h  AND  EVI > 0  → INTERVENE
      ELIF S(t_now) < θ_s               → LOST
      ELSE                               → WAIT

    Parameters
    ----------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model.
    df_scaled : pd.DataFrame
        Scaled feature DataFrame with T and E columns.
    customer_df : pd.DataFrame
        Original customer DataFrame (for Monetary values).
    t_now : float, optional
        Current time in days (defaults to median T).
    theta_h : float
        Hazard threshold for intervention trigger.
    theta_s : float
        Survival floor below which customer is considered lost.
    p_response : float
        Expected campaign response rate.
    cost_per_contact : float
        Cost per marketing contact (GBP).

    Returns
    -------
    pd.DataFrame
        Decision table with columns:
        [CustomerID, hazard_now, survival_now, evi, decision, optimal_window_days]
    """
    logger.info(
        f"Running intervention policy | θ_h={theta_h} | θ_s={theta_s} | "
        f"p_response={p_response} | cost_per_contact=£{cost_per_contact:.2f}"
    )

    signals = compute_intervention_signals(waf, df_scaled, customer_df, t_now)

    # ── Expected Value of Intervention ────────────────────────────────────────
    # EVI(t*, i) = p_response * Monetary_i * [1 - S(t* | x_i)] - C_contact
    signals["evi"] = (
        p_response * signals["Monetary"] * (1 - signals["survival_now"])
        - cost_per_contact
    )

    # ── Apply decision rule ───────────────────────────────────────────────────
    def _decide(row):
        if row["survival_now"] < theta_s:
            return "LOST"
        elif row["hazard_now"] > theta_h and row["evi"] > 0:
            return "INTERVENE"
        else:
            return "WAIT"

    signals["decision"] = signals.apply(_decide, axis=1)

    # ── Add CustomerID ────────────────────────────────────────────────────────
    signals.index.name = "CustomerID"
    signals = signals.reset_index()

    # ── Log decision distribution ─────────────────────────────────────────────
    dist = signals["decision"].value_counts()
    logger.info(f"Decision distribution:\n{dist.to_string()}")
    logger.info(
        f"Intervention rate: {(signals['decision'] == 'INTERVENE').mean() * 100:.1f}% | "
        f"Lost rate: {(signals['decision'] == 'LOST').mean() * 100:.1f}% | "
        f"Wait rate: {(signals['decision'] == 'WAIT').mean() * 100:.1f}%"
    )

    return signals[
        ["CustomerID", "hazard_now", "survival_now", "evi", "decision", "optimal_window_days"]
    ]


def rfm_intervention_decisions(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate intervention decisions from RFM segmentation (baseline policy).
    Used for comparison against the Weibull AFT policy.

    Rule:
      - At Risk  → INTERVENE
      - Lost     → LOST
      - Loyal    → WAIT
      - Champions → WAIT

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Output of models.rfm_segment() with RFM_Segment column.

    Returns
    -------
    pd.DataFrame
        Decision table with columns: [CustomerID, RFM_Segment, decision]
    """
    segment_to_decision = {
        "At Risk":   "INTERVENE",
        "Lost":      "LOST",
        "Loyal":     "WAIT",
        "Champions": "WAIT",
    }
    df = rfm_df.copy().reset_index()
    df["decision"] = df["RFM_Segment"].map(segment_to_decision)
    return df[["CustomerID", "RFM_Segment", "decision"]]
