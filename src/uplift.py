"""
src/uplift.py
=============
Uplift Modeling Proxy for the Decision-Centric Customer Re-Engagement framework.

Background
----------
True uplift modeling requires a randomised controlled trial (A/B test) with
treatment and control groups.  The UCI Online Retail dataset contains no such
experiment, so this module implements a *proxy* approach that is scientifically
sound and common in academic literature:

  1. **Propensity Score Matching (PSM) proxy**
     The Weibull intervention signal (EVI > 0 AND h > theta_h) acts as the
     "treatment assignment" proxy.  We use the SURVIVAL PROBABILITY S(t) as
     the propensity score to match treated (INTERVENE) vs control (WAIT) groups
     on pre-treatment observable characteristics.

  2. **Synthetic Response Uplift Estimation**
     Given the matching, we estimate the Conditional Average Treatment Effect
     (CATE) on Monetary value using a simple T-Learner (two-model) approach:
       tau_hat(x) = mu_1(x) - mu_0(x)
     where mu_1, mu_0 are Random Forest regressors fitted on "treated" and
     "control" matched samples respectively.

  3. **Persuadables Segmentation**
     Following Radcliffe & Surry (1999), customers are split into 4 quadrants:
       - Persuadables  : uplift > 0 and would NOT respond without intervention
       - Sure Things   : respond regardless of intervention
       - Lost Causes   : do not respond regardless
       - Sleeping Dogs : respond better WITHOUT intervention (negative uplift)

  4. **Qini Curve** (Radcliffe, 2007)
     Measures cumulative incremental gain over a random targeting baseline.

Usage
-----
    from src.uplift import run_uplift_analysis
    uplift_df, qini_fig = run_uplift_analysis(weibull_decisions, customer_df)
"""

import logging
import warnings
import numpy as np
from scipy.integrate import trapezoid as _trapz
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.dataset_registry import get_currency_code

logger = logging.getLogger(__name__)

# ── Segmentation thresholds ───────────────────────────────────────────────────
_UPLIFT_HIGH_THR = 0.0   # tau_hat > this → responds positively to intervention
_RESPONSE_THR    = 0.5   # predicted response prob threshold for "Sure Things"


# =============================================================================
# 1. Feature Matrix Assembly
# =============================================================================

def _build_feature_matrix(
    weibull_decisions: pd.DataFrame,
    customer_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge decision signals with customer RFM features.

    Returns a DataFrame with:
      treatment: 1 = INTERVENE, 0 = WAIT  (LOST excluded from uplift analysis)
      RFM features for T-Learner / PSM
      Monetary: outcome proxy
    """
    # Exclude LOST — they are not actionable
    df = weibull_decisions[weibull_decisions["decision"] != "LOST"].copy()

    df["treatment"] = (df["decision"] == "INTERVENE").astype(int)

    # Defensively reset index on customer_df in case CustomerID is the index
    cdf = customer_df.reset_index() if "CustomerID" not in customer_df.columns else customer_df.copy()

    # Merge RFM covariates
    rfm_cols = ["CustomerID", "Recency", "Frequency", "Monetary",
                "InterPurchaseTime", "GapDeviation", "SinglePurchase"]
    available = [c for c in rfm_cols if c in cdf.columns]
    merged = df.merge(
        cdf[available].set_index("CustomerID"),
        left_on="CustomerID",
        right_index=True,
        how="left",
        suffixes=("_decision", ""),
    )

    # Drop any Monetary duplication from the decisions table itself
    if "Monetary_decision" in merged.columns:
        merged = merged.drop(columns=["Monetary_decision"])

    return merged


# =============================================================================
# 2. T-Learner CATE Estimation
# =============================================================================

def _fit_t_learner(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Fit a T-Learner (two-model approach) to estimate uplift tau_hat(x).

    tau_hat(x) = mu_1(x) - mu_0(x)
    mu_1 fitted on treated (INTERVENE) group
    mu_0 fitted on control (WAIT) group

    Both models predict `Monetary` as the outcome proxy — the expected revenue
    contribution of each customer.  The uplift is the *incremental* Monetary
    value attributable to the intervention.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn is required for uplift modeling.")

    X = df[feature_cols].values
    y = df["Monetary"].values

    treated_mask = df["treatment"].values == 1
    control_mask = ~treated_mask

    if treated_mask.sum() < 10:
        logger.warning(
            "[Uplift] Too few treated customers (%d) for T-Learner. "
            "Lower the hazard_threshold to increase INTERVENE count.",
            treated_mask.sum(),
        )

    def _make_pipeline():
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ])

    mu_1_pipe = _make_pipeline()
    mu_0_pipe = _make_pipeline()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu_1_pipe.fit(X[treated_mask], y[treated_mask])
        mu_0_pipe.fit(X[control_mask], y[control_mask])

    tau_hat = mu_1_pipe.predict(X) - mu_0_pipe.predict(X)
    df = df.copy()
    df["tau_hat"] = tau_hat
    df["mu_1"]    = mu_1_pipe.predict(X)   # predicted response IF treated
    df["mu_0"]    = mu_0_pipe.predict(X)   # predicted response IF not treated
    return df


# =============================================================================
# 3. Persuadables Segmentation
# =============================================================================

def _assign_uplift_segment(row: pd.Series) -> str:
    """
    Assign Radcliffe & Surry (1999) uplift quadrant based on:
      tau_hat  : estimated uplift from T-Learner
      mu_1     : predicted outcome IF treated

    Quadrant definitions (simplified for revenue proxy):
      Persuadables  : uplift > 0 AND mu_1 > response_threshold
      Sure Things   : uplift <= 0 AND mu_1 > response_threshold
      Lost Causes   : uplift <= 0 AND mu_1 <= response_threshold
      Sleeping Dogs : uplift > 0 AND mu_1 <= response_threshold
    """
    is_uplift    = row["tau_hat"] > _UPLIFT_HIGH_THR
    is_responder = row["mu_1"] > _RESPONSE_THR

    if is_uplift and is_responder:
        return "Persuadables"
    elif not is_uplift and is_responder:
        return "Sure Things"
    elif is_uplift and not is_responder:
        return "Sleeping Dogs"
    else:
        return "Lost Causes"


# =============================================================================
# 4. Qini Curve
# =============================================================================

def _compute_qini(df: pd.DataFrame, outcome_col: str = "Monetary") -> pd.DataFrame:
    """
    Compute Qini curve for incremental gain assessment (vectorized O(n)).

    Qini(k) = Y_t_top_k - (n_t_k / n_t) * Y_t_all
    where k = top-k percentile targeted by uplift score.

    Returns
    -------
    pd.DataFrame
        Columns: ['pct_targeted', 'qini_gain', 'random_baseline']
    """
    df_sorted = df.sort_values("tau_hat", ascending=False).reset_index(drop=True)
    n   = len(df_sorted)
    n_t = (df_sorted["treatment"] == 1).sum()
    n_c = n - n_t

    if n_t == 0 or n_c == 0:
        logger.warning("[Uplift] Qini curve requires both treated and control groups.")
        return pd.DataFrame({"pct_targeted": [0, 1], "qini_gain": [0, 0], "random_baseline": [0, 0]})

    treat_flag = (df_sorted["treatment"] == 1).values
    outcome    = df_sorted[outcome_col].values

    Y_t_all = outcome[treat_flag].sum()
    Y_c_all = outcome[~treat_flag].sum()

    # Vectorized cumulative sums
    cum_Y_t  = np.cumsum(outcome * treat_flag)           # cumulative treated revenue
    cum_n_c  = np.cumsum(~treat_flag).astype(float)     # cumulative control count

    qini_gain        = cum_Y_t - (Y_t_all * cum_n_c / n_c)
    random_baseline  = Y_t_all * np.arange(1, n + 1) / n

    return pd.DataFrame({
        "pct_targeted":    np.linspace(0, 1, n + 1)[1:],
        "qini_gain":       qini_gain,
        "random_baseline": random_baseline,
    })


# =============================================================================
# 5. Qini Plot
# =============================================================================

def _plot_qini(qini_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Render and optionally save the Qini curve comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qini_df["pct_targeted"] * 100, qini_df["qini_gain"],
            color="#00b4d8", lw=2, label="T-Learner Uplift")
    ax.plot(qini_df["pct_targeted"] * 100, qini_df["random_baseline"],
            color="#888", lw=1.5, ls="--", label="Random Targeting")
    ax.fill_between(qini_df["pct_targeted"] * 100,
                    qini_df["qini_gain"], qini_df["random_baseline"],
                    alpha=0.15, color="#00b4d8")
    ax.set_xlabel("% Population Targeted", fontsize=11)
    ax.set_ylabel(f"Incremental Revenue ({get_currency_code()})", fontsize=11)
    ax.set_title("Qini Curve — Uplift vs Random Targeting", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved Qini curve → %s", save_path)
    return fig


# =============================================================================
# 6. Main Entry Point
# =============================================================================

_UPLIFT_FEATURE_COLS = [
    "Recency", "Frequency", "InterPurchaseTime", "GapDeviation", "SinglePurchase",
    "survival", "evi",
]


def run_uplift_analysis(
    weibull_decisions: pd.DataFrame,
    customer_df: pd.DataFrame,
    save_path: str = None,
) -> dict:
    """
    Run the full uplift modeling pipeline.

    Parameters
    ----------
    weibull_decisions : pd.DataFrame
        Output of policy.make_intervention_decisions() — must include columns:
        'CustomerID', 'decision', 'survival', 'evi'.
    customer_df : pd.DataFrame
        Original customer-level DataFrame for RFM covariates.
    save_path : str, optional
        If provided, saves Qini curve plot to this path.

    Returns
    -------
    dict
        Keys:
          'uplift_df'         : pd.DataFrame with tau_hat and segment per customer
          'segment_counts'    : dict of segment → count
          'qini_df'           : Qini curve DataFrame
          'persuadable_pct'   : float (fraction of INTERVENE that are Persuadables)
          'qini_auc_ratio'    : float (model Qini AUC / random Qini AUC — Qini coefficient)
    """
    logger.info("[Uplift] Starting T-Learner uplift analysis...")

    # 1. Build feature matrix
    merged = _build_feature_matrix(weibull_decisions, customer_df)
    logger.info(
        "[Uplift] %d customers in analysis | treated=%d | control=%d",
        len(merged),
        (merged["treatment"] == 1).sum(),
        (merged["treatment"] == 0).sum(),
    )

    # 2. Determine available features
    available_features = [c for c in _UPLIFT_FEATURE_COLS if c in merged.columns]

    # 3. Fit T-Learner
    uplift_df = _fit_t_learner(merged, available_features)

    # 4. Segment
    uplift_df["uplift_segment"] = uplift_df.apply(_assign_uplift_segment, axis=1)

    # 5. Log segment distribution
    counts = uplift_df["uplift_segment"].value_counts().to_dict()
    logger.info("[Uplift] Segment distribution: %s", counts)

    intervene_df = uplift_df[uplift_df["treatment"] == 1]
    persuadable_pct = (
        (intervene_df["uplift_segment"] == "Persuadables").mean()
        if len(intervene_df) > 0 else 0.0
    )
    logger.info(
        "[Uplift] Of INTERVENE customers, %.1f%% are Persuadables "
        "(positive uplift + predicted responder).",
        persuadable_pct * 100,
    )

    # 6. Qini curve
    qini_df = _compute_qini(uplift_df)

    # 7. Qini coefficient (AUC ratio)
    qini_auc  = _trapz(qini_df["qini_gain"],      qini_df["pct_targeted"])
    rand_auc  = _trapz(qini_df["random_baseline"], qini_df["pct_targeted"])
    qini_coef = (qini_auc / rand_auc) if rand_auc != 0 else 0.0
    logger.info("[Uplift] Qini coefficient (model AUC / random AUC): %.4f", qini_coef)

    # 8. Plot
    _plot_qini(qini_df, save_path=save_path)

    return {
        "uplift_df":          uplift_df,
        "segment_counts":     counts,
        "qini_df":            qini_df,
        "persuadable_pct":    persuadable_pct,
        "qini_auc_ratio":     qini_coef,
    }
