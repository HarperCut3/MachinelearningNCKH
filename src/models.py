"""
src/models.py
=============
Implements and trains all four models in the Decision-Centric framework:

  1. WeibullAFTFitter  (lifelines) -- Primary survival model
  2. CoxPHFitter       (lifelines) -- Semi-parametric baseline
  3. LogisticRegression (sklearn)  -- Binary classification baseline
  4. RFM Segmentation  (custom)    -- Heuristic quintile-based baseline

Mathematical background:
  Weibull AFT: log(T) = beta'x + sigma*epsilon,  epsilon ~ Gumbel(0,1)
    S(t|x) = exp(-(t/lambda(x))^rho),  lambda(x) = exp(beta'x),  rho = 1/sigma
    h(t|x) = (rho/lambda(x)) * (t/lambda(x))^(rho-1)

  CoxPH: h(t|x) = h0(t) * exp(beta'x)   [partial likelihood, no dist. assumption]

  Logistic: P(E=1 | x) = sigma(beta'x)   [binary horizon-based label]
    NOTE: Recency is EXCLUDED to prevent data leakage (E = Recency > tau)

  RFM Score: Quintile rank on Recency (inverted), Frequency, Monetary
"""

import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lifelines import WeibullAFTFitter, CoxPHFitter

logger = logging.getLogger(__name__)

# Feature columns used by survival models (Weibull AFT, CoxPH)
SURVIVAL_FEATURES = [
    "Recency", "Frequency", "Monetary",
    "InterPurchaseTime", "GapStability", "SinglePurchase",
]

# Feature columns for Logistic Regression (Recency EXCLUDED to prevent leakage)
# E = (Recency > tau) => including Recency gives AUC = 1.0 trivially
LOGISTIC_FEATURES = [
    "Frequency", "Monetary",
    "InterPurchaseTime", "GapStability", "SinglePurchase",
]


def _get_preprocessor() -> Pipeline:
    """Return a sklearn preprocessing pipeline (impute -> scale)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# =============================================================================
# 1. Weibull AFT Model
# =============================================================================

def train_weibull_aft(
    customer_df: pd.DataFrame,
    penalizer: float = 0.01,
) -> tuple:
    """
    Fit a Weibull Accelerated Failure Time model.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level DataFrame with features + T + E columns.
    penalizer : float
        L2 regularization strength (default: 0.01).

    Returns
    -------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model.
    df_scaled : pd.DataFrame
        Scaled feature DataFrame used for fitting (preserves T, E).
    preprocessor : Pipeline
        Fitted sklearn preprocessing pipeline.
    """
    logger.info("Training Weibull AFT model...")

    preprocessor = _get_preprocessor()
    X_scaled = preprocessor.fit_transform(customer_df[SURVIVAL_FEATURES])
    df_scaled = pd.DataFrame(X_scaled, columns=SURVIVAL_FEATURES, index=customer_df.index)
    df_scaled["T"] = customer_df["T"].values
    df_scaled["E"] = customer_df["E"].values

    waf = WeibullAFTFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waf.fit(df_scaled, duration_col="T", event_col="E")

    rho_val = waf.params_["rho_"]["Intercept"]
    logger.info(f"Weibull AFT fitted | rho = {rho_val:.4f}")
    logger.info(f"  Shape param rho > 1 -> increasing hazard over time: {rho_val > 1}")
    return waf, df_scaled, preprocessor


# =============================================================================
# 2. Cox Proportional Hazards Model
# =============================================================================

def train_coxph(
    customer_df: pd.DataFrame,
    penalizer: float = 0.1,
    check_assumptions: bool = True,
) -> tuple:
    """
    Fit a Cox Proportional Hazards model.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level DataFrame with features + T + E columns.
    penalizer : float
        Ridge regularization (default: 0.1).
    check_assumptions : bool
        If True, run Schoenfeld residuals test for PH assumption.

    Returns
    -------
    cph : CoxPHFitter
        Fitted CoxPH model.
    df_scaled : pd.DataFrame
        Scaled feature DataFrame used for fitting.
    preprocessor : Pipeline
        Fitted sklearn preprocessing pipeline.
    """
    logger.info("Training Cox Proportional Hazards model...")

    preprocessor = _get_preprocessor()
    X_scaled = preprocessor.fit_transform(customer_df[SURVIVAL_FEATURES])
    df_scaled = pd.DataFrame(X_scaled, columns=SURVIVAL_FEATURES, index=customer_df.index)
    df_scaled["T"] = customer_df["T"].values
    df_scaled["E"] = customer_df["E"].values

    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df_scaled, duration_col="T", event_col="E", show_progress=False)

    logger.info("CoxPH fitted.")
    logger.info(f"\n{cph.summary[['coef', 'exp(coef)', 'p']].to_string()}")

    if check_assumptions:
        logger.info("Running Schoenfeld residuals test (PH assumption check)...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.check_assumptions(df_scaled, p_value_threshold=0.05, show_plots=False)
        except Exception as e:
            logger.warning(f"Assumption check raised: {e}")

    return cph, df_scaled, preprocessor


# =============================================================================
# 3. Logistic Regression Baseline (Data-Leakage-Free)
# =============================================================================

def train_logistic(
    customer_df: pd.DataFrame,
    cv_folds: int = 5,
) -> tuple:
    """
    Train a Logistic Regression binary classifier as a baseline.
    Target: E (1 = churned within tau days, 0 = still active).

    DATA LEAKAGE FIX:
    -----------------
    Recency is EXCLUDED from the Logistic Regression feature set.
    Rationale: The churn label E is defined as (Recency > tau), making
    Recency a tautological predictor that inflates AUC to 1.0.

    Survival models (Weibull AFT, CoxPH) are exempt from this rule because
    they model the full time-to-event distribution T, not a binary label
    derived from Recency.

    Logistic features: Frequency, Monetary, InterPurchaseTime,
                       GapStability, SinglePurchase
    (Recency deliberately excluded to prevent data leakage)

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level DataFrame with features + E column.
    cv_folds : int
        Number of stratified cross-validation folds (default: 5).

    Returns
    -------
    lr : LogisticRegression
        Fitted logistic regression model (on full data).
    pipeline : Pipeline
        Full preprocessing + model pipeline.
    cv_metrics : dict
        Cross-validated AUC and accuracy scores.
    """
    logger.info("Training Logistic Regression baseline (Recency excluded to prevent leakage)...")
    logger.info(f"  Logistic features: {LOGISTIC_FEATURES}")

    X = customer_df[LOGISTIC_FEATURES].values
    y = customer_df["E"].values

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    cv_metrics = {
        "auc_mean":  cv_auc.mean(),
        "auc_std":   cv_auc.std(),
        "acc_mean":  cv_acc.mean(),
        "acc_std":   cv_acc.std(),
        "features":  LOGISTIC_FEATURES,
    }

    logger.info(
        f"Logistic CV AUC (no Recency): {cv_metrics['auc_mean']:.4f} +/- {cv_metrics['auc_std']:.4f} | "
        f"Accuracy: {cv_metrics['acc_mean']:.4f} +/- {cv_metrics['acc_std']:.4f}"
    )

    # Fit on full dataset for downstream use
    pipeline.fit(X, y)
    lr = pipeline.named_steps["lr"]

    return lr, pipeline, cv_metrics


# =============================================================================
# 4. RFM Segmentation Baseline
# =============================================================================

def rfm_segment(customer_df: pd.DataFrame, n_quintiles: int = 5) -> pd.DataFrame:
    """
    Compute RFM quintile scores and assign customer segments.

    Scoring logic:
      - Recency:   lower is better -> inverted quintile (5 = most recent)
      - Frequency: higher is better -> normal quintile (5 = most frequent)
      - Monetary:  higher is better -> normal quintile (5 = highest spend)
      - RFM_Score = R_score + F_score + M_score  (range: 3-15)

    Segments (based on RFM_Score):
      - Champions : 13-15
      - Loyal     : 10-12
      - At Risk   : 7-9
      - Lost      : 3-6

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level DataFrame with Recency, Frequency, Monetary columns.
    n_quintiles : int
        Number of quantile bins (default: 5).

    Returns
    -------
    pd.DataFrame
        customer_df with added columns: R_score, F_score, M_score,
        RFM_Score, RFM_Segment, intervention_priority (1=highest).
    """
    logger.info("Computing RFM segmentation...")

    df = customer_df.copy()
    labels = list(range(1, n_quintiles + 1))

    # Recency: lower recency -> higher score (more recent = better)
    df["R_score"] = pd.qcut(
        df["Recency"], q=n_quintiles, labels=labels[::-1], duplicates="drop"
    ).astype(int)

    # Frequency: higher frequency -> higher score
    df["F_score"] = pd.qcut(
        df["Frequency"].rank(method="first"), q=n_quintiles, labels=labels, duplicates="drop"
    ).astype(int)

    # Monetary: higher spend -> higher score
    df["M_score"] = pd.qcut(
        df["Monetary"].rank(method="first"), q=n_quintiles, labels=labels, duplicates="drop"
    ).astype(int)

    df["RFM_Score"] = df["R_score"] + df["F_score"] + df["M_score"]

    def _assign_segment(score: int) -> str:
        if score >= 13:
            return "Champions"
        elif score >= 10:
            return "Loyal"
        elif score >= 7:
            return "At Risk"
        else:
            return "Lost"

    df["RFM_Segment"] = df["RFM_Score"].apply(_assign_segment)

    # Intervention priority: "At Risk" customers are the primary targets
    priority_map = {"At Risk": 1, "Lost": 2, "Loyal": 3, "Champions": 4}
    df["intervention_priority"] = df["RFM_Segment"].map(priority_map)

    segment_counts = df["RFM_Segment"].value_counts()
    logger.info(f"RFM Segments:\n{segment_counts.to_string()}")

    return df
