"""
main.py
=======
Pipeline Orchestrator -- Decision-Centric Customer Re-Engagement
================================================================
Runs the full end-to-end pipeline:

  1. Load & clean UCI Online Retail data
  2. Engineer customer-level RFM + Survival features
  3. Train Weibull AFT, CoxPH, Logistic Regression, RFM models
  4. Apply intervention policy (Weibull + RFM baseline)
  5. Evaluate: C-index, IBS, AUC, Outreach Efficiency, Revenue Lift
  6. Serialize models and processed data for dashboard
  7. Generate all publication-quality figures
  8. Export decision table to CSV

Usage:
  python main.py
  python main.py --tau 60
  python main.py --tau 120 --no-shap
  python main.py --sensitivity
"""

import os
import sys
import argparse
import logging
import warnings
import joblib
import numpy as np
import pandas as pd

# Suppress verbose third-party warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader    import load_and_clean, get_snapshot_date
from src.feature_engine import build_customer_features, sensitivity_analysis_tau
from src.models         import (
    train_weibull_aft, train_coxph, train_logistic, rfm_segment,
    SURVIVAL_FEATURES,
)
from src.policy         import make_intervention_decisions, rfm_intervention_decisions
from src.evaluation     import (
    compute_c_index,
    compute_integrated_brier_score,
    compute_time_dependent_auc,
    compute_outreach_efficiency,
    compute_revenue_lift,
    print_full_report,
)
from src.visualization  import (
    plot_kaplan_meier_by_segment,
    plot_weibull_survival_curves,
    plot_hazard_trajectories,
    plot_shap_summary,
    plot_decision_distribution,
    plot_brier_score_over_time,
)

# Configuration
DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Online Retail.xlsx")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reports")
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
LOG_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "pipeline.log")


def setup_logging():
    """Configure logging to both stdout and file."""
    os.makedirs("outputs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decision-Centric Customer Re-Engagement Pipeline"
    )
    parser.add_argument(
        "--tau", type=int, default=90,
        help="Inactivity threshold in days to define churn (default: 90)"
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Skip SHAP computation (faster run for testing)"
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run sensitivity analysis across tau in {60, 90, 120}"
    )
    return parser.parse_args()


def save_artifacts(
    waf,
    preprocessor,
    customer_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    weibull_decisions: pd.DataFrame,
    df_scaled_waf: pd.DataFrame,
    tau: int,
) -> None:
    """
    Serialize trained models and processed data for the Streamlit dashboard.

    Saves to outputs/models/:
      - weibull_model.pkl       : Fitted WeibullAFTFitter
      - preprocessor.pkl        : Fitted sklearn preprocessing pipeline
      - processed_data.pkl      : Customer-level DataFrame (features + T, E, RFM)
      - decisions.pkl           : Intervention decision table
      - df_scaled.pkl           : Scaled feature DataFrame (for model input)
      - pipeline_meta.pkl       : Metadata dict (tau, snapshot, feature names)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger = logging.getLogger("main")

    # Reset index so CustomerID becomes a column (it's the index in customer_df)
    merged = customer_df.reset_index()  # CustomerID index -> column
    for col in ["R_score", "F_score", "M_score", "RFM_Score", "RFM_Segment", "intervention_priority"]:
        if col in rfm_df.columns:
            merged[col] = rfm_df[col].values

    # Add decision and EVI columns from weibull_decisions
    if "CustomerID" in weibull_decisions.columns:
        decision_map = weibull_decisions.set_index("CustomerID")["decision"].to_dict()
        evi_map      = weibull_decisions.set_index("CustomerID")["evi"].to_dict()
        merged["decision"] = merged["CustomerID"].map(decision_map)
        merged["evi"]      = merged["CustomerID"].map(evi_map)

    meta = {
        "tau":               tau,
        "survival_features": SURVIVAL_FEATURES,
        "n_customers":       len(customer_df),
        "churn_rate":        customer_df["E"].mean(),
    }

    artifacts = {
        "weibull_model.pkl":  waf,
        "preprocessor.pkl":   preprocessor,
        "processed_data.pkl": merged,
        "decisions.pkl":      weibull_decisions,
        "df_scaled.pkl":      df_scaled_waf,
        "pipeline_meta.pkl":  meta,
    }

    for filename, obj in artifacts.items():
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(obj, path)
        logger.info(f"  Saved -> {path}")

    logger.info(f"All artifacts saved to {MODELS_DIR}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  DECISION-CENTRIC CUSTOMER RE-ENGAGEMENT PIPELINE")
    logger.info(f"  tau = {args.tau} days | SHAP = {not args.no_shap}")
    logger.info("=" * 70)

    # ── STEP 1: Load & Clean Data ─────────────────────────────────────────────
    logger.info("\n[STEP 1] Loading and cleaning dataset...")
    df_clean = load_and_clean(DATA_PATH)
    snapshot = get_snapshot_date(df_clean)

    # ── STEP 2: Feature Engineering ───────────────────────────────────────────
    logger.info(f"\n[STEP 2] Engineering customer features (tau={args.tau}d)...")
    customer_df = build_customer_features(df_clean, snapshot, tau=args.tau)

    if args.sensitivity:
        logger.info("\n[STEP 2b] Running sensitivity analysis across tau in {60, 90, 120}...")
        sensitivity_results = sensitivity_analysis_tau(df_clean, snapshot)
        for tau_val, cdf in sensitivity_results.items():
            churn_rate = cdf["E"].mean() * 100
            logger.info(f"  tau={tau_val}d -> churn rate: {churn_rate:.1f}%")

    # ── STEP 3: RFM Segmentation (Baseline) ──────────────────────────────────
    logger.info("\n[STEP 3] Computing RFM segmentation...")
    rfm_df = rfm_segment(customer_df)

    # ── STEP 4: Train Models ──────────────────────────────────────────────────
    logger.info("\n[STEP 4] Training survival models...")

    waf, df_scaled_waf, preprocessor_waf = train_weibull_aft(customer_df)
    cph, df_scaled_cph, preprocessor_cph = train_coxph(customer_df)
    lr, lr_pipeline, lr_cv_metrics       = train_logistic(customer_df)

    # ── STEP 5: Intervention Policy ───────────────────────────────────────────
    logger.info("\n[STEP 5] Applying intervention policy...")
    t_now = float(df_scaled_waf["T"].median())

    weibull_decisions = make_intervention_decisions(
        waf, df_scaled_waf, customer_df, t_now=t_now
    )
    rfm_decisions = rfm_intervention_decisions(rfm_df)

    decision_path = os.path.join(REPORTS_DIR, "intervention_decisions.csv")
    weibull_decisions.to_csv(decision_path, index=False)
    logger.info(f"Decision table saved -> {decision_path}")

    # ── STEP 6: Evaluation ────────────────────────────────────────────────────
    logger.info("\n[STEP 6] Evaluating models...")

    c_index_weibull  = compute_c_index(waf, df_scaled_waf, model_name="WeibullAFT")
    c_index_cox      = compute_c_index(cph, df_scaled_cph, model_name="CoxPH")
    ibs              = compute_integrated_brier_score(waf, df_scaled_waf)
    auc_df           = compute_time_dependent_auc(waf, df_scaled_waf)
    outreach_metrics = compute_outreach_efficiency(weibull_decisions, rfm_decisions)
    revenue_metrics  = compute_revenue_lift(weibull_decisions, rfm_decisions)

    print_full_report(
        c_index_weibull=c_index_weibull,
        c_index_cox=c_index_cox,
        ibs=ibs,
        lr_cv_metrics=lr_cv_metrics,
        auc_df=auc_df,
        outreach_metrics=outreach_metrics,
        revenue_metrics=revenue_metrics,
        tau=args.tau,
    )

    # ── STEP 7: Serialize Models & Data for Dashboard ─────────────────────────
    logger.info("\n[STEP 7] Serializing models and processed data...")
    save_artifacts(
        waf=waf,
        preprocessor=preprocessor_waf,
        customer_df=customer_df,
        rfm_df=rfm_df,
        weibull_decisions=weibull_decisions,
        df_scaled_waf=df_scaled_waf,
        tau=args.tau,
    )

    # ── STEP 8: Visualizations ────────────────────────────────────────────────
    logger.info("\n[STEP 8] Generating figures...")

    try:
        plot_kaplan_meier_by_segment(
            customer_df, rfm_df,
            save_path=os.path.join(FIGURES_DIR, "01_kaplan_meier_by_segment.png")
        )
    except Exception as e:
        logger.warning(f"KM plot failed: {e}")

    try:
        plot_weibull_survival_curves(
            waf, df_scaled_waf,
            save_path=os.path.join(FIGURES_DIR, "02_weibull_survival_curves.png")
        )
    except Exception as e:
        logger.warning(f"Weibull survival plot failed: {e}")

    try:
        plot_hazard_trajectories(
            waf, df_scaled_waf, rfm_df,
            save_path=os.path.join(FIGURES_DIR, "03_hazard_trajectories.png")
        )
    except Exception as e:
        logger.warning(f"Hazard trajectory plot failed: {e}")

    try:
        plot_decision_distribution(
            weibull_decisions, rfm_decisions,
            save_path=os.path.join(FIGURES_DIR, "04_decision_distribution.png")
        )
    except Exception as e:
        logger.warning(f"Decision distribution plot failed: {e}")

    try:
        plot_brier_score_over_time(
            waf, df_scaled_waf,
            save_path=os.path.join(FIGURES_DIR, "05_brier_score_over_time.png")
        )
    except Exception as e:
        logger.warning(f"Brier score plot failed: {e}")

    if not args.no_shap:
        try:
            plot_shap_summary(
                waf, df_scaled_waf,
                feature_cols=SURVIVAL_FEATURES,
                save_path=os.path.join(FIGURES_DIR, "06_shap_summary.png")
            )
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")

    logger.info("\n[DONE] Pipeline completed successfully.")
    logger.info(f"  Figures  -> {FIGURES_DIR}")
    logger.info(f"  Reports  -> {REPORTS_DIR}")
    logger.info(f"  Models   -> {MODELS_DIR}")
    logger.info(f"  Log      -> {LOG_PATH}")
    logger.info("\nTo launch the dashboard, run:")
    logger.info("  streamlit run app.py")


if __name__ == "__main__":
    main()
