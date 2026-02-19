"""
app.py
======
Streamlit Decision Engine Dashboard
====================================
Interactive dashboard for the Decision-Centric Customer Re-Engagement project.

Features:
  - Customer lookup by ID
  - Real-time survival S(t) and hazard h(t) computation
  - INTERVENE / WAIT / LOST recommendation with EVI
  - Individual survival and hazard curves with current-recency marker
  - Portfolio-level metrics and segment distribution
  - 🧠 Explainable AI (SHAP Risk Breakdown) per customer
  - 📊 Monte Carlo Policy Simulation confidence intervals

Usage:
  streamlit run app.py

Prerequisites:
  Run the pipeline first to generate model artifacts:
    python main.py --no-shap
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Engine | Customer Re-Engagement",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1d27; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 0.75rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #ccd6f6; }
    .metric-unit  { font-size: 0.8rem; color: #8892b0; margin-top: 2px; }

    /* Decision banners */
    .decision-intervene {
        background: linear-gradient(135deg, #3d0000, #6b0000);
        border: 2px solid #ff4444;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 30px rgba(255,68,68,0.3);
    }
    .decision-wait {
        background: linear-gradient(135deg, #003d1a, #006b2e);
        border: 2px solid #00cc66;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0,204,102,0.3);
    }
    .decision-lost {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #888;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 20px rgba(136,136,136,0.2);
    }
    .decision-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 8px; }
    .decision-subtitle { font-size: 0.95rem; opacity: 0.85; }

    /* EVI box */
    .evi-box {
        background: linear-gradient(135deg, #0d2137, #0a3d62);
        border: 1px solid #1a6fa8;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 16px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #64ffda;
        border-bottom: 1px solid #2e3250;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* CI box */
    .ci-box {
        background: linear-gradient(135deg, #0d1f2d, #0a2e40);
        border: 1px solid #1a5f7a;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .ci-label { font-size: 0.7rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .ci-median { font-size: 1.6rem; font-weight: 700; color: #64ffda; }
    .ci-range  { font-size: 0.75rem; color: #8892b0; margin-top: 2px; }

    /* Streamlit metric override */
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ─────────────────────────────────────────────────────

def get_available_runs() -> list:
    """Scan outputs/ directory for available run folders (e.g. UCI_tau90)."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    if not os.path.exists(base_dir):
        return []
    
    runs = []
    try:
        for d in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, d)) and "tau" in d:
                runs.append(d)
    except Exception:
        pass
        
    return sorted(runs, reverse=True)  # Newest (by name if timestamped) or Alphabetical


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts(run_dir_name: str):
    """Load all serialized artifacts from outputs/{run_dir_name}/models/."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_path, "outputs", run_dir_name, "models")
    
    required = [
        "weibull_model.pkl", "preprocessor.pkl",
        "processed_data.pkl", "pipeline_meta.pkl",
    ]
    
    if not os.path.exists(models_dir):
        return None, None, None, None, None, f"Models directory not found: {models_dir}"

    missing = [f for f in required if not os.path.exists(os.path.join(models_dir, f))]
    if missing:
        return None, None, None, None, None, f"Missing model files in {run_dir_name}: {missing}"

    try:
        waf        = joblib.load(os.path.join(models_dir, "weibull_model.pkl"))
        prep       = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
        data       = joblib.load(os.path.join(models_dir, "processed_data.pkl"))
        meta       = joblib.load(os.path.join(models_dir, "pipeline_meta.pkl"))

        # Load scaled training data for SHAP background (optional)
        df_scaled_path = os.path.join(models_dir, "df_scaled.pkl")
        df_scaled = joblib.load(df_scaled_path) if os.path.exists(df_scaled_path) else None

        return waf, prep, data, meta, df_scaled, None
    except Exception as e:
        return None, None, None, None, None, str(e)



# ── Policy Engine ─────────────────────────────────────────────────────────────
def compute_customer_risk(
    waf,
    prep,
    customer_row: pd.Series,
    survival_features: list,
    t_eval: float,
    hazard_threshold: float = 0.015,
    p_response: float = 0.12,
    contact_cost: float = 1.0,
) -> dict:
    """
    Compute survival probability, hazard rate, EVI, and decision for one customer.

    Parameters
    ----------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model.
    prep : Pipeline
        Fitted sklearn preprocessor.
    customer_row : pd.Series
        Single customer's raw features.
    survival_features : list
        Feature column names the model was trained on (VIF-pruned).
    t_eval : float
        Time point to evaluate (customer's current Recency in days).
    hazard_threshold : float
        Hazard rate above which we consider intervention.
    p_response : float
        Assumed email response rate (default: 12%).
    contact_cost : float
        Cost per outreach contact in GBP (default: 1.0).

    Returns
    -------
    dict with keys: survival, hazard, evi, decision, t_eval, t_grid, S_curve, h_curve
    """
    # Prepare scaled input — strictly match the active_features list
    X_raw    = customer_row[survival_features].values.reshape(1, -1)
    X_scaled = prep.transform(X_raw)
    df_input = pd.DataFrame(X_scaled, columns=survival_features)

    # Time grid for curves (0 to 400 days)
    t_max  = 400
    t_grid = np.linspace(1, t_max, 300)

    # Survival function S(t)
    S_matrix = waf.predict_survival_function(df_input, times=t_grid).values  # (T, 1)
    S_curve  = S_matrix[:, 0]

    # Hazard function h(t) = -dS/dt / S(t)  [numerical differentiation]
    dt = t_grid[1] - t_grid[0]
    dS = np.gradient(S_curve, dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_curve = np.where(S_curve > 1e-10, -dS / S_curve, 0.0)
    h_curve = np.clip(h_curve, 0, None)

    # Point estimates at t_eval
    t_idx = np.searchsorted(t_grid, t_eval)
    t_idx = min(t_idx, len(t_grid) - 1)
    S_t   = float(S_curve[t_idx])
    h_t   = float(h_curve[t_idx])

    # EVI = p_response * Monetary * (1 - S(t)) - C_contact
    monetary = float(customer_row.get("Monetary", 0))
    evi = p_response * monetary * (1.0 - S_t) - contact_cost

    # Decision logic
    if S_t < 0.05:
        decision = "LOST"
    elif h_t > hazard_threshold and evi > 0:
        decision = "INTERVENE"
    else:
        decision = "WAIT"

    return {
        "survival":  S_t,
        "hazard":    h_t,
        "evi":       evi,
        "decision":  decision,
        "t_eval":    t_eval,
        "t_grid":    t_grid,
        "S_curve":   S_curve,
        "h_curve":   h_curve,
    }


def _build_shap_section(
    waf,
    prep,
    customer_row: pd.Series,
    survival_features: list,
    df_scaled_bg,
):
    """
    Render the SHAP Risk Breakdown expander section in Streamlit.

    Strategy: wrap WeibullAFTFitter.predict_median() as a plain callable
    so shap.Explainer can treat it as a black-box model.  SHAP values on the
    predicted median survival time (log-scale) give an interpretable
    decomposition: a high positive SHAP value for a feature means that feature
    *extended* the customer's predicted lifetime (lower risk), while negative
    SHAP values compress it (higher risk).
    """
    with st.expander("🧠 Explainable AI (SHAP Risk Breakdown)", expanded=False):
        try:
            import shap  # lazy import — optional dependency
        except ImportError:
            st.warning(
                "**SHAP not installed.** Run `pip install shap` then restart the app.",
                icon="⚠️",
            )
            return

        try:
            # ── Prepare the background dataset ──────────────────────────────
            if df_scaled_bg is not None and all(f in df_scaled_bg.columns for f in survival_features):
                # Use a sample of 100 rows from the training set as background
                bg_size  = min(100, len(df_scaled_bg))
                bg_data  = df_scaled_bg[survival_features].sample(bg_size, random_state=42).values
            else:
                # Fallback: single zero-vector background
                bg_data = np.zeros((1, len(survival_features)))

            # ── Scale the selected customer ──────────────────────────────────
            X_raw    = customer_row[survival_features].values.reshape(1, -1)
            X_scaled = prep.transform(X_raw)

            # ── Define black-box predict function ────────────────────────────
            # Output: log(predicted_median) so SHAP values are additive on the
            # log scale.  Positive SHAP ↑ → feature extends survival (lower risk).
            def _predict_log_median(X: np.ndarray) -> np.ndarray:
                df_in = pd.DataFrame(X, columns=survival_features)
                medians = waf.predict_median(df_in)
                return np.log1p(np.maximum(medians.values, 1e-6))

            with st.spinner("Computing SHAP values…"):
                explainer   = shap.Explainer(_predict_log_median, bg_data)
                shap_values = explainer(X_scaled)

            sv = shap_values.values[0]          # shape: (n_features,)
            base_val = float(shap_values.base_values[0])

            # ── Sort features by |SHAP| ──────────────────────────────────────
            order     = np.argsort(np.abs(sv))
            feat_names_sorted = [survival_features[i] for i in order]
            sv_sorted         = sv[order]
            colors            = ["#ff4444" if v < 0 else "#00cc66" for v in sv_sorted]

            st.markdown(
                "<p style='color:#8892b0; font-size:0.85rem;'>"
                "SHAP values decompose <em>why</em> this customer's predicted "
                "survival time is higher or lower. "
                "🟢 = feature <b>extends</b> predicted lifetime (lower churn risk) | "
                "🔴 = feature <b>compresses</b> lifetime (higher churn risk)."
                "</p>",
                unsafe_allow_html=True,
            )

            fig, ax = plt.subplots(figsize=(9, max(3, len(survival_features) * 0.55)))
            fig.patch.set_facecolor("#0f1117")
            ax.set_facecolor("#1a1d27")

            bars = ax.barh(feat_names_sorted, sv_sorted, color=colors, edgecolor="#0f1117", linewidth=0.8)
            ax.axvline(0, color="#8892b0", linewidth=0.8, linestyle="--")

            for bar, val in zip(bars, sv_sorted):
                x_pos  = bar.get_width() + (0.003 if val >= 0 else -0.003)
                ha_val = "left" if val >= 0 else "right"
                ax.text(
                    x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}", va="center", ha=ha_val,
                    color="#ccd6f6", fontsize=8.5,
                )

            ax.set_xlabel("SHAP value  (impact on log predicted median survival)", color="#8892b0", fontsize=9)
            ax.set_title(
                f"SHAP Feature Contributions  |  Base value = {base_val:.4f}",
                color="#ccd6f6", fontsize=10, fontweight="bold",
            )
            ax.tick_params(colors="#8892b0", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#2e3250")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Feature contribution table
            shap_df = pd.DataFrame({
                "Feature":        [survival_features[i] for i in np.argsort(np.abs(sv))[::-1]],
                "Raw Value":      [float(customer_row[survival_features[i]]) for i in np.argsort(np.abs(sv))[::-1]],
                "SHAP Value":     sorted(sv, key=abs, reverse=True),
                "Direction":      ["🔴 Higher Risk" if v < 0 else "🟢 Lower Risk"
                                   for v in sorted(sv, key=abs, reverse=True)],
            })
            st.dataframe(
                shap_df.style.format({"Raw Value": "{:.4f}", "SHAP Value": "{:+.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(f"SHAP computation failed: {exc}")
            st.info("This may happen if the model artifacts are from an older run. Re-run `python main.py --no-shap` to refresh.")


def _render_monte_carlo_section(mc: dict):
    """Render the Monte Carlo Policy Simulation results panel."""
    st.markdown("### 📊 Monte Carlo Policy Simulation — *Economic Reality Engine*")

    budget   = mc.get("marketing_budget",    500.0)
    penalty  = mc.get("sleeping_dog_penalty", 0.20)
    n_iters  = mc.get("n_iterations",        1000)
    st.markdown(
        f"<p style='color:#8892b0; font-size:0.85rem; margin-top:-8px;'>"
        f"{n_iters:,} iterations sampling uncertainty over response rate "
        f"(<i>μ=15%, σ=3%</i>) and outreach cost (<i>μ=£1.00, σ=£0.10</i>). "
        f"<b style='color:#ff9f43;'>Budget cap: £{budget:,.0f}/campaign.</b> "
        f"RFM contacts with low hazard incur a "
        f"<b style='color:#ff4444;'>{penalty:.0%} Sleeping Dog penalty</b> "
        f"(brand-damage / annoyance churn). Reported as 95% Confidence Intervals."
        f"</p>",
        unsafe_allow_html=True,
    )

    w_ci      = mc.get("weibull_profit_ci",   (0, 0, 0))
    r_ci      = mc.get("rfm_profit_ci",        (0, 0, 0))
    eg_ci     = mc.get("efficiency_gain_ci",   (0, 0, 0))
    n_w_pool  = mc.get("n_weibull_intervene",  "—")
    n_w_fund  = mc.get("n_weibull_funded",     n_w_pool)
    n_r_pool  = mc.get("n_rfm_intervene",      "—")
    n_r_fund  = mc.get("n_rfm_funded",         n_r_pool)
    n_sleep   = mc.get("n_rfm_sleeping_dogs",  "—")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"<div class='ci-box'>"
            f"<div class='ci-label'>Weibull (Precision) — Total Profit</div>"
            f"<div class='ci-median'>£{w_ci[1]:,.0f}</div>"
            f"<div class='ci-range'>95% CI: £{w_ci[0]:,.0f} — £{w_ci[2]:,.0f}</div>"
            f"<div class='ci-range' style='margin-top:6px; color:#64ffda;'>"
            f"{n_w_fund:,} funded (pool: {n_w_pool:,})</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<div class='ci-box'>"
            f"<div class='ci-label'>RFM Baseline (Penalised) — Total Profit</div>"
            f"<div class='ci-median'>£{r_ci[1]:,.0f}</div>"
            f"<div class='ci-range'>95% CI: £{r_ci[0]:,.0f} — £{r_ci[2]:,.0f}</div>"
            f"<div class='ci-range' style='margin-top:6px; color:#ff9f43;'>"
            f"{n_r_fund:,} funded | "
            f"<span style='color:#ff6b6b;'>⚠ {n_sleep:,} Sleeping Dogs</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col3:
        gain_pct_med = eg_ci[1] * 100
        gain_color   = "#00cc66" if gain_pct_med >= 0 else "#ff4444"
        st.markdown(
            f"<div class='ci-box'>"
            f"<div class='ci-label'>Weibull vs RFM — Efficiency Gain</div>"
            f"<div class='ci-median' style='color:{gain_color};'>{gain_pct_med:+.1f}%</div>"
            f"<div class='ci-range'>95% CI: {eg_ci[0]*100:+.1f}% — {eg_ci[2]*100:+.1f}%</div>"
            f"<div class='ci-range' style='margin-top:6px;'>(per £ of budget spent)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Summary callout
    st.markdown("<br>", unsafe_allow_html=True)
    desc = (
        f"Under a **£{budget:,.0f} campaign budget**, Weibull precision targeting "
        f"funds **{n_w_fund:,} high-risk customers** (EVI-ranked), earning "
        f"**£{w_ci[1]:,.0f}** median profit. "
        f"RFM funds **{n_r_fund:,} customers** but penalises **{n_sleep:,} Sleeping Dogs** "
        f"({penalty:.0%} future value loss each), yielding only **£{r_ci[1]:,.0f}**. "
        f"Median efficiency gain: **{gain_pct_med:+.1f}%**."
    )
    if gain_pct_med >= 0:
        st.success(desc)
    else:
        st.warning(desc + " Consider lowering `hazard_threshold` to expand the INTERVENE pool.")


def plot_individual_curves(result: dict, customer_id, recency: float) -> plt.Figure:
    """Plot survival and hazard curves for a single customer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f1117")

    t_grid  = result["t_grid"]
    S_curve = result["S_curve"]
    h_curve = result["h_curve"]
    t_eval  = result["t_eval"]

    # Color by decision
    color_map  = {"INTERVENE": "#ff4444", "WAIT": "#00cc66", "LOST": "#888888"}
    curve_color = color_map.get(result["decision"], "#64ffda")

    # ── Survival Curve ────────────────────────────────────────────────────────
    ax1.set_facecolor("#1a1d27")
    ax1.plot(t_grid, S_curve, color=curve_color, linewidth=2.5, label="S(t) — Survival Prob.")
    ax1.fill_between(t_grid, S_curve, alpha=0.15, color=curve_color)
    ax1.axvline(x=t_eval, color="#ffd700", linewidth=1.8, linestyle="--",
                label=f"Current Recency = {recency:.0f}d")
    ax1.axhline(y=result["survival"], color="#ffd700", linewidth=0.8, linestyle=":", alpha=0.6)
    ax1.scatter([t_eval], [result["survival"]], color="#ffd700", s=80, zorder=5)
    ax1.set_xlabel("Days", color="#8892b0", fontsize=10)
    ax1.set_ylabel("Survival Probability S(t)", color="#8892b0", fontsize=10)
    ax1.set_title(f"Customer {customer_id} — Survival Curve", color="#ccd6f6", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, 400)
    ax1.tick_params(colors="#8892b0")
    ax1.spines["bottom"].set_color("#2e3250")
    ax1.spines["left"].set_color("#2e3250")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=9, labelcolor="#ccd6f6", facecolor="#1a1d27", edgecolor="#2e3250")
    ax1.annotate(
        f"S({recency:.0f}) = {result['survival']:.3f}",
        xy=(t_eval, result["survival"]),
        xytext=(t_eval + 20, result["survival"] + 0.08),
        color="#ffd700", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.2),
    )

    # ── Hazard Curve ──────────────────────────────────────────────────────────
    ax2.set_facecolor("#1a1d27")
    ax2.plot(t_grid, h_curve, color="#ff9f43", linewidth=2.5, label="h(t) — Hazard Rate")
    ax2.fill_between(t_grid, h_curve, alpha=0.15, color="#ff9f43")
    ax2.axvline(x=t_eval, color="#ffd700", linewidth=1.8, linestyle="--",
                label=f"Current Recency = {recency:.0f}d")
    ax2.axhline(y=result["hazard"], color="#ffd700", linewidth=0.8, linestyle=":", alpha=0.6)
    ax2.scatter([t_eval], [result["hazard"]], color="#ffd700", s=80, zorder=5)
    ax2.set_xlabel("Days", color="#8892b0", fontsize=10)
    ax2.set_ylabel("Hazard Rate h(t)", color="#8892b0", fontsize=10)
    ax2.set_title(f"Customer {customer_id} — Hazard Curve", color="#ccd6f6", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 400)
    ax2.tick_params(colors="#8892b0")
    ax2.spines["bottom"].set_color("#2e3250")
    ax2.spines["left"].set_color("#2e3250")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=9, labelcolor="#ccd6f6", facecolor="#1a1d27", edgecolor="#2e3250")
    ax2.annotate(
        f"h({recency:.0f}) = {result['hazard']:.4f}",
        xy=(t_eval, result["hazard"]),
        xytext=(t_eval + 20, result["hazard"] * 1.3 + 0.001),
        color="#ffd700", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.2),
    )

    plt.tight_layout(pad=2.0)
    return fig


def plot_portfolio_overview(data: pd.DataFrame) -> plt.Figure:
    """Plot portfolio-level segment distribution and decision breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0f1117")

    # Segment distribution
    if "RFM_Segment" in data.columns:
        seg_counts = data["RFM_Segment"].value_counts()
        colors = {"Champions": "#64ffda", "Loyal": "#00cc66", "At Risk": "#ff9f43", "Lost": "#ff4444"}
        bar_colors = [colors.get(s, "#888") for s in seg_counts.index]
        ax1.set_facecolor("#1a1d27")
        bars = ax1.bar(seg_counts.index, seg_counts.values, color=bar_colors, edgecolor="#0f1117", linewidth=1.5)
        for bar, val in zip(bars, seg_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     str(val), ha="center", va="bottom", color="#ccd6f6", fontsize=10, fontweight="bold")
        ax1.set_title("RFM Segment Distribution", color="#ccd6f6", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Customers", color="#8892b0")
        ax1.tick_params(colors="#8892b0")
        ax1.spines["bottom"].set_color("#2e3250")
        ax1.spines["left"].set_color("#2e3250")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

    # Decision distribution
    if "decision" in data.columns:
        dec_counts = data["decision"].value_counts()
        dec_colors = {"INTERVENE": "#ff4444", "WAIT": "#00cc66", "LOST": "#888888"}
        pie_colors = [dec_colors.get(d, "#888") for d in dec_counts.index]
        wedges, texts, autotexts = ax2.pie(
            dec_counts.values,
            labels=dec_counts.index,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="#0f1117", linewidth=2),
        )
        for text in texts:
            text.set_color("#ccd6f6")
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color("#0f1117")
            autotext.set_fontweight("bold")
        ax2.set_facecolor("#1a1d27")
        ax2.set_title("Weibull Policy Decision Distribution", color="#ccd6f6", fontsize=11, fontweight="bold")

    plt.tight_layout(pad=2.0)
    return fig


# ── Uplift Analysis Panel ────────────────────────────────────────────────────
def _render_uplift_section(run_dir_name: str):
    """
    Render the Uplift Modeling results panel.
    Reads outputs/{run_dir_name}/reports/uplift_segments.csv.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    UPLIFT_CSV = os.path.join(base_path, "outputs", run_dir_name, "reports", "uplift_segments.csv")
    QINI_PNG   = os.path.join(base_path, "outputs", run_dir_name, "figures", "07_qini_curve.png")

    with st.expander("🎯 Uplift Modeling — Persuadables Analysis (T-Learner)", expanded=False):
        if not os.path.exists(UPLIFT_CSV):
            st.info(
                "Uplift analysis not yet run.\n\n"
                "Run: `python main.py --no-shap --uplift` to generate Persuadables data."
            )
            return

        try:
            uplift_df = pd.read_csv(UPLIFT_CSV)

        except Exception as exc:
            st.error(f"Could not load uplift_segments.csv: {exc}")
            return

        st.markdown(
            "<p style='color:#8892b0; font-size:0.85rem;'>"
            "T-Learner CATE estimation (Gradient Boosting). Weibull INTERVENE decisions "
            "used as treatment proxy. Customers split into 4 Radcliffe &amp; Surry (1999) quadrants."
            "</p>",
            unsafe_allow_html=True,
        )

        # Segment KPI cards
        seg_counts = uplift_df["uplift_segment"].value_counts().to_dict()
        seg_colors = {
            "Persuadables":  "#00cc66",
            "Sure Things":   "#4a9eff",
            "Sleeping Dogs": "#ff9f43",
            "Lost Causes":   "#888888",
        }
        total = len(uplift_df)
        c1, c2, c3, c4 = st.columns(4)
        for col, seg in zip([c1, c2, c3, c4],
                            ["Persuadables", "Sure Things", "Sleeping Dogs", "Lost Causes"]):
            n = seg_counts.get(seg, 0)
            col.markdown(
                f"<div style='background:linear-gradient(135deg,#0f1117,#1a1d27);"
                f"border:1px solid {seg_colors[seg]}; border-radius:12px; padding:16px; text-align:center;'>"
                f"<div style='color:{seg_colors[seg]}; font-size:0.75rem; font-weight:600;'>{seg.upper()}</div>"
                f"<div style='font-size:1.6rem; font-weight:700; color:#e0e0e0;'>{n:,}</div>"
                f"<div style='color:#8892b0; font-size:0.7rem;'>{n/total*100:.1f}% of portfolio</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Qini curve if available
        if os.path.exists(QINI_PNG):
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(QINI_PNG, caption="Qini Curve — Uplift vs Random Targeting",
                     use_container_width=True)

        # Top Persuadables table
        persuadables = uplift_df[uplift_df["uplift_segment"] == "Persuadables"].copy()
        if not persuadables.empty:
            st.markdown("**Top Persuadables (highest uplift estimate)**")
            st.dataframe(
                persuadables.sort_values("tau_hat", ascending=False)
                .head(15)[["CustomerID", "tau_hat", "uplift_segment"]]
                .rename(columns={"tau_hat": "CATE Estimate (τ̂)"})
                .reset_index(drop=True),
                use_container_width=True,
            )


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📊 Decision Engine")
        st.caption("Decision-Centric Customer Re-Engagement")
        
        # Run Selection (Dynamic)
        avail_runs = get_available_runs()
        if not avail_runs:
            st.error("No analysis runs found in `outputs/`. Run `python main.py` first.")
            st.stop()

        selected_run = st.selectbox(
            "Select Analysis Run",
            options=avail_runs,
            index=0,
            help="Select which dataset/configuration output to view."
        )
        st.markdown("---")

        # Load artifacts for selected run
        waf, prep, data, meta, df_scaled, error = load_artifacts(selected_run)

        if error:
            st.error(f"**Error loading {selected_run}:**\n{error}")
            st.stop()

        st.success(f"**{meta['n_customers']:,} customers** loaded")
        st.info(f"Churn threshold: **tau = {meta['tau']} days**")
        st.markdown("---")

        # Customer selector
        st.markdown("### Customer Lookup")
        customer_ids = sorted(data["CustomerID"].astype(int).tolist())
        selected_id = st.selectbox(
            "Select Customer ID",
            options=customer_ids,
            index=0,
            help="Select a customer to analyze their churn risk"
        )

        # Or type directly
        typed_id = st.number_input(
            "Or type Customer ID",
            min_value=int(min(customer_ids)),
            max_value=int(max(customer_ids)),
            value=int(selected_id),
            step=1,
        )
        customer_id = int(typed_id)

        st.markdown("---")
        st.markdown("### Policy Parameters")
        hazard_threshold = st.slider(
            "Hazard Threshold", min_value=0.001, max_value=0.05,
            value=0.015, step=0.001, format="%.3f",
            help="Minimum hazard rate to trigger INTERVENE"
        )
        p_response = st.slider(
            "Email Response Rate", min_value=0.05, max_value=0.30,
            value=0.12, step=0.01, format="%.2f",
            help="Assumed probability customer responds to outreach"
        )
        contact_cost = st.number_input(
            "Contact Cost (GBP)", min_value=0.1, max_value=10.0,
            value=1.0, step=0.1,
        )

        st.markdown("---")
        st.caption("Built with Weibull AFT | lifelines | Streamlit")

    # ── Main Content ──────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='color:#64ffda; font-size:2rem; margin-bottom:0;'>"
        "📊 Customer Re-Engagement Decision Engine"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#8892b0; margin-top:4px;'>"
        "Survival Analysis-Powered Intervention Timing | Weibull AFT Model"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Resolve active feature list (crash-proofing) ──────────────────────────
    # Use the VIF-pruned feature list saved during training.  Falls back to
    # the full survival_features list (pre-Phase 4 artifacts), then to the
    # hard-coded default.  This ensures the UI never passes a feature that
    # was dropped by VIF pruning, which would cause a model input shape
    # mismatch and crash the app.
    _default_feats = [
        "Recency", "Frequency", "Monetary",
        "InterPurchaseTime", "GapDeviation", "SinglePurchase",
    ]
    survival_features = (
        meta.get("active_features_waf")
        or meta.get("survival_features")
        or _default_feats
    )

    # ── Customer Lookup ───────────────────────────────────────────────────────
    customer_mask = data["CustomerID"].astype(int) == customer_id
    if not customer_mask.any():
        st.error(f"Customer ID **{customer_id}** not found in the dataset.")
        st.info("Please select a valid Customer ID from the sidebar.")
        st.stop()

    customer_row = data[customer_mask].iloc[0]

    # Validate that all active features are present in the loaded data
    missing_feats = [f for f in survival_features if f not in customer_row.index]
    if missing_feats:
        st.error(
            f"Loaded data is missing features: **{missing_feats}**. "
            "Please re-run `python main.py --no-shap` to regenerate artifacts."
        )
        st.stop()

    # Compute risk at current recency
    recency = float(customer_row["Recency"])
    t_eval  = max(recency, 1.0)

    try:
        result = compute_customer_risk(
            waf=waf,
            prep=prep,
            customer_row=customer_row,
            survival_features=survival_features,
            t_eval=t_eval,
            hazard_threshold=hazard_threshold,
            p_response=p_response,
            contact_cost=contact_cost,
        )
    except Exception as e:
        st.error(f"Risk computation failed: {e}")
        st.stop()

    # ── Customer Metrics Header ───────────────────────────────────────────────
    st.markdown(f"### Customer **#{customer_id}** — Profile")

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, "Recency",   f"{customer_row['Recency']:.0f}",       "days since last buy"),
        (col2, "Frequency", f"{customer_row['Frequency']:.0f}",      "total orders"),
        (col3, "Monetary",  f"£{customer_row['Monetary']:,.0f}",     "total spend"),
        (col4, "Survival T", f"{customer_row['T']:.0f}",              "days observed"),
        (col5, "RFM Segment", customer_row.get("RFM_Segment", "N/A"),  ""),
    ]
    for col, label, value, unit in metrics:
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{value}</div>"
                f"<div class='metric-unit'>{unit}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Decision Panel ────────────────────────────────────────────────────────
    st.markdown("### 🎯 Intervention Recommendation")

    col_dec, col_metrics = st.columns([1.2, 1])

    with col_dec:
        decision = result["decision"]

        if decision == "INTERVENE":
            st.markdown(
                "<div class='decision-intervene'>"
                "<div class='decision-title' style='color:#ff4444;'>🔴 RECOMMENDATION: INTERVENE</div>"
                "<div class='decision-subtitle'>High churn risk detected. Send targeted re-engagement email NOW.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif decision == "WAIT":
            st.markdown(
                "<div class='decision-wait'>"
                "<div class='decision-title' style='color:#00cc66;'>🟢 RECOMMENDATION: WAIT</div>"
                "<div class='decision-subtitle'>Customer is still engaged. Monitor and re-evaluate next cycle.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:  # LOST
            st.markdown(
                "<div class='decision-lost'>"
                "<div class='decision-title' style='color:#888;'>⚫ RECOMMENDATION: LOST</div>"
                "<div class='decision-subtitle'>Survival probability near zero. Intervention not cost-effective.</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        # EVI Box
        evi_color = "#00cc66" if result["evi"] > 0 else "#ff4444"
        st.markdown(
            f"<div class='evi-box'>"
            f"<div style='color:#8892b0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Expected Value of Intervention (EVI)</div>"
            f"<div style='font-size:2.2rem; font-weight:800; color:{evi_color}; margin:8px 0;'>£{result['evi']:,.2f}</div>"
            f"<div style='color:#8892b0; font-size:0.8rem;'>EVI = p_response × Monetary × [1 - S(t)] - C_contact</div>"
            f"<div style='color:#8892b0; font-size:0.75rem; margin-top:4px;'>"
            f"= {p_response:.2f} × £{customer_row['Monetary']:,.0f} × {1-result['survival']:.3f} - £{contact_cost:.2f}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_metrics:
        st.markdown("<div class='section-header'>Risk Metrics at Current Recency</div>", unsafe_allow_html=True)

        # Survival probability gauge
        s_val   = result["survival"]
        s_color = "#00cc66" if s_val > 0.5 else ("#ff9f43" if s_val > 0.2 else "#ff4444")
        st.metric(
            label=f"Survival Probability S({recency:.0f}d)",
            value=f"{s_val:.4f}",
            delta=f"{'Still active' if s_val > 0.5 else 'High risk'}",
            delta_color="normal" if s_val > 0.5 else "inverse",
        )

        h_val = result["hazard"]
        st.metric(
            label=f"Hazard Rate h({recency:.0f}d)",
            value=f"{h_val:.5f}",
            delta=f"{'Above threshold' if h_val > hazard_threshold else 'Below threshold'}",
            delta_color="inverse" if h_val > hazard_threshold else "normal",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Policy Logic:**")
        st.markdown(f"""
        ```
        IF h(t) > {hazard_threshold:.3f}  → {h_val:.5f} {'✓' if h_val > hazard_threshold else '✗'}
        AND EVI > 0         → £{result['evi']:.2f} {'✓' if result['evi'] > 0 else '✗'}
        AND S(t) > 0.05     → {s_val:.4f} {'✓' if s_val > 0.05 else '✗'}
        => {decision}
        ```
        """)

    st.markdown("---")

    # ── Individual Curves ─────────────────────────────────────────────────────
    st.markdown("### 📈 Survival & Hazard Curves")
    st.caption("Vertical gold line = current customer recency (days since last purchase)")

    fig_curves = plot_individual_curves(result, customer_id, recency)
    st.pyplot(fig_curves, use_container_width=True)
    plt.close(fig_curves)

    st.markdown("---")

    # ── SHAP Section ──────────────────────────────────────────────────────────
    _build_shap_section(waf, prep, customer_row, survival_features, df_scaled)

    st.markdown("---")

    # ── Monte Carlo Policy Simulation ─────────────────────────────────────────
    mc_results = meta.get("monte_carlo_results")
    if mc_results:
        _render_monte_carlo_section(mc_results)
        st.markdown("---")

    # ── Uplift Analysis ───────────────────────────────────────────────────────
    # Pass the selected run directory so it knows where to look for CSVs
    _render_uplift_section(selected_run)
    st.markdown("---")

    # ── Portfolio Overview ────────────────────────────────────────────────────
    st.markdown("### 🌐 Portfolio Overview")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        n_intervene = (data["decision"] == "INTERVENE").sum() if "decision" in data.columns else 0
        st.metric("INTERVENE", f"{n_intervene:,}", f"{n_intervene/len(data)*100:.1f}%")
    with kpi2:
        n_wait = (data["decision"] == "WAIT").sum() if "decision" in data.columns else 0
        st.metric("WAIT", f"{n_wait:,}", f"{n_wait/len(data)*100:.1f}%")
    with kpi3:
        n_lost = (data["decision"] == "LOST").sum() if "decision" in data.columns else 0
        st.metric("LOST", f"{n_lost:,}", f"{n_lost/len(data)*100:.1f}%")
    with kpi4:
        churn_rate = meta.get("churn_rate", 0) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%", f"tau={meta['tau']}d")

    fig_portfolio = plot_portfolio_overview(data)
    st.pyplot(fig_portfolio, use_container_width=True)
    plt.close(fig_portfolio)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8892b0; font-size:0.8rem;'>"
        "Decision-Centric Customer Re-Engagement | Weibull AFT Survival Analysis | "
        "UCI Online Retail Dataset | International Council Presentation"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
