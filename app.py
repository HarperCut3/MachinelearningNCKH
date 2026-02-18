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

    /* Streamlit metric override */
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    """Load all serialized artifacts from outputs/models/."""
    required = [
        "weibull_model.pkl", "preprocessor.pkl",
        "processed_data.pkl", "pipeline_meta.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None, None, None, None, f"Missing model files: {missing}"

    try:
        waf        = joblib.load(os.path.join(MODELS_DIR, "weibull_model.pkl"))
        prep       = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
        data       = joblib.load(os.path.join(MODELS_DIR, "processed_data.pkl"))
        meta       = joblib.load(os.path.join(MODELS_DIR, "pipeline_meta.pkl"))
        return waf, prep, data, meta, None
    except Exception as e:
        return None, None, None, None, str(e)


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
        Feature column names used by the model.
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
    # Prepare scaled input
    X_raw = customer_row[survival_features].values.reshape(1, -1)
    X_scaled = prep.transform(X_raw)
    df_input = pd.DataFrame(X_scaled, columns=survival_features)

    # Time grid for curves (0 to 400 days)
    t_max = 400
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


def plot_individual_curves(result: dict, customer_id, recency: float) -> plt.Figure:
    """Plot survival and hazard curves for a single customer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#0f1117")

    t_grid  = result["t_grid"]
    S_curve = result["S_curve"]
    h_curve = result["h_curve"]
    t_eval  = result["t_eval"]

    # Color by decision
    color_map = {"INTERVENE": "#ff4444", "WAIT": "#00cc66", "LOST": "#888888"}
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


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Load artifacts
    waf, prep, data, meta, error = load_artifacts()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📊 Decision Engine")
        st.markdown("*Decision-Centric Customer Re-Engagement*")
        st.markdown("---")

        if error:
            st.error(f"**Model not loaded:** {error}")
            st.info("Run `python main.py --no-shap` first to generate model artifacts.")
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

    # ── Customer Lookup ───────────────────────────────────────────────────────
    customer_mask = data["CustomerID"].astype(int) == customer_id
    if not customer_mask.any():
        st.error(f"Customer ID **{customer_id}** not found in the dataset.")
        st.info("Please select a valid Customer ID from the sidebar.")
        st.stop()

    customer_row = data[customer_mask].iloc[0]
    survival_features = meta.get("survival_features", [
        "Recency", "Frequency", "Monetary",
        "InterPurchaseTime", "GapStability", "SinglePurchase",
    ])

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
        (col1, "Recency",          f"{customer_row['Recency']:.0f}",       "days since last buy"),
        (col2, "Frequency",        f"{customer_row['Frequency']:.0f}",      "total orders"),
        (col3, "Monetary",         f"£{customer_row['Monetary']:,.0f}",     "total spend"),
        (col4, "Survival T",       f"{customer_row['T']:.0f}",              "days observed"),
        (col5, "RFM Segment",      customer_row.get("RFM_Segment", "N/A"),  ""),
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
        s_val = result["survival"]
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
