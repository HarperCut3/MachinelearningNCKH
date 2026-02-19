"""
src/simulator.py  (Phase 10 — Economic Reality Engine)
=======================================================
Budget-Constrained Monte Carlo Policy Simulation with Uplift Penalty.

PURPOSE
-------
Fixes the "Profitability Paradox" where a naive RFM baseline appears to
outperform the Weibull-optimised policy simply by contacting 6x more
customers (mass spray-and-pray vs. precision targeting).

Economic Realities enforced:
  1. BUDGET CONSTRAINT
       max_contacts = marketing_budget / cost_per_contact
       Both policies must operate within THIS budget.  This eliminates the
       volume advantage of RFM and forces a fair, like-for-like comparison.

  2. SLEEPING DOG PENALTY  (Radcliffe & Surry 1999 / Section 5.2)
       When RFM contacts a customer whose current hazard is LOW:
         - They were unlikely to churn anyway (Sure Thing / Sleeping Dog).
         - The unsolicited email ANNOYS them → triggers annoyance churn.
         - Penalty = sleeping_dog_penalty × Monetary (future value lost).
       This reflects the brand-damage cost of mass marketing.

  3. WEIBULL PRECISION (no Sleeping Dog penalty)
       Weibull triggers only on h(t) > θ_h AND EVI > 0 — it never contacts
       low-hazard customers by construction.

SIMULATION LOGIC (per iteration)
---------------------------------
1. Sample p_response  ~ N(resp_mean, resp_std)  clipped to [0, 1]
2. Sample cost        ~ max(N(cost_mean, cost_std), 0.1)
3. max_contacts       = floor(marketing_budget / cost)

Weibull Policy
  - Pool : decision == "INTERVENE"
  - Sort  : descending EVI
  - Take  : top min(|pool|, max_contacts) customers
  - Profit: sum[ Monetary_i * p_response * (1-survival_i) - cost ]

RFM Policy
  - Pool : top rfm_top_pct% by Monetary  (naïve heuristic)
  - Sort  : descending Monetary
  - Take  : top min(|pool|, max_contacts) customers
  - Profit per customer:
      IF hazard_now_i > hazard_threshold  →  TRUE PERSUADABLE
          profit = Monetary_i * p_response - cost
      ELSE                                →  SLEEPING DOG / SURE THING
          profit = -(Monetary_i * sleeping_dog_penalty) - cost

Returns
-------
dict with 95% CIs for Weibull Profit, RFM (penalised) Profit, Efficiency
Gain, plus budget metadata and customer counts.
"""

import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Configuration Loader
# =============================================================================

def _load_full_config() -> dict:
    """Load the complete simulation_params.yaml. Falls back to empty dict."""
    _cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "simulation_params.yaml"
    )
    try:
        import yaml
        with open(_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.debug("[MonteCarlo] Loaded config from config/simulation_params.yaml")
        return cfg if cfg else {}
    except Exception as exc:
        logger.debug("[MonteCarlo] Could not load YAML config (%s). Using defaults.", exc)
        return {}


_CFG             = _load_full_config()
_MC_CFG          = _CFG.get("monte_carlo",  {})
_SIM_CFG         = _CFG.get("simulation",   {})
_POLICY_CFG      = _CFG.get("policy",       {})

# Monte Carlo sampling params
_MC_N_ITER            = _MC_CFG.get("n_iterations",       1000)
_MC_RFM_TOP_PCT       = _MC_CFG.get("rfm_top_pct",        0.40)
_MC_RESP_MEAN         = _MC_CFG.get("response_rate_mean",  0.15)
_MC_RESP_STD          = _MC_CFG.get("response_rate_std",   0.03)
_MC_COST_MEAN         = _MC_CFG.get("cost_mean",           1.0)
_MC_COST_STD          = _MC_CFG.get("cost_std",            0.10)

# Economic reality constraints  (Phase 10)
_MARKETING_BUDGET     = _SIM_CFG.get("marketing_budget",    500.0)
_SLEEPING_DOG_PENALTY = _SIM_CFG.get("sleeping_dog_penalty", 0.20)

# Hazard threshold (same as policy engine) — used to classify Sleeping Dogs
_HAZARD_THRESHOLD     = _POLICY_CFG.get("hazard_threshold", 0.01)


# =============================================================================
# 2. Main Simulation
# =============================================================================

def run_monte_carlo_simulation(
    df_decisions: pd.DataFrame,
    n_iterations: int = None,
    rfm_top_pct: float = None,
    marketing_budget: float = None,
    sleeping_dog_penalty: float = None,
    seed: int = 42,
) -> dict:
    """
    Run a budget-constrained Monte Carlo simulation over policy outcomes.

    Parameters
    ----------
    df_decisions : pd.DataFrame
        Merged decision table from policy.make_intervention_decisions().
        Required columns:
          ``decision``    : "INTERVENE" | "WAIT" | "LOST"
          ``Monetary``    : customer lifetime spend (GBP)
          ``survival``    : S(t) from Weibull model  [0, 1]
          ``evi``         : Expected Value of Intervention (GBP)
          ``hazard_now``  : instantaneous hazard h(t) — used for Sleeping Dog test
    n_iterations : int, optional
        Monte Carlo draws (default from YAML — 1,000).
    rfm_top_pct : float, optional
        Fraction of customers by Monetary for RFM baseline (default 40%).
    marketing_budget : float, optional
        Maximum GBP spend per campaign (default from YAML — £500).
    sleeping_dog_penalty : float, optional
        Fraction of Monetary lost per Sleeping Dog contacted (default 0.20).
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    dict
        Keys:
          ``weibull_profit_ci``         : (lower, median, upper)  GBP
          ``rfm_profit_ci``             : (lower, median, upper)  GBP
          ``efficiency_gain_ci``        : (lower, median, upper)  fraction
          ``n_weibull_intervene``       : int — Weibull pool size
          ``n_weibull_funded``          : int — actually funded within budget
          ``n_rfm_intervene``           : int — RFM pool size
          ``n_rfm_funded``              : int — actually funded within budget
          ``n_rfm_sleeping_dogs``       : int — low-hazard RFM contacts (penalised)
          ``marketing_budget``          : float — budget used
          ``sleeping_dog_penalty``      : float — penalty rate used
          ``n_iterations``              : int
    """
    # ── Apply YAML defaults ───────────────────────────────────────────────────
    if n_iterations        is None: n_iterations        = _MC_N_ITER
    if rfm_top_pct         is None: rfm_top_pct         = _MC_RFM_TOP_PCT
    if marketing_budget    is None: marketing_budget     = _MARKETING_BUDGET
    if sleeping_dog_penalty is None: sleeping_dog_penalty = _SLEEPING_DOG_PENALTY

    rng = np.random.default_rng(seed)

    # ── Validate required columns ─────────────────────────────────────────────
    required = {"decision", "Monetary", "evi"}
    missing  = required - set(df_decisions.columns)
    if missing:
        raise ValueError(
            f"[MonteCarlo] df_decisions is missing columns: {missing}. "
            f"Available: {list(df_decisions.columns)}"
        )

    # ── Extract arrays ────────────────────────────────────────────────────────
    monetary = df_decisions["Monetary"].fillna(0).values.astype(float)

    # survival → churn probability
    if "survival" in df_decisions.columns:
        churn_prob = np.clip(1.0 - df_decisions["survival"].fillna(0.5).values, 0, 1).astype(float)
    else:
        logger.warning(
            "[MonteCarlo] 'survival' column missing — using uniform churn prob 0.50."
        )
        churn_prob = np.full(len(df_decisions), 0.50)

    # hazard_now → used to classify RFM contacts as Persuadable vs. Sleeping Dog
    if "hazard_now" in df_decisions.columns:
        hazard = df_decisions["hazard_now"].fillna(0).values.astype(float)
    else:
        logger.warning(
            "[MonteCarlo] 'hazard_now' column missing — all RFM contacts treated as "
            "Sleeping Dogs (conservative). Re-run make_intervention_decisions to fix."
        )
        hazard = np.zeros(len(df_decisions))

    evi_col = df_decisions["evi"].fillna(0).values.astype(float)

    # ── Define fixed candidate pools ──────────────────────────────────────────
    # Weibull pool: customers flagged INTERVENE, sorted by EVI descending
    weibull_mask    = (df_decisions["decision"] == "INTERVENE").values
    weibull_idx     = np.where(weibull_mask)[0]
    weibull_sort    = weibull_idx[np.argsort(evi_col[weibull_idx])[::-1]]   # best EVI first
    n_weibull_pool  = len(weibull_sort)

    # RFM pool: top rfm_top_pct% by Monetary, sorted descending
    rfm_threshold   = np.quantile(monetary, 1.0 - rfm_top_pct)
    rfm_mask        = monetary >= rfm_threshold
    rfm_idx         = np.where(rfm_mask)[0]
    rfm_sort        = rfm_idx[np.argsort(monetary[rfm_idx])[::-1]]          # richest first
    n_rfm_pool      = len(rfm_sort)

    # Pre-classify RFM contacts as Persuadable vs. Sleeping Dog (fixed per run)
    rfm_is_persuadable = hazard[rfm_sort] > _HAZARD_THRESHOLD   # boolean array

    logger.info(
        "[MonteCarlo] Economic Reality Engine | budget=£%.0f | "
        "sleeping_dog_penalty=%.0f%% | n_iterations=%d",
        marketing_budget, sleeping_dog_penalty * 100, n_iterations,
    )
    logger.info(
        "[MonteCarlo] Weibull pool: %d INTERVENE | RFM pool: %d (top %.0f%% by Monetary)",
        n_weibull_pool, n_rfm_pool, rfm_top_pct * 100,
    )

    # ── Arrays to collect per-iteration results ───────────────────────────────
    weibull_profits   = np.empty(n_iterations)
    rfm_profits       = np.empty(n_iterations)
    efficiency_gains  = np.empty(n_iterations)

    n_weibull_funded_arr = np.empty(n_iterations, dtype=int)
    n_rfm_funded_arr     = np.empty(n_iterations, dtype=int)

    # ==========================================================================
    # MAIN SIMULATION LOOP
    # ==========================================================================
    for i in range(n_iterations):

        # ── 1. Sample stochastic business parameters ──────────────────────────
        sim_p_resp = float(np.clip(rng.normal(_MC_RESP_MEAN, _MC_RESP_STD), 0.0, 1.0))
        sim_cost   = float(max(rng.normal(_MC_COST_MEAN, _MC_COST_STD), 0.1))

        # ── 2. Budget constraint: how many contacts can we fund? ──────────────
        max_contacts = max(int(np.floor(marketing_budget / sim_cost)), 1)

        # ── 3. WEIBULL POLICY — precision targeting ───────────────────────────
        # Take top-EVI contacts within budget
        n_w_funded    = min(n_weibull_pool, max_contacts)
        w_targets     = weibull_sort[:n_w_funded]

        # Net profit per contact: Monetary * p_response * P(churn) - cost
        # (Weibull contacts only have POSITIVE EVI by construction)
        w_profits_per = (
            monetary[w_targets] * sim_p_resp * churn_prob[w_targets]
        ) - sim_cost
        weibull_profits[i]     = float(np.sum(w_profits_per))
        n_weibull_funded_arr[i] = n_w_funded

        # ── 4. RFM POLICY — budget-capped, with Sleeping Dog penalty ─────────
        n_r_funded     = min(n_rfm_pool, max_contacts)
        r_targets_sort = rfm_sort[:n_r_funded]
        r_persuadable  = rfm_is_persuadable[:n_r_funded]

        # True Persuadables (h(t) > threshold): positive revenue contribution
        r_profit_persuadable = np.where(
            r_persuadable,
            monetary[r_targets_sort] * sim_p_resp - sim_cost,
            0.0
        )
        # Sleeping Dogs / Sure Things (h(t) <= threshold): annoyance churn
        # Future value LOST + contact cost
        r_profit_sleeping_dog = np.where(
            ~r_persuadable,
            -(monetary[r_targets_sort] * sleeping_dog_penalty) - sim_cost,
            0.0
        )
        rfm_profits[i]        = float(np.sum(r_profit_persuadable + r_profit_sleeping_dog))
        n_rfm_funded_arr[i]   = n_r_funded

        # ── 5. Efficiency gain: (Weibull - RFM) / max(|RFM|, 1) ─────────────
        denom = max(abs(rfm_profits[i]), 1.0)
        efficiency_gains[i] = (weibull_profits[i] - rfm_profits[i]) / denom

    # ==========================================================================
    # CONFIDENCE INTERVALS
    # ==========================================================================
    pcts  = [2.5, 50.0, 97.5]
    w_ci  = tuple(float(x) for x in np.percentile(weibull_profits,  pcts))
    r_ci  = tuple(float(x) for x in np.percentile(rfm_profits,      pcts))
    eg_ci = tuple(float(x) for x in np.percentile(efficiency_gains, pcts))

    # ── Count Sleeping Dogs in the median-budget-funded RFM cohort ───────────
    median_cost_approx = _MC_COST_MEAN
    n_median_funded    = min(n_rfm_pool, max(int(np.floor(marketing_budget / median_cost_approx)), 1))
    n_sleeping_dogs    = int((~rfm_is_persuadable[:n_median_funded]).sum())
    n_weibull_funded   = int(np.median(n_weibull_funded_arr))
    n_rfm_funded       = int(np.median(n_rfm_funded_arr))

    # ── Logging ───────────────────────────────────────────────────────────────
    logger.info("[MonteCarlo] 95%% Confidence Intervals (budget-constrained, Sleeping Dog penalised):")
    logger.info(
        "  Weibull Profit  (GBP): lower=£%.0f | median=£%.0f | upper=£%.0f  "
        "[%d contacts funded]",
        w_ci[0], w_ci[1], w_ci[2], n_weibull_funded,
    )
    logger.info(
        "  RFM Profit      (GBP): lower=£%.0f | median=£%.0f | upper=£%.0f  "
        "[%d contacts funded | %d Sleeping Dogs penalised]",
        r_ci[0], r_ci[1], r_ci[2], n_rfm_funded, n_sleeping_dogs,
    )
    logger.info(
        "  Efficiency Gain (frac): lower=%+.3f | median=%+.3f | upper=%+.3f",
        eg_ci[0], eg_ci[1], eg_ci[2],
    )

    median_gain_pct = eg_ci[1] * 100
    if median_gain_pct > 0:
        logger.info(
            "[MonteCarlo] Weibull OUTPERFORMS penalised RFM by %.1f%% "
            "(median efficiency gain under budget constraint).",
            median_gain_pct,
        )
    else:
        logger.warning(
            "[MonteCarlo] RFM still outperforms Weibull (median gain = %.1f%%). "
            "Consider lowering hazard_threshold or increasing marketing_budget.",
            median_gain_pct,
        )

    return {
        "weibull_profit_ci":      w_ci,
        "rfm_profit_ci":          r_ci,
        "efficiency_gain_ci":     eg_ci,
        "n_weibull_intervene":    n_weibull_pool,
        "n_weibull_funded":       n_weibull_funded,
        "n_rfm_intervene":        n_rfm_pool,
        "n_rfm_funded":           n_rfm_funded,
        "n_rfm_sleeping_dogs":    n_sleeping_dogs,
        "marketing_budget":       marketing_budget,
        "sleeping_dog_penalty":   sleeping_dog_penalty,
        "n_iterations":           n_iterations,
    }
