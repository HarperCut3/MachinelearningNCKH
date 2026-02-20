"""
src/sensitivity.py  (Phase E — E1: Sleeping Dog Penalty Sensitivity Analysis)
==============================================================================
Sweeps the sleeping_dog_penalty across [5%, 10%, 15%, 20%, 30%, 40%, 50%] and
runs Monte Carlo simulation at each level.

Scientific Purpose
------------------
If Weibull dominates RFM at ALL penalty levels, the policy is **robust** to the
heuristic choice → reviewer cannot attack the 20% assumption.

Output
------
  - DataFrame with columns: penalty, weibull_median, rfm_median, weibull_lo, weibull_hi,
    efficiency_gain, weibull_dominates
  - Line chart with 95% CI bands: ``figures/sleeping_dog_sensitivity.png``
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def sleeping_dog_sensitivity(
    df_decisions: pd.DataFrame,
    penalties: list = None,
    n_mc: int = 500,
    save_dir: str = None,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation at each penalty level and collect results.

    Parameters
    ----------
    df_decisions : pd.DataFrame
        Merged decision table from ``policy.make_intervention_decisions()``.
    penalties : list of float
        Sleeping Dog penalty values to sweep [0.05 .. 0.50].
    n_mc : int
        Monte Carlo iterations per penalty level (500 for speed).
    save_dir : str
        Directory to save plot + CSV. If None, no files saved.

    Returns
    -------
    pd.DataFrame
        One row per penalty level with profit CIs and dominance flag.
    """
    from src.simulator import run_monte_carlo_simulation

    if penalties is None:
        penalties = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    records = []
    for pen in penalties:
        logger.info(f"[Sensitivity] penalty={pen:.0%} ...")
        mc = run_monte_carlo_simulation(
            df_decisions,
            n_iterations=n_mc,
            sleeping_dog_penalty=pen,
        )
        w_ci = mc.get("weibull_profit_ci", (0, 0, 0))
        r_ci = mc.get("rfm_profit_ci", (0, 0, 0))
        eg   = mc.get("efficiency_gain_ci", (0, 0, 0))
        records.append({
            "penalty":           pen,
            "weibull_lo":        w_ci[0],
            "weibull_median":    w_ci[1],
            "weibull_hi":        w_ci[2],
            "rfm_lo":            r_ci[0],
            "rfm_median":        r_ci[1],
            "rfm_hi":            r_ci[2],
            "efficiency_gain":   eg[1] * 100 if eg[1] else 0,
            "weibull_dominates": w_ci[1] > r_ci[1],
        })

    result_df = pd.DataFrame(records)

    # ── Log summary ──────────────────────────────────────────────────────────
    all_dominant = result_df["weibull_dominates"].all()
    if all_dominant:
        logger.info(
            "[Sensitivity] ✅ Weibull DOMINATES RFM at ALL penalty levels "
            f"({penalties[0]:.0%} to {penalties[-1]:.0%}) — policy is ROBUST."
        )
    else:
        crossover = result_df[~result_df["weibull_dominates"]]["penalty"].values
        logger.warning(
            f"[Sensitivity] ⚠ Weibull loses to RFM at penalty={crossover} "
            "— policy is NOT robust. Consider adjusting hazard_threshold."
        )

    # ── Plot: line chart with CI bands ────────────────────────────────────────
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = result_df["penalty"] * 100

        # Weibull band
        ax.plot(x, result_df["weibull_median"], "o-", color="#e74c3c",
                linewidth=2.5, markersize=7, label="Weibull AFT (median)")
        ax.fill_between(x, result_df["weibull_lo"], result_df["weibull_hi"],
                        alpha=0.15, color="#e74c3c", label="Weibull 95% CI")

        # RFM band
        ax.plot(x, result_df["rfm_median"], "s--", color="#3498db",
                linewidth=2.5, markersize=7, label="RFM Baseline (median)")
        ax.fill_between(x, result_df["rfm_lo"], result_df["rfm_hi"],
                        alpha=0.15, color="#3498db", label="RFM 95% CI")

        ax.set_xlabel("Sleeping Dog Penalty (%)", fontsize=12)
        ax.set_ylabel("Median Profit (currency units)", fontsize=12)
        ax.set_title(
            "Sensitivity Analysis: Sleeping Dog Penalty\n"
            "Weibull AFT vs RFM Baseline — Monte Carlo 95% CI",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark dominance region
        if all_dominant:
            ax.text(0.5, 0.02, "✅ Weibull dominates at ALL penalty levels",
                    transform=ax.transAxes, ha="center", fontsize=11,
                    color="green", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "sleeping_dog_sensitivity.png")
        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"[Sensitivity] Plot saved → {plot_path}")

        csv_path = os.path.join(save_dir, "sensitivity_penalty.csv")
        result_df.to_csv(csv_path, index=False)
        logger.info(f"[Sensitivity] CSV saved → {csv_path}")

    return result_df
