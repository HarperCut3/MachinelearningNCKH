# Decision-Centric Customer Re-Engagement
### A Survival Analysis Approach in Retail E-Commerce

> **Weibull AFT model** to predict *when* customers will churn — not just *if* — enabling precision intervention timing over traditional RFM heuristics.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/HarperCut3/MachinelearningNCKH.git
cd MachinelearningNCKH

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (generates models + figures)
python main.py --no-shap

# 4. Launch the interactive dashboard
streamlit run app.py
```

> **Dataset required:** Place `Online Retail.xlsx` (UCI Online Retail Dataset) in the project root before running. Download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail).

---

## Project Structure

```
MachinelearningNCKH/
├── Online Retail.xlsx          ← UCI dataset (not tracked in git)
├── requirements.txt            ← Python dependencies
├── main.py                     ← Pipeline orchestrator (CLI)
├── app.py                      ← Streamlit dashboard
├── src/
│   ├── data_loader.py          ← Data ingestion & strict cleaning
│   ├── feature_engine.py       ← RFM + Survival feature engineering
│   ├── models.py               ← WeibullAFT, CoxPH, Logistic, RFM
│   ├── policy.py               ← EVI-based intervention policy engine
│   ├── evaluation.py           ← C-index, IBS, AUC, business metrics
│   └── visualization.py        ← Publication-quality figures
└── outputs/
    ├── figures/                ← Generated PNG plots
    ├── reports/                ← intervention_decisions.csv
    └── models/                 ← Serialized model artifacts (auto-generated)
```

---

## Pipeline CLI Options

```bash
python main.py                    # Full run (includes SHAP, ~5 min)
python main.py --no-shap          # Fast run (skip SHAP)
python main.py --tau 60           # Change churn threshold to 60 days
python main.py --tau 120          # Change churn threshold to 120 days
python main.py --sensitivity      # Test tau in {60, 90, 120} days
```

---

## Key Results (tau = 90 days, 4,338 customers)

| Metric | Value | Target |
|---|---|---|
| Weibull AFT C-index | **0.9815** | > 0.60 ✅ |
| CoxPH C-index | **0.9842** | > 0.58 ✅ |
| Logistic AUC (no leakage) | **0.7766** | > 0.65 ✅ |
| Integrated Brier Score | **0.0808** | < 0.25 ✅ |
| Time-AUC at 30 days | **0.893** | — |
| Outreach efficiency gain | **69.3%** | ≥ 20% ✅ |
| Weibull shape param ρ | **1.4524** | ρ > 1 (increasing hazard) ✅ |

---

## Methodology

### Framework
Shifts from binary churn classification to a **decision-centric survival analysis** framework:

```
Traditional:  P(churn | x)          → "Who will churn?"
This project: T(x) ~ Weibull AFT    → "When to intervene?"
```

### Feature Engineering
| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Total number of orders |
| Monetary | Total spend (GBP) |
| InterPurchaseTime | Average days between orders |
| GapStability | Std dev of inter-purchase gaps |
| SinglePurchase | Binary flag (1 = only one order ever) |

### Survival Target
- **T** = Days from first to last purchase (or observation end)
- **E** = 1 if inactive > tau days (churned), 0 if censored (still active)

### Policy Engine
```
EVI(t*, i) = p_response × Monetary_i × [1 - S(t* | x_i)] - C_contact

Decision:
  IF S(t) < 0.05              → LOST      (intervention not viable)
  IF h(t) > θ_h AND EVI > 0  → INTERVENE (send re-engagement email)
  ELSE                        → WAIT      (monitor next cycle)
```

### Models
| Model | Role | Library |
|---|---|---|
| **Weibull AFT** | Primary — continuous survival time | `lifelines` |
| **CoxPH** | Semi-parametric baseline | `lifelines` |
| **Logistic Regression** | Binary classification baseline (Recency excluded) | `scikit-learn` |
| **RFM Quintiles** | Heuristic baseline | custom |

---

## Dashboard Features

The Streamlit dashboard (`app.py`) provides:
- **Customer lookup** by ID with real-time risk computation
- **INTERVENE 🔴 / WAIT 🟢 / LOST ⚫** recommendation with EVI in GBP
- **Individual survival S(t) and hazard h(t) curves** with current-recency marker
- **Adjustable policy parameters** (hazard threshold, response rate, contact cost)
- **Portfolio overview** — segment distribution and decision breakdown

---

## Dependencies

```
pandas >= 1.5.0
numpy >= 1.23.0
openpyxl >= 3.0.10
scikit-learn >= 1.2.0
lifelines >= 0.27.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
scipy >= 1.10.0
shap >= 0.41.0
streamlit >= 1.20.0
joblib >= 1.2.0
```

---

## Citation

Dataset: Daqing Chen, Sai Liang Sain, and Kun Guo, *Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining*, Journal of Database Marketing and Customer Strategy Management, 2012.

---

*Project presented at the International Council — Seoul, 2026.*
