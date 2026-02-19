# Decision-Centric Customer Retention
### *Precision Survival Analysis & Uplift Modeling for E-Commerce*

Implementation of a **Weibull AFT model** to predict *when* customers will churn — not just *if* — enabling proactive, perfectly-timed re-engagement interventions.

---

## 🚀 Overview
Most churn models ask "Will this customer leave next month?". This framework asks **"When will this customer leave, and what is the optimal time to intervene?"**

Key capabilities:
- **Survival Analysis**: Weibull AFT model (C-index > 0.76) predicts exact churn timing.
- **Uplift Modeling**: Uses T-Learner to identify "Persuadables" (customers who respond *only* if treated).
- **Decision Engine**: Calculates Expected Value of Intervention (EVI) to maximize ROI.
- **Interactive Dashboard**: Real-time risk profiling & portfolio management.

---

## ✨ Features
- **Precision Targeting**: Intervene only when hazard is high AND projected ROI is positive.
- **Explainable AI**: SHAP values explain *why* a customer is at risk.
- **Multi-Dataset Support**: UCI Online Retail, Ta Feng Grocery, and CDNOW datasets.
- **Cross-Validation**: Stratified K-Fold CV for survival model robustness.
- **Experiment Tracking**: MLflow integration for model metrics & artifact logging.
- **Comparison Reports**: Auto-generate cross-dataset performance comparisons.
- **Dockerized**: Production-ready container support.
- **Configurable**: Business rules centralized in YAML Configuration.

---

## 🛠️ Installation

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Setup
```bash
# Clone repository
git clone https://github.com/HarperCut3/MachinelearningNCKH.git
cd MachinelearningNCKH

# Create virtual environment (recommended)
python -m venv .venv
# Activate: Windows -> .venv\Scripts\activate | Linux/Mac -> source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development (Jupyter, pytest):
pip install -r requirements-dev.txt
```

### 3. Data Setup
1. Download **Online Retail.xlsx** from [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail).
2. Place it in `data/raw/Online Retail.xlsx`.

---

## 🚦 Usage

### Run Pipeline
```bash
# Minimal run (fastest, ~2 mins)
python main.py --no-shap --no-mlflow

# Full run with SHAP analysis
python main.py

# Include Uplift Modeling (T-Learner)
python main.py --uplift

# Run on different datasets
python main.py --dataset cdnow --tau 90
python main.py --dataset tafeng --tau 60

# Enable Cross-Validation (5-fold)
python main.py --cv

# Full run with all features
python main.py --dataset uci --tau 90 --cv --uplift
```

### CLI Flags
| Flag | Default | Description |
|---|---|---|
| `--dataset` | `uci` | Dataset to use: `uci`, `tafeng`, or `cdnow` |
| `--tau` | `90` | Churn threshold in days |
| `--cv` | off | Enable 5-fold stratified cross-validation |
| `--uplift` | off | Enable T-Learner uplift modeling |
| `--no-shap` | off | Skip SHAP computation (faster) |
| `--no-mlflow` | off | Disable MLflow experiment tracking |

### Launch Dashboard
Start the interactive Streamlit app:
```bash
streamlit run app.py
# Access at http://localhost:8501
```

### Generate Comparison Report
Compare results across all dataset runs:
```bash
python src/comparison.py
```

---

## 📂 Project Structure

```
MachinelearningNCKH/
├── config/
│   └── simulation_params.yaml     # Business parameters (hazard threshold, costs, etc.)
├── data/
│   ├── raw/                       # Raw datasets (gitignored — download separately)
│   └── processed/                 # Feature cache (gitignored — auto-generated)
├── src/
│   ├── data_loader.py             # UCI Online Retail loader
│   ├── data_loader_tafeng.py      # Ta Feng Grocery loader
│   ├── data_loader_cdnow.py       # CDNOW loader
│   ├── feature_engine.py          # Vectorized RFM + survival features
│   ├── models.py                  # Weibull AFT, CoxPH, Logistic, RFM
│   ├── policy.py                  # EVI Decision Engine
│   ├── simulator.py               # Monte Carlo Simulation
│   ├── evaluation.py              # C-index, IBS, AUC, business metrics
│   ├── uplift.py                  # Uplift Modeling (T-Learner)
│   ├── comparison.py              # Cross-run comparison report
│   └── visualization.py           # Publication-ready plots
├── tests/
│   └── test_components.py         # Unit tests for core modules
├── outputs/                       # All pipeline outputs (gitignored)
│   └── {DATASET}_tau{N}/          # Isolated per-run output directory
│       ├── figures/               # Generated plots
│       ├── reports/               # intervention_decisions.csv
│       ├── models/                # Serialized .pkl artifacts
│       └── logs/                  # Timestamped pipeline logs
├── notebooks/                     # Jupyter exploration (gitignored)
├── app.py                         # Streamlit Dashboard
├── main.py                        # Pipeline Orchestrator
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Dashboard + MLflow services
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Dev/notebook dependencies
└── README.md
```

---

## ⚙️ Configuration
Adjust business parameters in `config/simulation_params.yaml`:

```yaml
policy:
  hazard_threshold: 0.01    # Daily hazard trigger
  cost_per_contact: 1.0     # Cost per intervention (£)
  response_rate: 0.15       # Expected campaign success rate
```

---

## 📊 Key Results

### UCI Online Retail (n=4,338)
| Metric | Score | Target |
|---|---|---|
| **Weibull C-index (OOS)** | **0.829** | > 0.60 ✅ |
| **IBS Score** | **0.162** | < 0.25 ✅ |

### CDNOW (n=23,502)
| Metric | Score | Target |
|---|---|---|
| **Weibull C-index (OOS)** | **0.773** | > 0.60 ✅ |
| **IBS Score** | **0.077** | < 0.25 ✅ |
| **CV Mean C-index** | **0.746** | > 0.60 ✅ |

> **Monte Carlo Simulation**: The Weibull policy achieves significantly higher revenue precision per contact compared to standard RFM targeting, while reducing outreach costs by ~77%.

---

## 🐳 Docker Deployment

**Run everything (Dashboard + MLflow):**
```bash
docker compose up --build
```
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5000

---

## 📝 Citation
*D. Chen et al., "Data mining for the online retail industry: A case study of RFM model-based customer segmentation", 2012.*
