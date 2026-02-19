"""
src/comparison.py
=================
Generates a comparison report across different datasets and tau values.
Reads pipeline_meta.pkl from outputs/ directories and aggregates metrics.
"""

import os
import glob
import joblib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def generate_comparison_report(outputs_dir="outputs"):
    """
    Scans outputs directory for pipeline runs and compiles a comparison table.
    """
    # Look for all pipeline_meta.pkl files
    search_path = os.path.join(outputs_dir, "*", "models", "pipeline_meta.pkl")
    meta_files = glob.glob(search_path)
    
    if not meta_files:
        logger.warning(f"No pipeline_meta.pkl files found in {outputs_dir}")
        return

    records = []
    
    for meta_path in meta_files:
        try:
            meta = joblib.load(meta_path)
            
            # Extract run name from path (e.g. outputs/UCI_tau90/models/...)
            # parent: UCI_tau90
            run_dir = os.path.dirname(os.path.dirname(meta_path))
            run_name = os.path.basename(run_dir)
            
            dataset_name = run_name.split("_tau")[0]
            
            # Get metrics dict (added in Phase 11)
            metrics = meta.get("metrics", {})
            
            # If metrics missing (old runs), skip or fill N/A
            if not metrics:
                continue
            
            # Build record
            rec = {
                "Run":              run_name,
                "Dataset":          dataset_name,
                "Tau":              meta.get("tau"),
                "Customers":        meta.get("n_customers"),
                "Churn Rate (%)":   meta.get("churn_rate", 0) * 100,
                
                # Technical Metrics
                "C-index (Train)":  metrics.get("c_index_weibull_train"),
                "C-index (OOS)":    metrics.get("c_index_weibull_oos"),
                "CV Mean C":        metrics.get("cv_mean_c_index"),
                "IBS":              metrics.get("ibs_train"),
                
                # Business Metrics
                "Efficiency Gain %": metrics.get("outreach_efficiency"),
                "Revenue Lift %":    metrics.get("revenue_lift"),
                "Uplift Qini":       metrics.get("uplift_qini"),
            }
            records.append(rec)
            
        except Exception as e:
            logger.warning(f"Error reading {meta_path}: {e}")

    if not records:
        logger.warning("No valid metrics found in the scanned runs.")
        return

    df = pd.DataFrame(records)
    
    # Sort by Dataset then Tau
    df = df.sort_values(["Dataset", "Tau"])
    
    # Select and rename columns for display
    display_cols = [
        "Dataset", "Tau", "Customers", "Churn Rate (%)", 
        "C-index (Train)", "C-index (OOS)", "CV Mean C", 
        "Efficiency Gain %", "Revenue Lift %"
    ]
    
    # Format floats
    pd.options.display.float_format = "{:,.4f}".format
    
    print("\n" + "="*100)
    print("  CROSS-DATASET COMPARISON REPORT")
    print("="*100)
    print(df[display_cols].to_string(index=False))
    print("="*100 + "\n")
    
    # Save to CSV
    out_csv = os.path.join(outputs_dir, "comparison_report.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Full report saved to {out_csv}")

if __name__ == "__main__":
    generate_comparison_report()
