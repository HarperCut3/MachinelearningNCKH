"""
src/data_loader_cdnow.py
========================
Data loader specifically for the CDNOW dataset.

Source Format
-------------
Text file (space-delimited), no header.
Columns:
  1. Customer ID (string/int)
  2. Date (YYYYMMDD)
  3. Quantity (int)
  4. Dollar Value (float)

Target Schema (Standard)
------------------------
  - CustomerID   (string)
  - InvoiceNo    (string) — synthetic
  - InvoiceDate  (datetime)
  - Quantity     (int)
  - SalesAmount  (float)
"""

import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Constants for cleaner code
_COLS_RAW = ["CustomerID", "DateStr", "Quantity", "SalesAmount"]

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean CDNOW transaction data.

    Parameters
    ----------
    file_path : str
        Path to CDNOW_master.txt

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns:
        ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'SalesAmount']
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[CDNOW] File not found: {file_path}")

    logger.info(f"Loading dataset from: {file_path}")

    # 1. Parse space-delimited file
    #    Use regex separator for variable whitespace (\s+)
    try:
        # header=None means first row is data
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=_COLS_RAW,
            dtype={
                "CustomerID": str,
                "DateStr": str,  # Keep as string for parsing
                "Quantity": int,
                "SalesAmount": float
            },
            engine="python"  # required for regex sep
        )
    except Exception as e:
        logger.error(f"[CDNOW] Parsing failed: {e}")
        raise

    total_rows_raw = len(df)

    # 2. Parse Dates (YYYYMMDD)
    df["InvoiceDate"] = pd.to_datetime(df["DateStr"], format="%Y%m%d", errors="coerce")
    
    # 3. Drop invalid dates if any
    mask_date = df["InvoiceDate"].notna()
    if (~mask_date).sum() > 0:
        logger.warning(f"[CDNOW] Dropped {(~mask_date).sum()} rows with invalid dates")
        df = df[mask_date].copy()

    # 4. Standardize Columns
    #    Create synthetic InvoiceNo = CustomerID + "-" + DateStr
    df["InvoiceNo"] = df["CustomerID"] + "-" + df["DateStr"]

    # 5. Filter for positive Quantity/SalesAmount
    #    (Standard LTV approach usually ignores returns unless explicit)
    mask_valid = (df["Quantity"] > 0) & (df["SalesAmount"] > 0)
    n_invalid  = (~mask_valid).sum()
    if n_invalid > 0:
        logger.info(f"[CDNOW] Dropped {n_invalid} rows with <= 0 Quantity/Amount")
        df = df[mask_valid].copy()

    # 6. Select final 5 columns
    out_df = df[[
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "Quantity",
        "SalesAmount"
    ]].rename(columns={"SalesAmount": "TotalSpend"}).copy()

    # 7. Log summary stats
    n_cust = out_df["CustomerID"].nunique()
    n_inv  = out_df["InvoiceNo"].nunique()
    d_min  = out_df["InvoiceDate"].min().date()
    d_max  = out_df["InvoiceDate"].max().date()

    logger.info(
        f"[CDNOW] Loaded {len(out_df):,} rows (from {total_rows_raw:,} raw) | "
        f"Unique customers: {n_cust:,} | "
        f"Date range: {d_min} → {d_max}"
    )

    return out_df
