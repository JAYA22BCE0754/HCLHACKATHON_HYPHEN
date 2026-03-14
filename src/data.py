from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def create_sample_dataset(output_path: Path) -> Path:
    """Create a realistic house-pricing sample dataset for regression demos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    df["IncomeBand"] = pd.cut(
        df["MedInc"],
        bins=[-np.inf, 2.5, 4.0, 6.0, np.inf],
        labels=["low", "medium", "high", "premium"],
    ).astype(str)
    df["RegionCluster"] = np.where(df["Latitude"] >= 36.0, "north", "south")

    df["SalePrice"] = (df["MedHouseVal"] * 100000).round(2)
    df = df.drop(columns=["MedHouseVal"])

    df.to_csv(output_path, index=False)
    return output_path


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load data from CSV or Excel file."""
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Excel.")


def validate_schema(df: pd.DataFrame, target_col: str) -> None:
    """Validate minimum schema requirements for regression training."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in dataset columns.")

    if df.shape[1] < 3:
        raise ValueError("Dataset must have at least one target and two feature columns.")

    if not np.issubdtype(df[target_col].dtype, np.number):
        raise ValueError("Target column must be numeric for regression.")


def remove_outliers_iqr(df: pd.DataFrame, target_col: str, iqr_multiplier: float = 1.5) -> Tuple[pd.DataFrame, int]:
    """Remove outlier rows using IQR thresholds on numeric feature columns."""
    numeric_cols = [
        col for col in df.select_dtypes(include=["number"]).columns.tolist() if col != target_col
    ]
    if not numeric_cols:
        return df.copy(), 0

    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask &= df[col].between(lower, upper)

    cleaned = df.loc[mask].copy()
    removed = int((~mask).sum())
    return cleaned, removed
