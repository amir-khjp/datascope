"""Dataset profiling: missing values, dtype inference, outlier detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class ProfileReport:
    n_rows: int
    n_cols: int
    missing_by_column: dict[str, int]
    dtypes: dict[str, str]
    numeric_summary: dict[str, dict[str, float]]
    outliers: dict[str, Any] = field(default_factory=dict)


def profile(df: pd.DataFrame) -> ProfileReport:
    """Summarize shape, missingness, dtypes, and per-column numeric stats."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_summary = {
        col: {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
        for col in numeric_cols
    }

    return ProfileReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        missing_by_column={col: int(df[col].isna().sum()) for col in df.columns},
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        numeric_summary=numeric_summary,
    )


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Boolean mask of rows outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Boolean mask of rows with |z-score| above `threshold`."""
    z = (series - series.mean()) / series.std(ddof=0)
    return z.abs() > threshold


def detect_outliers_isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 0) -> pd.Series:
    """Boolean mask over rows, using an Isolation Forest on all numeric columns."""
    numeric = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).median())
    if numeric.shape[1] == 0 or len(numeric) < 2:
        return pd.Series(False, index=df.index)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(numeric)
    return pd.Series(preds == -1, index=df.index)
