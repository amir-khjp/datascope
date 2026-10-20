"""Simple, explicit cleaning helpers: impute and cap/drop outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def impute(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill missing values column-by-column: numeric with mean/median, else mode."""
    out = df.copy()
    for col in out.columns:
        if out[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            fill_value = out[col].mean() if strategy == "mean" else out[col].median()
        else:
            mode = out[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else ""
        out[col] = out[col].fillna(fill_value)
    return out


def cap_outliers(df: pd.DataFrame, columns: list[str], k: float = 1.5) -> pd.DataFrame:
    """Winsorize each column to [Q1 - k*IQR, Q3 + k*IQR]."""
    out = df.copy()
    for col in columns:
        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        out[col] = out[col].clip(lower, upper)
    return out


def drop_rows(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Drop rows where `mask` is True (e.g. an outlier mask)."""
    return df.loc[~mask].reset_index(drop=True)


def encode_categorical(df: pd.DataFrame, columns: list[str], method: str = "onehot") -> pd.DataFrame:
    """One-hot or label-encode the given categorical columns."""
    out = df.copy()
    if method == "onehot":
        return pd.get_dummies(out, columns=columns, drop_first=False)
    if method == "label":
        for col in columns:
            out[col] = out[col].astype("category").cat.codes
        return out
    raise ValueError(f"Unknown encoding method: {method!r}")


def scale_numeric(df: pd.DataFrame, columns: list[str], method: str = "standard") -> pd.DataFrame:
    """Standard or min-max scale the given numeric columns."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    out = df.copy()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    out[columns] = scaler.fit_transform(out[columns].astype(np.float64))
    return out
