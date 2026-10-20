"""Bundled sample datasets so every module in DataScope is runnable with zero setup."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_classification_sample() -> pd.DataFrame:
    """Wine dataset: 13 numeric features + a `target` class column."""
    from sklearn.datasets import load_wine

    data = load_wine(as_frame=True)
    df = data.frame.rename(columns={"target": "target"})
    return df


def load_regression_sample(n_samples: int = 400, seed: int = 0) -> pd.DataFrame:
    """California housing: numeric features + a continuous `target`.

    Subsampled to `n_samples` rows by default (the full dataset has ~20,600
    rows, which is too slow for interactive techniques like t-SNE or
    hierarchical clustering). Pass `n_samples=None` for the full dataset.
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data.frame.rename(columns={"MedHouseVal": "target"})
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n_samples, random_state=seed).reset_index(drop=True)
    return df


def load_transactions_sample(n_transactions: int = 200, seed: int = 0) -> list[list[str]]:
    """Synthetic market-basket transactions for association rule mining."""
    rng = np.random.default_rng(seed)
    catalog = ["bread", "milk", "eggs", "butter", "cheese", "beer", "diapers", "coffee", "sugar", "tea"]
    # bias a few co-occurrences so rule mining has something to find
    transactions = []
    for _ in range(n_transactions):
        basket = set(rng.choice(catalog, size=rng.integers(2, 6), replace=False).tolist())
        if "bread" in basket and rng.random() < 0.8:
            basket.add("butter")
        if "coffee" in basket and rng.random() < 0.7:
            basket.add("sugar")
        transactions.append(sorted(basket))
    return transactions


def load_timeseries_sample(n_points: int = 180, seed: int = 0) -> pd.Series:
    """Synthetic daily series with trend + weekly seasonality + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    trend = 0.05 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 1, n_points)
    values = 50 + trend + seasonal + noise
    index = pd.date_range("2024-01-01", periods=n_points, freq="D")
    return pd.Series(values, index=index, name="value")
