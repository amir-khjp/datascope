"""Time series analysis: decomposition and autocorrelation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf


@dataclass
class DecompositionResult:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    observed: pd.Series


def decompose(series: pd.Series, period: int, model: str = "additive") -> DecompositionResult:
    result = seasonal_decompose(series, period=period, model=model)
    return DecompositionResult(result.trend, result.seasonal, result.resid, result.observed)


def autocorrelation(series: pd.Series, n_lags: int = 20) -> pd.DataFrame:
    acf_values = acf(series.dropna(), nlags=n_lags)
    pacf_values = pacf(series.dropna(), nlags=n_lags)
    return pd.DataFrame({"lag": range(n_lags + 1), "acf": acf_values, "pacf": pacf_values})
