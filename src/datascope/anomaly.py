"""Anomaly/outlier detection as a first-class technique (beyond simple IQR profiling)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class AnomalyResult:
    method: str
    is_anomaly: np.ndarray
    scores: np.ndarray


def isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 0) -> AnomalyResult:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(df)
    scores = model.score_samples(df)
    return AnomalyResult("isolation_forest", preds == -1, scores)


def local_outlier_factor(df: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.05) -> AnomalyResult:
    model = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(df) - 1), contamination=contamination)
    preds = model.fit_predict(df)
    scores = model.negative_outlier_factor_
    return AnomalyResult("local_outlier_factor", preds == -1, scores)
