"""Baseline supervised modeling with cross-validation, for classification and regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score

CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(random_state=0),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=0),
}

REGRESSORS = {
    "linear_regression": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(random_state=0),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=0),
}


@dataclass
class ModelResult:
    model_name: str
    task: str
    cv_scores: np.ndarray
    mean_score: float
    fitted_model: object


def run_classifiers(X: pd.DataFrame, y: pd.Series, cv: int = 5, models: list[str] | None = None) -> list[ModelResult]:
    names = models or list(CLASSIFIERS.keys())
    results = []
    for name in names:
        estimator = CLASSIFIERS[name]()
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
        estimator.fit(X, y)
        results.append(ModelResult(name, "classification", scores, float(scores.mean()), estimator))
    return sorted(results, key=lambda r: r.mean_score, reverse=True)


def run_regressors(X: pd.DataFrame, y: pd.Series, cv: int = 5, models: list[str] | None = None) -> list[ModelResult]:
    names = models or list(REGRESSORS.keys())
    results = []
    for name in names:
        estimator = REGRESSORS[name]()
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
        estimator.fit(X, y)
        results.append(ModelResult(name, "regression", scores, float(scores.mean()), estimator))
    return sorted(results, key=lambda r: r.mean_score, reverse=True)
