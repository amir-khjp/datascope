"""Feature engineering and selection."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


def polynomial_features(df: pd.DataFrame, columns: list[str], degree: int = 2) -> pd.DataFrame:
    """Add polynomial/interaction features for the given numeric columns."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    transformed = poly.fit_transform(df[columns])
    names = poly.get_feature_names_out(columns)
    poly_df = pd.DataFrame(transformed, columns=names, index=df.index)
    return pd.concat([df.drop(columns=columns), poly_df], axis=1)


def mutual_info_ranking(X: pd.DataFrame, y: pd.Series, task: str = "classification") -> pd.DataFrame:
    """Rank features by mutual information with the target."""
    scorer = mutual_info_classif if task == "classification" else mutual_info_regression
    scores = scorer(X, y, random_state=0)
    return (
        pd.DataFrame({"feature": X.columns, "mutual_info": scores})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )


def rfe_ranking(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 5) -> list[str]:
    """Select the top `n_features_to_select` features via recursive feature elimination."""
    estimator = LogisticRegression(max_iter=1000)
    selector = RFE(estimator, n_features_to_select=min(n_features_to_select, X.shape[1]))
    selector.fit(X, y)
    return list(X.columns[selector.support_])
