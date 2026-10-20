"""Model interpretability: permutation importance (core) and SHAP (optional extra)."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_feature_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 0) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    return (
        pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def shap_values(model, X: pd.DataFrame):
    """Requires the optional `shap` dependency: pip install datascope[interpret]."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap_values requires shap: pip install datascope[interpret]") from exc

    explainer = shap.Explainer(model, X)
    return explainer(X)
