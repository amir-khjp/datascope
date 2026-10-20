from sklearn.ensemble import RandomForestClassifier

from datascope.interpretability import permutation_feature_importance


def test_permutation_importance_ranks_features(numeric_df):
    X, y = numeric_df[["f1", "f2", "f3"]], numeric_df["target"]
    model = RandomForestClassifier(random_state=0).fit(X, y)
    importance = permutation_feature_importance(model, X, y, n_repeats=5)
    assert len(importance) == 3
    assert "importance_mean" in importance.columns
