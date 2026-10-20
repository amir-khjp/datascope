from datascope.features import mutual_info_ranking, polynomial_features, rfe_ranking


def test_mutual_info_ranking_orders_features(numeric_df):
    ranking = mutual_info_ranking(numeric_df[["f1", "f2", "f3"]], numeric_df["target"])
    assert len(ranking) == 3
    assert list(ranking.columns) == ["feature", "mutual_info"]


def test_polynomial_features_adds_columns(numeric_df):
    expanded = polynomial_features(numeric_df[["f1", "f2"]], columns=["f1", "f2"], degree=2)
    assert expanded.shape[1] > 2


def test_rfe_ranking_selects_subset(numeric_df):
    selected = rfe_ranking(numeric_df[["f1", "f2", "f3"]], numeric_df["target"], n_features_to_select=2)
    assert len(selected) == 2
