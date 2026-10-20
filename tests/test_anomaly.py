from datascope.anomaly import isolation_forest, local_outlier_factor


def test_isolation_forest_flags_some_rows(numeric_df):
    result = isolation_forest(numeric_df[["f1", "f2", "f3"]], contamination=0.1)
    assert result.is_anomaly.sum() > 0


def test_local_outlier_factor_flags_some_rows(numeric_df):
    result = local_outlier_factor(numeric_df[["f1", "f2", "f3"]], contamination=0.1)
    assert result.is_anomaly.sum() > 0
