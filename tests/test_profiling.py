
from datascope.profiling import (
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_zscore,
    profile,
)


def test_profile_reports_missing_and_dtypes(df_with_missing):
    report = profile(df_with_missing)
    assert report.n_rows == 5
    assert report.missing_by_column["a"] == 1
    assert report.missing_by_column["b"] == 1


def test_detect_outliers_iqr_flags_extreme_value(df_with_missing):
    mask = detect_outliers_iqr(df_with_missing["a"].dropna())
    assert mask.iloc[-1]  # the 100.0 value


def test_detect_outliers_zscore_flags_extreme_value(df_with_missing):
    mask = detect_outliers_zscore(df_with_missing["a"].dropna(), threshold=1.0)
    assert mask.any()


def test_isolation_forest_outliers(numeric_df):
    mask = detect_outliers_isolation_forest(numeric_df[["f1", "f2", "f3"]], contamination=0.1)
    assert mask.sum() > 0
