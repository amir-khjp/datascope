from datascope.data import (
    load_classification_sample,
    load_regression_sample,
    load_timeseries_sample,
    load_transactions_sample,
)


def test_load_classification_sample_has_target():
    df = load_classification_sample()
    assert "target" in df.columns
    assert df["target"].nunique() > 1


def test_load_regression_sample_has_target():
    df = load_regression_sample()
    assert "target" in df.columns


def test_load_transactions_sample_shape():
    transactions = load_transactions_sample(n_transactions=50)
    assert len(transactions) == 50
    assert all(len(t) >= 2 for t in transactions)


def test_load_timeseries_sample_length():
    series = load_timeseries_sample(n_points=30)
    assert len(series) == 30
