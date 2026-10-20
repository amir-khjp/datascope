from datascope.data import load_timeseries_sample
from datascope.timeseries import autocorrelation, decompose


def test_decompose_extracts_trend_and_seasonal():
    series = load_timeseries_sample(n_points=90)
    result = decompose(series, period=7)
    assert len(result.trend) == len(series)
    assert len(result.seasonal) == len(series)


def test_autocorrelation_returns_expected_lags():
    series = load_timeseries_sample(n_points=90)
    acf_df = autocorrelation(series, n_lags=10)
    assert len(acf_df) == 11
