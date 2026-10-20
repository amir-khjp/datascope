from datascope.modeling import run_classifiers, run_regressors


def test_run_classifiers_ranks_models(numeric_df):
    results = run_classifiers(numeric_df[["f1", "f2", "f3"]], numeric_df["target"], cv=3)
    assert len(results) == 3
    assert results[0].mean_score >= results[-1].mean_score


def test_run_regressors_ranks_models(numeric_df):
    results = run_regressors(numeric_df[["f1", "f2", "f3"]], numeric_df["f1"] * 2, cv=3)
    assert len(results) == 3
