from datascope.cleaning import cap_outliers, encode_categorical, impute, scale_numeric


def test_impute_fills_numeric_and_categorical(df_with_missing):
    cleaned = impute(df_with_missing)
    assert cleaned["a"].isna().sum() == 0
    assert cleaned["b"].isna().sum() == 0


def test_cap_outliers_limits_extreme_value(df_with_missing):
    filled = df_with_missing.fillna({"a": df_with_missing["a"].median()})
    capped = cap_outliers(filled, columns=["a"])
    assert capped["a"].max() < 100.0


def test_encode_categorical_onehot(df_with_missing):
    filled = df_with_missing.fillna({"b": "x"})
    encoded = encode_categorical(filled, columns=["b"], method="onehot")
    assert any(col.startswith("b_") for col in encoded.columns)


def test_scale_numeric_standard(numeric_df):
    scaled = scale_numeric(numeric_df, columns=["f1", "f2", "f3"])
    assert abs(scaled["f1"].mean()) < 1e-6
