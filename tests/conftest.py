import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0, scale=0.5, size=(30, 3))
    cluster_b = rng.normal(loc=5, scale=0.5, size=(30, 3))
    data = np.vstack([cluster_a, cluster_b])
    df = pd.DataFrame(data, columns=["f1", "f2", "f3"])
    df["target"] = [0] * 30 + [1] * 30
    return df


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 100.0],
            "b": ["x", "y", "x", None, "y"],
        }
    )
