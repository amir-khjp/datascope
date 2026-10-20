from datascope.clustering import dbscan, elbow_curve, hierarchical, kmeans


def test_kmeans_finds_two_clusters(numeric_df):
    result = kmeans(numeric_df[["f1", "f2", "f3"]], n_clusters=2)
    assert result.n_clusters == 2
    assert result.silhouette > 0.5


def test_elbow_curve_has_expected_shape(numeric_df):
    curve = elbow_curve(numeric_df[["f1", "f2", "f3"]], k_range=range(2, 5))
    assert len(curve) == 3
    assert set(curve.columns) == {"k", "inertia", "silhouette"}


def test_dbscan_separates_clusters(numeric_df):
    result = dbscan(numeric_df[["f1", "f2", "f3"]], eps=1.0, min_samples=3)
    assert result.n_clusters >= 1


def test_hierarchical_clusters(numeric_df):
    result = hierarchical(numeric_df[["f1", "f2", "f3"]], n_clusters=2)
    assert result.n_clusters == 2
