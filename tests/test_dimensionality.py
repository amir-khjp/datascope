from datascope.dimensionality import pca, tsne


def test_pca_reduces_dimensions(numeric_df):
    result = pca(numeric_df[["f1", "f2", "f3"]], n_components=2)
    assert result.embedding.shape == (60, 2)
    assert result.explained_variance_ratio is not None


def test_tsne_reduces_dimensions(numeric_df):
    result = tsne(numeric_df[["f1", "f2", "f3"]], n_components=2, perplexity=5)
    assert result.embedding.shape == (60, 2)
