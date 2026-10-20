"""Dimensionality reduction: PCA (core), t-SNE (core), UMAP (optional extra)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class ReductionResult:
    method: str
    embedding: np.ndarray
    explained_variance_ratio: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        cols = [f"dim_{i + 1}" for i in range(self.embedding.shape[1])]
        return pd.DataFrame(self.embedding, columns=cols)


def pca(df: pd.DataFrame, n_components: int = 2, random_state: int = 0) -> ReductionResult:
    model = PCA(n_components=n_components, random_state=random_state)
    embedding = model.fit_transform(df)
    return ReductionResult("pca", embedding, model.explained_variance_ratio_)


def tsne(df: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0, random_state: int = 0) -> ReductionResult:
    perplexity = min(perplexity, max(5.0, (len(df) - 1) / 3))
    model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init="pca")
    embedding = model.fit_transform(df)
    return ReductionResult("tsne", embedding)


def umap_reduce(df: pd.DataFrame, n_components: int = 2, random_state: int = 0) -> ReductionResult:
    """Requires the optional `umap-learn` dependency: pip install datascope[advanced]."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap_reduce requires umap-learn: pip install datascope[advanced]") from exc

    model = umap.UMAP(n_components=n_components, random_state=random_state)
    embedding = model.fit_transform(df)
    return ReductionResult("umap", embedding)
