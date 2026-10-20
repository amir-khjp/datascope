"""Clustering with built-in diagnostics (elbow, silhouette, dendrogram linkage)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusterResult:
    method: str
    labels: np.ndarray
    n_clusters: int
    silhouette: float | None = None
    extra: dict | None = None


def kmeans(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 0) -> ClusterResult:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(df)
    score = silhouette_score(df, labels) if n_clusters > 1 else None
    return ClusterResult("kmeans", labels, n_clusters, score, {"inertia": float(model.inertia_)})


def elbow_curve(df: pd.DataFrame, k_range: range = range(2, 10), random_state: int = 0) -> pd.DataFrame:
    """Inertia and silhouette score for each k, to pick a cluster count."""
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(df)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(df, labels)) if k > 1 else None,
            }
        )
    return pd.DataFrame(rows)


def dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> ClusterResult:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    score = silhouette_score(df, labels) if n_clusters > 1 else None
    n_noise = int((labels == -1).sum())
    return ClusterResult("dbscan", labels, n_clusters, score, {"n_noise": n_noise})


def hierarchical(df: pd.DataFrame, n_clusters: int = 3, linkage: str = "ward") -> ClusterResult:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(df)
    score = silhouette_score(df, labels) if n_clusters > 1 else None
    return ClusterResult("hierarchical", labels, n_clusters, score)
