"""Interactive Streamlit dashboard tying every DataScope technique together.

Run with: `datascope app` or `streamlit run src/datascope/app.py`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from datascope import (
    anomaly,
    clustering,
    dimensionality,
    features,
    interpretability,
    modeling,
    profiling,
    timeseries,
)
from datascope.data import (
    load_classification_sample,
    load_regression_sample,
    load_timeseries_sample,
    load_transactions_sample,
)

st.set_page_config(page_title="DataScope", layout="wide")
st.title("DataScope")
st.caption("Data mining, modeling, and visualization techniques, applied to your data.")

with st.sidebar:
    st.header("Dataset")
    source = st.radio("Source", ["Sample: classification", "Sample: regression", "Upload CSV"])
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV file", type="csv")
        df = pd.read_csv(uploaded) if uploaded else None
    elif source == "Sample: classification":
        df = load_classification_sample()
    else:
        df = load_regression_sample()

if df is None:
    st.info("Upload a CSV or pick a sample dataset to get started.")
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = st.sidebar.selectbox("Target column (optional)", ["(none)"] + list(df.columns))
target_col = None if target_col == "(none)" else target_col
feature_cols = [c for c in numeric_cols if c != target_col]

tabs = st.tabs(
    ["Profiling", "Features", "Dimensionality", "Clustering", "Association Rules", "Modeling", "Interpretability", "Anomalies", "Time Series"]
)

with tabs[0]:
    st.subheader("Dataset profile")
    report = profiling.profile(df)
    col1, col2 = st.columns(2)
    col1.metric("Rows", report.n_rows)
    col2.metric("Columns", report.n_cols)
    st.write("Missing values by column")
    st.dataframe(pd.Series(report.missing_by_column, name="missing").to_frame())
    st.write("Numeric summary")
    st.dataframe(pd.DataFrame(report.numeric_summary).T)

with tabs[1]:
    st.subheader("Feature engineering")
    if len(feature_cols) >= 2 and target_col:
        task = "classification" if df[target_col].nunique() <= 10 else "regression"
        ranking = features.mutual_info_ranking(df[feature_cols].fillna(0), df[target_col], task=task)
        st.write(f"Mutual information with target ({task})")
        st.plotly_chart(px.bar(ranking, x="feature", y="mutual_info"), width="stretch")
    else:
        st.info("Pick a target column in the sidebar to see feature rankings.")

with tabs[2]:
    st.subheader("Dimensionality reduction")
    if len(feature_cols) >= 2:
        method = st.selectbox("Method", ["pca", "tsne"])
        clean = df[feature_cols].fillna(df[feature_cols].median())
        result = dimensionality.pca(clean) if method == "pca" else dimensionality.tsne(clean)
        embedding_df = result.to_frame()
        if target_col:
            embedding_df["target"] = df[target_col].astype(str).values
            fig = px.scatter(embedding_df, x="dim_1", y="dim_2", color="target")
        else:
            fig = px.scatter(embedding_df, x="dim_1", y="dim_2")
        st.plotly_chart(fig, width="stretch")
        if result.explained_variance_ratio is not None:
            st.caption(f"Explained variance: {result.explained_variance_ratio.round(3).tolist()}")
    else:
        st.info("Need at least 2 numeric columns.")

with tabs[3]:
    st.subheader("Clustering")
    if len(feature_cols) >= 2:
        clean = df[feature_cols].fillna(df[feature_cols].median())
        k = st.slider("Number of clusters (k)", 2, 10, 3)
        result = clustering.kmeans(clean, n_clusters=k)
        st.write(f"Silhouette score: {result.silhouette:.3f}" if result.silhouette else "")
        elbow = clustering.elbow_curve(clean, k_range=range(2, 9))
        st.plotly_chart(px.line(elbow, x="k", y="inertia", title="Elbow curve"), width="stretch")
        pca_2d = dimensionality.pca(clean, n_components=2).to_frame()
        pca_2d["cluster"] = result.labels.astype(str)
        st.plotly_chart(px.scatter(pca_2d, x="dim_1", y="dim_2", color="cluster", title="Clusters (PCA projection)"), width="stretch")
    else:
        st.info("Need at least 2 numeric columns.")

with tabs[4]:
    st.subheader("Association rule mining")
    st.caption("Demonstrated on a synthetic market-basket dataset (requires `pip install datascope[association]`).")
    try:
        transactions = load_transactions_sample()
        rules = __import__("datascope.association", fromlist=["mine_rules"]).mine_rules(transactions, min_support=0.1, min_confidence=0.4)
        st.dataframe(rules.head(20))
    except ImportError as exc:
        st.warning(str(exc))

with tabs[5]:
    st.subheader("Baseline modeling")
    if target_col and feature_cols:
        clean_X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col]
        task = "classification" if y.nunique() <= 10 else "regression"
        results = modeling.run_classifiers(clean_X, y) if task == "classification" else modeling.run_regressors(clean_X, y)
        summary = pd.DataFrame([{"model": r.model_name, "mean_cv_score": r.mean_score} for r in results])
        st.dataframe(summary)
        st.session_state["best_model"] = results[0].fitted_model
        st.session_state["best_model_X"] = clean_X
        st.session_state["best_model_y"] = y
    else:
        st.info("Pick a target column in the sidebar to train models.")

with tabs[6]:
    st.subheader("Interpretability")
    if "best_model" in st.session_state:
        importance = interpretability.permutation_feature_importance(
            st.session_state["best_model"], st.session_state["best_model_X"], st.session_state["best_model_y"], n_repeats=5
        )
        st.plotly_chart(px.bar(importance, x="feature", y="importance_mean", title="Permutation importance"), width="stretch")
    else:
        st.info("Run the Modeling tab first.")

with tabs[7]:
    st.subheader("Anomaly detection")
    if len(feature_cols) >= 2:
        clean = df[feature_cols].fillna(df[feature_cols].median())
        result = anomaly.isolation_forest(clean)
        pca_2d = dimensionality.pca(clean, n_components=2).to_frame()
        pca_2d["anomaly"] = np.where(result.is_anomaly, "anomaly", "normal")
        st.plotly_chart(px.scatter(pca_2d, x="dim_1", y="dim_2", color="anomaly", title="Isolation Forest anomalies (PCA projection)"), width="stretch")
        st.caption(f"{int(result.is_anomaly.sum())} anomalies flagged out of {len(df)} rows.")
    else:
        st.info("Need at least 2 numeric columns.")

with tabs[8]:
    st.subheader("Time series")
    st.caption("Demonstrated on a synthetic daily series with trend + weekly seasonality.")
    series = load_timeseries_sample()
    decomposition = timeseries.decompose(series, period=7)
    st.plotly_chart(px.line(decomposition.observed, title="Observed"), width="stretch")
    st.plotly_chart(px.line(decomposition.trend, title="Trend"), width="stretch")
    st.plotly_chart(px.line(decomposition.seasonal, title="Seasonal"), width="stretch")
    acf_df = timeseries.autocorrelation(series, n_lags=20)
    st.plotly_chart(px.bar(acf_df, x="lag", y="acf", title="Autocorrelation (ACF)"), width="stretch")
