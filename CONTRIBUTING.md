# Contributing to DataScope

Thanks for considering a contribution! DataScope is organized as one module
per data-mining/data-science technique, so it's straightforward to add a new
one without touching the rest of the codebase.

## Setup

```bash
git clone https://github.com/amir-khjp/datascope.git
cd datascope
pip install -e ".[dev,association]"
pytest -q
ruff check src tests
```

## Layout

- `src/datascope/<technique>.py` — one file per technique (profiling, clustering,
  dimensionality, modeling, ...), each exposing plain functions over
  pandas DataFrames/Series, with a matching `tests/test_<technique>.py`.
- `src/datascope/data/` — bundled sample datasets so every technique is
  runnable with zero setup.
- `src/datascope/app.py` — the Streamlit dashboard that ties every technique
  together with interactive Plotly charts.

## Adding a new technique

1. Add `src/datascope/<name>.py` with plain, testable functions (see
   `clustering.py` or `anomaly.py` for the expected shape: take a DataFrame
   in, return a small dataclass or DataFrame out).
2. Add `tests/test_<name>.py`.
3. Wire it into a new tab in `app.py` if it's visualizable.
4. If it needs a heavy dependency, add it as an optional extra in
   `pyproject.toml` (see how `association`/`advanced`/`interpret` are done)
   rather than a core dependency.

## Good first issues

- **Gradient boosting hyperparameter search**: add a `tune.py` module with
  grid/random search over `modeling.py`'s estimators.
- **UMAP tab in the dashboard**: wire `dimensionality.umap_reduce` into
  `app.py`'s Dimensionality tab, gated on the optional extra being installed.
- **SHAP tab in the dashboard**: same idea for `interpretability.shap_values`.
- **ARIMA forecasting**: extend `timeseries.py` with a basic
  `statsmodels.tsa.arima.model.ARIMA` forecast and plot it in the dashboard.
- **PDF/HTML export of the dashboard's current view**: let a user export
  their analysis as a static report, similar to how
  [CorpusLens](https://github.com/amir-khjp/corpuslens) does it for text data.

Open an issue before starting large changes so we can agree on the approach.

## Code style

- Format/lint with `ruff check src tests`.
- Keep core dependencies light; anything heavy (umap-learn, shap, mlxtend)
  should be an optional extra.
- Every new technique needs at least one test in `tests/`.
