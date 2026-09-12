# Examples

- [`dx/describe.py`](dx/describe.py) — `probably.dx.describe` at both depths.
- [`dx/is_this_normal.py`](dx/is_this_normal.py) — `probably.dx.is_this_normal` on two samples.
- [`dx/what_is_this.py`](dx/what_is_this.py) — `probably.dx.what_is_this` identifying a sample's
  distribution, returning the fitted scipy object and the diagnostics behind it.
- [`dx/is_this_different.py`](dx/is_this_different.py) — `probably.dx.is_this_different`
  comparing two samples against one control.
- [`predict/ProbablyConformalRegressor.py`](predict/ProbablyConformalRegressor.py) —
  `probably.predict.ProbablyConformalRegressor` around a random forest, checking that the
  intervals cover at the requested rate.
- [`logo.py`](logo.py) — generates the project logo (`assets/logo.png`).
- [`data/iris.csv`](data/iris.csv) — the classic iris dataset, used by every `viz/` example
  so they all plot the same, familiar data.
- [`viz/simple_histogram.py`](viz/simple_histogram.py) — `probably.viz.simple_histogram`.
- [`viz/simple_scatter.py`](viz/simple_scatter.py) — `probably.viz.simple_scatter`, with a
  fitted line and points colored by species.
- [`viz/simple_bar.py`](viz/simple_bar.py) — `probably.viz.simple_bar`.
- [`viz/simple_kde.py`](viz/simple_kde.py) — `probably.viz.simple_kde`, pooled and by species.
- [`viz/simple_kde_2d.py`](viz/simple_kde_2d.py) — `probably.viz.simple_kde_2d`.
- [`viz/simple_ecdf.py`](viz/simple_ecdf.py) — `probably.viz.simple_ecdf`, pooled and by
  species.
- [`viz/simple_qq_normal.py`](viz/simple_qq_normal.py) — `probably.viz.simple_qq_normal`.
- [`viz/simple_pp_normal.py`](viz/simple_pp_normal.py) — `probably.viz.simple_pp_normal`.
- [`viz/simple_roc.py`](viz/simple_roc.py) — `probably.viz.simple_roc`, one model and two.
- [`viz/simple_2x2.py`](viz/simple_2x2.py) — `probably.viz.simple_2x2`.

This directory otherwise holds one runnable example per public function/class in `probably`.

Conventions:

- One file per public symbol: `examples/<submodule>/<symbol_name>.py` (or
  `examples/<symbol_name>.py` for top-level exports).
- Each example is a standalone, runnable script — it should execute end to end with
  `python examples/<path>.py` inside the `probably` conda environment.
- **Keep them bare.** Make the data, call the function, show the result — nothing else.
  No module docstring, no explanatory comments, no narrated `print` headers, no helper
  functions, no formatting of the output. Print the raw return value. Assume the reader
  is new to the package: every extra line is something they must read past to reach the
  call itself.
- Prefer small, realistic inputs (e.g. `numpy` arrays, a toy sklearn dataset) over
  placeholder/dummy data.
- Every public function/class must *also* carry an example in its docstring's `Examples`
  section. That is where context and explanation belong — not here.
