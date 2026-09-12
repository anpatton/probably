# Examples

- [`logo.py`](logo.py) — generates the project logo (`assets/logo.png`), a series of
  beta distributions.
- [`data/iris.csv`](data/iris.csv) — the classic iris dataset (sepal/petal measurements and
  species), used by every `viz/` example below so they all plot the same, familiar data.
- [`viz/simple_histogram.py`](viz/simple_histogram.py) — plots iris petal length with
  `probably.viz.simple_histogram`.
- [`viz/simple_scatter.py`](viz/simple_scatter.py) — plots iris petal length vs. width,
  colored by species, with `probably.viz.simple_scatter`.
- [`viz/simple_bar.py`](viz/simple_bar.py) — plots iris species counts with
  `probably.viz.simple_bar`.
- [`viz/simple_kde.py`](viz/simple_kde.py) — plots the density of iris petal length with
  `probably.viz.simple_kde`, both pooled and as one overlapping curve per species via
  `color_by`.
- [`viz/simple_kde_2d.py`](viz/simple_kde_2d.py) — plots the joint density of iris petal
  length and width with `probably.viz.simple_kde_2d`.
- [`viz/simple_roc.py`](viz/simple_roc.py) — plots the ROC curve of a logistic regression
  separating iris virginica from the rest with `probably.viz.simple_roc`, both alone and
  overlaid against a second model.
- [`viz/simple_2x2.py`](viz/simple_2x2.py) — plots the confusion matrix for that same
  classifier with `probably.viz.simple_2x2`.
- [`viz/simple_table.py`](viz/simple_table.py) — tabulates iris petal length with
  `probably.viz.simple_table`, at both depths.

This directory otherwise holds one runnable example per public function/class in `probably`.

Conventions:

- One file per public symbol: `examples/<submodule>/<symbol_name>.py` (or
  `examples/<symbol_name>.py` for top-level exports).
- Each example is a standalone, runnable script — it should execute end to end with
  `python examples/<path>.py` inside the `probably` conda environment.
- Prefer small, realistic inputs (e.g. `numpy` arrays, a toy sklearn dataset) over
  placeholder/dummy data, and print or plot something that shows the result.
- Every public function/class must *also* carry the same example (or a condensed version
  of it) in its docstring's `Examples` section — the files here are the extensive,
  narrative version; the docstring version is the quick-reference version.
