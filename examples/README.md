# Examples

- [`logo.py`](logo.py) — generates the project logo (`assets/logo.png`), a series of
  beta distributions.

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
