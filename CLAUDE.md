# probably

Helper functions and wrappers for modeling and data science work involving probability
distributions and Monte Carlo simulation. Compatible with scikit-learn. All agent python work is to be conducted in conda environment `probably`.

## Project layout

- `src/probably/` — package source (src layout)
- `tests/` — pytest tests
- `pyproject.toml` — build config (hatchling), dependencies, tool config (ruff, mypy, pytest)

## Design goals

- All modeling-adjacent code (estimators, transformers) must be sklearn-compatible:
  subclass `sklearn.base.BaseEstimator` (and the relevant mixins, e.g.
  `TransformerMixin`, `ClassifierMixin`, `RegressorMixin`), implement `fit`/`transform`/
  `predict` as appropriate, follow the `get_params`/`set_params` conventions (params set
  in `__init__` without mutation), and pass `sklearn.utils.estimator_checks.check_estimator`.
- Any such sklearn-style class must have `Probably` as the first word of its name
  (`ProbablyConformalRegressor`, never `ConformalRegressor`), so it is always obvious at
  the call site which objects come from this package and which come from sklearn.
- If sklearn compatibility ever breaks (e.g. a `check_estimator` failure, an upstream API
  change, or a design tradeoff that would violate the conventions above), raise the concern
  explicitly rather than silently working around it or skipping the check.
- This package is a specialized bolt-on to scikit-learn, numpy, and scipy — never a
  replacement for them or a "better version" of something they already provide. Before
  adding a class or function, check whether those libraries already have it or the piece
  it would be built on; if so, write a thin entry point that uses the existing object
  (a constructor that returns a scipy distribution, a function that reads an sklearn
  estimator's attributes) rather than wrapping or reimplementing it.
- Keep runtime dependencies to a minimum. Only add a new dependency when the functionality
  can't reasonably be achieved with what's already required (numpy, scipy, scikit-learn,
  matplotlib); prefer stdlib or these existing deps over pulling in something new.
- Every public function/class (anything exposed to the end user, i.e. importable from
  `probably` or its public submodules and not prefixed with `_`) must have a clear,
  runnable example: at minimum an `Examples` section in its docstring (doctest-style or
  plain code block), and it should also be represented in an `examples/` directory and/or
  the docs so users can find extensive, realistic usage examples — not just a signature.

## Common commands

Run these inside the `probably` conda environment (`conda activate probably`):

- Install editable: `pip install -e ".[dev]"`
- Run tests: `pytest`
- Lint: `ruff check .`
- Type check: `mypy src`
