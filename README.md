<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="examples/assets/logo-dark.png">
    <img src="examples/assets/logo.png" alt="probably logo" width="320">
  </picture>
</p>

# probably

[![CI](https://github.com/andrewpatton/probably/actions/workflows/ci.yml/badge.svg)](https://github.com/andrewpatton/probably/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Helper functions and wrappers for modeling and data science work involving probability
distributions and Monte Carlo simulation. Modeling-adjacent components are designed to be
[scikit-learn](https://scikit-learn.org/) compatible.

## Status

🚧 Early development — the public API is still taking shape. Pin versions if you depend on
this package, and expect breaking changes before `1.0`.

## Installation

```bash
pip install probably
```

## Quick start

```python
import probably

print(probably.__version__)
```

As modules land, this section will grow to cover each of them with a runnable example. See
[`examples/`](examples/) for extensive, standalone examples of every public function and
class as they're added.

## Design goals

- **Distributions & Monte Carlo first** — helpers and wrappers built around probability
  distributions and Monte Carlo simulation workflows.
- **scikit-learn compatible** — modeling-adjacent estimators/transformers follow sklearn's
  `BaseEstimator`/mixin conventions so they drop into existing sklearn pipelines.
- **Minimal dependencies** — built on `numpy`, `scipy`, and `scikit-learn`; new dependencies
  are added only when necessary.
- **Documented by example** — every public function/class ships with a runnable example.
- **`simple_*` functions favor simplicity over configuration** — the `simple_` prefix (e.g.
  `probably.viz.simple_histogram`) marks functions designed with as few parameters as
  possible, no configuration for its own sake, and a consistent, predictable signature
  shape across every `simple_*` function — accept a plain vector input, take only the
  handful of options a first-time user would expect, and return the plotted object.

## Development

Requires the `probably` conda environment (see [CLAUDE.md](CLAUDE.md)).

```bash
conda activate probably
pip install -e ".[dev]"

pytest              # run tests
ruff check .        # lint
mypy src            # type check
```

## License

[MIT](LICENSE)
