"""Descriptive statistics for a single vector."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

from probably._coerce import coerce_to_1d_array

# Layer 1 answers "where is it and how wide is it"; layer 2 answers "what
# shape is it". Splitting them keeps the common call short without hiding
# anything -- depth=2 returns layer 1 as well, never instead of it.
LAYER_1_KEYS = ("n", "mean", "median", "std", "variance", "min", "q1", "q3", "max", "range")
LAYER_2_KEYS = ("skew", "kurtosis", "iqr", "mad", "cv", "sem")


def _layer_1(values: NDArray[np.floating[Any]]) -> dict[str, float]:
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        # ddof=1: these are samples, not populations. n=1 gives nan, not 0.
        "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        "variance": float(values.var(ddof=1)) if values.size > 1 else float("nan"),
        "min": float(values.min()),
        "q1": float(q1),
        "q3": float(q3),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def _layer_2(values: NDArray[np.floating[Any]], layer_1: dict[str, float]) -> dict[str, float]:
    n = values.size
    std = layer_1["std"]
    mean = layer_1["mean"]
    return {
        # bias=False matches the sample estimators used above; kurtosis is
        # excess kurtosis, so a normal sits at 0 rather than 3.
        "skew": float(skew(values, bias=False)) if n > 2 else float("nan"),
        "kurtosis": float(kurtosis(values, bias=False)) if n > 3 else float("nan"),
        "iqr": layer_1["q3"] - layer_1["q1"],
        "mad": float(np.median(np.abs(values - np.median(values)))),
        # Coefficient of variation is meaningless once the mean crosses zero,
        # so it is reported as nan rather than as a huge or negative number.
        "cv": std / mean if mean != 0 else float("nan"),
        "sem": float(std / np.sqrt(n)) if n > 1 else float("nan"),
    }


def describe(x: Any, depth: Literal[1, 2] = 1) -> dict[str, float]:
    """Summarize a single vector as a dictionary of statistics.

    Returns data rather than a rendered table, so the result can be sorted,
    filtered, compared, or loaded straight into a ``DataFrame``.

    Parameters
    ----------
    x : array-like
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``. Must be all finite.
    depth : {1, 2}, default 1
        ``1`` returns location and spread: ``n``, ``mean``, ``median``,
        ``std``, ``variance``, ``min``, ``q1``, ``q3``, ``max``, ``range``.
        ``2`` returns those *and* the shape statistics: ``skew``,
        ``kurtosis`` (excess), ``iqr``, ``mad``, ``cv``, ``sem``.

    Returns
    -------
    dict[str, float]
        Statistic name to value, layer 1 first. Statistics that are
        undefined for the sample size (e.g. ``skew`` of two points) are
        ``nan`` rather than absent, so the keys for a given ``depth`` are
        always the same.

    Examples
    --------
    >>> summary = describe([1, 2, 3, 4, 5])
    >>> summary["mean"], summary["median"], summary["range"]
    (3.0, 3.0, 4.0)
    >>> sorted(describe([1, 2, 3, 4, 5], depth=2)) == sorted(describe([1], depth=2))
    True
    >>> round(describe([1, 2, 3, 4, 100], depth=2)["skew"], 3)
    2.232
    """
    if depth not in (1, 2):
        raise ValueError(f"depth must be 1 or 2, got {depth!r}")

    values = coerce_to_1d_array(x)
    if values.size == 0:
        raise ValueError("x must have at least 1 value, got an empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must be all finite values")

    summary = _layer_1(values)
    if depth == 2:
        summary.update(_layer_2(values, summary))
    return summary
