"""Compare one or more samples against a reference sample."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from probably._coerce import coerce_to_1d_array
from probably.dx.compare._effects import overlap, prob_greater, verdict_for_difference
from probably.dx.dists._engine import labelled_inputs

TEST_NAMES = ("ks", "cramer_von_mises", "mann_whitney")


def _run_tests(
    reference: NDArray[np.floating[Any]], other: NDArray[np.floating[Any]]
) -> dict[str, tuple[float, float]]:
    """Three tests sensitive to different kinds of difference."""
    results: dict[str, tuple[float, float]] = {}
    for name, run in (
        ("ks", lambda: stats.ks_2samp(reference, other)),
        ("cramer_von_mises", lambda: stats.cramervonmises_2samp(reference, other)),
        ("mann_whitney", lambda: stats.mannwhitneyu(other, reference)),
    ):
        try:
            result = run()
            results[name] = (float(result.statistic), float(result.pvalue))
        except Exception:
            # A test that cannot run on these samples reports nan rather than
            # dropping its keys, so every record from one call has one shape.
            results[name] = (float("nan"), float("nan"))
    return results


def _adjust(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg across comparisons, leaving non-finite values alone."""
    finite = [i for i, p in enumerate(pvalues) if np.isfinite(p)]
    if len(finite) < 2:
        return list(pvalues)
    adjusted = list(pvalues)
    corrected = stats.false_discovery_control([pvalues[i] for i in finite])
    for index, value in zip(finite, corrected, strict=True):
        adjusted[index] = float(value)
    return adjusted


def is_this_different(reference: Any, other: Any, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Ask whether samples differ from a reference, and by how much.

    Answers ``"yes"``, ``"no"``, or ``"maybe"`` per comparison, alongside
    effect sizes that say *how* different rather than only *whether*. For a
    practitioner the effect sizes usually matter more: a large enough sample
    makes almost any difference "significant", so ``prob_greater`` and
    ``overlap`` are the numbers to read first.

    Parameters
    ----------
    reference : array-like
        The sample everything is compared against, e.g. a control group.
    other : array-like or mapping
        One vector to compare, or several. Pass a mapping to name them;
        a list of vectors is named by position. Every one is compared against
        ``reference``, not against each other.
    alpha : float, default 0.05
        Significance level each test is judged against.

    Returns
    -------
    list[dict]
        One record per comparison. A record list, so it loads straight into a
        ``DataFrame``. Each record carries:

        ``name``, ``n_reference``, ``n_other``
            Which comparison, and the two sample sizes.
        ``ks_*``, ``cramer_von_mises_*``, ``mann_whitney_*``
            A ``statistic``, a ``pvalue``, and a ``pvalue_adjusted`` for each
            test. The three look for different kinds of difference: the largest
            gap between the two distributions, the gap totalled across their
            whole range, and a consistent shift up or down.
        ``prob_greater``
            Chance a draw from ``other`` exceeds a draw from ``reference``.
            0.5 means interchangeable.
        ``overlap``
            Share of density the two have in common, 1 for identical.
        ``median_shift``, ``wasserstein``
            How far apart they are, in the data's own units.
        ``verdict``, ``reason``
            ``"yes"`` they differ, ``"no"`` they look the same, or ``"maybe"``.

    Notes
    -----
    **P-values are corrected for multiple comparisons.** When one call compares
    more than one sample against the reference, the p-values are adjusted with
    the Benjamini-Hochberg procedure (``scipy.stats.false_discovery_control``)
    and the verdict is based on the adjusted values. Comparing five samples at
    once otherwise makes a false positive likely by chance alone. A single
    comparison has nothing to correct against, so ``pvalue_adjusted`` equals
    ``pvalue`` -- which means the same pair of samples can read as ``"maybe"``
    in a batch and ``"yes"`` on its own. That is the correction working, not an
    inconsistency.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(size=200)
    >>> treated = rng.normal(loc=2, size=200)
    >>> result = is_this_different(control, treated)[0]
    >>> result["verdict"]
    'yes'
    >>> round(result["prob_greater"], 2)
    0.92

    Compare several against the one reference, named:

    >>> results = is_this_different(control, {"low": rng.normal(size=200), "high": treated})
    >>> [(r["name"], r["verdict"]) for r in results]
    [('low', 'no'), ('high', 'yes')]
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")

    reference_values = _clean(reference, "reference")
    comparisons = [(name, _clean(item, name)) for name, item in labelled_inputs(_as_inputs(other))]

    records: list[dict[str, Any]] = []
    for name, values in comparisons:
        tests = _run_tests(reference_values, values)
        record: dict[str, Any] = {
            "name": name,
            "n_reference": reference_values.size,
            "n_other": values.size,
        }
        for test_name in TEST_NAMES:
            statistic, pvalue = tests[test_name]
            record[f"{test_name}_statistic"] = statistic
            record[f"{test_name}_pvalue"] = pvalue
            record[f"{test_name}_pvalue_adjusted"] = pvalue
        record["prob_greater"] = prob_greater(reference_values, values)
        record["overlap"] = overlap(reference_values, values)
        record["median_shift"] = float(np.median(values) - np.median(reference_values))
        record["wasserstein"] = float(stats.wasserstein_distance(reference_values, values))
        record["alpha"] = alpha
        records.append(record)

    # Correct each test's p-values across the comparisons in this call, so a
    # batch of five is not five independent chances at a false positive.
    if len(records) > 1:
        for test_name in TEST_NAMES:
            key = f"{test_name}_pvalue"
            for record, adjusted in zip(records, _adjust([r[key] for r in records]), strict=True):
                record[f"{key}_adjusted"] = adjusted

    for record in records:
        smallest = min(record["n_reference"], record["n_other"])
        pvalues = [record[f"{name}_pvalue_adjusted"] for name in TEST_NAMES]
        record["verdict"], record["reason"] = verdict_for_difference(pvalues, smallest, alpha)

    return records


def _as_inputs(other: Any) -> tuple[Any, ...]:
    """Present `other` the way ``labelled_inputs`` expects to receive it.

    ``labelled_inputs`` was built for ``*args``, where each vector arrives as
    its own argument. Here they arrive inside one argument instead, so a
    sequence holding sub-sequences is spread back out; a mapping is passed
    through whole, since that is the form it reads names from.
    """
    if isinstance(other, Mapping):
        return (other,)
    if _holds_sub_vectors(other):
        return tuple(other)
    return (other,)


def _holds_sub_vectors(other: Any) -> bool:
    """Is this several vectors rather than one?"""
    if isinstance(other, np.ndarray):
        return other.ndim > 1
    try:
        first = next(iter(other))
    except (TypeError, StopIteration):
        return False
    return not isinstance(first, (str, bytes)) and hasattr(first, "__len__")


def _clean(x: Any, name: str) -> NDArray[np.floating[Any]]:
    values = coerce_to_1d_array(x, name=name)
    if values.size < 2:
        raise ValueError(f"{name} must have at least 2 values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be all finite values")
    return values
