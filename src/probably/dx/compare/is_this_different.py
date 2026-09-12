"""Compare one or more samples against a reference sample."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from probably._coerce import clean_vector, holds_sub_vectors
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

    Parameters
    ----------
    reference : array-like
        The sample everything is compared against.
    other : array-like or mapping
        One vector to compare, or several. Pass a mapping to name them.
    alpha : float, default 0.05
        Significance level each test is judged against.


    Returns
    -------
    list[dict]
        One record per comparison, holding ``name``, ``n_reference``,
        ``n_other``, a ``statistic``/``pvalue``/``pvalue_adjusted`` for each
        of ``ks``, ``cramer_von_mises`` and ``mann_whitney``, the effect
        sizes ``prob_greater``, ``overlap``, ``median_shift`` and
        ``wasserstein``, plus ``verdict`` and ``reason``.


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
    >>> results = is_this_different(control, {"low": rng.normal(size=200), "high": treated})
    >>> [(r["name"], r["verdict"]) for r in results]
    [('low', 'no'), ('high', 'yes')]
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")

    reference_values = clean_vector(reference, name="reference", minimum=2)
    comparisons = [
        (name, clean_vector(item, name=name, minimum=2))
        for name, item in labelled_inputs(_as_inputs(other))
    ]

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
    if holds_sub_vectors(other):
        return tuple(other)
    return (other,)
