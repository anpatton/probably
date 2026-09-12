"""Simple table of summary statistics and 1-D distribution tests."""

import warnings
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
from numpy.typing import NDArray
from scipy import stats

from probably.viz._utils import (
    BACKGROUND_COLOR,
    INK_COLOR,
    MUTED_COLOR,
    apply_simple_style,
    coerce_to_1d_array,
    format_number,
)

ALPHA = 0.05

_HEADER_COLOR = "#F1E6D2"
_ROW_HEIGHT = 0.34

_DEPTH_1_COLUMNS = ("Statistic", "Value")
_DEPTH_1_WIDTHS = (0.6, 0.4)
_DEPTH_2_COLUMNS = ("Statistic", "Value", "p", f"α = {ALPHA}")
_DEPTH_2_WIDTHS = (0.44, 0.18, 0.16, 0.22)

_Row = tuple[str, ...]
_TestRunner = Callable[[NDArray[np.floating[Any]]], tuple[float, float]]


def _scipy_test(func: Callable[..., Any]) -> _TestRunner:
    def run(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
        result = func(values)
        return float(result.statistic), float(result.pvalue)

    return run


def _anderson_test(dist: str) -> _TestRunner:
    def run(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
        # method="interpolate" returns a p-value; the critical-value attributes
        # it replaces are removed in scipy 1.19. scipy-stubs spells the literal
        # "interpolated", which scipy itself rejects at runtime -- the stub is
        # wrong, so the call is correct as written and the overload is ignored.
        result = stats.anderson(values, dist=dist, method="interpolate")  # type: ignore[call-overload]
        return float(result.statistic), float(result.pvalue)

    return run


def _fitted_test(
    dist_name: str,
    floc: float | None = None,
    support: str | None = None,
) -> _TestRunner:
    """A Kolmogorov-Smirnov test against `dist_name` fitted to the sample.

    `floc` pins the location. Families with positive support need it: left
    free, a 3-parameter lognormal collapses to a normal (tiny shape, huge
    negative loc) and happily "fits" data that is not lognormal at all.
    """

    def run(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
        if support == "positive" and not np.all(values > 0):
            raise ValueError(f"{dist_name} requires strictly positive values")
        if support == "non-negative" and not np.all(values >= 0):
            raise ValueError(f"{dist_name} requires non-negative values")

        dist = getattr(stats, dist_name)
        params = dist.fit(values) if floc is None else dist.fit(values, floc=floc)
        # Freeze and pass the bound cdf: `kstest(values, name, args=params)`
        # feeds the params straight to the raw cdf and raises.
        result = stats.kstest(values, dist(*params).cdf)
        return float(result.statistic), float(result.pvalue)

    return run


def _cramer_von_mises(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
    params = stats.norm.fit(values)
    result = stats.cramervonmises(values, "norm", args=params)
    return float(result.statistic), float(result.pvalue)


# Depth 1: where the data sits and how far it spreads.
_DESCRIPTIVES: list[tuple[str, Callable[[NDArray[np.floating[Any]]], float]]] = [
    ("n", lambda v: float(v.size)),
    ("Mean", lambda v: float(np.mean(v))),
    ("Median", lambda v: float(np.median(v))),
    ("Std. dev.", lambda v: float(np.std(v, ddof=1))),
    ("Variance", lambda v: float(np.var(v, ddof=1))),
    ("Min", lambda v: float(np.min(v))),
    ("25%", lambda v: float(np.percentile(v, 25))),
    ("75%", lambda v: float(np.percentile(v, 75))),
    ("Max", lambda v: float(np.max(v))),
    ("IQR", lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))),
    ("Skewness", lambda v: float(stats.skew(v))),
    ("Excess kurtosis", lambda v: float(stats.kurtosis(v))),
]

# Depth 2 adds these. Each name carries the family under its null hypothesis,
# so "reject" means the data is not that distribution.
_TESTS: list[tuple[str, _TestRunner]] = [
    ("Shapiro–Wilk (normal)", _scipy_test(stats.shapiro)),
    ("D'Agostino–Pearson (normal)", _scipy_test(stats.normaltest)),
    ("Jarque–Bera (normal)", _scipy_test(stats.jarque_bera)),
    ("Anderson–Darling (normal)", _anderson_test("norm")),
    ("Kolmogorov–Smirnov (normal)", _fitted_test("norm")),
    ("Cramér–von Mises (normal)", _cramer_von_mises),
    ("Skew test (normal)", _scipy_test(stats.skewtest)),
    ("Kurtosis test (normal)", _scipy_test(stats.kurtosistest)),
    ("Kolmogorov–Smirnov (lognormal)", _fitted_test("lognorm", floc=0, support="positive")),
    ("Kolmogorov–Smirnov (exponential)", _fitted_test("expon", floc=0, support="non-negative")),
    ("Kolmogorov–Smirnov (uniform)", _fitted_test("uniform")),
    ("Anderson–Darling (logistic)", _anderson_test("logistic")),
]

_DEPTHS = (1, 2)


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 0.0001:
        return "<0.0001"
    return format_number(value)


def _descriptive_rows(values: NDArray[np.floating[Any]], width: int) -> list[_Row]:
    rows: list[_Row] = []
    for label, measure in _DESCRIPTIVES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                text = format_number(measure(values))
        except Exception:
            text = "n/a"
        rows.append((label, text) + ("",) * (width - 2))
    return rows


def _test_rows(values: NDArray[np.floating[Any]]) -> list[_Row]:
    rows: list[_Row] = []
    for label, run in _TESTS:
        try:
            # Fits against the wrong family emit noisy invalid-value warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = run(values)
        except Exception:
            # A test that cannot run on this sample -- too few points, or the
            # wrong support -- is reported as n/a rather than failing the panel.
            rows.append((label, "n/a", "n/a", "n/a"))
            continue

        if not np.isfinite(p_value):
            # No p-value means no verdict; never report "fail to reject" here.
            rows.append((label, format_number(statistic), "n/a", "n/a"))
            continue

        verdict = "reject" if p_value < ALPHA else "fail to reject"
        rows.append((label, format_number(statistic), _format_p(p_value), verdict))
    return rows


def simple_table(
    x: Any,
    depth: int = 1,
    title: str | None = None,
) -> Axes:
    """Tabulate a summary of a single vector, optionally with distribution tests.

    Parameters
    ----------
    x : array-like
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    depth : {1, 2}, default 1
        ``1`` reports central tendency and spread: n, mean, median, standard
        deviation, variance, min/max, quartiles, IQR, skewness, and excess
        kurtosis. ``2`` keeps all of that and adds a panel of twelve
        distribution tests -- normality tests plus fits against the
        lognormal, exponential, uniform, and logistic families. Each test
        names the family under its null hypothesis, so ``reject`` means the
        data is not that distribution at ``alpha = 0.05``.
    title : str, optional
        Table title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the table was drawn on.

    Notes
    -----
    Anything that cannot be computed for the sample -- a test needing more
    points than it has, or values outside a family's support -- is reported
    as ``n/a`` rather than failing the whole table. The goodness-of-fit tests
    estimate their parameters from the same sample they test, which makes
    their p-values conservative; read them as a screen, not as exact
    significance.

    Examples
    --------
    >>> import numpy as np
    >>> axes = simple_table(np.random.default_rng(0).normal(size=200))
    >>> with_tests = simple_table(np.random.default_rng(0).normal(size=200), depth=2)
    """
    if depth not in _DEPTHS:
        raise ValueError(f"depth must be one of {list(_DEPTHS)}, got {depth!r}")

    values = coerce_to_1d_array(x)

    columns = _DEPTH_1_COLUMNS if depth == 1 else _DEPTH_2_COLUMNS
    widths = _DEPTH_1_WIDTHS if depth == 1 else _DEPTH_2_WIDTHS

    rows = _descriptive_rows(values, len(columns))
    section_rows: set[int] = set()
    if depth == 2:
        # +1 for the column-label row above the body.
        section_rows.add(len(rows) + 1)
        rows = [*rows, ("Distribution tests",) + ("",) * (len(columns) - 1), *_test_rows(values)]

    figure, axes = plt.subplots(figsize=(8, _ROW_HEIGHT * (len(rows) + 1)))

    table = axes.table(
        cellText=[list(row) for row in rows],
        colLabels=list(columns),
        colWidths=list(widths),
        cellLoc="left",
        # Fill the axes exactly, rather than floating in the middle of it.
        bbox=Bbox.from_bounds(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor(MUTED_COLOR)
        cell.set_linewidth(0.8)
        if row == 0 or row in section_rows:
            cell.set_facecolor(_HEADER_COLOR)
            cell.set_text_props(color=INK_COLOR, fontweight="bold")
        else:
            cell.set_facecolor(BACKGROUND_COLOR)
            cell.set_text_props(color=INK_COLOR)

    apply_simple_style(axes)
    axes.axis("off")

    if title is not None:
        axes.set_title(title)

    figure.tight_layout()

    return axes
