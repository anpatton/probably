import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from probably.viz import (
    simple_2x2,
    simple_bar,
    simple_ecdf,
    simple_histogram,
    simple_kde,
    simple_kde_2d,
    simple_pp_normal,
    simple_qq_normal,
    simple_roc,
    simple_scatter,
)
from probably.viz._utils import BACKGROUND_COLOR, DENSITY_COLORMAPS, SERIES_COLORS

matplotlib.use("Agg")


def _call(func, filepath=None):
    """Invoke each chart with its minimal valid arguments."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    calls = {
        simple_histogram: lambda: simple_histogram(x, filepath=filepath),
        simple_bar: lambda: simple_bar(["a", "b"], [3, 1], filepath=filepath),
        simple_kde: lambda: simple_kde(x, filepath=filepath),
        simple_kde_2d: lambda: simple_kde_2d(x, y, filepath=filepath),
        simple_scatter: lambda: simple_scatter(x, y, filepath=filepath),
        simple_roc: lambda: simple_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], filepath=filepath),
        simple_2x2: lambda: simple_2x2([0, 0, 1, 1], [0, 1, 1, 1], filepath=filepath),
        simple_qq_normal: lambda: simple_qq_normal(x, filepath=filepath),
        simple_pp_normal: lambda: simple_pp_normal(x, filepath=filepath),
        simple_ecdf: lambda: simple_ecdf(x, filepath=filepath),
    }
    return calls[func]()


ALL_CHARTS = [
    simple_histogram,
    simple_bar,
    simple_kde,
    simple_kde_2d,
    simple_scatter,
    simple_roc,
    simple_2x2,
    simple_qq_normal,
    simple_pp_normal,
    simple_ecdf,
]


@pytest.mark.parametrize("func", ALL_CHARTS)
@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg"])
def test_filepath_writes_an_image(func, suffix, tmp_path):
    target = tmp_path / f"chart{suffix}"
    axes = _call(func, filepath=target)
    assert isinstance(axes, Axes)
    assert target.exists()
    assert target.stat().st_size > 0


@pytest.mark.parametrize("func", ALL_CHARTS)
def test_filepath_is_optional(func, tmp_path):
    assert isinstance(_call(func), Axes)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("func", ALL_CHARTS)
@pytest.mark.parametrize("name", ["chart.pdf", "chart.svg", "chart.txt", "chart"])
def test_rejects_non_image_extensions(func, name, tmp_path):
    with pytest.raises(ValueError, match="filepath must end in"):
        _call(func, filepath=tmp_path / name)


@pytest.mark.parametrize("func", ALL_CHARTS)
def test_bad_extension_writes_nothing(func, tmp_path):
    with pytest.raises(ValueError):
        _call(func, filepath=tmp_path / "chart.pdf")
    assert list(tmp_path.iterdir()) == []


def test_filepath_accepts_a_plain_string(tmp_path):
    target = tmp_path / "chart.png"
    simple_histogram([1.0, 2.0, 3.0, 4.0], filepath=str(target))
    assert target.exists()


def test_uppercase_extension_is_accepted(tmp_path):
    target = tmp_path / "chart.PNG"
    simple_histogram([1.0, 2.0, 3.0, 4.0], filepath=target)
    assert target.exists()


@pytest.mark.parametrize("flag", ["r", "y", "b"])
def test_histogram_color_flag(flag):
    axes = simple_histogram([1.0, 2.0, 2.0, 3.0], color=flag)
    assert axes.patches[0].get_facecolor() == matplotlib.colors.to_rgba(SERIES_COLORS[flag])


@pytest.mark.parametrize("flag", ["r", "y", "b"])
def test_bar_color_flag(flag):
    axes = simple_bar(["a", "b"], [3, 1], color=flag)
    assert axes.patches[0].get_facecolor() == matplotlib.colors.to_rgba(SERIES_COLORS[flag])


@pytest.mark.parametrize("flag", ["r", "y", "b"])
def test_kde_2d_color_flag(flag):
    rng = np.random.default_rng(0)
    axes = simple_kde_2d(rng.normal(size=60), rng.normal(size=60), color=flag)
    assert axes.collections[0].get_cmap() is DENSITY_COLORMAPS[flag]


def test_kde_2d_ramps_are_distinct_per_flag():
    ramps = {flag: DENSITY_COLORMAPS[flag](0.8) for flag in ("r", "y", "b")}
    assert len(set(ramps.values())) == 3


def test_density_ramps_start_at_the_parchment_ground():
    # the low end must dissolve into the page, not wash it with flat colour
    for flag in ("r", "y", "b"):
        assert DENSITY_COLORMAPS[flag](0.0) == matplotlib.colors.to_rgba(BACKGROUND_COLOR)


def test_single_series_charts_default_to_different_colors():
    rng = np.random.default_rng(0)
    histogram = simple_histogram([1.0, 2.0, 2.0, 3.0])
    bar = simple_bar(["a", "b"], [3, 1])
    kde_2d = simple_kde_2d(rng.normal(size=60), rng.normal(size=60))

    assert histogram.patches[0].get_facecolor() == matplotlib.colors.to_rgba(SERIES_COLORS["r"])
    assert bar.patches[0].get_facecolor() == matplotlib.colors.to_rgba(SERIES_COLORS["b"])
    assert kde_2d.collections[0].get_cmap() is DENSITY_COLORMAPS["y"]


@pytest.mark.parametrize("func", [simple_qq_normal, simple_pp_normal])
@pytest.mark.parametrize("flag", ["r", "y", "b"])
def test_diagnostic_scatter_color_flag(func, flag):
    axes = func(np.random.default_rng(0).normal(size=40), color=flag)
    assert axes.collections[0].get_facecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba(SERIES_COLORS[flag], alpha=0.85)
    )


@pytest.mark.parametrize("flag", ["r", "y", "b"])
def test_ecdf_color_flag(flag):
    axes = simple_ecdf([1.0, 2.0, 3.0], color=flag)
    assert axes.lines[0].get_color() == SERIES_COLORS[flag]


def test_diagnostic_charts_default_to_different_colors():
    values = np.random.default_rng(0).normal(size=40)
    assert simple_qq_normal(values).collections[0].get_facecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba(SERIES_COLORS["r"], alpha=0.85)
    )
    assert simple_pp_normal(values).collections[0].get_facecolor()[0] == pytest.approx(
        matplotlib.colors.to_rgba(SERIES_COLORS["b"], alpha=0.85)
    )
    assert simple_ecdf(values).lines[0].get_color() == SERIES_COLORS["y"]


def test_rejects_unknown_color():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="color must be one of"):
        simple_histogram([1.0, 2.0, 3.0], color="green")
    for func in (simple_qq_normal, simple_pp_normal, simple_ecdf):
        with pytest.raises(ValueError, match="color must be one of"):
            func([1.0, 2.0, 3.0], color="green")
    with pytest.raises(ValueError, match="color must be one of"):
        simple_bar(["a", "b"], [3, 1], color="green")
    with pytest.raises(ValueError, match="color must be one of"):
        simple_kde_2d(rng.normal(size=60), rng.normal(size=60), color="green")
