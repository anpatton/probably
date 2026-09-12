import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from probably.viz import simple_scatter

matplotlib.use("Agg")


def test_returns_axes():
    axes = simple_scatter([1, 2, 3], [4, 5, 6])
    assert isinstance(axes, Axes)


def test_accepts_numpy_arrays():
    axes = simple_scatter(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert len(axes.collections) > 0


def test_sets_labels_and_title():
    axes = simple_scatter([1, 2], [3, 4], xlabel="x", ylabel="y", title="my scatter")
    assert axes.get_xlabel() == "x"
    assert axes.get_ylabel() == "y"
    assert axes.get_title() == "my scatter"


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        simple_scatter([1, 2, 3], [1, 2])


def test_rejects_unknown_line():
    with pytest.raises(ValueError):
        simple_scatter([1, 2, 3], [1, 2, 3], line="bogus")


@pytest.mark.parametrize("line", ["ab", "lm", "loess"])
def test_line_options_draw_a_black_line_and_caption(line):
    rng = np.random.default_rng(0)
    axes = simple_scatter(rng.normal(size=30), rng.normal(size=30), line=line)
    lines = axes.get_lines()
    assert len(lines) == 1
    assert lines[0].get_color() == "black"
    caption_texts = [t.get_text() for t in axes.figure.texts]
    assert any(caption.startswith("Black line:") for caption in caption_texts)


def test_rejects_mismatched_color_by_length():
    with pytest.raises(ValueError):
        simple_scatter([1, 2, 3], [1, 2, 3], color_by=["a", "b"])


def test_color_by_categorical_uses_palette_and_legend():
    axes = simple_scatter([1, 2, 3, 4], [1, 2, 3, 4], color_by=["a", "b", "a", "b"])
    # one scatter collection per category, plus a legend
    assert len(axes.collections) == 2
    assert axes.get_legend() is not None
    colors = {tuple(c.get_facecolor()[0]) for c in axes.collections}
    assert len(colors) == 2


def test_color_by_continuous_uses_sequential_ramp_and_colorbar():
    rng = np.random.default_rng(0)
    x = rng.normal(size=30)
    y = rng.normal(size=30)
    values = rng.normal(size=30)
    axes = simple_scatter(x, y, color_by=values)
    assert len(axes.collections) == 1
    assert axes.collections[0].get_cmap().name == "probably_retro"
    assert len(axes.figure.axes) == 2  # main axes + colorbar axes


def test_caption_sits_clear_of_the_x_axis_label():
    # the caption used to be pinned to the figure bottom, landing on the label
    rng = np.random.default_rng(0)
    axes = simple_scatter(
        rng.normal(size=30), rng.normal(size=30), line="lm", xlabel="Petal Length (cm)"
    )
    figure = axes.get_figure(root=True)
    figure.canvas.draw()

    captions = [t for t in figure.texts if "Black line" in t.get_text()]
    assert len(captions) == 1
    caption_top = captions[0].get_window_extent().y1
    label_bottom = axes.xaxis.label.get_window_extent().y0
    assert caption_top < label_bottom
