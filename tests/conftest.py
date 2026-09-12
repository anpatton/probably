import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close figures after each test so the suite does not leak them."""
    yield
    plt.close("all")
