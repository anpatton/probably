from pathlib import Path

import numpy as np

from probably.viz import simple_histogram

iris = np.genfromtxt(
    Path(__file__).parent.parent / "data" / "iris.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
assets = Path(__file__).parent.parent / "assets"

simple_histogram(
    iris["petal_length"],
    bins=20,
    xlabel="Petal Length (cm)",
    title="Iris petal length",
    filepath=assets / "simple_histogram.png",
)
