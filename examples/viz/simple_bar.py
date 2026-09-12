from pathlib import Path

import numpy as np

from probably.viz import simple_bar

iris = np.genfromtxt(
    Path(__file__).parent.parent / "data" / "iris.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
assets = Path(__file__).parent.parent / "assets"

species, counts = np.unique(iris["species"], return_counts=True)

simple_bar(
    species,
    counts,
    xlabel="Count",
    title="Iris species counts",
    filepath=assets / "simple_bar.png",
)
