from pathlib import Path

import numpy as np

from probably.viz import simple_pp_normal

iris = np.genfromtxt(
    Path(__file__).parent.parent / "data" / "iris.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
assets = Path(__file__).parent.parent / "assets"

simple_pp_normal(
    iris["sepal_width"],
    title="Iris sepal width vs. a fitted normal",
    filepath=assets / "simple_pp_normal.png",
)
