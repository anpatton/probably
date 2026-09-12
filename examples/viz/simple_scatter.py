from pathlib import Path

import numpy as np

from probably.viz import simple_scatter

iris = np.genfromtxt(
    Path(__file__).parent.parent / "data" / "iris.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
assets = Path(__file__).parent.parent / "assets"

simple_scatter(
    iris["petal_length"],
    iris["petal_width"],
    line="lm",
    color_by=iris["species"],
    xlabel="Petal Length (cm)",
    ylabel="Petal Width (cm)",
    title="Iris petal length vs. width",
    filepath=assets / "simple_scatter.png",
)
