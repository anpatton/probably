from pathlib import Path

import numpy as np

from probably.viz import simple_ecdf

iris = np.genfromtxt(
    Path(__file__).parent.parent / "data" / "iris.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
assets = Path(__file__).parent.parent / "assets"

simple_ecdf(
    iris["petal_length"],
    xlabel="Petal Length (cm)",
    title="Iris petal length",
    filepath=assets / "simple_ecdf.png",
)
