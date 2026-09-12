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

by_species = {
    name: iris["petal_length"][iris["species"] == name] for name in np.unique(iris["species"])
}

simple_ecdf(
    iris["petal_length"],
    xlabel="Petal Length (cm)",
    title="Iris petal length",
    filepath=assets / "simple_ecdf.png",
)

simple_ecdf(
    by_species,
    xlabel="Petal Length (cm)",
    title="Iris petal length by species",
    filepath=assets / "simple_ecdf_grouped.png",
)
