from pathlib import Path

import numpy as np

from probably.viz import simple_kde

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

simple_kde(
    iris["petal_length"],
    xlabel="Petal Length (cm)",
    title="Iris petal length density",
    filepath=assets / "simple_kde.png",
)

simple_kde(
    by_species,
    xlabel="Petal Length (cm)",
    title="Iris petal length density by species",
    filepath=assets / "simple_kde_grouped.png",
)
