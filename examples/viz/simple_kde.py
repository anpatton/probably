"""Demonstrate `probably.viz.simple_kde` on the iris dataset.

Passing several series instead of one vector draws an overlapping density
curve for each, which pulls the pooled bimodal shape apart into per-species
peaks. A mapping is used here so the legend is labelled by species.

Run with:

    python examples/viz/simple_kde.py

Writes examples/assets/simple_kde.png and examples/assets/simple_kde_grouped.png.
"""

import csv
from pathlib import Path

from probably.viz import simple_kde

data_path = Path(__file__).parent.parent / "data" / "iris.csv"
with data_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

petal_lengths = [float(row["petal_length"]) for row in rows]

by_species: dict[str, list[float]] = {}
for row in rows:
    by_species.setdefault(row["species"], []).append(float(row["petal_length"]))

assets = Path(__file__).parent.parent / "assets"
assets.mkdir(parents=True, exist_ok=True)

# Pooled across every species.
axes = simple_kde(
    petal_lengths,
    xlabel="Petal Length (cm)",
    title="Iris petal length density",
    filepath=assets / "simple_kde.png",
)
print(f"wrote {assets / 'simple_kde.png'}")

# One overlapping curve per species.
grouped_axes = simple_kde(
    by_species,
    xlabel="Petal Length (cm)",
    title="Iris petal length density by species",
    filepath=assets / "simple_kde_grouped.png",
)
print(f"wrote {assets / 'simple_kde_grouped.png'}")
