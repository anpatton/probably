"""Demonstrate `probably.viz.simple_kde_2d` on the iris dataset.

Run with:

    python examples/viz/simple_kde_2d.py

Writes examples/assets/simple_kde_2d.png.
"""

import csv
from pathlib import Path

from probably.viz import simple_kde_2d

data_path = Path(__file__).parent.parent / "data" / "iris.csv"
with data_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

petal_length = [float(row["petal_length"]) for row in rows]
petal_width = [float(row["petal_width"]) for row in rows]

axes = simple_kde_2d(
    petal_length,
    petal_width,
    xlabel="petal length (cm)",
    ylabel="petal width (cm)",
    title="Iris petal dimensions joint density",
)

output_path = Path(__file__).parent.parent / "assets" / "simple_kde_2d.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
axes.figure.savefig(output_path, bbox_inches="tight")
print(f"wrote {output_path}")
