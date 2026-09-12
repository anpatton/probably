"""Demonstrate `probably.viz.simple_scatter` on the iris dataset.

Run with:

    python examples/viz/simple_scatter.py

Writes examples/assets/simple_scatter.png.
"""

import csv
from pathlib import Path

from probably.viz import simple_scatter

data_path = Path(__file__).parent.parent / "data" / "iris.csv"
with data_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

petal_length = [float(row["petal_length"]) for row in rows]
petal_width = [float(row["petal_width"]) for row in rows]
species = [row["species"] for row in rows]

axes = simple_scatter(
    petal_length,
    petal_width,
    line="lm",
    color_by=species,
    xlabel="Petal Length (cm)",
    ylabel="Petal Width (cm)",
    title="Iris petal dimensions",
)

output_path = Path(__file__).parent.parent / "assets" / "simple_scatter.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
axes.figure.savefig(output_path, bbox_inches="tight")
print(f"wrote {output_path}")
