"""Demonstrate `probably.viz.simple_kde` on the iris dataset.

Run with:

    python examples/viz/simple_kde.py

Writes examples/assets/simple_kde.png.
"""

import csv
from pathlib import Path

from probably.viz import simple_kde

data_path = Path(__file__).parent.parent / "data" / "iris.csv"
with data_path.open(newline="") as f:
    petal_lengths = [float(row["petal_length"]) for row in csv.DictReader(f)]

axes = simple_kde(petal_lengths, xlabel="petal length (cm)", title="Iris petal length density")

output_path = Path(__file__).parent.parent / "assets" / "simple_kde.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
axes.figure.savefig(output_path, bbox_inches="tight")
print(f"wrote {output_path}")
