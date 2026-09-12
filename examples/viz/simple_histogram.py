"""Demonstrate `probably.viz.simple_histogram` on the iris dataset.

Run with:

    python examples/viz/simple_histogram.py

Writes examples/assets/simple_histogram.png.
"""

import csv
from pathlib import Path

from probably.viz import simple_histogram

data_path = Path(__file__).parent.parent / "data" / "iris.csv"
with data_path.open(newline="") as f:
    petal_lengths = [float(row["petal_length"]) for row in csv.DictReader(f)]

output_path = Path(__file__).parent.parent / "assets" / "simple_histogram.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

axes = simple_histogram(
    petal_lengths,
    bins=20,
    xlabel="Petal Length (cm)",
    title="Iris petal length",
    filepath=output_path,
)

print(f"wrote {output_path}")
