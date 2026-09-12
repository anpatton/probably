"""Demonstrate `probably.viz.simple_bar` on the iris dataset.

Run with:

    python examples/viz/simple_bar.py

Writes examples/assets/simple_bar.png.
"""

import csv
from collections import Counter
from pathlib import Path

from probably.viz import simple_bar

data_path = Path(__file__).parent.parent / "data" / "iris.csv"
with data_path.open(newline="") as f:
    species = [row["species"] for row in csv.DictReader(f)]

counts = Counter(species)

axes = simple_bar(
    list(counts.keys()), list(counts.values()), xlabel="Count", title="Iris species counts"
)

output_path = Path(__file__).parent.parent / "assets" / "simple_bar.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
axes.figure.savefig(output_path, bbox_inches="tight")
print(f"wrote {output_path}")
