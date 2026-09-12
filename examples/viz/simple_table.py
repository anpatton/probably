"""Demonstrate `probably.viz.simple_table` on the iris dataset.

Depth 1 summarises where the data sits and how far it spreads. Depth 2 keeps
that and adds the distribution-test panel, which flags iris petal length as
decidedly non-normal -- it is the pooled mixture of three species.

Run with:

    python examples/viz/simple_table.py

Writes examples/assets/simple_table.png and examples/assets/simple_table_depth2.png.
"""

import csv
from pathlib import Path

from probably.viz import simple_table

data_path = Path(__file__).parent.parent / "data" / "iris.csv"
with data_path.open(newline="") as f:
    petal_lengths = [float(row["petal_length"]) for row in csv.DictReader(f)]

assets = Path(__file__).parent.parent / "assets"
assets.mkdir(parents=True, exist_ok=True)

# Depth 1: central tendency and spread.
axes = simple_table(petal_lengths, title="Iris petal length")
axes.figure.savefig(assets / "simple_table.png", bbox_inches="tight")
print(f"wrote {assets / 'simple_table.png'}")

# Depth 2: the same, plus the distribution tests.
deeper_axes = simple_table(
    petal_lengths,
    depth=2,
    title="Iris petal length, with distribution tests",
)
deeper_axes.figure.savefig(assets / "simple_table_depth2.png", bbox_inches="tight")
print(f"wrote {assets / 'simple_table_depth2.png'}")
