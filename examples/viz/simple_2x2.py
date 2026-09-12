"""Demonstrate `probably.viz.simple_2x2` on the iris dataset.

Trains a logistic regression to separate virginica from the other two
species, then plots the confusion matrix of its held-out predictions.

Run with:

    python examples/viz/simple_2x2.py

Writes examples/assets/simple_2x2.png.
"""

import csv
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from probably.viz import simple_2x2

data_path = Path(__file__).parent.parent / "data" / "iris.csv"
with data_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

features = [
    [float(row[name]) for name in ("sepal_length", "sepal_width", "petal_length", "petal_width")]
    for row in rows
]
is_virginica = [int(row["species"] == "virginica") for row in rows]

x_train, x_test, y_train, y_test = train_test_split(
    features, is_virginica, test_size=0.4, random_state=0, stratify=is_virginica
)
model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
predictions = model.predict(x_test)

axes = simple_2x2(y_test, predictions, title="Virginica vs. rest")

output_path = Path(__file__).parent.parent / "assets" / "simple_2x2.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
axes.figure.savefig(output_path, bbox_inches="tight")
print(f"wrote {output_path}")
