"""Demonstrate `probably.viz.simple_roc` on the iris dataset.

Trains models to separate virginica from the other two species, then plots
the ROC curve of their held-out scores. Passing several `(y_true, y_score)`
pairs keyed by model name overlays one curve each, with every AUC in the
legend. Each pair carries its own labels, so the models compared here sit on
different train/test splits.

Run with:

    python examples/viz/simple_roc.py

Writes examples/assets/simple_roc.png and examples/assets/simple_roc_compared.png.
"""

import csv
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from probably.viz import simple_roc

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
logistic = LogisticRegression(max_iter=1000).fit(x_train, y_train)
shallow_tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x_train, y_train)

assets = Path(__file__).parent.parent / "assets"
assets.mkdir(parents=True, exist_ok=True)

# A single model: the AUC goes in the caption.
axes = simple_roc(
    y_test,
    logistic.predict_proba(x_test)[:, 1],
    title="Virginica vs. rest",
)
axes.figure.savefig(assets / "simple_roc.png", bbox_inches="tight")
print(f"wrote {assets / 'simple_roc.png'}")

# A second model on its own, different split -- each pair brings its own labels.
other_x_train, other_x_test, other_y_train, other_y_test = train_test_split(
    features, is_virginica, test_size=0.4, random_state=7, stratify=is_virginica
)
other_tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(other_x_train, other_y_train)

compared_axes = simple_roc(
    {
        "logistic (split 0)": (y_test, logistic.predict_proba(x_test)[:, 1]),
        "depth-1 tree (split 0)": (y_test, shallow_tree.predict_proba(x_test)[:, 1]),
        "depth-1 tree (split 7)": (
            other_y_test,
            other_tree.predict_proba(other_x_test)[:, 1],
        ),
    },
    title="Virginica vs. rest, by model",
)
compared_axes.figure.savefig(assets / "simple_roc_compared.png", bbox_inches="tight")
print(f"wrote {assets / 'simple_roc_compared.png'}")
