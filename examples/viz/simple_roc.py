from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from probably.viz import simple_roc

iris = np.genfromtxt(
    Path(__file__).parent.parent / "data" / "iris.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
assets = Path(__file__).parent.parent / "assets"

features = np.column_stack(
    [iris["sepal_length"], iris["sepal_width"], iris["petal_length"], iris["petal_width"]]
)
is_virginica = (iris["species"] == "virginica").astype(int)

x_train, x_test, y_train, y_test = train_test_split(
    features, is_virginica, test_size=0.4, random_state=0, stratify=is_virginica
)
logistic = LogisticRegression(max_iter=1000).fit(x_train, y_train)
tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x_train, y_train)

simple_roc(
    y_test,
    logistic.predict_proba(x_test)[:, 1],
    title="Virginica vs. rest",
    filepath=assets / "simple_roc.png",
)

simple_roc(
    {
        "logistic": (y_test, logistic.predict_proba(x_test)[:, 1]),
        "depth-1 tree": (y_test, tree.predict_proba(x_test)[:, 1]),
    },
    title="Virginica vs. rest, by model",
    filepath=assets / "simple_roc_compared.png",
)
