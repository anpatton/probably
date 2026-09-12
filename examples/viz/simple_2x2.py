from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from probably.viz import simple_2x2

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
model = LogisticRegression(max_iter=1000).fit(x_train, y_train)

simple_2x2(
    y_test,
    model.predict(x_test),
    title="Virginica vs. rest",
    filepath=assets / "simple_2x2.png",
)
