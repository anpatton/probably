import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from probably.predict import ProbablyConformalRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(10_000, 1))
y = 3 * X[:, 0] + rng.normal(scale=2, size=10_000)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = ProbablyConformalRegressor(RandomForestRegressor(random_state=0), coverage=0.9)
model.fit(X_train, y_train)

low, high = model.predict_interval(X_test)

print("interval half-width:", model.quantile_)
print("fraction of test values inside their interval:", np.mean((low <= y_test) & (y_test <= high)))
