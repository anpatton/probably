import numpy as np

from probably.dx import describe

data = np.random.default_rng(0).normal(size=300)

print(describe(data))
print(describe(data, depth=2))
