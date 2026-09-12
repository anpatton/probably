import numpy as np

from probably.dx import what_is_this

data = np.random.default_rng(0).lognormal(0, 0.5, size=300)

distribution, diagnostics = what_is_this(data)

print(distribution.dist.name, distribution.kwds or distribution.args)

for row in diagnostics:
    print(row)
