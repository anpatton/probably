import numpy as np

from probably.dx import is_this_normal

rng = np.random.default_rng(0)
data = {"normal": rng.normal(size=300), "exponential": rng.exponential(size=300)}

for result in is_this_normal(data):
    print(result["name"], result["verdict"])
