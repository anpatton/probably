import numpy as np

from probably.dx import is_this_different

rng = np.random.default_rng(0)
control = rng.normal(size=300)
treatments = {"low": rng.normal(0.1, size=300), "high": rng.normal(2, size=300)}

for result in is_this_different(control, treatments):
    print(result["name"], result["verdict"], result["prob_greater"], result["overlap"])
