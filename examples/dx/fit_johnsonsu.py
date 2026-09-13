from scipy import stats

from probably.dx import fit_johnsonsu

data = stats.johnsonsu.rvs(-2, 2, size=300, random_state=0)

distribution, diagnostics = fit_johnsonsu(data)

print(distribution.dist.name, distribution.args)

for row in diagnostics:
    print(row)

print(fit_johnsonsu(data, quick=True).diagnostics[0])
