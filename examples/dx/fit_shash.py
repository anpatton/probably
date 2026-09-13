from scipy import stats

from probably.dx import fit_shash

data = stats.skewnorm.rvs(4, size=300, random_state=0)

distribution, diagnostics = fit_shash(data)

print(distribution.dist.name, distribution.args)

for row in diagnostics:
    print(row)

print(fit_shash(data, quick=True).diagnostics[0])
