"""Generate the `probably` project logo.

A series of beta distributions with peaks shifting left to right and
decreasing height, in rainbow colors — a small nod to the package's focus
on probability distributions.

Run with:

    python examples/logo.py

Writes examples/assets/logo.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from scipy import stats

BETA_PARAMS = [
    (2, 8),
    (3, 6),
    (4, 4),
    (5, 3.5),
    (7, 3),
    (9, 2.5),
    (12, 2),
]

x = np.linspace(0, 1, 400)
x_plot = 0.5 + (x - 0.5) * 0.75
colors = colormaps["rainbow"](np.linspace(1, 0, len(BETA_PARAMS)))
heights = np.linspace(0.6, 0.09, len(BETA_PARAMS))

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

for (a, b), color, height in zip(BETA_PARAMS, colors, heights, strict=True):
    pdf = stats.beta.pdf(x, a, b)
    y = pdf / pdf.max() * height
    ax.plot(x_plot, y, color=color, linewidth=3, solid_capstyle="round")
    ax.fill_between(x_plot, y, color=color, alpha=0.15)

ax.text(
    0.5,
    -0.12,
    "Pr(obably)",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=68,
    fontweight="bold",
    color="#0b0b0b",
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.tight_layout(pad=0.2)

output_path = Path(__file__).parent / "assets" / "logo.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, transparent=True, bbox_inches="tight")
plt.close(fig)
print(f"wrote {output_path}")
